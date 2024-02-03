# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import random
from typing import Tuple
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.utils.events import get_event_storage
from .modeling.cluster2former_criterion import VideoSetCluster2FormerCriterion
from . import VideoMaskFormer
from .utils.memory import retry_if_cuda_oom


logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoCluster2Former(VideoMaskFormer):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        cluster_eval: bool,
        # video
        num_frames,
        vis_period: int,
        vis_conf_threshold: int,
        test_inference_threshold: float,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            cluster_eval: wheter do cluster panoptic inference or not
            vis_period: int, visualize the gt and the prediction in tensorboard in every 'vis_period' 
                iteration
            vis_conf_threshold: int, confidence threshold for the instances (prediction) in 
                tensorboard visualization
            test_inference_threshold: thresholding the softmax in inference to generate mask
        """
        super().__init__(backbone = backbone,
        sem_seg_head = sem_seg_head,
        criterion = criterion,
        num_queries = num_queries,
        object_mask_threshold = object_mask_threshold,
        overlap_threshold = overlap_threshold,
        metadata = metadata,
        size_divisibility = size_divisibility,
        sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference,
        pixel_mean = pixel_mean,
        pixel_std = pixel_std,
        num_frames = num_frames,
        vis_period = vis_period,
        vis_conf_threshold = vis_conf_threshold)
        
        self.cluster_eval = cluster_eval
        self.test_inference_threshold = test_inference_threshold

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        cluster_weight = cfg.MODEL.CLUSTER_2_FORMER.CLUSTER_WEIGHT

        weight_dict = {"loss_ce": class_weight, "loss_cluster": cluster_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if cfg.INPUT.DATASET_MAPPER_NAME == 'mask_former_scribble':
            losses = ["clusters_labels_scribble"]
        else:
            losses = ["clusters_labels"]
        
        cluster_eval = False
        
        if "clusters_labels" in losses or "clusters_labels_scribble" in losses:
            cluster_eval = True

        criterion = VideoSetCluster2FormerCriterion(
            sem_seg_head.num_classes,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            total_instance_pixel_sample_num=cfg.MODEL.CLUSTER_2_FORMER.TOTAL_INSTANCE_PIXEL_SAMPLE_NUM,
            min_num_sample=cfg.MODEL.CLUSTER_2_FORMER.MIN_NUM_SAMPLE,
            max_num_sample=cfg.MODEL.CLUSTER_2_FORMER.MAX_NUM_SAMPLE,
            max_inst=cfg.MODEL.CLUSTER_2_FORMER.MAX_INST,
            beta=cfg.MODEL.CLUSTER_2_FORMER.BETA,
            delta=cfg.MODEL.CLUSTER_2_FORMER.DELTA,
            make_inter_frame_bg_points_connections=cfg.MODEL.CLUSTER_2_FORMER.MAKE_INTER_FRAME_BG_POINTS_CONNECTIONS,
            make_inter_frame_point_connections=cfg.MODEL.CLUSTER_2_FORMER.MAKE_INTER_FRAME_POINT_CONNECTIONS,
            min_point_pair_weight=cfg.MODEL.CLUSTER_2_FORMER.MIN_POINT_PAIR_WEIGHT,
            make_positive_bg_pairs=cfg.MODEL.CLUSTER_2_FORMER.MAKE_POSITIVE_BG_PAIRS,
            cos_sim_clustering_loss=cfg.MODEL.CLUSTER_2_FORMER.COS_SIM_CLUSTERING_LOSS,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cluster_eval": cluster_eval,
            "test_inference_threshold": cfg.MODEL.CLUSTER_2_FORMER.TEST.INFERENCE_THRESHOLD,
            # tensorboard visualization
            "vis_period": cfg.TB_VISUALIZATION.VIS_PERIOD,
            "vis_conf_threshold": cfg.TB_VISUALIZATION.VIS_CONF_THRESHOLD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
        }
        
    def tensorboard_visualization(self, batched_inputs, proposals, image_size, size0, size1):
        """
        A function used to visualize frames with ground truth annotations and 
        the predictions in tensorboard.
        
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
            outputs: a dictionary that contains predicted outputs of the model.
                The dictionary contains:
                   * "pred_logits": Tensor in (B, Q, Class+1) format.
                   * "pred_masks": Tensor in (B, Q, T, H, W) format.
                batched_inputs and outputs should have the same length.
            image_size: a tuple of Integers, which defines the raw image size
                in (H_raw,W_raw) format
            size0: an Integer, which defines the height of the upsampled prediction
            size1: an Integer, which defines the width of the upsampled prediction
        """ 
        from demo.visualizer import VisualizerGT
        from demo_video.visualizer import TrackVisualizer
        from detectron2.utils.visualizer import ColorMode
        
        storage = get_event_storage()

        mask_cls_results = proposals["pred_logits"]
        mask_pred_results = proposals["pred_masks"]

        mask_cls_result = mask_cls_results[0]
        # upsample masks
        mask_pred_result = retry_if_cuda_oom(F.interpolate)(
            mask_pred_results[0],
            size=(size0, size1),
            mode="bilinear",
            align_corners=False,
        )
        
        del proposals

        input_per_video = batched_inputs[0] 

        height = input_per_video.get("height", image_size[0])  # raw image size before data augmentation
        width = input_per_video.get("width", image_size[1])
        
        predictions = retry_if_cuda_oom(self.inference_video_cluster)(mask_cls_result, mask_pred_result, image_size, height, width)
        thresholded_idxs = np.array(predictions["pred_scores"]) >= self.vis_conf_threshold
        
        image_size = predictions["image_size"]
        pred_scores = [predictions["pred_scores"][idx] for idx, v in enumerate(thresholded_idxs) if v]
        pred_labels = [predictions["pred_labels"][idx] for idx, v in enumerate(thresholded_idxs) if v]
        pred_masks = [predictions["pred_masks"][idx] for idx, v in enumerate(thresholded_idxs) if v]

        frames = input_per_video['image']
        frame_masks = list(zip(*pred_masks))
        gt_vis_output = []
        pred_vis_output = []
        for frame_idx in range(len(frames)):
            # gt
            img = input_per_video["image"][frame_idx]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            if self.training:
                visualizer = VisualizerGT(img, self.metadata)
                v_gt = visualizer.draw_gt_instances(gt=input_per_video["instances"][frame_idx].to(torch.device("cpu")))
                gt_vis_output.append(v_gt.get_image())
            
            visualizer = TrackVisualizer(img, self.metadata, instance_mode=ColorMode.IMAGE)
            ins = Instances(image_size)

            # visualize confident predictions only
            if len(pred_scores) > 0:
                ins.scores = pred_scores
                ins.pred_classes = pred_labels
                pred_masks = torch.stack(frame_masks[frame_idx], dim=0)
                ins.pred_masks = retry_if_cuda_oom(F.interpolate)(
                    pred_masks[None].to(torch.float32),
                    size=(img.shape[0],img.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                
                v_pred = visualizer.draw_instance_predictions(predictions=ins)
                pred_vis_output.append(v_pred.get_image())
                
        if self.training:
            gt_vis_concat = np.concatenate(gt_vis_output, axis=1)
        
        if len(pred_scores) > 0:
            pred_vis_concat = np.concatenate(pred_vis_output, axis=1)
            if self.training:
                vis_img = np.concatenate([gt_vis_concat,pred_vis_concat], axis=0)
                vis_name = "Training - Top: GT masks;  Bottom: Predicted outputs"
            else:
                vis_img = pred_vis_concat
                vis_name = "Evaluation - Predicted outputs"
            
            vis_img = vis_img.transpose(2, 0, 1)
            storage.put_image(vis_name, vis_img)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
    
        features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                    
            size0 = images.tensor.shape[-2]
            size1 = images.tensor.shape[-1]
            
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.tensorboard_visualization(batched_inputs, outputs, images.image_sizes[0], size0, size1)
            
            if self.val_img_num != 0:
                self.to_count_val_img_num = False
                self.rand_img_num = random.randint(1,self.val_img_num) 
                self.val_img_num_act = 0 
            return losses
        else:
            # counting the images in the validation set
            if self.to_count_val_img_num:
                self.val_img_num += 1
            else:
                self.val_img_num_act += 1
                
            size0 = images.tensor.shape[-2]
            size1 = images.tensor.shape[-1]
                
            # visualize the prediction in a randomly picked validation img 
            if self.val_img_num_act == self.rand_img_num and self.val_img_num_act != 0:
                self.tensorboard_visualization(batched_inputs, outputs, images.image_sizes[0], size0, size1)
                
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            # upsample masks
            mask_pred_result = retry_if_cuda_oom(F.interpolate)(
                mask_pred_results[0],
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            del outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            if not self.cluster_eval:
                return retry_if_cuda_oom(self.inference_video)(mask_cls_result, mask_pred_result, image_size, height, width)
            else:
                return retry_if_cuda_oom(self.inference_video_cluster)(mask_cls_result, mask_pred_result, image_size, height, width)
    
    def inference_video_cluster(self, pred_cls, pred_masks, img_size, output_height, output_width):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            
            pred_masks = F.softmax(pred_masks, dim=0)
            pred_masks = pred_masks[topk_indices]
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]

            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            
            masks = pred_masks > self.test_inference_threshold

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
