# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Tuple

from torch import nn

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from .modeling.maskcluster2former_criterion import VideoSetMaskCluster2FormerCriterion
from .modeling.matcher import VideoHungarianMatcher
from . import VideoMaskFormer

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class VideoMaskCluster2Former(VideoMaskFormer):
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
        # video
        num_frames,
        vis_period: int,
        vis_conf_threshold: int
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
            vis_period: int, visualize the gt and the prediction in tensorboard in every 'vis_period' 
                iteration
            vis_conf_threshold: int, confidence threshold for the instances (prediction) in 
                tensorboard visualization
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
        # video
        num_frames = num_frames,
        vis_period = vis_period,
        vis_conf_threshold = vis_conf_threshold)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        cluster_weight = cfg.MODEL.CLUSTER_2_FORMER.CLUSTER_WEIGHT
        
        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, "loss_cluster": cluster_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks", "clusters"]

        criterion = VideoSetMaskCluster2FormerCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
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
            # tensorboard visualization
            "vis_period": cfg.TB_VISUALIZATION.VIS_PERIOD,
            "vis_conf_threshold": cfg.TB_VISUALIZATION.VIS_CONF_THRESHOLD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
        }