# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Adrián Szlatincsán from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskCluster2Former criterion.
"""

import torch
import math

from detectron2.utils.events import get_event_storage
from detectron2.projects.point_rend.point_features import (
    point_sample,
)

from .criterion import VideoSetCriterion
from mask2former.modeling.cluster2former_criterion import ccl_loss_jit, cos_sim_clustering_loss_jit

class VideoSetMaskCluster2FormerCriterion(VideoSetCriterion):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box) 
        and self-supervise the selected pixel pairs (clustering)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, total_instance_pixel_sample_num, 
                 min_num_sample, max_num_sample, max_inst, beta, delta, 
                 make_inter_frame_bg_points_connections, make_inter_frame_point_connections, 
                 min_point_pair_weight,make_positive_bg_pairs,cos_sim_clustering_loss):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            total_instance_pixel_sample_num: total number of pixels to sample from the instances in a frame,
            sample the same amount from the background, if <= 0, then use min_num_sample and max_num_sample
            min_num_sample: if total_instance_pixel_sample_num <= 0, minimum number of points to sample from each object and from the background
            max_num_sample: if total_instance_pixel_sample_num <= 0, maximum number of points to sample from each object and from the background
            max_inst: maximum number of instances to cluster
            beta: parameter for the point sampling equation for clustering (to see more, check the equation)
            delta: parameter for the point sampling equation for clustering (to see more, check the equation)
            make_inter_frame_bg_points_connections: whether to make point pair connections for clustering using 
            background points from an other frame or not
            make_inter_frame_point_connections: whether to make inter frame (temporal) point pair connections 
            for clustering or not
            min_point_pair_weight: the minimum value by which the loss is multiplied for each point pairs
            make_positive_bg_pairs: whether to make connections between background points or not 
            (connections will be positive)
            cos_sim_clustering_loss: If True use cos_similarity_clustering loss, else use ccl_loss
        """
        super().__init__(num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio)
        # for clustering
        self.total_instance_pixel_sample_num = total_instance_pixel_sample_num
        self.min_num_sample = min_num_sample
        self.max_num_sample = max_num_sample
        self.max_inst = max_inst
        self.beta = beta
        self.delta = delta
        self.make_inter_frame_bg_points_connections = make_inter_frame_bg_points_connections
        self.make_inter_frame_point_connections = make_inter_frame_point_connections
        self.min_point_pair_weight = min_point_pair_weight
        self.make_positive_bg_pairs = make_positive_bg_pairs
        self.cos_sim_clustering_loss = cos_sim_clustering_loss
    
    def loss_clusters(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the cosine similiraty clustering loss or the ccl loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        
        target_masks = [t["masks"] for t in targets]
        src_masks = outputs['pred_masks'] # (B,Q,T,H_src,W_src)
        
        # initialize required tensors to calculate the loss for the batch
        b_pairs_in_frame = torch.empty((0,2),dtype=torch.int32,device=src_masks.device)
        b_pairs_inter_frames = torch.empty((0,2),dtype=torch.int32,device=src_masks.device)
        b_pairs_weights_in_frame = torch.empty((0),dtype=torch.float64,device=src_masks.device)
        b_pairs_weights_inter_frames = torch.empty((0),dtype=torch.float64,device=src_masks.device)
        frame_num = target_masks[0].shape[1]
        b_target_clusters = torch.empty((0),dtype=torch.int16,device=src_masks.device)
        b_src_clusters = torch.empty((0,src_masks.shape[1]),dtype=torch.float32,device=src_masks.device)
        comb_frame = torch.combinations(torch.arange(0,frame_num),with_replacement=True)
        
        diagonal_length = math.sqrt((target_masks[0].shape[-2])**2 + (target_masks[0].shape[-1])**2)
        mask_pixel_num = target_masks[0].shape[-1]*target_masks[0].shape[-2]
        b_pix_sel_size = 0
        
        # iterate through the batch
        for b_idx, target_mask, src_mask in zip(range(len(target_masks)), target_masks, src_masks):
            # target_mask (N,T,H,W)
            # src_mask (Q,T,H_src,W_src)
            
            point_coords_list = []
            
            with torch.no_grad():
                # get target cluster mask
                cluster_mask = torch.argmax(torch.cat((torch.zeros((1, target_mask.shape[1], target_mask.shape[2], target_mask.shape[3]), dtype=torch.int8).to(target_mask), target_mask.to(torch.int8)), 0), dim=0).float() # (T,H,W)
                
                pix_sel_infos = []
                pix_sel_frame_start_idx = 0
                bg_point_samples_num_list = []
                pix_sel_all_frames = torch.empty((0,2),dtype=torch.int64, device=src_masks.device)
                
                # iterate through the frames
                for frame_idx, cluster_frame_mask in enumerate(cluster_mask):
                    pix_sel = torch.empty((0,2),dtype=torch.int64, device=src_masks.device)
                    target_mask_frame = target_mask[:,frame_idx,:,:] # (N,H,W)

                    if self.total_instance_pixel_sample_num > 0:
                        instance_num = target_mask_frame.shape[0]
                        if instance_num <= 0:
                            instance_num = 1
                        point_num_per_instance = int(self.total_instance_pixel_sample_num / instance_num)
   
                        # select maximum self.max_inst instances
                        ins_idx = (torch.randperm(len(target_mask_frame),device=target_mask.device))[:self.max_inst]

                        # sampling pixel points from the instances
                        for m in target_mask_frame[ins_idx]:
                            whr = (m > 0).nonzero()
                            ins_point_samples_idx = torch.randperm(len(whr),device=target_mask.device)[:point_num_per_instance]
                            sampled = (whr[ins_point_samples_idx])
                            pix_sel = torch.cat((pix_sel,sampled))
                            del ins_point_samples_idx
                            del whr
                            del sampled
                        del ins_idx
                        
                        # sampling pixel point from the background
                        if target_mask_frame.shape[0] != 0:
                            whr = (cluster_frame_mask == 0).nonzero()
                            bg_point_samples_idx = torch.randperm(len(whr),device=target_mask.device)[:pix_sel.shape[0]]
                            bg_point_samples_num = len(bg_point_samples_idx)
                            bg_point_samples_num_list.append(bg_point_samples_num)
                            sampled = (whr[bg_point_samples_idx]) 
                            pix_sel = torch.cat((sampled,pix_sel))
                            del bg_point_samples_idx
                            del whr
                            del sampled
                        else:
                            bg_point_samples_num_list.append(0)
                    else:
                        # sampling pixel points from the background
                        whr = (cluster_frame_mask == 0).nonzero()
                        bg_point_samples_idx = torch.randperm(len(whr),device=target_mask.device)[:min(self.min_num_sample+round(100*len(whr)/mask_pixel_num),self.max_num_sample)]
                        bg_point_samples_num = len(bg_point_samples_idx)
                        bg_point_samples_num_list.append(bg_point_samples_num)
                        sampled = (whr[bg_point_samples_idx])  
                        pix_sel = torch.cat((pix_sel,sampled))
                        del bg_point_samples_idx
                        del whr
                        del sampled
                    
                        # select maximum self.max_inst instances
                        ins_idx = (torch.randperm(len(target_mask_frame),device=target_mask.device))[:self.max_inst]

                        # sampling pixel points from the instances
                        for m in target_mask_frame[ins_idx]:
                            whr = (m > 0).nonzero()
                            ins_point_samples_idx = torch.randperm(len(whr),device=target_mask.device)[:min(self.min_num_sample+round(100*len(whr)/mask_pixel_num),self.max_num_sample)]
                            sampled = (whr[ins_point_samples_idx])
                            pix_sel = torch.cat((pix_sel,sampled))
                            del ins_point_samples_idx
                            del whr
                            del sampled
                        del ins_idx

                    pix_sel_all_frames = torch.cat((pix_sel_all_frames,pix_sel))
                    pix_sel_infos.append({'pix_sel_size':pix_sel.shape[0],'start_idx':pix_sel_frame_start_idx})
                    pix_sel_frame_start_idx += pix_sel.shape[0]

                    # get target cluster labels for the selected pixel points
                    point_labels_clusters = cluster_frame_mask[pix_sel[:, 0], pix_sel[:, 1]].to(torch.int16)
                    del cluster_frame_mask
                    b_target_clusters = torch.cat((b_target_clusters,point_labels_clusters))
                    del point_labels_clusters
                    
                    # convert indices to coordinates
                    pix_sel = torch.fliplr(pix_sel)
                    pix_sel = pix_sel.float()
                    eps = 1
                    pix_sel/=torch.Tensor([target_mask.shape[-1]+eps, target_mask.shape[-2]+eps]).to(pix_sel)
                    point_coords = pix_sel[None]
                    point_coords_list.append(point_coords)

                    del pix_sel
                    del point_coords
                    
                point_pairs_in_frame = torch.empty((0,2),dtype=torch.int64,device=target_mask.device)
                point_pairs_inter_frames = torch.empty((0,2),dtype=torch.int64,device=target_mask.device)
                
                # making pixel-pair connections
                for frame_pair in comb_frame:
                    # inter frames
                    if (frame_pair[0] != frame_pair[1]) and self.make_inter_frame_point_connections:
                        if self.make_inter_frame_bg_points_connections:
                            # pairing background points with object points
                            # 1.frame bg <-> 2. frame fg
                            point_pairs_bg_obj = torch.cartesian_prod(torch.arange(pix_sel_infos[frame_pair[0]]['start_idx'], pix_sel_infos[frame_pair[0]]['start_idx']+bg_point_samples_num_list[frame_pair[0]], device=target_mask.device), torch.arange(pix_sel_infos[frame_pair[1]]['start_idx']+bg_point_samples_num_list[frame_pair[1]], pix_sel_infos[frame_pair[1]]['start_idx']+pix_sel_infos[frame_pair[1]]['pix_sel_size'], device=target_mask.device)) 
                            point_pairs_inter_frames = torch.cat((point_pairs_inter_frames,point_pairs_bg_obj))
                            # pairing background points with object points
                            # 2.frame bg <-> 1. frame fg
                            point_pairs_bg_obj = torch.cartesian_prod(torch.arange(pix_sel_infos[frame_pair[1]]['start_idx'], pix_sel_infos[frame_pair[1]]['start_idx']+bg_point_samples_num_list[frame_pair[1]], device=target_mask.device), torch.arange(pix_sel_infos[frame_pair[0]]['start_idx']+bg_point_samples_num_list[frame_pair[0]], pix_sel_infos[frame_pair[0]]['start_idx']+pix_sel_infos[frame_pair[0]]['pix_sel_size'], device=target_mask.device)) 
                            point_pairs_inter_frames = torch.cat((point_pairs_inter_frames,point_pairs_bg_obj))
                        if self.make_positive_bg_pairs:
                            # pairing background points with background points
                            # 1.frame bg <-> 2. frame bg
                            point_pairs_bg_bg = torch.cartesian_prod(torch.arange(pix_sel_infos[frame_pair[0]]['start_idx'], pix_sel_infos[frame_pair[0]]['start_idx']+bg_point_samples_num_list[frame_pair[0]], device=target_mask.device), torch.arange(pix_sel_infos[frame_pair[1]]['start_idx'], pix_sel_infos[frame_pair[1]]['start_idx']+bg_point_samples_num_list[frame_pair[1]], device=target_mask.device)) 
                            point_pairs_inter_frames = torch.cat((point_pairs_inter_frames,point_pairs_bg_bg))
                        # pairing object points with object points
                        # 1.frame fg <-> 2. frame fg
                        point_pairs_obj_obj = torch.cartesian_prod(torch.arange(pix_sel_infos[frame_pair[0]]['start_idx']+bg_point_samples_num_list[frame_pair[0]], pix_sel_infos[frame_pair[0]]['start_idx']+pix_sel_infos[frame_pair[0]]['pix_sel_size'], device=target_mask.device), torch.arange(pix_sel_infos[frame_pair[1]]['start_idx']+bg_point_samples_num_list[frame_pair[1]], pix_sel_infos[frame_pair[1]]['start_idx']+pix_sel_infos[frame_pair[1]]['pix_sel_size'], device=target_mask.device)) 
                        point_pairs_inter_frames = torch.cat((point_pairs_inter_frames,point_pairs_obj_obj))
                        del point_pairs_obj_obj
                    # in frame
                    elif (frame_pair[0] == frame_pair[1]): 
                        if self.make_positive_bg_pairs:
                            # pairing background points with background points
                            # bg <-> bg
                            point_pairs_bg_bg = torch.combinations(torch.arange(pix_sel_infos[frame_pair[0]]['start_idx'], pix_sel_infos[frame_pair[0]]['start_idx']+bg_point_samples_num_list[frame_pair[0]], device=target_mask.device)) 
                            point_pairs_in_frame = torch.cat((point_pairs_in_frame,point_pairs_bg_bg))
                        # pairing background points with object points
                        # bg <-> fg
                        point_pairs_bg_obj = torch.cartesian_prod(torch.arange(pix_sel_infos[frame_pair[0]]['start_idx'], pix_sel_infos[frame_pair[0]]['start_idx']+bg_point_samples_num_list[frame_pair[0]], device=target_mask.device), torch.arange(pix_sel_infos[frame_pair[1]]['start_idx']+bg_point_samples_num_list[frame_pair[1]], pix_sel_infos[frame_pair[1]]['start_idx']+pix_sel_infos[frame_pair[1]]['pix_sel_size'], device=target_mask.device)) 
                        point_pairs_in_frame = torch.cat((point_pairs_in_frame,point_pairs_bg_obj))
                        # pairing object points with object points
                        # fg <-> fg
                        point_pairs_obj_obj = torch.combinations(torch.arange(pix_sel_infos[frame_pair[0]]['start_idx']+bg_point_samples_num_list[frame_pair[0]], pix_sel_infos[frame_pair[0]]['start_idx']+pix_sel_infos[frame_pair[0]]['pix_sel_size'], device=target_mask.device))
                        point_pairs_in_frame = torch.cat((point_pairs_in_frame,point_pairs_obj_obj))
                        del point_pairs_bg_obj
                        del point_pairs_obj_obj
                
                # Drop out in frame point pairs based on L2 norm 
                # and using Bernoulli distribution to get the mask
                point_pair_distances = torch.norm(pix_sel_all_frames[point_pairs_in_frame[:,0]].float() - pix_sel_all_frames[point_pairs_in_frame[:,1]].float(), dim=1)
                point_pair_mask_in_frame = torch.bernoulli(torch.min(torch.tensor(1),self.delta + (1 - point_pair_distances / diagonal_length) / self.beta)).bool()
                point_pairs_in_frame = point_pairs_in_frame[point_pair_mask_in_frame]
                b_pairs_in_frame = torch.cat((b_pairs_in_frame,point_pairs_in_frame.to(torch.int32) + b_pix_sel_size))
                
                # Calculating the loss weights for each point pairs
                if self.min_point_pair_weight < 1:
                    point_pair_distances = point_pair_distances[point_pair_mask_in_frame]
                    in_frame_eq_cluster_mask = torch.eq(b_target_clusters[point_pairs_in_frame[:,0] + b_pix_sel_size],b_target_clusters[point_pairs_in_frame[:,1] + b_pix_sel_size])
                    
                    point_pairs_weights_in_frame = torch.zeros(point_pairs_in_frame.shape[0],device=point_pairs_in_frame.device)
                    point_pairs_weights_in_frame[in_frame_eq_cluster_mask] += self.min_point_pair_weight + point_pair_distances[in_frame_eq_cluster_mask] * (1-self.min_point_pair_weight)/diagonal_length
                    point_pairs_weights_in_frame[(in_frame_eq_cluster_mask == False)] += 1 - point_pair_distances[(in_frame_eq_cluster_mask == False)] * (1-self.min_point_pair_weight)/diagonal_length
                    del in_frame_eq_cluster_mask
                    b_pairs_weights_in_frame = torch.cat((b_pairs_weights_in_frame,point_pairs_weights_in_frame))
                    del point_pairs_weights_in_frame
                
                del point_pairs_in_frame
                
                # Drop out inter frames point pairs based on L2 norm 
                # and using Bernoulli distribution to get the mask
                point_pair_distances = torch.norm(pix_sel_all_frames[point_pairs_inter_frames[:,0]].float() - pix_sel_all_frames[point_pairs_inter_frames[:,1]].float(), dim=1)
                point_pair_mask_inter_frames = torch.bernoulli(torch.min(torch.tensor(1),self.delta + (1 - point_pair_distances / diagonal_length) / self.beta)).bool()
                point_pairs_inter_frames = point_pairs_inter_frames[point_pair_mask_inter_frames]
                b_pairs_inter_frames = torch.cat((b_pairs_inter_frames,point_pairs_inter_frames.to(torch.int32) + b_pix_sel_size))
                
                # Calculating the loss weights for each point pairs
                if self.min_point_pair_weight < 1:
                    point_pair_distances = point_pair_distances[point_pair_mask_inter_frames]
                    inter_frames_eq_cluster_mask = torch.eq(b_target_clusters[point_pairs_inter_frames[:,0] + b_pix_sel_size],b_target_clusters[point_pairs_inter_frames[:,1] + b_pix_sel_size])
                    
                    point_pairs_weights_inter_frames = torch.zeros(point_pairs_inter_frames.shape[0],device=point_pairs_inter_frames.device)
                    point_pairs_weights_inter_frames[inter_frames_eq_cluster_mask] += self.min_point_pair_weight + point_pair_distances[inter_frames_eq_cluster_mask] * (1-self.min_point_pair_weight)/diagonal_length
                    point_pairs_weights_inter_frames[(inter_frames_eq_cluster_mask == False)] += 1 - point_pair_distances[(inter_frames_eq_cluster_mask == False)] * (1-self.min_point_pair_weight)/diagonal_length
                    del inter_frames_eq_cluster_mask
                    b_pairs_weights_inter_frames = torch.cat((b_pairs_weights_inter_frames,point_pairs_weights_inter_frames))
                    del point_pairs_weights_inter_frames
                    
                del point_pairs_inter_frames
                del point_pair_distances
                b_pix_sel_size += pix_sel_frame_start_idx
                point_pair_mask = torch.cat((point_pair_mask_in_frame,point_pair_mask_inter_frames))
                
                # plot informations about filtered points in tensorboard 
                values,counts = torch.unique(point_pair_mask,return_counts=True,sorted=True)
                if point_pair_mask.shape[0] != 0:
                    storage = get_event_storage()
                    storage.put_scalar('Number of point pairs', point_pair_mask.shape[0])
                    if values.shape[0] == 2:
                        filtered_num = counts[1].item()
                        ratio = 100*filtered_num/point_pair_mask.shape[0]
                    elif values[0].item() == True:
                        filtered_num = counts[0].item()
                        ratio = 100*filtered_num/point_pair_mask.shape[0]
                    elif values[0].item() == False:   
                        filtered_num = 0
                        ratio = 0
                            
                    storage.put_scalar('Number of filtered point pairs', filtered_num)
                    storage.put_scalar('Filtered Point pairs - original point pairs ratio (%)', ratio)
                
                del pix_sel_all_frames
                del point_pair_mask

            # sample the points from the model output
            for frame_idx, point_coords in enumerate(point_coords_list):
                point_logits_clusters = point_sample(
                        src_mask[:,frame_idx,:,:][:,None],
                        point_coords.repeat(src_mask.shape[0],1,1),
                        align_corners=False
                    ).squeeze(1) # (Q,Point_num)
                
                b_src_clusters = torch.cat((b_src_clusters,point_logits_clusters.transpose(0, 1))) # (Point_num,Q)
                del point_logits_clusters
        
        del comb_frame
        b_src_clusters = b_src_clusters.softmax(1)
        
        with torch.no_grad():
            storage = get_event_storage()
            #storage.put_scalar('loss_cluster_in_frame', cos_sim_clustering_loss_jit(b_src_clusters, b_target_clusters, b_pairs_in_frame.to(torch.int64)))
            #storage.put_scalar('loss_cluster_inter_frames', cos_sim_clustering_loss_jit(b_src_clusters, b_target_clusters, b_pairs_inter_frames.to(torch.int64)))
        
        b_pairs = torch.cat((b_pairs_in_frame,b_pairs_inter_frames)) # (Pair_num,2)
        if self.min_point_pair_weight < 1:
            b_weights = torch.cat((b_pairs_weights_in_frame,b_pairs_weights_inter_frames))
        else:
            b_weights = torch.ones(b_pairs.shape[0],device=b_pairs.device)
        
        if self.cos_sim_clustering_loss:
            losses = {
                "loss_cluster": cos_sim_clustering_loss_jit(b_src_clusters, b_target_clusters, b_pairs.to(torch.int64),b_weights) if b_pairs.shape[0] > 0 else torch.tensor(0.0,device=src_masks.device),
            }
        else:
            losses = {
                "loss_cluster": ccl_loss_jit(b_src_clusters, b_target_clusters, b_pairs.to(torch.int64),b_weights) if b_pairs.shape[0] > 0 else torch.tensor(0.0,device=src_masks.device),
            }

        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_masks):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'clusters': self.loss_clusters,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
            "total_instance_pixel_sample_num: {}".format(self.total_instance_pixel_sample_num),
            "min_num_sample: {}".format(self.min_num_sample),
            "max_num_sample: {}".format(self.max_num_sample),
            "max_inst: {}".format(self.max_inst),
            "beta: {}".format(self.beta),
            "delta: {}".format(self.delta),
            "make_inter_frame_bg_points_connections: {}".format(self.make_inter_frame_bg_points_connections),
            "make_inter_frame_point_connections: {}".format(self.make_inter_frame_point_connections),
            "min_point_pair_weight: {}".format(self.min_point_pair_weight),
            "make_positive_bg_pairs: {}".format(self.make_positive_bg_pairs),
            "cos_sim_clustering_loss: {}".format(self.cos_sim_clustering_loss),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
