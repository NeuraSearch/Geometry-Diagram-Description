# coding:utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(MAIN_PATH))

import torch
import torch.nn as nn

from .backbone import build_mnv2_fpn_backbone
from .fcos import FCOSModule
from .segmentation import SEGModule
from .vis_emb import VisEmb

from image_structure import to_image_list

class GeneralizedRCNN(nn.Module):
    """We modify this class to support segmentation and detection.
        NOTE: the relation prediction module is not here.
    """
    
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        
        self.backbone = build_mnv2_fpn_backbone(cfg)
        
        self.rpn = FCOSModule(cfg, self.backbone.out_channels)
        
        self.seg = SEGModule(cfg, self.backbone.out_channels)
        
        self.visemb = VisEmb(cfg, self.backbone.out_channels)
        
        # 1200 / 4 + 50 = 350
        w = h = int(cfg.max_size_train / cfg.seg_fpn_strides) + 50
        # [1, 1, 350] -> [1, 350, 350]
        xm = torch.linspace(0, 1.0, w).view(1, 1, -1).expand(1, h, w)
        # [1, 1, 350] -> [1, 350, 350]
        ym = torch.linspace(0, 1.0, h).view(1, -1, 1).expand(1, h, w)
        # [2, 350, 350]
        self.location_map = torch.cat((xm, ym), 0)
    
    def forward(self, images, targets_det=None, targets_seg=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets_det (list[BoxList]): ground-truth boxes present in the image (optional)
            targets_seg (list[GeoList]): ground-truth masks present in the image (optional)

        Returns:
            result (proposals or losses): the output from the model.
                    During training, it returns a dict[Tensor] which contains the losses.
                    During testing, it returns dict[list[BoxList], list[GeoList], list[RelList]] contains additional fields
        """
        if self.training and (targets_det is None or targets_seg is None):
            raise ValueError("In training mode, targets should be passed")

        # 返回ImageList: geo_parse/structures/image_list.py
        #   - tensors: [b, C, H, W], batch好的所有图像
        #   - image_sizes: [(H, W), ...]
        images = to_image_list(images)
        # ??? 我觉得backbone的返回应该是一个list, 每个元素是[b, c, h, h]
        features = self.backbone(images.tensors) 
        
        # # # # # # # # # Detection Part # # # # # # # # #
        """
        proposals_det (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                                image.
            - boxlists: List[ [BoxList, ...] ], len(boxlists) = bsz
        proposal_det_losses (dict[Tensor]): the losses for the model during training. During
                                   testing, it is an empty dict.
            - loss_box_cls: [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7), 16]
            - loss_box_reg: [?]
            - loss_centerness: [?]
        """
        proposals_det, proposal_det_losses, all_labels_to_layer = self.rpn(images, features[1:], targets_det)

        # # # # # # # # # # # # # # # # # # # # # # # # # # #


        # # # # # # # # # Segmentation Part # # # # # # # # #
        # self.location_map: [2, 350, 350] -(切割)> [2, H, W], H, W代表first FPN network的H,W
        # 因为输入图像是batch且padding过的, 因此features[0].shape大家都一样
        loc_map = self.location_map[:,:features[0].shape[2],:features[0].shape[3]].contiguous()
        # 扩充到所有batch上, [b, 2, H, W]
        loc_map = loc_map.expand(len(features[0]),-1,-1,-1).cuda()
        
        # 将FPN第一层与loc_map结合, [b, c+2, H, W]
        features_share = torch.cat((features[0], loc_map), 1)

        """
        images:
            - tensors: [b, C, H, W], batch好的所有图像
            - image_sizes: [(H, W), ...]
        feature_shape: [b, c+2, H, W]
        targets_seg:
            - masks: [H, W, geo数量]
            - size: (W,H)
            - extra_fields:
                {
                    "labels": [geo_idx, ...],
                    "locs": [[[x, y]], [[x1, y1], [x2, y2]], [[x,y], radius, "1111"]],
                    "ids": [p0, l1, c0]
                }
        """
        """
            - proposals_seg: [GeoList]
                geolists: [GeoList, ...]
                    - masks: [H, W, geo数量]
                    - size: (W,H)
                    - extra_fields:
                        {
                            "labels": [geo_idx, ...],
                            "locs": [[[x, y]], [[x1, y1], [x2, y2]], [[x,y], radius, "1111"]],
                            "ids": [p0, l1, c0]
                        }
           
        """
        proposals_seg, proposal_seg_losses, gt_mask = self.seg(images, features_share, targets_seg)

        # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # # [b, 64, H, W], H, W refer to the P2
        # visemb_features = self.visemb(features_share)
  
        if self.training:
            
            losses = {}
            
            # proposal_det_losses: {"loss_cls", "loss_reg", "loss_centerness"}
            losses.update(proposal_det_losses)
            # proposal_seg_losses: {"loss_binary_seg", "loss_var", "loss_dist", "loss_mean_reg"}
            losses.update(proposal_seg_losses)

            rel_metadata = {"gt_point_mask": gt_mask[0],
                            "gt_line_mask": gt_mask[1],
                            "gt_circle_mask": gt_mask[2],
                            "targets_det": targets_det,
                            "all_labels_to_layer": all_labels_to_layer,
                            "geo_feature_map": features[0],
                            "sym_feature_maps": features[1:]}
            
            return losses, rel_metadata
        
        else:
            
            rel_metadata = {"proposals_det": proposals_det,
                            "proposals_seg": proposals_seg,
                            "geo_feature_map": features[0],
                            "sym_feature_maps": features[1:]}
            
            return rel_metadata