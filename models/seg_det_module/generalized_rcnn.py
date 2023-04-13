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
    
    def forward(self, images, targets_det=None, targets_seg=None, targets_rel=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets_det (list[BoxList]): ground-truth boxes present in the image (optional)
            targets_seg (list[GeoList]): ground-truth masks present in the image (optional)
            targets_rel (list[RelList]): ground-truth relation map present in the image (optional)

        Returns:
            result (proposals or losses): the output from the model.
                    During training, it returns a dict[Tensor] which contains the losses.
                    During testing, it returns dict[list[BoxList], list[GeoList], list[RelList]] contains additional fields
        """
        if self.training and (targets_det is None or targets_seg is None or targets_rel is None):
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
        
        # TODO: Train 拿到box特征
        """
        循环每个data
            取得all_labels_to_layer[data]
            循环每个target
                得到bbox [sym数量, 4]
                得到labels [sym数量]
                得到ids [sym数量]
                
                循环i, ids
                    得到bbox[i] (4), labels[i],
                    得到layer_num = all_labels_to_layer[data][ids]
                    取得features[1:][layer_num - 3][i] # P3-3=0 P4-3=1 ... P7-3=4
                    然后使用RoIAlign求得vector[1, c, 7, 7] -> Conv2d(c, kernel_size=7) | 或者直接拉平通过TwoMLPHead -> 变成[1, c_2, 1, 1]
                        - input: features[1:][layer_num - 3][i].unsqueeze(0) (1, c, h, w)
                        - boxes: [ bbox[i].unsqueeze(0) ] -> [ Tensor(1, 4) ], 第0-axis必须是box个数
                        - output_size: todo
                        - spatial_scale = 1 / cfg.fpn_strides[layer_num - 3]
                    
                    if labels[i] is "text:
                        加入text_symbol
                    elif labels[i] is "other":
                        加入other_symbol
                    elif labels[i] is "head":
                        加入head_symbol
        """
        
        # TODO: test 拿到box特征
        """
        循环proposals_det
            循环每个boxlist
                拿到boxlist.bbox
                拿到boxlist.labels
                拿到boxlist.layers
                拿到boxlist.ids
                
                然后和Train一样的步骤     
        """

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

        # [b, 64, H, W], H, W refer to the P2
        visemb_features = self.visemb(features_share)
        
        # TODO: train 拿到seg特征
        """
        gt_point_mask_for_rel, gt_line_mask_for_rel, gt_circle_mask_for_rel = gt_mask
        这里拿gt_point_mask_for_rel举例
        循环每个gt_point_mask_for_rel
            循环每个point, 得到mask
                然后在 (visemb_features[b,64,h,w] + features_share[b,c+2,h,w]) # TODO: 怎么结合? concat?
                上面进行 mask.unsqueeze(0).repeat(C, 1, 1) * (visemb_features + features_share)
                -> [C, h, w] -> Conv(C, C_2, kernel_size=3, padding=1, stride=1, bias=True) -> 变成 [C_2, h, w]    
                -> 拉平 [C_2, h * w] -> sum(dim=-1) -> [C_2]    
        """
        
        # TODO: test 拿到seg特征
        """
        循环proposals_seg, 拿到每个data的geolist
            lables=geolist.get_field["labels"]
            ids=geolist.get_field["ids"]
            masks=geolist.masks # [h, w, #geo]
            循环i, geo每个geolist
                lables[i] == "point"
                    通过train的方法得到vector, 加入
        """