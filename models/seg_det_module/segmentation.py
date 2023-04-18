# coding:utf-8

import torch
import torch.nn as nn

from .loss import make_seg_loss_evaluator
from .inference import make_seg_postprocessor

class SEGHead(torch.nn.Module):
    
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SEGHead, self).__init__()
        
        num_classes = cfg.seg_num_classes-1   # 4-1=3
        emb_dim = cfg.seg_emb_dims    # 8

        # binary_seg_tower: 以下架构*3
        #   - Conv2d, C变为输入-2, H,W不变
        #   - GroupNorm
        #   - ReLU
        binary_seg_tower = []
        # embedding_tower: 以下架构*3
        #   - Conv2d, C变为输入-2, H,W不变
        #   - GroupNorm
        #   - ReLU
        embedding_tower = []

        # 3
        for index in range(cfg.seg_num_convs):

            if index==0:
                # !!! 很重要, 因为输入的concat了loc_map, channel为FPN P2 + 2
                in_channels_new = in_channels + 2
            else:
                in_channels_new = in_channels

            binary_seg_tower.append(
                nn.Conv2d(
                    in_channels_new,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                ))
            # 将in_channels分配到32个groups, 对每一组就行归一化
            binary_seg_tower.append(nn.GroupNorm(32, in_channels))
            binary_seg_tower.append(nn.ReLU())

            embedding_tower.append(
                nn.Conv2d(
                    in_channels_new,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                ))
            embedding_tower.append(nn.GroupNorm(32, in_channels))
            embedding_tower.append(nn.ReLU())


        self.add_module('binary_seg_tower', nn.Sequential(*binary_seg_tower))
        self.add_module('embedding_tower', nn.Sequential(*embedding_tower))

        # pixel分类网络, num_classes=3
        self.binary_seg_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        # pixel embedding网络, emb_dim=8
        self.embedding_pred = nn.Conv2d(
            in_channels, emb_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):

        binary_seg = self.binary_seg_logits(self.binary_seg_tower(x))
        embedding = self.embedding_pred(self.embedding_tower(x))

        # binary_seg: [b, 3, h, w]
        # embedding: [b, 8, h, w]
        return binary_seg, embedding

class SEGModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SEGModule, self).__init__()

        self.head = SEGHead(cfg, in_channels)
        self.loss_evaluator = make_seg_loss_evaluator(cfg)
        self.ggeo_selector = make_seg_postprocessor(cfg)
        self.cfg = cfg

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (Tensor): features of last level computed from the images that are
                                used for computing the predictions. 
            targets (list[GeoList]): ground-truth masks present in the image (optional)
            
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

        Returns:
            ggeos (list[GeoList]): the predicted mask from the SEGModule, one GeoList per
                                    image.
            losses (dict[Tensor]): the losses for the model during training. During
                                    testing, it is an empty dict.
        """
        # ???直觉认为features(b, c+2, h, w), h & w 和image的不同, 这里为了减少内存占用, 采用P2的输出大小
        # binary_seg: [b, 3, h, w]
        # embedding: [b, 8, h, w]
        binary_seg, embedding = self.head(features)
        
        if self.training:
            return self._forward_train(
                binary_seg, embedding, targets
            )
        else:
            return self._forward_test(
                binary_seg, embedding, images.image_sizes, self.cfg.MODEL.SEG.FPN_STRIDES
            )

    def _forward_train(self,  binary_seg, embedding, targets):

        binary_seg_loss, var_loss, dist_loss, reg_loss, \
            gt_point_mask_for_rel, gt_line_mask_for_rel, gt_circle_mask_for_rel = self.loss_evaluator(
            binary_seg, embedding, targets
        )
        losses = {
            "loss_binary_seg": binary_seg_loss * self.cfg.loss_ratio_bs,
            "loss_var": var_loss * self.cfg.loss_ratio_var,
            "loss_dist": dist_loss * self.cfg.loss_ratio_dist,
            "loss_mean_reg": reg_loss * self.cfg.loss_ratio_reg
        }
        return None, losses, (gt_point_mask_for_rel, gt_line_mask_for_rel, gt_circle_mask_for_rel)

    def _forward_test(self, binary_seg, embedding, image_sizes, fpn_stride):
        
        # fpn_stride: 4
        ggeos = self.ggeo_selector(
            binary_seg, embedding, image_sizes, fpn_stride
        )
        return ggeos, {}, None