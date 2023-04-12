# coding:utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Scale
from .loss import make_fcos_loss_evaluator
from .inference import make_fcos_postprocessor

class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        num_classes = cfg.rpn_num_classes - 1    # 17-1=16
        self.fpn_strides = cfg.fpn_strides   # [8, 16, 32, 64, 128]
        self.norm_reg_targets = cfg.rpn_norm_reg_targets     # False
        self.centerness_on_reg = cfg.rpn_centerness_on_reg   # False
        self.use_dcn_in_tower = cfg.rpn_use_dcn_in_tower     # False

        cls_tower = []
        bbox_tower = []
        # 3
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        # 16
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        # 4
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        # 1
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB  # 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        # x: [P3, P4, P5, P6, P7], [b, c, h, w]
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)     # [b, c, h, w]
            box_tower = self.bbox_tower(feature)    # [b, c, h, w]

            logits.append(self.cls_logits(cls_tower))   # [b, 16, h, w]
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))   # [b, 1, h, w]

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))   # [b, 4, h, w]
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        
        """
            - logits: [[b, 16, h, w], ...] x 5
            - bbox_reg: [[b, 4, h, w], ...] x 5
            - centerness: [[b, 1, h, w], ...] x 5
        """
        return logits, bbox_reg, centerness

class FCOSModule(nn.Module):
    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        self.head = FCOSHead(cfg, in_channels)
        self.box_selector_test = make_fcos_postprocessor(cfg)
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)
        # [8, 16, 32, 64, 128]
        self.fpn_strides = cfg.fpn_strides
    
    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                                    used for computing the predictions. Each tensor in the list
                                    correspond to different feature levels
            targets (list[BoxList]): ground-truth boxes present in the image (optional)
        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                                   image.
            losses (dict[Tensor]): the losses for the model during training. During
                                   testing, it is an empty dict.
        """
        
        """
            - box_cls: [[b, 16, h, w], ...] x 5
            - box_regression: [[b, 4, h, w], ...] x 5
            - centerness: [[b, 1, h, w], ...] x 5
        """
        box_cls, box_regression, centerness = self.head(features)
        # locations: List[Tensor(h_n * w_n, 2)], len(locations)==5
        # !!! 注意h_n,w_n根据不同的feature level, 是不同的
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, images.image_sizes
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, targets):
        """
            - locations: List[Tensor(h_n * w_n, 2), ...], len(locations)==5
            - box_cls:  List[Tensor(b, 16, h_n, w_n), ...], len(locations)==5
            - box_regression: List[Tensor(b, 4, h_n, w_n), ...], len(locations)==5
            - centerness: List[Tensor(b, 1, h_n, w_n), ...], len(locations)==5
            - targets: 
        """
        loss_box_cls, loss_box_reg, loss_centerness, all_labels_to_layer = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        # loss_box_cls: [1]
        # loss_box_reg: [?]
        # loss_centerness: [?]
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        return None, losses, all_labels_to_layer

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
            - locations: List[Tensor(h_n * w_n, 2), ...], len(locations)==5
            - box_cls:  List[Tensor(b, 16, h_n, w_n), ...], len(locations)==5
            - box_regression: List[Tensor(b, 4, h_n, w_n), ...], len(locations)==5
            - centerness: List[Tensor(b, 1, h_n, w_n), ...], len(locations)==5
            - image_sizes: [H, W]
        """
        # boxlists: List[ [BoxList, ...] ], len(boxlists) = bsz
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            # [points_per_level, 2]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        # locations: List[Tensor(h_n * w_n, 2)], len(locations)==5
        # !!! 注意h_n,w_n根据不同的feature level, 是不同的
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        # suppose: h=w=80, stride=8
        # [0, 8, 16, ..., 624, 632] (80)
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        # [0, 8, 16, ..., 624, 632] (80)
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        """
            a, b = torch.meshgrid(m, n) -> m*n的矩阵
                - a: 每一行的数字相同
                - b: 每一列的数字相同
            
            shift_y: (80, 80)
                0   0  ... 0  0 -> 80
                8   8  ... 8  8 -> 80
                .   .      .  .
                .   .      .  .
                .   .      .  .
                632 632    632 632
            shift_x: (80, 80)
                0  8  16  ...  624  632  -> 80
                0  8  16  ...  624  632
                .  .  .         .    .
                .  .  .         .    .
                .  .  .         .    .
                0  8  16  ...  624  632
        """
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        # 80个[0  8  16  ...  624  630]循环
        shift_x = shift_x.reshape(-1)
        # 80个[0   0  ... 0  0] + 80个[8   8  ... 8  8] + ...
        shift_y = shift_y.reshape(-1)
        """像是一列(80个)接着一列
            (0,0)   (8,0)   (16, 0) ... (624, 0)  (632, 0)
            (0,8)   (8,8)   (16, 8) ... (624, 8)  (632, 8)
            ...
            ...
            ...
            (0, 632)  (8, 632)  ...  (624, 632)  (632, 632)
        """
        # 这个 "+stride//2" 是为了得到感受野的中心点
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations