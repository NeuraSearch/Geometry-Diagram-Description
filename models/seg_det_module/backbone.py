# coding:utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torch.nn as nn

from collections import OrderedDict

from . import mobilenet
from . import fpn as fpn_module
from .layers import conv_with_kaiming_uniform

def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.backbone_out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2[0], # The FPN layer used for segmentation and GNN
            in_channels_stage2[1], # P3
            in_channels_stage2[2], # P4
            in_channels_stage2[3], # P5
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.fpn_use_gn, cfg.fpn_use_relu
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model