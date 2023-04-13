# coding:utf-8

import torch
import torch.nn as nn

from .geo_vector import GeoVectorBuild
from .sym_vector import SymVectorBuild

class RelGenerator(nn.Module):
    """This class is for relation construction of the sym, geo
        extracted from the GeneralizedRCNN object.
    """
    
    def __init__(self, cfg):
        super(RelGenerator, self).__init__()
        
        # build sym vector module
        self.build_sym = SymVectorBuild(cfg)
        
        # build geo vector module
        self.build_geo = GeoVectorBuild(cfg)
        
        # build rel predict module
    
    def forward(self,
                geo_feature_map, sym_feature_maps,
                gt_point_mask=None, gt_line_mask=None, gt_circle_mask=None,
                targets_det=None, all_labels_to_layer=None,
                proposals_seg=None, proposals_det=None,
                images_not_tensor=None):
        """
        Args:
            geo_feature_map (Tensor(B, c, h, w)): feature of features_share and visemb_features.
            sym_feature_maps (Tensor(B, C, H, W)): P3 - P7 features from the FPN.
            gt_point_mask (List[Tensor(h, w)]): all points mask result, in bool. bsz len
            gt_line_mask (List[Tensor(h, w)]): all lines mask result, in bool. bsz len
            gt_circle_mask (List[Tensor(h, w)]): all circles mask result, in bool. bsz len
            targets_det (List[BoxList]): ground-truth boxes present in the image
            all_labels_to_layer (List[Dict]): the idx of feature layer on which the GT box lied.
            proposals_seg ([GeoList]): predicted GeoList.
            proposals_det (list[BoxList]): the predicted boxes from the RPN, one BoxList per image.
            images_not_tensor (list[(W, H), (W, H), ...]): len==bsz
        """
        
        # # # # # # # # # Build Geo Feature # # # # # # # # #
        
        # all_geo_info: List[Dict]: Contain batch data geo information,
        #   each dict contains geo information regarding to different classes, in Tensor([N, cfg.geo_embed_size])
        all_geo_info = self.build_geo(
            feature_map=geo_feature_map,
            gt_point_mask=gt_point_mask,
            gt_line_mask=gt_line_mask,
            gt_circle_mask=gt_circle_mask,
            proposals_seg=proposals_seg,
        )
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # # # # # # # # # Build Sym Feature # # # # # # # # #
        
        #   all_symbols_info: List[Dict]: Contain batch data symbols information,
        #     each dict contains symbols information regarding to different classes,
        #     except for the "text_symbols_str" key, other keys' values are in Tensor[?, cfg.sym_embed_size].
        all_sym_info = self.build_sym(
            feature_maps=sym_feature_maps,
            targets_det=targets_det,
            all_labels_to_layer=all_labels_to_layer,
            proposals_det=proposals_det,
            images_not_tensor=images_not_tensor,
        )