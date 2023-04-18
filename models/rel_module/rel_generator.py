# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(MAIN_PATH))

import torch
import torch.nn as nn

from .geo_vector import GeoVectorBuild
from .sym_vector import SymVectorBuild
from .construct_rel import ConstructRel
from .parse_rel import parse_rel
from image_structure import convert_parse_to_natural_language

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
        self.construc_rel = ConstructRel(cfg)
        
        self.cfg = cfg
    
    def forward(self,
                geo_feature_map, sym_feature_maps,
                gt_point_mask=None, gt_line_mask=None, gt_circle_mask=None,
                targets_det=None, all_labels_to_layer=None,
                proposals_seg=None, proposals_det=None,
                images_not_tensor=None,
                targets_geo=None, targets_sym=None):
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
            targets_geo (List[Dict], optional): The golden relation between (points and lines), (points and circles),
                The Dict keys must be: "pl_rels" (P, L), "pc_rels" (P, C)
            targets_sym (List[Dict]): The golden relations between sym and geo, (#sym, #relevant_geo)
                The Dict keys must be: "text_symbol_geo_rel", "head_symbol_geo_rel", 
                    "[None|double_|triple_|quad_|penta_]angle_symbols_geo_rel",
                    "[None|double_|triple_|quad_]bar_symbols_geo_rel",
                    "[None|double_|parallel_]parallel_symbols_geo_rel",
                    "perpendicular_symbols_geo_rel".
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
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # # # # # # # # # Construct Rel # # # # # # # # #
        
        geo_rels_predictions, sym_geo_rels_predictions, losses = self.construc_rel(
            all_geo_info=all_geo_info,
            all_sym_info=all_sym_info,
            targets_geo=targets_geo,
            targets_sym=targets_sym,
        )
        
        # # # # # # # # # # # # # # # # # # # # # # # # # # #
        
        # # # # # # # # # Parse Rel # # # # # # # # #
        
        if self.training:
            return losses
        else:
            # parse_results (List(Dict)): each dict contains the parsed relations:
            #   keys: {"angle", "length", "congruent_angle", "congruent_bar", "parallel", "perpendicular"}
            parse_results = parse_rel(
                geo_rels=geo_rels_predictions, 
                sym_geo_rels=sym_geo_rels_predictions, 
                ocr_results=all_sym_info["text_symbols_str"],
                threshold=self.cfg.threshold
            )
            
            # List[Dict]:  {"angle", "length", "congruent_angle", "congruent_bar", "parallel", "perpendicular"}
            # !!! From here, if the rel doesn't exist, there will be no such key.
            natural_language_results = []    
            for per_parse_res in parse_results:
                natural_language_results.append(convert_parse_to_natural_language(per_parse_res))
            
            return parse_results, natural_language_results