# coding:utf-8

import torch
import torch.nn as nn

from .seg_det_module.generalized_rcnn import GeneralizedRCNN
from .rel_module.rel_generator import RelGenerator

class DiagramDescribe(nn.Module):
    """This class is for generating natural langauge description for Diagram.
        Including: 1. Symbols Detection, 
                   2. Geometry Primitives Segmentation,
                   3. Relations Build,
    """
    
    def __init__(self, cfg):
        super(DiagramDescribe, self).__init__()
        
        self.det_seg_model = GeneralizedRCNN(cfg)
        
        self.rel_generator = RelGenerator(cfg)
        
    def forward(self, 
                images, images_not_tensor, 
                targets_det=None, targets_seg=None,
                targets_geo=None, targets_sym=None):
        """
        Args:
            images (list[Tensor] or ImageList): image in [W, H, C] format, after Transformations.
            images_not_tensor (list[Tensor] or ImageList): image in [W, H] format, after Transformations except for ToTensor. We need this foramt for OCR.
            targets_det (list[BoxList]): ground-truth boxes present in the image (optional).
            targets_seg (list[GeoList]): ground-truth masks present in the image (optional).
            targets_geo (List[Dict], optional): The golden relation between (points and lines), (points and circles). 
                The Dict keys must be: "pl_rels" (P, L), "pc_rels" (P, C)
            targets_sym (List[Dict]): The golden relations between sym and geo, (#sym, #relevant_geo)
                The Dict keys must be: "text_symbol_geo_rel", "head_symbol_geo_rel", 
                    "[None|double_|triple_|quad_|penta_]angle_symbols_geo_rel",
                    "[None|double_|triple_|quad_]bar_symbols_geo_rel",
                    "[None|double_|parallel_]parallel_symbols_geo_rel",
                    "perpendicular_symbols_geo_rel".
        """
        
        """ *** 1. Symbols Detection & 2. Geometry Primitives Segmentation  *** """
        
        if self.training:
            
            # det_seg_losses: {"loss_cls", "loss_reg", "loss_centerness", "loss_binary_seg", "loss_var", "loss_dist", "loss_mean_reg"}
            det_seg_losses, rel_metatdata = self.det_seg_model(
                images=images,
                targets_det=targets_det,
                targets_seg=targets_seg,
            )
            
            # rel_losses: {"pl_loss": Tensor, "pc_loss": Tensor,
            #              "text_symbol_geo_rel_loss", "head_symbol_geo_rel_loss",
            #              "angle_symbols_geo_rel_loss", "bar_symbols_geo_rel_loss",
            #              "parallel_symbols_geo_rel_loss", "perpendicular_symbols_geo_rel_loss"}
            rel_losses = self.rel_generator(
                geo_feature_map=rel_metatdata["geo_feature_map"],
                sym_feature_maps=rel_metatdata["sym_feature_maps"],
                gt_point_mask=rel_metatdata["gt_point_mask"],
                gt_line_mask=rel_metatdata["gt_line_mask"],
                gt_circle_mask=rel_metatdata["gt_circle_mask"],
                targets_det=rel_metatdata["targets_det"],
                all_labels_to_layer=rel_metatdata["all_labels_to_layer"],
                targets_geo=targets_geo,
                targets_sym=targets_sym,
            )
            
            losses = {}
            losses.update(det_seg_losses)
            losses.update(rel_losses)
            
            return losses
        
        else:
            
            rel_metatdata = self.det_seg_model(images=images)
            
            # !!! All keys below doesn't exist if such relation is None, unlike others.
            # parse_results (List(Dict)): each Dict is a data prediction, containing keys: 
            # {"angle", "length", "congruent_angle", "congruent_bar", "parallel", "perpendicular"}.
            # # List[Dict]:  each Dict is a data prediction, containing keys: 
            # {"angle", "length", "congruent_angle", "congruent_bar", "parallel", "perpendicular"}
            parse_results, natural_language_results = self.rel_generator(
                geo_feature_map=rel_metatdata["geo_feature_map"],
                sym_feature_maps=rel_metatdata["sym_feature_maps"],
                proposals_seg=rel_metatdata["proposals_seg"],
                proposals_det=rel_metatdata["proposals_det"],
            )
            
            return parse_results, natural_language_results