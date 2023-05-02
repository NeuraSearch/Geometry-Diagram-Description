# coding:utf-8

import os
import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent.parent
sys.path.insert(0, str(MAIN_PATH))

import torch
import torch.nn as nn

from .seg_det_module.generalized_rcnn import GeneralizedRCNN
from .rel_module.rel_generator import RelGenerator

import numpy as np
from train_utils import draw_objs
CLASSES_SYM = [
    "__background__", 
    "text", 
    "perpendicular", "head", "head_len",
    "angle","bar","parallel", 
    "double angle","double bar","double parallel", 
    "triple angle","triple bar","triple parallel",
    "quad angle", "quad bar", 
    "penta angle", 
    "arrow"
]

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
        
        self.cfg = cfg
        
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
                    "[None|double_|triple_]parallel_symbols_geo_rel",
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
            
            if not self.cfg.only_parse:
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
            
            if not self.cfg.only_parse:
                losses.update(rel_losses)
            
            return losses
        
        else:
            
            rel_metatdata = self.det_seg_model(images=images)
            # category_index = {str(i): v for i, v in enumerate(CLASSES_SYM)}
            # print("Haha")
            # # print(rel_metatdata)
            # for data_i, per_data_det in enumerate(rel_metatdata["proposals_det"]):
            #     bboxes = per_data_det.bbox.detach().cpu().numpy()  # [N, 4]
            #     labels = per_data_det.get_field("labels") # [N]
            #     scores = per_data_det.get_field("scores").detach().cpu().numpy()
                
            #     # golden_det = targets_det[i]
            #     # bboxes = golden_det.bbox.detach().cpu().numpy()
            #     # labels = golden_det.get_field("labels") # [N]
                
            #     masks = np.array(rel_metatdata["proposals_seg"][data_i].masks)
            #     # masks = np.array(targets_seg[data_i].masks)
                
            #     x_axis = masks.shape[0]
            #     y_axis = masks.shape[1]
            #     all_masks = []
            #     for mask_i in range(masks.shape[-1]):
            #         mask_ = masks[:, :, mask_i]
            #         masks_full = np.ones((images_not_tensor[data_i].size[-1], images_not_tensor[data_i].size[0]))
            #         for x in range(x_axis):
                        
            #             for i in range(0, 4):
            #                 x_ = x + i
                        
            #                 for y in range(y_axis):
            #                     val = mask_[x, y]
            #                     for i in range(0, 4):
            #                         masks_full[x_, y+i] = 1 if val == 255 else 0

            #         # masks_full = np.transpose(masks_full)
            #         all_masks.append(masks_full)
                            
                
            #     all_masks = np.array(all_masks)
            #     # print(bboxes)
            #     # print(labels)
            #     images = draw_objs(images_not_tensor[data_i], bboxes, labels, scores, category_index=category_index, masks=all_masks)
            #     images.save(f"{data_i+10}.png")
            # exit()
            
            if not self.cfg.only_parse:
                # !!! All keys below doesn't exist if such relation is None, unlike others.
                # parse_results (List(Dict)): each Dict is a data prediction, containing keys: 
                # {"angle", "length", "congruent_angle", "congruent_bar", "parallel", "perpendicular"}.
                # # List[Dict]:  each Dict is a data prediction, containing keys: 
                # {"angle", "length", "congruent_angle", "congruent_bar", "parallel", "perpendicular"}
                natural_language_results = self.rel_generator(
                    geo_feature_map=rel_metatdata["geo_feature_map"],
                    sym_feature_maps=rel_metatdata["sym_feature_maps"],
                    proposals_seg=rel_metatdata["proposals_seg"],
                    proposals_det=rel_metatdata["proposals_det"],
                    images_not_tensor=images_not_tensor,
                )
            
            else:
                dummy_results = [{"A": 1} for _ in range(self.cfg.test_img_per_batch)]
                return dummy_results
            
            return natural_language_results