# coding:utf-8

import torch
import torch.nn as nn
from torchvision.ops import roi_align

import numpy as np
from collections import defaultdict

import easyocr

class SymVectorHead(nn.Module):
    
    def __init__(self, inp_channel, out_channel, kernel_size):
        super(SymVectorHead, self).__init__()
        
        head_tower = []
        head_tower.append(nn.Conv2d(
            in_channels=inp_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=1,
            bias=False,
        ))
        head_tower.append(nn.GroupNorm(8, out_channel))
        head_tower.append(nn.ReLU())
        self.add_module("sym_head_tower", nn.Sequential(*head_tower))
        
        self.sym_feat_nn = nn.Linear(out_channel, out_channel)
        self.sym_feat_ac = nn.ReLU()
            
    def forward(self, feature):
        """
            feature: [N, C, cfg.sym_output_size, cfg.sym_output_size],
                    N is the number of one symbol class.
        """
        # [b, out_channel, cfg.sym_output_size, cfg.sym_output_size]
        head_out = self.sym_head_tower(feature)
        
        # [b, out_channel, cfg.sym_output_size * cfg.sym_output_size] -> [b, out_channel]
        head_out = head_out.flatten(start_dim=2).mean(dim=-1)
        
        # [b, out_channel]
        linear_out = self.sym_feat_nn(head_out)
        linear_out = self.sym_feat_ac(linear_out)
        
        return linear_out
        
class SymVectorBuild(nn.Module):
    
    # this is not used
    sym_lists = ["text_symbols", "text_symbols_str",
                 "perpendicular_symbols", "head_symbols",
                 "angle_symbols", "double_angle_symbols", "triple_angle_symbols", "quad_angle_symbols", "penta_angle_symbols",
                 "bar_symbols", "double_bar_symbols", "triple_bar_symbols", "quad_bar_symbols",
                 "parallel_symbols", "double_parallel_symbols", "triple_parallel_symbols"]
    
    def __init__(self, cfg):
        super(SymVectorBuild, self).__init__()
        
        self.roi_output_size = cfg.sym_roi_output_size
        self.fpn_strides = cfg.fpn_strides
        
        # [1200, 1, h]
        self.row_embeddings = nn.Parameter(torch.randn(1200, cfg.backbone_out_channels).unsqueeze(1))
        # [1, 1200, h]
        self.col_embeddings = nn.Parameter(torch.randn(1200, cfg.backbone_out_channels).unsqueeze(0))
        # [1200, 1200, h] -> [h, 1200, 1200] -> [1, h, 1200, 1200]
        self.sym_spatial_embeddings = nn.Parameter(torch.permute(self.row_embeddings + self.col_embeddings, (2, 0, 1)).unsqueeze(0))

        self.sym_head = SymVectorHead(
            inp_channel=cfg.backbone_out_channels,
            out_channel=cfg.sym_embed_size,
            kernel_size=self.roi_output_size,
        )
        
        self.sym_embeddings = nn.Embedding(15, cfg.sym_embed_size)
        
        self.fuse_nn = nn.Linear(cfg.sym_embed_size, cfg.sym_embed_size)
        self.fuse_ac = nn.ReLU()

        self.easyocr = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        # self.easyocr = easyocr.Reader(["en"], gpu=next(self.parameters()).device)
        
    def forward(self, 
                feature_maps,
                targets_det=None,
                all_labels_to_layer=None,
                proposals_det=None,
                images_not_tensor=None):
        """
        Args:
            feature_maps (Tensor(B, C, H, W)): P3 - P7 features from the FPN.
            targets_det (List[BoxList]): ground-truth boxes present in the image
            all_labels_to_layer (List[Dict]): the idx of feature layer on which the GT box lied.
            proposals_det (list[BoxList]): the predicted boxes from the RPN, one BoxList per image.
            images_not_tensor (list[(W, H), (W, H), ...]): len==bsz
        
        Returns:
            all_symbols_info: List[Dict]: Contain batch data symbols information,
                each dict contains symbols information regarding to different classes,
                except for the "text_symbols_str" key, other keys' values are in Tensor[?, cfg.sym_embed_size].
        """
        
        if self.training:
            assert all_labels_to_layer != None and targets_det != None
            all_symbols_info = self._forward_train(feature_maps, targets_det, all_labels_to_layer)
        else:
            assert proposals_det != None and images_not_tensor != None
            all_symbols_info = self._forward_test(feature_maps, proposals_det, images_not_tensor)
    
        return all_symbols_info
    
    def _forward_train(self, feature_maps, targets_det, all_labels_to_layer):
        """
        Returns:
            all_symbols_info: List[Dict]: Contain batch data symbols information,
                each dict contains symbols information regarding to different classes,
                except for the "text_symbols_str" key, other keys' values are in Tensor[?, cfg.sym_embed_size].
        """
        
        all_symbols_info = []
        for b_id, targets in enumerate(targets_det):
            labels_to_layer = all_labels_to_layer[b_id]
            
            bboxes = targets.bbox                                # [#sym, 4]
            labels = targets.get_field("labels")                 # [#sym]
            ids = targets.get_field("ids")                       # [#sym]
            text_contents = targets.get_field("text_contents")   # [#sym]
            
            symbols_info = defaultdict(list)
            # id_: "s0", "s1", etc.
            for i, id_ in enumerate(ids):
                box = bboxes[i].unsqueeze(0)            # [1, 4]
                layer_num = labels_to_layer[id_] - 3    # original layer_num starts from 3
                layer_feature_map = feature_maps[layer_num][b_id].unsqueeze(0)    # [1, c, h, 2]
                
                # feature: [1, c, output_size, output_size]
                feature = roi_align(input=layer_feature_map, boxes=[box], 
                                   output_size=self.roi_output_size,
                                   spatial_scale=1 / self.fpn_strides[layer_num])
                
                # spatial_feature: [1, c, output_size, output_size]
                spatial_feature = roi_align(input=self.sym_spatial_embeddings, boxes=[box],
                                            output_size=self.roi_output_size, spatial_scale=1 / self.fpn_strides[layer_num])
                feature += spatial_feature

                label = labels[i]
                if label == 1:      # text_symbol
                    symbols_info["text_symbols"].append(feature)
                    symbols_info["text_symbols_str"].append(text_contents[i])
                elif label == 2:    # perpendicular
                    symbols_info["perpendicular_symbols"].append(feature)
                elif label == 3:    # head
                    symbols_info["head_symbols"].append(feature)
                elif label in [5, 8, 11, 14, 16]:   # angle
                    if label == 5:
                        symbols_info["angle_symbols"].append(feature)
                    elif label == 8:
                        symbols_info["double_angle_symbols"].append(feature)
                    elif label == 11:
                        symbols_info["triple_angle_symbols"].append(feature)
                    elif label == 14:
                        symbols_info["quad_angle_symbols"].append(feature)
                    elif label == 16:
                        symbols_info["penta_angle_symbols"].append(feature)
                elif label in [6, 9, 12, 15]:   # bar
                    if label == 6:
                        symbols_info["bar_symbols"].append(feature)
                    elif label == 9:
                        symbols_info["double_bar_symbols"].append(feature)
                    elif label == 12:
                        symbols_info["triple_bar_symbols"].append(feature)
                    elif label == 15:
                        symbols_info["quad_bar_symbols"].append(feature)
                elif label in [7, 10, 13]:  # parallel
                    if label == 7:
                        symbols_info["parallel_symbols"].append(feature)
                    elif label == 10:
                        symbols_info["double_parallel_symbols"].append(feature)
                    elif label == 13:
                        symbols_info["triple_parallel_symbols"].append(feature)
                elif label in [4, 17]:
                    pass
                else:
                    raise ValueError(f"Unknown symbol label: ({label})")
            
            for key, value in symbols_info.items():                
                if len(value) != 0:
                    if key == "text_symbols":
                        # [[1, c, output_size, output_size], [1, c, output_size, output_size], ...] 
                        # -> [N, c, output_size, output_size] -> [N, c_2]
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=0)
                    elif key == "perpendicular_symbols":
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=1)
                    elif key == "head_symbols":
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=2)
                    elif key in ["angle_symbols", "double_angle_symbols", "triple_angle_symbols", "quad_angle_symbols", "penta_angle_symbols"]:
                        if key == "angle_symbols": sym_ids = 3
                        elif key == "double_angle_symbols": sym_ids = 4
                        elif key == "triple_angle_symbols": sym_ids = 5
                        elif key == "quad_angle_symbols": sym_ids = 6
                        elif key == "penta_angle_symbols": sym_ids = 7
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=sym_ids)
                    elif key in ["bar_symbols", "double_bar_symbols", "triple_bar_symbols", "quad_bar_symbols"]:
                        if key == "bar_symbols": sym_ids = 8
                        if key == "double_bar_symbols": sym_ids = 9
                        if key == "triple_bar_symbols": sym_ids = 10
                        if key == "quad_bar_symbols": sym_ids = 11
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=sym_ids)
                    elif key in ["parallel_symbols", "double_parallel_symbols", "triple_parallel_symbols"]:
                        if key == "parallel_symbols": sym_ids = 12
                        if key == "double_parallel_symbols": sym_ids = 13
                        if key == "triple_parallel_symbols": sym_ids = 14
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=sym_ids)
                    elif key == "text_symbols_str":
                        pass
                    else:
                        raise ValueError(f"Unknown key: ({key})")
                else:
                    # no need to assign empty [], because "symbols_info" is "defaultdict(list)",
                    # so even key doesn't exist, it wil return [].
                    pass
                        
            all_symbols_info.append(symbols_info)
        
        return all_symbols_info

    def _forward_test(self, feature_maps, proposals_det, images):
        
        all_symbols_info = []
        
        for b_id, batch_det in enumerate(proposals_det):
            bboxes = batch_det.bbox
            labels = batch_det.get_field("labels")
            layers = batch_det.get_field("layers")
            ids = batch_det.get_field("ids")
            
            symbols_info = defaultdict(list)
            for i in range(len(ids)):
                box = bboxes[i].unsqueeze(0) # [1, 4]
                layer_num = layers[i] - 3
                layer_feature_map = feature_maps[layer_num][b_id].unsqueeze(0) # [1, c, h, 2]
                
                # feature: [1, c, output_size, output_size]
                feature = roi_align(input=layer_feature_map, boxes=[box], 
                                   output_size=self.roi_output_size,
                                   spatial_scale=1 / self.fpn_strides[layer_num]) 

                # spatial_feature: [1, c, output_size, output_size]
                spatial_feature = roi_align(input=self.sym_spatial_embeddings, boxes=[box],
                                            output_size=self.roi_output_size, spatial_scale=1 / self.fpn_strides[layer_num])
                feature += spatial_feature
                
                label = labels[i]
                if label == 1:
                    symbols_info["text_symbols"].append(feature)
                    # OCR
                    symbols_info["text_symbols_str"].append(self.ocr(images[b_id], box))
                elif label == 2:
                    symbols_info["perpendicular_symbols"].append(feature)
                elif label == 3:
                    symbols_info["head_symbols"].append(feature)
                elif label in [5, 8, 11, 14, 16]:
                    if label == 5:
                        symbols_info["angle_symbols"].append(feature)
                    elif label == 8:
                        symbols_info["double_angle_symbols"].append(feature)
                    elif label == 11:
                        symbols_info["triple_angle_symbols"].append(feature)
                    elif label == 14:
                        symbols_info["quad_angle_symbols"].append(feature)
                    elif label == 16:
                        symbols_info["penta_angle_symbols"].append(feature)
                elif label in [6, 9, 12, 15]:
                    if label == 6:
                        symbols_info["bar_symbols"].append(feature)
                    elif label == 9:
                        symbols_info["double_bar_symbols"].append(feature)
                    elif label == 12:
                        symbols_info["triple_bar_symbols"].append(feature)
                    elif label == 15:
                        symbols_info["quad_bar_symbols"].append(feature)
                elif label in [7, 10, 13]:
                    if label == 7:
                        symbols_info["parallel_symbols"].append(feature)
                    elif label == 10:
                        symbols_info["double_parallel_symbols"].append(feature)
                    elif label == 13:
                        symbols_info["triple_parallel_symbols"].append(feature)

            for key, value in symbols_info.items():                
                if len(value) != 0:
                    if key == "text_symbols":
                        # [[1, c, output_size, output_size], [1, c, output_size, output_size], ...] 
                        # -> [N, c, output_size, output_size] -> [N, c_2]
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=0)
                    elif key == "perpendicular_symbols":
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=1)
                    elif key == "head_symbols":
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=2)
                    elif key in ["angle_symbols", "double_angle_symbols", "triple_angle_symbols", "quad_angle_symbols", "penta_angle_symbols"]:
                        if key == "angle_symbols": sym_ids = 3
                        elif key == "double_angle_symbols": sym_ids = 4
                        elif key == "triple_angle_symbols": sym_ids = 5
                        elif key == "quad_angle_symbols": sym_ids = 6
                        elif key == "penta_angle_symbols": sym_ids = 7
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=sym_ids)
                    elif key in ["bar_symbols", "double_bar_symbols", "triple_bar_symbols", "quad_bar_symbols"]:
                        if key == "bar_symbols": sym_ids = 8
                        if key == "double_bar_symbols": sym_ids = 9
                        if key == "triple_bar_symbols": sym_ids = 10
                        if key == "quad_bar_symbols": sym_ids = 11
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=sym_ids)
                    elif key in ["parallel_symbols", "double_parallel_symbols", "triple_parallel_symbols"]:
                        if key == "parallel_symbols": sym_ids = 12
                        if key == "double_parallel_symbols": sym_ids = 13
                        if key == "triple_parallel_symbols": sym_ids = 14
                        symbols_info[key] = self.fuse(torch.cat(value, dim=0), sym_ids=sym_ids)
                    elif key == "text_symbols_str":
                        pass
                    else:
                        raise ValueError(f"Unknown key: ({key})")
                else:
                    # no need to assign empty [], because "symbols_info" is "defaultdict(list)",
                    # so even key doesn't exist, it wil return [].
                    pass

            all_symbols_info.append(symbols_info)
        
        return all_symbols_info

    def fuse(self, sym_feat, sym_ids):
        symbols_num = sym_feat.size(0)
        symbols_ids = torch.LongTensor([sym_ids]).repeat(symbols_num).to(sym_feat.device)
        
        symbols_embeddings = self.sym_embeddings(symbols_ids)   # [N, h]
        symbols_feat = self.sym_head(sym_feat)                  # [N, h]
        
        return self.fuse_ac(self.fuse_nn(symbols_embeddings + symbols_feat))
        

    def ocr(self, image, box):
        # 这里加一个transforms只要resize, RandomHorizontalFlip,
        # 这里img: [W, H], -> 我们在用的时候使用np.array(img)-> [H, W, 3],
        # 然后把通道转一下, np.array(img)[:, :, ::-1]
        # 因为原图是"RGB", 我们转为"BGR"
        
        # [W, H] -> [w_c, h_c]
        image_crop = image.crop(box[0].tolist())
        
        # [w_c, h_c] -> [h_c, w_c, 3] -> RGB -> BGR
        image_crop_array = np.array(image_crop)[:, :, ::-1]
        
        ocr_res = self.easyocr.readtext(image_crop_array, detail=0)
        
        return " ".join(ocr_res)