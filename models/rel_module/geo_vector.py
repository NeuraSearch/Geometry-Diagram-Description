# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import defaultdict

class GeoVectorHead(nn.Module):
    
    def __init__(self, inp_channel, out_channel):
        super(GeoVectorHead, self).__init__()
        
        point_head_tower = []
        point_head_tower.append(
            nn.Conv2d(
                in_channels=inp_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        point_head_tower.append(nn.GroupNorm(32, out_channel))
        point_head_tower.append(nn.ReLU())
        self.add_module("geo_point_head_tower", nn.Sequential(*point_head_tower))

        line_head_tower = []
        line_head_tower.append(
            nn.Conv2d(
                in_channels=inp_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        line_head_tower.append(nn.GroupNorm(32, out_channel))
        line_head_tower.append(nn.ReLU())
        self.add_module("geo_line_head_tower", nn.Sequential(*line_head_tower))

        circle_head_tower = []
        circle_head_tower.append(
            nn.Conv2d(
                in_channels=inp_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        circle_head_tower.append(nn.GroupNorm(32, out_channel))
        circle_head_tower.append(nn.ReLU())
        self.add_module("geo_circle_head_tower", nn.Sequential(*circle_head_tower))
        
    def forward(self, feature, geo_type):
        """
            feature: [N, c, h, w], N is the aggregate of the points (or lines, or circles) in one data.
            geo_type: str, "point" or "line" or "circle".
        """
        
        # tower_out: [N, geo_embed_size, h, w]
        if geo_type == "point":
            tower_out = self.geo_point_head_tower(feature)
        elif geo_type == "line":
            tower_out = self.geo_line_head_tower(feature)
        elif geo_type == "circle":
            tower_out = self.geo_circle_head_tower(feature)
        else:
            raise ValueError(f"Unknown geo_type: ({geo_type})")
            
        # flatten to [N, geo_embed_size, h*w]
        tower_out = tower_out.flatten(start_dim=2)
        
        # adopot GlobalMeanPooling, [N, geo_embed_size]
        # ??? alternative: GlobalMaxPooling, tower_out.max(dim=-1)[0]
        return tower_out.mean(dim=-1)

class GeoVectorBuild(nn.Module):
    
    def __init__(self, cfg):
        super(GeoVectorBuild, self).__init__()
        
        self.geo_head = GeoVectorHead(inp_channel=64,
                                      out_channel=cfg.geo_embed_size)
        
    
    def forward(self,
                feature_map,
                gt_point_mask=None,
                gt_line_mask=None,
                gt_circle_mask=None,
                proposals_seg=None):
        """
        Args:
            feature_map (Tensor(B, c, h, w)): feature of features_share and visemb_features.
            gt_point_mask (List[Tensor(h, w)]): all points mask result, in bool. bsz len
            gt_line_mask (List[Tensor(h, w)]): all lines mask result, in bool. bsz len
            gt_circle_mask (List[Tensor(h, w)]): all circles mask result, in bool. bsz len
            proposals_seg ([GeoList]): predicted GeoList.
        
        Returns:
            all_geo_info: List[Dict]: Contain batch data geo information,
                each dict contains geo information regarding to different classes, in Tensor([N, cfg.geo_embed_size])
        """

        if self.training:
            assert gt_point_mask != None and gt_line_mask != None and gt_circle_mask != None
            all_geo_info = self._forward_train(feature_map, gt_point_mask, gt_line_mask, gt_circle_mask)
        else:
            assert proposals_seg != None
            all_geo_info = self._forward_test(feature_map, proposals_seg)
        
        return all_geo_info
    
    def _forward_train(self, feature_map, gt_point_mask, gt_line_mask, gt_circle_mask):
        """
        Returns:
            all_geo_info: List[Dict]: Contain batch data geo information,
                each dict contains geo information regarding to different classes, in Tensor([N, cfg.geo_embed_size])
        """
        
        all_geo_info = []
        for b_id, (points_mask, lines_mask, circles_mask) in enumerate(zip(gt_point_mask, gt_line_mask, gt_circle_mask)):
            all_geo_info.append(
                {
                    "points": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=points_mask, geo_type="point"),
                    "lines": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=lines_mask, geo_type="line"),
                    "circles": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=circles_mask, geo_type="circle"),
                }
            )
            
        return all_geo_info

    def _forward_test(self, feature_map, proposals_seg):
        
        all_geo_info = []
        for b_id, geolist in enumerate(proposals_seg):
            
            geo_info = defaultdict(list)
            if geolist == None:
                geo_info["points"] = []
                geo_info["lines"] = []
                geo_info["circles"] = []
            else:
                labels = geolist.get_field("labels")    # [#geo]
                seg_masks = geolist.masks               # [h, w, #geo]
                seg_masks = torch.from_numpy(seg_masks).float().cuda()
                # [h, w, N] -> [H, W, N]
                seg_masks = self.pad_mask_to_feature_map_size(seg_masks, feature_map[b_id].size()[1:])
                
                
                for i, label_idx in enumerate(labels):
                    mask = seg_masks[:, :, i]   # [h, w] !!! We assume the mask is not empty during inference.
                    if label_idx == 1:
                        # [1, geo_embed_size]
                        geo_info["points"].append(self.get_mask_map(feature_map[b_id], [mask], geo_type="point"))
                    elif label_idx == 2:
                        # [1, geo_embed_size]
                        geo_info["lines"].append(self.get_mask_map(feature_map[b_id], [mask], geo_type="line"))
                    elif label_idx == 3:
                        # [1, geo_embed_size]
                        geo_info["circles"].append(self.get_mask_map(feature_map[b_id], [mask], geo_type="circle"))
                
            for key in ["points", "lines", "circles"]:
                val = geo_info[key]
                if len(val) != 0:
                    # [N, geo_embed_size]
                    geo_info[key] = torch.cat(val, dim=0)
                else:
                    geo_info[key] = []
            
            all_geo_info.append(geo_info)
        
        return all_geo_info

    def pad_mask_to_feature_map_size(self, mask, reshape_size):
        """Pad mask to the size of feature map.
            During inference, the mask is cropped according to the actual image size,
            which might be smaller than the feature map. We need to pad it.

        Args:
            mask (torch.Tensor): [h, w, N]
            reshape_size (Tuple): [H, W]
        Returns:
            pad_mask (torch.Tensor): [H, W, N]
        """
        
        mask_w = mask.size(1)
        mash_h = mask.size(0)
        reshape_w = reshape_size[1]
        reshape_h = reshape_size[0]
        
        # pad starts padding from the last dimenstion
        pad = (0, 0, 0, max(0, reshape_w - mask_w), 0, max(0, reshape_h - mash_h))
        pad_mask = F.pad(input=mask, pad=pad, mode="constant")

        return pad_mask
                
    def get_mask_map(self, feature_map, batch_mask, geo_type):
        """_summary_

        Args:
            feature_map (Tensor(feat_channel, h, w)): feature map of a data.
            batch_mask (List[Tensor(h, w)]): mask for a bunch of geos.
            geo_type: str, "point" or "line" or "circle".
        """
        all_mask_map = []
        for mask in batch_mask:
            # expand mask[h,w] to channel size [c,h,w]
            # [h, w] -> [feat_channel, h, w]
            mask_expand = mask.unsqueeze(0).repeat(feature_map.size(0), 1, 1)
            # [feat_channel, h, w] * [feat_channel, h, w] = [feat_channel, h, w]
            all_mask_map.append(feature_map * mask_expand)
                
        if len(all_mask_map) > 0:
            # [N, feat_channel, h, w] -> [N, geo_embed_size]
            geo_feature = self.geo_head(feature=torch.stack(all_mask_map, dim=0), geo_type=geo_type)
            return geo_feature
        else:
            return []