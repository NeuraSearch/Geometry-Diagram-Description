# coding:utf-8

import torch
import torch.nn as nn

from collections import defaultdict

class GeoVectorHead(nn.Module):
    
    def __init__(self, inp_channel, out_channel):
        super(GeoVectorHead, self).__init__()
        
        head_tower = []
        head_tower.append(
            nn.Conv2d(
                in_channels=inp_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
        )
        head_tower.append(nn.GroupNorm(32, out_channel))
        head_tower.append(nn.ReLU())
        self.add_module("geo_head_tower", nn.Sequential(*head_tower))

    def forward(self, feature):
        """
            feature: [N, c, h, w], N is the aggregate of the points (or lines, or circles) in one data.
        """
        
        # tower_out: [N, geo_embed_size, h, w]
        tower_out = self.geo_head_tower(feature)
        
        # flatten to [N, geo_embed_size, h*w]
        tower_out = tower_out.flatten(start_dim=2)
        
        # [N, geo_embed_size]
        return tower_out.sum(dim=-1)

class GeoVectorBuild(nn.Module):
    
    def __init__(self, cfg):
        super(GeoVectorBuild, self).__init__()
        
        self.geo_head = GeoVectorHead(inp_channel=cfg.backbone_out_channels+64+2,
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
        for b_id, (points_mask, lines_mask, circles_mask) in zip(gt_point_mask, gt_line_mask, gt_circle_mask):
            all_geo_info.append(
                {
                    "points": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=points_mask),
                    "lines": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=lines_mask),
                    "circles": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=circles_mask),
                }
            )
            
        return all_geo_info

    def _forward_test(self, feature_map, proposals_seg):
        
        all_geo_info = []
        for b_id, geolist in enumerate(proposals_seg):
            
            labels = geolist.get_field("labels")    # [#geo]
            seg_masks = geolist.masks               # [h, w, #geo]
            
            geo_info = defaultdict(list)
            for i, label_idx in enumerate(labels):
                mask = seg_masks[:, :, i]   # [h, w]
                
                if label_idx == 1:
                    # [1, geo_embed_size]
                    geo_info["points"].append(self.get_mask_map(feature_map[b_id], [mask]))
                elif label_idx == 2:
                    # [1, geo_embed_size]
                    geo_info["line"].append(self.get_mask_map(feature_map[b_id], [mask]))
                elif label_idx == 3:
                    # [1, geo_embed_size]
                    geo_info["circle"].append(self.get_mask_map(feature_map[b_id], [mask]))
            
            for key, val in geo_info.items():
                if len(val) != 0:
                    # [N, geo_embed_size]
                    geo_info[key] = torch.cat(val, dim=0)
            
            all_geo_info.append(geo_info)
        
        return all_geo_info
                
    def get_mask_map(self, feature_map, batch_mask):
        """_summary_

        Args:
            feature_map (Tensor(feat_channel, h, w)): feature map of a data.
            batch_mask (List[Tensor(h, w)]): mask for a bunch of geos.
        """
        all_mask_map = []
        for mask in batch_mask:
            # [h, w] -> [feat_channel, h, w]
            mask_expand = mask.unsqueeze(0).repeat(feature_map.size(0), 1, 1)
            # [feat_channel, h, w] * [feat_channel, h, w] = [feat_channel, h, w]
            all_mask_map.append(feature_map * mask_expand)
                
        if len(all_mask_map) > 0:
            # [N, feat_channel, h, w] -> [N, geo_embed_size]
            geo_feature = self.geo_head(torch.stack(all_mask_map, dim=0))
            return geo_feature
    
        return None