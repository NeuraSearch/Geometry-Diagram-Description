# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from collections import defaultdict

class GeoVectorHead(nn.Module):
    
    def __init__(self, inp_channel, out_channel):
        super(GeoVectorHead, self).__init__()

        # self.row_embeddings = nn.Embedding(350, inp_channel)
        # self.col_embeddings = nn.Embedding(350, inp_channel)
        # self.register_buffer("row_ids", torch.arange(350))
        # self.register_buffer("col_ids", torch.arange(350))
        self.spatial_embedddings = nn.Linear(2, inp_channel)
        
        self.geo_feature_nn = nn.Linear(inp_channel, out_channel, bias=False)
        self.geo_feature_ac = nn.ReLU()
             
        self.geo_embeddings = nn.Embedding(3, out_channel)
    
        self.fuse_nn = nn.Linear(out_channel, out_channel)
        self.fuse_ac = nn.ReLU()
    
    def forward(self, feature, all_mask_tensor, geo_type):
        """
            feature: [N, c, h, w], N is the aggregate of the points (or lines, or circles) in one data.
            all_mask_tensor: [N, c, h, w]
            geo_type: str, "point" or "line" or "circle".
        """
        
        # tower_out: [N, geo_embed_size, h, w]
        if geo_type == "point":
           points_num = feature.size(0)
           points_ids = torch.LongTensor([0]).repeat(points_num).to(feature.device)
           embeddings = self.geo_embeddings(points_ids)  # [N, geo_embed_size]
        elif geo_type == "line":
           lines_num = feature.size(0)
           lines_ids = torch.LongTensor([1]).repeat(lines_num).to(feature.device)
           embeddings = self.geo_embeddings(lines_ids)  # [N, geo_embed_size]
        elif geo_type == "circle":
           circles_num = feature.size(0)
           circles_ids = torch.LongTensor([2]).repeat(circles_num).to(feature.device)
           embeddings = self.geo_embeddings(circles_ids)  # [N, geo_embed_size]
        else:
            raise ValueError(f"Unknown geo_type: ({geo_type})")

        
        # [N, c, h, w] -> [[N, h, w, c]
        geo_feature = torch.permute(feature, (0, 2, 3, 1))
        
        # row = geo_feature.size(1)
        # col = geo_feature.size(2)
        # this_row_embeds = self.row_embeddings(self.row_ids[:row]).unsqueeze(1)   # [row, 1, h]
        # this_col_embeds = self.col_embeddings(self.col_ids[:col]).unsqueeze(0)   # [1, col, h]
        # feature_spatial_embeds = (this_row_embeds + this_col_embeds).unsqueeze(0).repeat(geo_feature.size(0), 1, 1, 1)  # [N, row, col, h]
        x_coords = torch.linspace(-1, 1, steps=feature.size(3)).to(geo_feature.device)
        y_coords = torch.linspace(-1, 1, steps=feature.size(2)).to(geo_feature.device)
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        grid = torch.stack([grid_x, grid_y], dim=-1)    # [h, w, 2]
        grid_flatten = torch.flatten(grid, start_dim=0, end_dim=1)  # [h*w, 2]
        pos_embs = self.spatial_embedddings(grid_flatten)   # [h*w, h]
        pos_embs = torch.reshape(pos_embs, (feature.size(2), feature.size(3), -1))  #[height, w, h]       
        pos_embs = pos_embs.unsqueeze(0).repeat(feature.size(0), 1, 1, 1)   # [N, height, w, h]
        
        # pos_embs = pos_embs * all_mask_tensor
        
        geo_feature += pos_embs   # [N, h, w, c]
        geo_feature = torch.permute(geo_feature, (0, 3, 1, 2))  # [N, c, h, w]
        
        # [N, c, h, w] -> [N, c, h * w] -> [N, c]
        geo_feature = geo_feature.flatten(start_dim=2).mean(dim=-1)
        geo_feature = self.geo_feature_nn(geo_feature)
        geo_feature = self.geo_feature_ac(geo_feature)
        
        return self.fuse(embeddings, geo_feature)

    def fuse(self, embeddings, geo_feat):
        """
            Args:
                embeddings: [N, h]
                geo_feat: [N, h]
        """
        return self.fuse_ac(self.fuse_nn(embeddings + geo_feat))

    
class GeoVectorBuild(nn.Module):
    
    def __init__(self, cfg):
        super(GeoVectorBuild, self).__init__()
        
        self.geo_head = GeoVectorHead(inp_channel=cfg.backbone_out_channels,
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
        all_mask_tensor = []
        for mask in batch_mask:
            # expand mask[h,w] to channel size [c,h,w]
            # [h, w] -> [feat_channel, h, w]
            mask_expand = mask.unsqueeze(0).repeat(feature_map.size(0), 1, 1)
            all_mask_tensor.append(mask_expand)
            # [feat_channel, h, w] * [feat_channel, h, w] = [feat_channel, h, w]
            all_mask_map.append(feature_map * mask_expand)
        
        if len(all_mask_map) > 0:
            all_mask_tensor = torch.stack(all_mask_tensor, dim=0)   # [N, feat_channel, h, w]
            # [N, feat_channel, h, w] -> [N, geo_embed_size]
            geo_feature = self.geo_head(feature=torch.stack(all_mask_map, dim=0), all_mask_tensor=all_mask_tensor, geo_type=geo_type)
            return geo_feature
        else:
            return []