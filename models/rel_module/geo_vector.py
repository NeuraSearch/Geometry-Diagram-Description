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

        self.point_spatial_embeddings = nn.Sequential(
            nn.Linear(2, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.ReLU(),
        )
        self.line_spatial_embeddings = nn.Sequential(
            nn.Linear(4, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.ReLU(),
        )
        self.circle_spatial_embeddings = nn.Sequential(
            nn.Linear(3, out_channel),
            nn.ReLU(),
            nn.Linear(out_channel, out_channel),
            nn.ReLU(),
        )

        self.geo_feature_nn = nn.Linear(inp_channel, out_channel, bias=False)
        self.geo_feature_ac = nn.ReLU()

        self.geo_embeddings = nn.Embedding(3, out_channel)

        self.fuse_nn = nn.Linear(out_channel, out_channel)
        self.fuse_ac = nn.ReLU()

    def forward(self, geo_feature, all_mask_tensor, geo_type, loc_info):
        """
            geo_feature: [N, c, h, w], N is the aggregate of the points (or lines, or circles) in one data.
            all_mask_tensor: [N, c, h, w]
            geo_type: str, "point" or "line" or "circle".
        """

        # tower_out: [N, geo_embed_size, h, w]
        if geo_type == "point":
           points_num = geo_feature.size(0)
           points_ids = torch.LongTensor([0]).repeat(points_num).to(geo_feature.device)
           embeddings = self.geo_embeddings(points_ids)  # [N, geo_embed_size]
           locs = [torch.FloatTensor(loc) for loc in loc_info]
           locs = torch.stack(locs, dim=0).to(geo_feature.device)   # [N, 2]
           spatial_embedding = self.point_spatial_embeddings(locs)  # [N, h]
        elif geo_type == "line":
           lines_num = geo_feature.size(0)
           lines_ids = torch.LongTensor([1]).repeat(lines_num).to(geo_feature.device)
           embeddings = self.geo_embeddings(lines_ids)  # [N, geo_embed_size]
           locs = []
           for line_info in loc_info:
               line_info = [torch.FloatTensor(p) for p in line_info]
               locs.append(torch.cat(line_info))    # [4]
           locs = torch.stack(locs, dim=0).to(geo_feature.device)  # [N, 4]
           spatial_embedding = self.line_spatial_embeddings(locs)  # [N, h]
        elif geo_type == "circle":
           circles_num = geo_feature.size(0)
           circles_ids = torch.LongTensor([2]).repeat(circles_num).to(geo_feature.device)
           embeddings = self.geo_embeddings(circles_ids)  # [N, geo_embed_size]
           locs = []
           for circle_info in loc_info:
               circle_info[0].append(circle_info[1])
               locs.append(torch.FloatTensor(circle_info[0]))
           locs = torch.stack(locs, dim=0).to(geo_feature.device)  # [N, 3]
           spatial_embedding = self.circle_spatial_embeddings(locs)  # [N, h]
        else:
            raise ValueError(f"Unknown geo_type: ({geo_type})")

        # x_coords = torch.linspace(-1, 1, steps=geo_feature.size(3)).to(geo_feature.device)
        # y_coords = torch.linspace(-1, 1, steps=geo_feature.size(2)).to(geo_feature.device)
        # grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        # grid = torch.stack([grid_x, grid_y], dim=-1)    # [h, w, 2]
        # grid_flatten = torch.flatten(grid, start_dim=0, end_dim=1)  # [h*w, 2]
        # pos_embs = self.spatial_embedddings(grid_flatten)   # [h*w, h]
        # pos_embs = torch.reshape(pos_embs, (geo_feature.size(2), geo_feature.size(3), -1))  #[height, w, h]
        # pos_embs = pos_embs.unsqueeze(0).repeat(geo_feature.size(0), 1, 1, 1)   # [N, height, w, h]
        # pos_embs = torch.permute(pos_embs, (0, 3, 1, 2))    # [N, h, w, c] -> [N, c, h, w]

        # pos_embs = pos_embs * all_mask_tensor

        # geo_feature += pos_embs   # [N, c, h, w]

        # [N, c, h, w] -> [N, c, h * w] -> [N, c]
        geo_feature = geo_feature.flatten(start_dim=2).mean(dim=-1)
        geo_feature = self.geo_feature_nn(geo_feature)
        geo_feature = self.geo_feature_ac(geo_feature)

        return self.fuse(embeddings, geo_feature, spatial_embedding)

    def fuse(self, embeddings, geo_feat, spatial_embedding):
        """
            Args:
                embeddings: [N, h]
                geo_feat: [N, h]
        """
        return self.fuse_ac(self.fuse_nn(geo_feat + spatial_embedding)) + spatial_embedding
        
        # return self.fuse_ac(self.fuse_nn(embeddings + geo_feat + spatial_embedding)) + spatial_embedding


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
                targets_seg=None,
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
            all_geo_info = self._forward_train(feature_map, gt_point_mask, gt_line_mask, gt_circle_mask, targets_seg)
        else:
            assert proposals_seg != None
            all_geo_info = self._forward_test(feature_map, proposals_seg)

        return all_geo_info

    def _forward_train(self, feature_map, gt_point_mask, gt_line_mask, gt_circle_mask, targets_seg):
        """
        Returns:
            all_geo_info: List[Dict]: Contain batch data geo information,
                each dict contains geo information regarding to different classes, in Tensor([N, cfg.geo_embed_size])
        """

        all_geo_info = []
        for b_id, (points_mask, lines_mask, circles_mask, target_seg) in enumerate(zip(gt_point_mask, gt_line_mask, gt_circle_mask, targets_seg)):

            points_loc = []
            lines_loc = []
            circles_loc = []
            for idx, la in enumerate(target_seg.get_field("labels")):
                if la == 1:
                    points_loc.append(target_seg.get_field("locs")[idx][0])
                elif la == 2:
                    lines_loc.append(target_seg.get_field("locs")[idx])
                elif la == 3:
                    circles_loc.append(target_seg.get_field("locs")[idx])
            
            all_geo_info.append(
                {
                    "points": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=points_mask, geo_type="point", loc_info=points_loc),
                    "lines": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=lines_mask, geo_type="line", loc_info=lines_loc),
                    "circles": self.get_mask_map(feature_map=feature_map[b_id], batch_mask=circles_mask, geo_type="circle", loc_info=circles_loc),
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
                locs = geolist.get_field("locs")
                seg_masks = geolist.masks               # [h, w, #geo]
                seg_masks = torch.from_numpy(seg_masks).float().cuda()
                # [h, w, N] -> [H, W, N]
                seg_masks = self.pad_mask_to_feature_map_size(seg_masks, feature_map[b_id].size()[1:])


                for i, label_idx in enumerate(labels):
                    mask = seg_masks[:, :, i]   # [h, w] !!! We assume the mask is not empty during inference.
                    if label_idx == 1:
                        # print("point: ", locs[i])
                        # [1, geo_embed_size]
                        geo_info["points"].append(self.get_mask_map(feature_map[b_id], [mask], geo_type="point", loc_info=locs[i]))
                    elif label_idx == 2:
                        # print("line: ", locs[i])
                        # [1, geo_embed_size]
                        geo_info["lines"].append(self.get_mask_map(feature_map[b_id], [mask], geo_type="line", loc_info=[locs[i]]))
                    elif label_idx == 3:
                        # print("circle: ", locs[i])
                        # [1, geo_embed_size]
                        geo_info["circles"].append(self.get_mask_map(feature_map[b_id], [mask], geo_type="circle", loc_info=[locs[i]]))

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

    def get_mask_map(self, feature_map, batch_mask, geo_type, loc_info):
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
            geo_feature = self.geo_head(geo_feature=torch.stack(all_mask_map, dim=0), all_mask_tensor=all_mask_tensor, geo_type=geo_type, loc_info=loc_info)
            return geo_feature
        else:
            return []