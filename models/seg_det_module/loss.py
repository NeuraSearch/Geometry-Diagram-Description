# coding:utf-8

"""
This file contains specific functions for computing losses of FCOS
file
"""

import os
import bisect
import operator
import itertools

import torch
import torch.nn.functional as F
from torch import nn
import torchvision
import numpy as np

from .layers import IOULoss

INF = 100000000

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    gt = torch.zeros_like(logits, device=logits.device)
    for i in range(gt.size(0)):
        # "labels" contain 0 as background, however, in "logits", we don't consider the probs for background,
        #  hence, the 0-idx in "logits" is "text" class
        la = labels[i] - 1
        if la >= 0:
            gt[i, la] = 1
    
    loss = torchvision.ops.sigmoid_focal_loss(inputs=logits, targets=gt, alpha=alpha, gamma=gamma)
    
    return loss.sum()

class FCOSLossComputation(object):
    """
    This class computes the FCOS losses.
    """
    def __init__(self, cfg):
        self.cls_loss_func = sigmoid_focal_loss
        self.fpn_strides = cfg.fpn_strides
        self.center_sampling_radius = cfg.rpn_center_sampling_radius
        self.iou_loss_type = cfg.rpn_iou_loss_type
        self.norm_reg_targets = cfg.rpn_norm_reg_targets

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[...,0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        """
            - points: List[Tensor(h_n * w_n, 2), ...], len(points)==5
        """
        # P3 -> P7
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        
        # # # # # # # # # 为每个点创建它所属的level的[min, max] # # # # # # # # #
        # 存每个level的object_sizes_of_interest_per_level: List[Tensor(h_n * w_n, 2(min,max)), ...], len(locations)==5
        # min, max是这个level的点如果(l,r,t,b)最大值在[min, max], 它就可以作为positive data
        expanded_object_sizes_of_interest = []
        # points_per_level: 每一个level的feature map映射回image上的坐标
        for l, points_per_level in enumerate(points):
            # object_sizes_of_interest_per_level: Tensor([-1, 64]),
            # 这个是用来判断每个点的(l,t,r,b)的最大值是否在这个范围内, 如果在那么这个点可当作GT Box的点
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            # [points_per_level, 2], 每个点都要检查一下是否在范围内, 从而判断这个level的这个点是否可以做GT Box的点
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # expanded_object_sizes_of_interest本身是list, 然后存了一堆tensor, 我们将它们整体变成tensor
        # [#P_all_n, 2], #P_all_n是所有feature level的点的个数相加得到的
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        # 既然我们将expanded_object_sizes_of_interest每层feature map合并了, 那么我们记录下每层点的个数
        # num_points_per_level: 每一层的points数量[#p3, #p4, #p5, #p6, #p7]
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        # 将points也合并成tensor
        # [#P_all_n, 2], 每个点都不一样
        points_all_level = torch.cat(points, dim=0)
        # !!! 这里注意, expanded_object_sizes_of_interest和points_all_level虽然shape相同,
        #     但是, 前者每个level的点的两个值是一样的, 后者每个点的两个值不同(坐标)
        
        # labels: [#points_per_level] x bsz, 对应的是feature map上每个点分配的label_idx
        #   这里每层的点都会分配一个label, 有些点直接就被分配了0(背景), 有些点分配的GT Box不一定是真的, 这里无所谓
        #   NOTE: 我想强调的是每个点都有一个lable, 从而labels.size(0) == reg_targets.size(0)
        # reg_targers: [#points_per_level, 4] x bsz, 对应的是feature map上每个点分配的GT Box
        labels, reg_targets, sym_ids = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )
        
        # we extract the layer from which the golden GT box is based on
        # all_labels_to_layer: [{"s1": P3, "s0": P5, ...}, ...] len==bsz
        all_labels_to_layer = self.extract_golden_label_layer_num(
            labels, reg_targets, sym_ids
        )

        # 循环每个batch
        for i in range(len(labels)):
            # lables[i]被分割成5部分(P3-P5), 每部分的大小都是[#_points_in_that_level]
            # [#_points_in_that_level] x 5
            # labels[i]代表的是一个data的所有点的标注信息[#points_per_level]
            # num_points_per_level代表每层的point的个数
            # labels[i]被变成 Tuple([#P3], [#P4], ... , [#P7])
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            # reg_target[i]: Tuple([#P3, 4], [#P4, 4], ... , [#P7, 4])
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)
        # 之后, labels和reg_targets变成, 每个data包含的每一层的点的信息
        # lables: [Tuple([#P3], [#P4], ... , [#P7])] x bsz
        # reg_targets: [Tuple([#P3, 4], [#P4, 4], ... , [#P7, 4])] x bsz

        labels_level_first = []
        reg_targets_level_first = []
        # points: List[Tensor(h_n * w_n, 2), ...], len(points)==5
        # 循环每一个level
        for level in range(len(points)):
            # labels: [#points_per_level] x bsz
            # 将每个data的第level层的labels整合到一起
            # labels_per_im[level]: [#P_n] 每个data的第level层的点的标注信息
            # labels_level_first[-1]: [bsz, #P_n]
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            # reg_targets_per_level: [bsz, #P_n, 4]
            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            # !!! 这一步这里没有使用, 我们的(l,t,r,b)都是基于原图尺寸做的
            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        """
            - labels_level_first: List[Tensor(bsz, #P_n)], len(labels_level_first)==5
            - reg_targets_level_first: List[Tensor(bsz, #P_n, 4)], len(reg_targets_level_first)==5
        """
        return labels_level_first, reg_targets_level_first, all_labels_to_layer

    def extract_golden_label_layer_num(self, labels, reg_targets, sym_ids):
        """For train, this is used to find out the feature map on which the GT box is mapped on,
            so that we could extract the feature vector for the GT box on the correct layer of feature map.
        
            Args:
                labels: [#points_per_level] x bsz
                reg_targets: [#points_per_level, 4] x bsz
                sym_ids: [#points_per_level] x bsz, each item is str
        """
        # e.g., [#P3, #P4, #P5] -> [#P3, #P3+#P4, #P3+#P4+#P5]
        layer_points_num_accu = itertools.accumulate(self.num_points_per_level, operator.add)
        assert layer_points_num_accu[-1] == labels.size(0)
        
        # [{"s1": P3, "s0": P5, ...}, ...] len==bsz
        all_labels_to_layer = []
        for b_idx, label in enumerate(labels):
            # select points positions assigned to non-background
            selected_pos = torch.nonzero(labels > 0)    # [?]
            
            reg_selected = reg_targets[b_idx][selected_pos]     # [?, 4]
            # a GT box might be assigned to multiple points, we select the one with highest centerness score
            center_scores = self.compute_centerness_targets(reg_selected)   # [?]
            
            la_to_layer = {}
            la_to_cs = {}
            for i, pos in enumerate(selected_pos):
                assert label[pos] != 0 and sym_ids[pos] != "sb"

                sym_ids  = sym_ids[pos]
                layer_num = bisect.bisect_right(layer_points_num_accu, pos)
                if sym_ids not in la_to_layer:
                    #  0  1  2  3  4
                    # P3 P4 P5 P6 P7
                    la_to_layer[sym_ids] = layer_num + 3
                    la_to_cs[sym_ids] = center_scores[i]
                else:
                    if center_scores[i] > la_to_cs[sym_ids]:
                        la_to_layer[sym_ids] = layer_num + 3
                        la_to_cs[sym_ids] = center_scores[i]
            
            all_labels_to_layer.append(la_to_layer)
        
        return all_labels_to_layer
        
    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        """
            - locations: [#points_per_level, 2], 每个点都不一样
            - targets:
            - object_sizes_of_interest: [#points_per_level, 2], 每个level的每个点的两个数字一样
        """
        labels = []
        reg_targets = []
        sym_ids = []    # NOTE: for recording "s0, s1, s2, ..."
        # locations里面的点是根据行数循环的, 固定列, 先循环行, 然后再改变列, 再循环行
        #   xs: 0 8 16 ... 624 632 0 8 16 ... 624 632 ...
        #   ys: 0 0 0  ... 0   0   8 8 8  ... 8   8   ...
        # [#points_per_level]
        xs, ys = locations[:, 0], locations[:, 1]

        # 处理batch中的每一个data
        for im_i in range(len(targets)):
            # # # # # # # # # 求解feature map的每个点与GT Box的(l,t,r,b) # # # # # # # # #
            targets_per_im = targets[im_i]
            if len(targets_per_im)==0: continue
            assert targets_per_im.mode == "xyxy"
            # bboxes: [sym数量, 4]
            bboxes = targets_per_im.bbox
            # labels_per_im: [idx, ...] [sym数量]
            labels_per_im = targets_per_im.get_field("labels")
            # area: 求得每个GT Box的面积, [sym数量]
            area = targets_per_im.area()
            # [points_per_level_3 + points_per_level_2 + ... + points_per_level_7, 1] - [1, #sym]
            # 每个feature map的每个点都和GT Box对应的坐标相减
            # xmin = cx - l*S ymin = cx - t*S xmax = cx + r*S ymax = cx + b*S
            # 其实我觉得应该是sl,st,sr,sb, 因为xs, bboxes都是在原图上的坐标
            # [points_per_level_3 + points_per_level_2 + ... + points_per_level_7, #sym]
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            # 虽然bboxes提前了, 他是[1, #sym], xs[:, None]是[points_per_level_3 + points_per_level_2 + ... + points_per_level_7, 1]
            # 仍然是每个xmax减去每一层中心点
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            # l,t,r,b 都是 [#points_per_level, #sym]
            # 将每一层每个sym的l,t,r,b拼接在一起: [#points_per_level, #sym, 4]
            # 如此得到feature map上的每个点与GT Box的(l,t,r,b)
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            
            # # # # # # # # # 判断feature map上的每个点是否在GT Box中 # # # # # # # # #
            if self.center_sampling_radius > 0:  # only computing center point loss
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # 根据文章, 只要这个点落在GT box就算positive box,
                # 那这里, 既然(l,t,r,b)中的四个值中的最小值都>0,那么其他四个值也>0, 从而这个点在GT Box中
                # reg_targets_per_im.min(dim=2), 返回最小值和index, 因此取[0], 得到最小值[#points_per_level, #sym]
                # is_in_boxes: [#points_per_level, #sym], 每一行代表每个点是{True, False}, 代表这个点(i)是否被分给GT Box(j)
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # # # # # # # # # 用每个点与GT Box的(l,t,r,b)的最大值判断GT Box可以被哪一层的点接收 # # # # # # # # #
            # [#points_per_level, #sym]
            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            # object_sizes_of_interest[:, [0]]: [#points_per_level, 1]
            # is_cared_in_the_level: [#points_per_level, #sym]
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

            # area: 每个GT Box的面积, 和#sym一个长度 -> area[None]: [1, #sym]
            # -> locations_to_gt_area: [#points_per_level, #sym]
            # !!! 相当于每一行都一样, 都是#sym个GT Box, 每一行每一列的数字就是一个点对应的某个GT Box的面积
            locations_to_gt_area = area[None].repeat(len(locations), 1)
            # !!! 若一个点被分给不同GT Box, 将这个点分配给面积最小的,
            #     于是, 第一步我们先把上面(1)不再GT Box, (2)不满足层感受野大小设置的, 面积置为一个很大的值INF
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # 每一个level的每一个中心点只能有一个分配的真实box
            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            # locations_to_min_area: [#points_per_level], 每个点分配的GT Box的面积
            # locations_to_gt_inds: [#points_per_level], 每个点分配的GT Box的index
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            # reg_targets_per_im: [#points_per_level, #sym, 4], 每个level的每个中心点对应所有真实box的(l,t,r,b)
            # !!! locations_to_gt_inds是一维的, range(len(locations))我们是一行一行取的, 如果用`:`, 则会在每一行取locations_to_gt_inds
            # reg_targets_per_im: [#points_per_level, 4], 取得每个点分配的GT Box的(l,t,r,b)
            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            # 将针对reg_targets_per_im的每个level每个点对应的sym_idx提取出来
            # 因为labels_per_im([#sym])是按照顺序排列的, 但是我们针对每个点选取的是哪个GT Box可不是按顺序排列的
            # 将每个点选取的GT Box的label_id取出来[#points_per_level]
            # labels_per_im此时在输入模型的时候已经被转为tensor了
            labels_per_im = labels_per_im[locations_to_gt_inds]
            # 但是总有一些点不属于任何一个GT Box, 那么它的对应的所有GT Box的area已经在上面被置为INF,
            # 对这些点我们将它们的对应的labes设置为背景(0)
            # labels_per_im: [#points_per_level]
            # locations_to_min_area: [#points_per_level]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)
            
            # NOTE: for each point, we record which sym_ids is allocated to it
            # e.g., temp: [s1, s5, s2, ..., sb, s3], in #points_per_level length
            ids_per_im = targets_per_im.get_field("ids")
            temp = []
            for sym_idx in locations_to_gt_inds:
                if locations_to_min_area[sym_idx] == INF:
                    temp.append("sb")
                else:
                    temp.append(ids_per_im[sym_idx])
            sym_ids.append(temp)
            
        # labels: [[#points_per_level], ...] x bsz, 对应的是feature map上每个点分配的label_idx
        # reg_targers: [[#points_per_level, 4], ...] x bsz, 对应的是feature map上每个点分配的GT Box
        return labels, reg_targets, sym_ids

    def compute_centerness_targets(self, reg_targets):
        # reg_targets: [?, 4]

        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]

        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        # [?]
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, targets):
        """
            Arguments:
                locations (list[local tensor])
                box_cls (list[Tensor])
                box_regression (list[Tensor])
                centerness (list[Tensor])
                targets (list[BoxList])
            Returns:
                cls_loss (Tensor)
                reg_loss (Tensor)
                centerness_loss (Tensor)
        """
        """
            - locations: List[Tensor(h_n * w_n, 2), ...], len(locations)==5
            - box_cls:  List[Tensor(b, 16, h_n, w_n), ...], len(locations)==5
            - box_regression: List[Tensor(b, 4, h_n, w_n), ...], len(locations)==5
            - centerness: List[Tensor(b, 1, h_n, w_n), ...], len(locations)==5
            - targets: 
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)    # 16, 没考虑背景

        # 我们有GT box, 还有feature map,
        # 1. 这里是看feature map上的哪些点落入到GT box中
        # 2. 看GT box应该是哪个level的feature map
        # - labels: List[Tensor(bsz, #P_n)], len(labels)==5, 五层labels, 其中一个Tensor(bsz, #P_n)代表每一层所有batch的信息
        # - reg_targets: List[Tensor(bsz, #P_n, 4)], len(reg_targets)==5
        # - all_labels_to_layer: List[Dict{"s0": P5, "s2": P3, "s1": P6}, ...] len(all_labels_to_layer)==bsz
        labels, reg_targets, all_labels_to_layer = self.prepare_targets(locations, targets)

        box_cls_flatten = []            # pred: List[Tensor(b * #P_n, 16), ...],  5个
        box_regression_flatten = []     # pred: List[Tensor(b * #P_n, 4), ...],   5个
        centerness_flatten = []         # pred: List[Tensor([b * #P_n]), ...],    5个
        labels_flatten = []             # gold: List[Tensor([b * #P_n]), ...],    5个
        reg_targets_flatten = []        # gold: List[Tensor([b * #P_n, 4]), ...], 5个
        # 循环每一层
        for l in range(len(labels)):
            # box_cls_flatten中每一个是[b * #P_n, 16]
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            # box_regression_flatten中每一个是[b * #P_n, 4]
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            # labels_flatten中每一个是[b * #P_n]
            labels_flatten.append(labels[l].reshape(-1))
            # reg_targets_flatten中每一个是[b * #P_n, 4]
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            # [b * #P_n]
            centerness_flatten.append(centerness[l].reshape(-1))

        # [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7), 16]
        box_cls_flatten = torch.cat(box_cls_flatten, dim=0) 
        # [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7), 4]
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        # [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7)]
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        # [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7)]
        labels_flatten = torch.cat(labels_flatten, dim=0)
        # [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7), 4]
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        # 找到被分配到GT Box的点的idx
        # pos_inds: [?] 这里我也不确定?是不是等于GT Box的数量???
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        # 只关心postive point?
        # [?, 4]
        box_regression_flatten = box_regression_flatten[pos_inds]
        # [?, 4]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        # [?]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        # box_cls_flatten: [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7), 16]
        # labels_flatten: [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7)]
        # cls_loss: [(b * #P_3) + (b * #P_4) + (b * #P_5) + (b * #P_6) + (b * #P_7), 16]
        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            # reg_targets_flatten: [?, 4]
            # centerness_targets: [?], 每个postive point的 golden centerness
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            # box_regression_flatten: [?, 4]
            # reg_targets_flatten: [?, 4]
            # [?]
            # reg_loss: [?]
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            # centerness_loss: [?]
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        # cls_loss: [1]
        # reg_loss: [1]
        # centerness_loss: [1]
        # all_labels_to_layer: List[Dict{"s0": P5, "s2": P3, "s1": P6}, ...] len(all_labels_to_layer)==bsz
        return cls_loss, reg_loss, centerness_loss, all_labels_to_layer

class SegLossComputation(object):
    """
    This class computes the geo segmentation losses.
    """

    def __init__(self, cfg):

        self.discriminative_loss_func = self.discriminative_loss
        self.seg_loss_func=[]
        self.class_num = cfg.seg_num_classes-1    # 3
        self.embed_dim = cfg.seg_emb_dims        # 8
        self.delta_v = 0.5    
        self.delta_d = 3.0
        for i in range(self.class_num):
            weight_ratio = torch.tensor([10, 1, 4])   # [10, 1, 4]
            # point, line, circle的weight为什么不一样
            self.seg_loss_func.append(nn.BCEWithLogitsLoss(pos_weight=weight_ratio))

    def __call__(self, binary_seg, embedding, targets):
        """
        Arguments:
            binary_seg: N*3*H*W [Tensor]
            embedding: N*EMB_DIMS*H*W [Tensor]
            targets (list[GeoList])
        Returns:
            binary_seg_loss: [Tensor]
            var_loss: [Tensor]
            dist_loss: [Tensor]
            reg_loss: [Tensor]
        """

        reshape_size = binary_seg.shape[2:]

        ########################binary_seg################################
        binary_seg_loss = binary_seg.new([0]).zero_().squeeze()
        for i in range(self.class_num):
            seg_loss_func_cur = self.seg_loss_func[i].cuda()
            input = binary_seg[:,i,:,:]
            target = []
            for ggeo in targets:
                target.append(ggeo.get_binary_seg_target(
                                        class_index = i+1,
                                        reshape_size = reshape_size)
                            )
            target = torch.from_numpy(np.stack(target,axis=0)).float().cuda()
            binary_seg_loss = binary_seg_loss + seg_loss_func_cur(input, target)

        ########################embedding_seg################################
        seg_gt = []
        for ggeo in targets:
            # get instances of line and circle
            # instance of point obtained by connected component analysis from binary_seg
            seg_gt.append(ggeo.get_inst_seg_target(
                            class_index_list = [2,3],  
                            reshape_size = reshape_size)
                        )     
        var_loss, dist_loss, reg_loss = self.discriminative_loss(embedding, seg_gt)
        
        ########################get GT seg mask################################
        gt_point_mask_for_rel = []  # len(gt_point_mask_for_rel)==bsz
        gt_line_mask_for_rel = []
        gt_circle_mask_for_rel = []
        for ggeo in targets:
            gt_point_mask_for_rel.append(ggeo.get_inst_seg_for_rel(
                                            class_index=1,
                                            reshape_size=reshape_size)
                                        )
            gt_line_mask_for_rel.append(ggeo.get_inst_seg_for_rel(
                                            class_index=2,
                                            reshape_size=reshape_size)
                                        )
            gt_circle_mask_for_rel.append(ggeo.get_inst_seg_for_rel(
                                            class_index=3,
                                            reshape_size=reshape_size)
                                        )
              
        return binary_seg_loss, var_loss, dist_loss, reg_loss, gt_point_mask_for_rel, gt_line_mask_for_rel, gt_circle_mask_for_rel

    def discriminative_loss(self, embedding, seg_gt):
        
        batch_size = embedding.shape[0]
        assert batch_size == len(seg_gt), 'embedding batch size is not equal to seg_gt batch size'

        var_loss = embedding.new([0]).zero_().squeeze()
        dist_loss = embedding.new([0]).zero_().squeeze()
        reg_loss = embedding.new([0]).zero_().squeeze()

        for b in range(batch_size):
            embedding_b = embedding[b] # [embed_dim, H, W]
            seg_gt_b = seg_gt[b]
            num_lanes = len(seg_gt_b)
            if num_lanes==0:
                continue
            centroid_mean = []

            for seg_mask_i in seg_gt_b:
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[:, seg_mask_i]
                mean_i = torch.mean(embedding_i, dim=1)
                centroid_mean.append(mean_i)
                var_loss = var_loss + torch.mean(F.relu(torch.norm(embedding_i-mean_i.reshape(self.embed_dim,1) \
                                                , dim=0) - self.delta_v)**2 ) / num_lanes
            
            centroid_mean = torch.stack(centroid_mean)  # [n_lane, embed_dim]

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, self.embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, self.embed_dim)
                dist = torch.norm(centroid_mean1-centroid_mean2, dim=2)  # shape [num_lanes, num_lanes]
                # diagonal elements are 0, now mask above delta_d
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype, device=embedding.device) * self.delta_d  
                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_d)**2) / (num_lanes * (num_lanes-1)) / 2

            reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size

        return var_loss, dist_loss, reg_loss

def make_fcos_loss_evaluator(cfg):
    loss_evaluator = FCOSLossComputation(cfg)
    return loss_evaluator

def make_seg_loss_evaluator(cfg):
    loss_evaluator = SegLossComputation(cfg)
    return loss_evaluator