# coding:utf-8

import os
import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import cv2
import math
import torch
import torch.nn as nn
import numpy as np

from sklearn.cluster import MeanShift
from skimage import measure
from scipy import optimize

from image_structure import BoxList, GeoList
from image_structure import remove_small_boxes, cat_boxlist, choose_confident_boxes, boxlist_nms, boxlist_ml_nms

class FCOSPostProcessor(nn.Module):
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float) 0.05
            pre_nms_top_n (int) 1000
            nms_thresh (float) 0.6
            fpn_post_nms_top_n (int) 100
            min_size (int) 0
            num_classes (int) 81
            box_coder (BoxCoder) False
        """
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled

    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            image_sizes, layer_num):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
            *********************************************
            locations: Tensor(h_n * w_n, 2)
            box_cls: Tensor(b, 16, h_n, w_n)
            box_regression: Tensor(b, 4, h_n, w_n)
            centerness: Tensor(b, 1, h_n, w_n)
        """
        
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)                  # [b, h, w, c]
        box_cls = box_cls.reshape(N, -1, C).sigmoid()                           # [b, h*w, c]
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)    # [b, h, w, 4]
        box_regression = box_regression.reshape(N, -1, 4)                       # [b, h*w, 4]
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)            # [b, h*w, 1]
        centerness = centerness.reshape(N, -1).sigmoid()                        # [b, h*w]

        candidate_inds = box_cls > self.pre_nms_thresh  # [b, h*w, c]
        pre_nms_top_n = candidate_inds.contiguous().view(N, -1).sum(1)  # [b], number of qualified cand for each data
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)     # [b]

        # multiply the classification scores with centerness scores
        # [b, h*w, c]
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]    # [h*w, c]
            per_candidate_inds = candidate_inds[i]  # [h*w, c]
            per_box_cls = per_box_cls[per_candidate_inds]   # [?] select point

            per_candidate_nonzeros = per_candidate_inds.nonzero()   # [?, 2(x,y)]
            per_box_loc = per_candidate_nonzeros[:, 0]              # [?]
            per_class = per_candidate_nonzeros[:, 1] + 1            # [?]

            per_box_regression = box_regression[i]                  # [h*w, 4]
            per_box_regression = per_box_regression[per_box_loc]    # [?, 4] 
            per_locations = locations[per_box_loc]                  # [?, 2] 

            per_pre_nms_top_n = pre_nms_top_n[i]                    # [1] 

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            # [?, 4]
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist.add_field("layers", [layer_num for _ in range(per_class.size(0))])

            boxlist = boxlist.clip_to_image(remove_empty=False)     # 根据image的大小要把过大的box crop
            boxlist = remove_small_boxes(boxlist, self.min_size)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, image_sizes):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                                    applying box decoding and NMS
        """
        """
            - locations: List[Tensor(h_n * w_n, 2), ...], len(locations)==5
            - box_cls:  List[Tensor(b, 16, h_n, w_n), ...], len(locations)==5
            - box_regression: List[Tensor(b, 4, h_n, w_n), ...], len(locations)==5
            - centerness: List[Tensor(b, 1, h_n, w_n), ...], len(locations)==5
            - image_sizes: [H, W]
        """
        
        sampled_boxes = []  # [ [BoxList, ...] x bsz, ... ] x 5
        for layer_num, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            """
                l: Tensor(h_n * w_n, 2)
                o: Tensor(b, 16, h_n, w_n)
                b: Tensor(b, 4, h_n, w_n)
                c: Tensor(b, 1, h_n, w_n)
            """
            # sampled_boxes[-1]: [BoxList, ...] x bsz
            sampled_boxes.append(
                self.forward_for_single_feature_map(
                    l, o, b, c, image_sizes, layer_num+3
                )
            )
        
        # [ (P3_B1, P3_B2, ... , P3_BN), (P4_B1, P4_B2, ... , P4_BN), (P7_B1, P7_B2, ... , P7_BN)]
        # [ (P3_B1, P4_B1, ... , P7_B1), (P3_B2, P4_B2, ... , P7_B2), ... ] 5层
        boxlists = list(zip(*sampled_boxes))
        # [ BoxList(#box, 4), BoxList(#box, 4), ... ] len==bsz
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not self.bbox_aug_enabled:
            # boxlists: List[ [BoxList, ...] ], len(boxlists) = bsz
            boxlists = self.select_over_all_levels(boxlists)
        
        # TODO
        # Filter out candidates with low scores 
        score_thresh = 0.50
        boxlists = [choose_confident_boxes(boxlist, score_thresh) for boxlist in boxlists]
        # Filter out the overlapped candidates
        nms_thresh=0.9
        boxlists = [boxlist_nms(boxlist, nms_thresh) for boxlist in boxlists]
        
        for index in range(len(boxlists)):
            ids = []
            # generate ids of non-geometric primitives
            for num in range(len(boxlists[index])):
                ids.append('s'+str(num))
            boxlists[index].add_field("ids", ids)
            # get labels of text primitives
            labels_text = [0 if item==1 else -1 for item in boxlists[index].get_field('labels')]
            boxlists[index].add_field("labels_text", labels_text)

        return boxlists

    def select_over_all_levels(self, boxlists):
        # boxlists: [ BoxList(#box, 4), BoxList(#box, 4), ... ]
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            # result: [BoxList, BoxList, ...] 通过NMS选出来的点
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)
            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")     # cls_scores已经slice过了, BoxList的__getitem__()实现
                # 因此cls_scores与result是一样的长度, 虽然不一一对应, 但是result保存的Box的cls_score一定在cls_scores中
                # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
                # 如果number_of_detections=10, self.fpn_post_nms_top_n=5
                # 我们只保存60-100, 那么找到第6(10-5+1)小的值(60), 所有大于60的都满足
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        # List[ [BoxList, ...] ], len(results) = bsz
        return results

class GEOPostProcessor(nn.Module):
    """
        Performs post-processing on the outputs of the geo masks.
        This is only used in the testing.
    """
    def __init__(self, cfg):

        super(GEOPostProcessor, self).__init__()
        self.cfg = cfg
        self.band_width = 3.0/2.2
        self.min_area_ratio = [0.3, 0.16, 0.2]
        self.bin_seg_th = [0.5, 0.5, 0.5]
        
    def forward(self, bin_seg, embedding, image_sizes, fpn_stride):
        """
        Arguments:
            binary_seg: N*3*H*W [Tensor]
            embedding: N*EMB_DIMS*H*W [Tensor]
            image_sizes: 
        Returns:
            geolists (list[GeoList]): the post-processed masks, after
                applying mean-shift cluster and other methods
        """
        geolists = []

        for bin_seg_i, embedding_i, img_size in zip(bin_seg, embedding, image_sizes):
            # [H, W, E]
            embedding_i = embedding_i.permute(1,2,0).cpu().numpy()
            # [H, W, C], C=3
            bin_seg_i = torch.sigmoid(bin_seg_i).permute(1,2,0).cpu().numpy()
            # astype(np.uint8), False->0->0, True->1->255
            bin_seg_i = (bin_seg_i>self.bin_seg_th).astype(np.uint8)*255
            img_size = (round(img_size[1]/fpn_stride), round(img_size[0]/fpn_stride))
            geolist = self.forward_for_single_img(bin_seg_i, embedding_i, img_size)
            geolists.append(geolist)

        # geolists: [GeoList, ...]
        #   - masks: [H, W, geo数量]
        #   - size: (W,H)
        #   - extra_fields:
        #       {
        #         "labels": [geo_idx, ...],
        #         "locs": [[[x, y]], [[x1, y1], [x2, y2]], [[x,y], radius, "1111"]],
        #         "ids": [p0, l1, c0]
        #       }
        return geolists


    def forward_for_single_img(self, bin_seg, embedding, img_size):
        """
        First use mean shift to find dense cluster center.
        Arguments:
            bin_seg: numpy [H, W, 3], each pixel is 0 or 1, 0 is background pixel
            embedding: numpy [H, W, embed_dim]
            band_width: cluster pixels of the same instance within the range of band_width from the center, 
                        less small band_width is better for distinguishing similar instances  
        Return:
            cluster_result: numpy [H, W], index of different instances on each pixel
        """

        masks, classes = [], []
        width, height = img_size[0], img_size[1]

        # get point instance
        """
        1-connectivity    2-connectivity     diagonal connection close-up

            [ ]           [ ]  [ ]  [ ]             [ ]
             |               \  |  /                 |  <- hop 2
       [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
             |               /  |  \             hop 1
            [ ]           [ ]  [ ]  [ ]
        如果一个点在"2-connectivity"下他们相连, 那代表他们属于一类,
        ???这样做的目的因为instance segmentation的结果可能有多个点,
        不同的点应该是不连通的, 因此用这种方式区分他们
        """
        label_img = measure.label(bin_seg[:,:,0], connectivity = 2)
        avg_area_point = (label_img>0).sum()/label_img.max()
        # np.unique(label_img)相互连通的点会用一个数字表示
        for idx in np.unique(label_img):
            if idx!=0:
                # (label_img==idx)取出这个点的mask
                # 用这个image自己的的(H,W)去得到和原始image一样大小的mask
                mask = (label_img==idx)[:height, :width]
                # 不是谁都可以算成mask, 满足下列条件才行
                if mask.sum() > avg_area_point*self.min_area_ratio[0]:
                    masks.append(mask)
                    classes.append(1)
        
        # get line and circle instance
        # [H, W]
        bin_seg_lc = np.sum(bin_seg[:,:,1:], axis=-1)
        # [H, W]
        cluster_result = np.zeros(bin_seg_lc.shape, dtype=np.int32)
        # 这里是把pixel大于0的点全部取了出来,
        # 这里embedding求的时候是使用FPN的输出加loc_map的输出
        # [?, emb]
        cluster_list = embedding[bin_seg_lc>0]
        if len(cluster_list)==0:
            return self.construct_GeoList(masks, classes, img_size)
        # cluster pixels into instances
        # 使用MeanShift将每个点进行cluster, 判断是否属于同样的instance
        # 为什么line和circle放在一起呢 ???
        # Pros: 不用像K-means一样提供K
        mean_shift = MeanShift(bandwidth=self.band_width, bin_seeding=True)
        mean_shift.fit(cluster_list)
        # [?, ]
        labels = mean_shift.labels_
        # [H, W]
        cluster_result[bin_seg_lc>0] = labels + 1
        avg_area_lc = (cluster_result>0).sum()/cluster_result.max()

        # screen out line, circle and noise instance
        for idx in np.unique(cluster_result):
            if idx!=0:
                # [H, W]
                mask = (cluster_result==idx)[:height, :width]
                # [H, W, 3] -> [h, w] sum -> num_of_line_pixels
                line_pixels = bin_seg[:height, :width, 1][mask].sum()
                # [H, W, 3] -> [h, w] sum -> num_of_circle_pixels
                # ??? 这里line_pixels不等于circle_pixels, 虽然他们用的一个mask, 但是取得地方分别是它们各自的[H,W](sigmoid)>threshold矩阵,
                #     并且取得的值存在0的情况, 因为mask是只要是line或circle的pixel超过threshold就可以的
                circle_pixels = bin_seg[:height, :width ,2][mask].sum()
                # filter out noise
                label_img = measure.label(mask, connectivity = 2)
                props = measure.regionprops(label_img)
                max_area = 0
                for prop in props:
                    max_area = max(max_area, prop.area)
                    if prop.area<avg_area_lc*0.01: 
                        mask[label_img==prop.label]=False
                # classify geo instances (line and circle)  
                if line_pixels>circle_pixels and max_area>avg_area_lc*self.min_area_ratio[1]:
                    # masks.append(mask)
                    # classes.append(2)
                    self.seperate_ins(mask, 2, masks, classes, avg_area_lc)
                if line_pixels<=circle_pixels and max_area>avg_area_lc*self.min_area_ratio[2]:
                    masks.append(mask)
                    classes.append(3)
                    # self.seperate_ins(mask, 3, masks, classes, avg_area_lc)
        
        # masks: [H, W], ...
        # classes: {1,2,3}, ...
        return self.construct_GeoList(masks, classes, img_size)

    def seperate_ins(self, mask, label, masks, classes, avg_area_lc):
        '''
            some case: diffenent similar instances are clustered into one instance 
            this function sperate them according to spatial location
        '''
        # 用rectangle(9*9)作为kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        # 执行dilate操作, 填补缺陷
        mask_new = cv2.dilate(mask.astype(np.uint8)*255, kernel) 
        label_img = measure.label(mask_new, connectivity = 2)
        for idx in np.unique(label_img):
            if idx!=0:
                mask_ins = (label_img==idx)&mask
                if mask_ins.sum()>avg_area_lc*self.min_area_ratio[label-1]:
                    masks.append(mask_ins)
                    classes.append(label)

    def get_parse_loc(self, classes, masks, ids):
        """
            obtain the parsing location of geo instance
        """
        def f_2(center_loc, data_loc):
            """ 
            Arguments:
                data_loc: ndarray[num, 2]
                center_loc: ndarray[2]
            Returns:
                fitting error
            """
            Ri = np.linalg.norm(data_loc-center_loc,axis=1)
            return Ri - Ri.mean()

        def get_point_dist(p0, p1):
            dist = math.sqrt((p0[0]-p1[0])**2
                    +(p0[1]-p1[1])**2)
            return dist

        """
        classes: {0,1,2}, ...
        masks: [H, W], ...
        ids: [p0, l0, p1, c0, l1, c1, ...]
        """

        locs = []
        point_locs = {}
        point_name_list = []
        
        for class_index, mask, id in zip(classes, masks, ids):
            if class_index==1: # point
                y_list, x_list = np.where(mask>0) 
                point_locs[id]=[x_list.mean(), y_list.mean()] # center of the pixel set
                point_name_list.append(id)

        for class_index, mask, id in zip(classes, masks, ids):
            if class_index==1: # point
                locs.append([point_locs[id]])   # [x, y]
            elif class_index==2: # line
                mask_loc = np.where(mask>0)
                x_min, x_min_index = np.min(mask_loc[1]), np.argmin(mask_loc[1])
                x_max, x_max_index = np.max(mask_loc[1]), np.argmax(mask_loc[1])
                y_min, y_min_index = np.min(mask_loc[0]), np.argmin(mask_loc[0])
                y_max, y_max_index = np.max(mask_loc[0]), np.argmax(mask_loc[0])
                if x_max-x_min>y_max-y_min:
                    endpoint_loc_1 = [x_min, mask_loc[0][x_min_index]]
                    endpoint_loc_2 = [x_max, mask_loc[0][x_max_index]]
                else:
                    endpoint_loc_1 = [mask_loc[1][y_min_index], y_min]
                    endpoint_loc_2 = [mask_loc[1][y_max_index], y_max]
                nearst_point = None
                line_loc = []
                # the endpoints of line are determined by the point instances
                for endpoint_loc in [endpoint_loc_1, endpoint_loc_2]:
                    point_name_list.sort(key=lambda x: get_point_dist(point_locs[x], endpoint_loc))
                    if point_name_list[0]!=nearst_point: 
                        nearst_point = point_name_list[0]
                    else:
                        nearst_point = point_name_list[1]
                    line_loc.append(point_locs[nearst_point])
                locs.append(line_loc)   # [[x1, y1], [x2, y2]]
            elif class_index==3: # circle
                point_list = np.stack(list(np.where(mask>0)),axis=-1)
                center_estimate = np.mean(point_list, axis=0) 
                # The center location of circles are obtained by the least square method
                center, _ = optimize.leastsq(f_2, center_estimate, args=(point_list))
                radius = np.linalg.norm(point_list-center,axis=1).mean()
                # TODO get explicit quadrant
                locs.append([[center[1], center[0]], radius, '1111'])
        
        return locs

    def construct_GeoList(self, masks, classes, size):
        # masks: [H, W], ...
        # classes: {1,2,3}, ...
        
        # [H, W, num_of_instance]
        masks_new = np.stack(masks, axis=-1).astype(np.uint8)*255
        ids = []
        id_num = [0, 0, 0]
        # 分配 p0,p1, l0,l1, c0,c1
        for item in classes:
            if item==1: 
                ids.append('p'+str(id_num[item-1]))
            elif item==2: 
                ids.append('l'+str(id_num[item-1]))
            elif item==3:
                ids.append('c'+str(id_num[item-1]))
            id_num[item-1] +=1

        pred_seg = GeoList(masks_new, size)
        pred_seg.add_field("labels", classes)
        pred_seg.add_field("locs", self.get_parse_loc(classes, masks, ids))
        pred_seg.add_field("ids", ids)

        return pred_seg

def make_fcos_postprocessor(config):

    pre_nms_thresh = config.rpn_inference_th        # 0.05
    pre_nms_top_n = config.rpn_pre_nms_top_n        # 1000
    nms_thresh = config.rpn_nms_th                  # 0.5
    fpn_post_nms_top_n = config.rpn_det_per_img     # 100
    bbox_aug_enabled = config.rpn_bbox_aug_enabled  # Fasle
    box_selector = FCOSPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.FCOS.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector

def make_seg_postprocessor(config):

    geo_selector = GEOPostProcessor(config)

    return geo_selector
