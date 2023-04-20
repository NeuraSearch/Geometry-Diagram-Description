# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from .bounding_box import BoxList
import torchvision

import itertools

def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = torchvision.ops.nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)

def boxlist_nms_imb(boxlist, nms_thresh, CLASSES_INSTANCE_NUM, ratio, \
                    max_proposals=-1, score_field="scores"): 
    """
        Performs non-maximum suppression on a boxlist, with scores specified
        in a boxlist field via score_field.

        Arguments:
            boxlist(BoxList)
            nms_thresh (float)
            max_proposals (int): if > 0, then only the top max_proposals are kept
                after non-maximum suppression
            score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    label = boxlist.get_field('labels')
    # adjust the score according to imbalance ratio
    score_final = score / (CLASSES_INSTANCE_NUM[label]**ratio)

    keep = torchvision.ops.nms(boxes, score_final, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)

def boxlist_ml_nms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
        Performs non-maximum suppression on a boxlist, with scores specified
        in a boxlist field via score_field.

        Arguments:
            boxlist(BoxList)
            nms_thresh (float)
            max_proposals (int): if > 0, then only the top max_proposals are kept
                after non-maximum suppression
            score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox    # [#box, 4]
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)
    # tensor([n0, n1, n2 ... ]) NMS选的点
    keep = torchvision.ops.batched_nms(boxes, scores, labels.float(), nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]     # BoxList的__getitem__()已经实现slice
    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
        Only keep boxes with both sides >= min_size
        Arguments:
            boxlist (Boxlist)
            min_size (int)
    """
    # TODO maybe add an API for querying the ws / hs
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = (
        (ws >= min_size) & (hs >= min_size)
    ).nonzero().squeeze(1)
    return boxlist[keep]

def choose_confident_boxes(boxlist, score_thresh):
    """
    Only keep boxes with score >= score_thresh

    Arguments:
        boxlist (Boxlist)
        score_thresh (float)
    """
    scores_list = boxlist.get_field("scores")
    keep = (scores_list>score_thresh).nonzero().squeeze(1)
    return boxlist[keep]

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """
        Compute the intersection over union of two set of boxes.
        The box order must be (xmin, ymin, xmax, ymax).

        Arguments:
        box1: (BoxList) bounding boxes, sized [N,4].
        box2: (BoxList) bounding boxes, sized [M,4].

        Returns:
        (tensor) iou, sized [N,M].

        Reference:
        https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)
    area1 = boxlist1.area()
    area2 = boxlist2.area()
    box1, box2 = boxlist1.bbox, boxlist2.bbox
    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]
    TO_REMOVE = 1
    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
        Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


def cat_boxlist(bboxes):
    """
        Concatenates a list of BoxList (having the same image size) into a
        single BoxList
        Arguments:
            bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    # bboxes: list[BoxList], each BoxList is all boxes in one layer,
    # here, "size" refers to the image size.
    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)
    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)
    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)
    # bbox.bbox: [#_box_in_that_layer, 4]
    # _cat([bbox.bbox for bbox in bboxes], dim=0): [#_box_in_all_laters, 4]
    # cat_boxes: new created BoxList
    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        content = [bbox.get_field(field) for bbox in bboxes]
        if isinstance(content[0], torch.Tensor):
            content_new = torch.cat(content, dim=0)
            cat_boxes.add_field(field, content_new)
        else:
            # !!! to add non-tensor field, e.g., "layers"
            try:
                content = list(itertools.chain(*content))
                cat_boxes.add_field(field, content)
            except TypeError:
                print(field)
                print(content)
                exit()
                
    return cat_boxes
