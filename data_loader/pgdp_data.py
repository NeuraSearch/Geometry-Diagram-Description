# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import os
import json
import torch
import torch.utils.data
from PIL import Image
import numpy as np
import cv2

from image_structure import BoxList, GeoList

class GEODataset(torch.utils.data.Dataset):
    
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
    CLASSES_GEO = [
        "__background__", 
        "point","line","circle"
    ]
    CLASSES_TEXT = [
        "point", "line", "len", "angle", "degree", "area", 
    ]
    CLASSES_SEM = [
        "point","line","circle",
        "text", 
        "perpendicular", "head", "head_len",
        "angle", "bar", "parallel"
    ]
        
    THICKNESS = 3

    def __init__(self, root, ann_file, transforms=None, is_train=False, cfg=None): 
        """_summary_

        Args:
            root: image存的文件夹
            ann_file: train.json, val.json, test.json文件
            transforms: 对图片进行转换, geo_parse/data/transforms/build.py(build_transforms())
            is_train: Defaults to False.
            cfg: configs/PGDP5K/geo_MNV2_FPN.yaml(主要是模型参数), 默认的cfg定义是在geo_parse/config/defaults.py,
                !!! 但是具体文件的位置存在geo_parse/config/paths_catalog.py, 是在geo_parse/data/build.py(make_data_loader)中使用
        """
        
        self.img_root = root
        self.transforms = transforms
        self.ids = []   # 下面存每个标注的id
        self.is_train = is_train
        self.cfg = cfg
        # symbol -> idx
        self.class_sym_to_ind = dict(zip(GEODataset.CLASSES_SYM, range(len(GEODataset.CLASSES_SYM))))
        # geometry -> idx
        self.class_geo_to_ind = dict(zip(GEODataset.CLASSES_GEO, range(len(GEODataset.CLASSES_GEO))))
        # symbol有可能有文字(text)描述, text描述对应的类型 -> idx
        self.class_text_to_ind = dict(zip(GEODataset.CLASSES_TEXT, range(len(GEODataset.CLASSES_TEXT))))

        with open(MAIN_PATH / ann_file, 'rb') as file:
            self.contents = json.load(file)
        for key in self.contents.keys():
            self.ids.append(key)
            
        if self.cfg.toy_data:
            self.ids = self.ids[:50]

    def __getitem__(self, index):
        
        # 1. 取得标注信息
        img_id = self.ids[index]
        annot_each = self.contents[img_id]

        # Samples without non-geometric primitives do not participate in the training
        while self.is_train and len(annot_each['symbols'])==0:
            index = np.random.randint(0, len(self.ids))
            img_id = self.ids[index]
            annot_each = self.contents[img_id]

        # 2. 打开图片
        img_org = Image.open(os.path.join(str(MAIN_PATH / self.img_root), annot_each['file_name'])).convert("RGB")

        target_det = None
        target_seg = None
        targets_geo = None
        targets_sym = None
        if self.is_train:
            # 3.1 得到symbol的golden detection box
            """
            target_det (BoxList):
                - bbox: [sym数量, 4]
                - size: (W, H)
                - mode: "xyxy"
                - extra_fields:
                    {
                        "labels": classes: [idx, ...] symbol class idx,
                        "text_contents": [str, null, ...] 非"text"&"arrow"应该是null,
                        "ids": [s0, s1, s2, ...],
                        "labels_text": [len_idx, -1], 对于"text"&"arrow"是对应的text的class, 其他的对应的是-1,
                    }
            """
            target_det = self.get_target_det(annot_each, img_org.size)
            
            # 3.2 得到geometry的golden segmantation map
            """
            target_seg (GeoList):
                - masks: [H, W, geo数量]
                - size: (W,H)
                - extra_fields:
                    {
                        "labels": [geo_idx, ...],
                        "locs": [[[x, y]], [[x1, y1], [x2, y2]], [[x,y], radius, "1111"]],
                        "ids": [p0, l1, c0]
                    }
            """
            target_seg = self.get_target_seg(annot_each, img_org.size)
            
            # 3.3 get targets_geo
            targets_geo, (points_num, lines_num, circles_num) = self.get_geo2geo_rel(annot_each, target_seg)
        
            targets_sym = self.get_sym2geo_rel(
                annot_each=annot_each,
                target_det=target_det,
                points_num=points_num,
                lines_num=lines_num,
                circles_num=circles_num,
            )
            
        if self.transforms is not None:
            img, target_det, target_seg, images_not_tensor = self.transforms(img_org, target_det, target_seg, is_train=self.is_train)
            # # only apply Resize, RandomHorizontalFlip, so that keep [W, H] format, suitable for OCR
            # images_not_tensor = self.transforms.trans_image_no_tensor(img_org)

        return img, images_not_tensor, target_det, target_seg, targets_geo, targets_sym, img_id, index

    def __len__(self):
        return len(self.ids)
    
    def get_height_and_width(self, index):
        img_id = self.ids[index]
        annot_each = self.contents[img_id]
        return annot_each["height"], annot_each["width"]
        
    def get_DET_GT_each(self, symbols):
        """_summary_

        Args:
            symbols: 输入为一个list, 其中包含每个symbol的标注信息, 每一个是一个字典,
                    包含{"id": str, "sym_class": str, 
                        "text_class": str, "text_content": str, 
                        "bbox": [x,y,w,h]}
        """
        boxes, classes, text_contents, ids, classes_text = [], [], [], [], []
        # arrow and others primitives are excluded 
        arrow_offset = 0
        for obj in symbols:
            # 若不是"text"&"arrow":
            #   classes: 加入symbol class idx
            #   classes_text: text的class认为是-1
            #   boxes: 加入[x,y,w,h]
            #   text_contents: 加入 text_content
            #   ids: s0, s1, s2
            # !!! Not sure why original code remove "arrow", we add here
            
            if obj["sym_class"] == "arrow":
                classes.append(self.class_sym_to_ind[obj["sym_class"]])
                classes_text.append(-1)
                boxes.append(obj["bbox"])
                text_contents.append(obj["text_content"])
                ids.append(obj["id"])
                continue              
            if obj["sym_class"]!='text':
                classes.append(self.class_sym_to_ind[obj["sym_class"]])
                classes_text.append(-1)
                boxes.append(obj["bbox"])
                text_contents.append(obj["text_content"])
                ids.append(obj["id"])
            elif obj["text_class"]!='others':
                # 否则就可能是"text" or "arrow"
                #   classes: 加入symbol class idx
                #   class_text: 加入text所属的类别
                #   boxes: 加入[x,y,w,h]
                #   text_contents: 加入 text_content
                #   ids: s0, s1, s2
                classes.append(self.class_sym_to_ind['text'])
                classes_text.append(self.class_text_to_ind[obj["text_class"]])
                boxes.append(obj["bbox"])
                text_contents.append(obj["text_content"])
                ids.append(obj["id"])

        return boxes, classes, text_contents, ids, classes_text
    
    def get_SEG_GT_each(self, geos, img_size):
        """_summary_

        Args:
            geos: 输入为一个字典, key为"points", "lines", "circles", 每个value是一个列表, 包含所有所属geometry的标注信息,
                每个标注信息是一个字典, {"id": str, 
                                     "loc":(points) [[x, y]] (lines) [[x1,y1], [x2,y2]] (circles) [[x,y],radius,"1111"]}
            img_size: (W,H)
        """
        masks, locs, classes, ids = [], [], [], []
        width, height = img_size

        for key, values in geos.items():
            for item in values:
                mask = np.zeros((height,width), dtype=np.uint8)
                loc = self.get_round_loc(item['loc'])   # [(x,y)], [(x1,y1),(x2,y2)], [(x,y),radius,"1111"]
                if key=='points':
                    cv2.circle(mask, center=loc[0], radius=GEODataset.THICKNESS, color=255, thickness=-1)
                if key=='lines':
                    cv2.line(mask, pt1=loc[0], pt2=loc[1], color=255, thickness=GEODataset.THICKNESS)
                if key=='circles':
                    cv2.circle(mask, center=loc[0], radius=loc[1], color=255, thickness=GEODataset.THICKNESS)
                    if loc[2]!='1111': # consider the case of a nonholonomic circle
                        draw_pixel_loc = np.where(mask>0)
                        for x, y in zip(draw_pixel_loc[1],draw_pixel_loc[0]):
                            delta_x = x-loc[0][0] # + - - +
                            delta_y = y-loc[0][1] # - - + +
                            if delta_x>=0 and delta_y<=0 and loc[2][0]=='0': mask[y,x] = 0
                            if delta_x<=0 and delta_y<=0 and loc[2][1]=='0': mask[y,x] = 0
                            if delta_x<=0 and delta_y>=0 and loc[2][2]=='0': mask[y,x] = 0
                            if delta_x>=0 and delta_y>=0 and loc[2][3]=='0': mask[y,x] = 0
                
                # mask是一个(H, W)的array
                masks.append(mask)
                # classes: [geo_idx, ...]
                classes.append(self.class_geo_to_ind[key[:-1]])
                # locs: [[[x, y]], [[x1, y1], [x2, y2]], [[x,y], radius, "1111"]]
                locs.append(item['loc'])
                # [p0, l1, c0]
                ids.append(item['id'])
        
        # np.stack(masks, axis=-1) -> [H, W, geo的数量]
        return np.stack(masks, axis=-1), locs, classes, ids

    def get_round_loc(self, loc):
        loc_copy = []
        for item in loc:
            if isinstance(item, float):
                item = round(item)
            if isinstance(item, list):
                item = tuple([round(v) for v in item])
            loc_copy.append(item)
        return loc_copy 

    def get_img_info(self, index):
        img_id = self.ids[index]
        annot_each = self.contents[img_id]
        return {"height": annot_each['height'], "width": annot_each['width']}

    def get_groundtruth(self, index):

        img_id = self.ids[index]
        annot_each = self.contents[img_id]
        height, width = annot_each['height'], annot_each['width']
        target_det = self.get_target_det(annot_each, (width, height))
        target_seg = self.get_target_seg(annot_each, (width, height))
        target_rel = self.get_target_rel(annot_each, target_det, target_seg)
        return target_det, target_seg, target_rel

    def get_target_det(self, annot_each, img_size):
        """_summary_

        Args:
            annot_each: json中一个图片的标注数据
            img_size: 图片大小
        
        Rerurns:
            target_det (BoxList):
                - bbox: [sym数量, 4]
                - size: (W, H)
                - mode: "xyxy"
                - extra_fields:
                    {
                        "labels": classes: [idx, ...] symbol class idx,
                        "text_contents": [str, null, ...] 非"text"&"arrow"应该是null,
                        "ids": [s0, s1, s2, ...],
                        "labels_text": [len_idx, -1], 对于"text"&"arrow"是对应的text的class, 其他的对应的是-1,
                    }
        """
        # boxes: [ [x, y, w, h], ... ]
        # classes: [idx, ...] symbol class idx
        # text_contents: [str, null, ...] 非"text"&"arrow"应该是null
        # ids: [s0, s1, s2, ...]
        # classes_text: [len_idx, -1], 对于"text"&"arrow"是对应的text的class, 其他的对应的是-1
        boxes, classes, text_contents, ids, classes_text = self.get_DET_GT_each(annot_each['symbols'])
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        # BoxList包含: box ([sym数量, 4]), size (W, H), mode ("xyxy"), extra_fields ({})
        # geo_parse/structures/bounding_box.py
        target_det = BoxList(boxes, img_size, mode="xywh").convert("xyxy")
        target_det.add_field("labels", torch.tensor(classes))
        target_det.add_field("text_contents", text_contents)
        target_det.add_field("ids", ids)
        target_det.add_field("labels_text", classes_text)
        return  target_det

    def get_target_seg(self, annot_each, img_size):
        """_summary_

        Args:
            annot_each: json中一个图片的标注数据
            img_size: 图片大小
        Returns:
            target_seg (GeoList):
                - masks: [H, W, geo数量]
                - size: (W,H)
                - extra_fields:
                    {
                        "labels": [geo_idx, ...],
                        "locs": [[[x, y]], [[x1, y1], [x2, y2]], [[x,y], radius, "1111"]],
                        "ids": [p0, l1, c0]
                    }
        """
        # masks: [H, W, geo数量]
        # locs: [[[x, y]], [[x1, y1], [x2, y2]], [[x,y], radius, "1111"]]
        # classes: [geo_idx, ...]
        # ids: [p0, l1, c0]
        masks, locs, classes, ids = self.get_SEG_GT_each(annot_each['geos'], img_size)
        # GeoList包含: masks: [H, W, geo数量], size: (W,H), extra_fields({})
        # geo_parse/structures/geo_ins.py
        target_seg = GeoList(masks, img_size)
        target_seg.add_field("labels", classes)
        target_seg.add_field("locs", locs)
        target_seg.add_field("ids", ids)
        return target_seg

    def get_geo2geo_rel(self, annot_each, target_seg):
        geo_classes = target_seg.get_field("labels")
        
        points_num = geo_classes.count(self.CLASSES_GEO.index("point"))
        lines_num = geo_classes.count(self.CLASSES_GEO.index("line"))
        circles_num = geo_classes.count(self.CLASSES_GEO.index("circle"))
        
        pl_rels = None
        pc_rels = None
        if points_num != 0 and lines_num != 0:
            pl_rels = torch.zeros((points_num, lines_num), dtype=torch.float)
        if points_num != 0 and circles_num != 0:
            pc_rels = torch.zeros((points_num, circles_num), dtype=torch.float)
        
        geo2geo = annot_each["relations"]["geo2geo"]
        for rel_list in geo2geo:
            assert len(rel_list) == 3
            geo_a = rel_list[0]
            geo_b = rel_list[1]
            rel = rel_list[2]
            
            assert geo_a[0] == "p"
            p_idx = int(geo_a[1:])
            assert geo_b[0] in ["l", "c"]
            if geo_b[0] == "l":
                l_idx = int(geo_b[1:])
                if rel == "endpoint":
                    pl_rels[p_idx, l_idx] = 1.
                elif rel == "online":
                    pl_rels[p_idx, l_idx] = 2.
                else:
                    raise ValueError
            elif geo_b[0] == "c":
                c_idx = int(geo_b[1:])
                if rel == "oncircle":
                    pc_rels[p_idx, c_idx] = 1.
                elif rel == "center":
                    pc_rels[p_idx, c_idx] = 2.
                else:
                    raise ValueError
        
        return {"pl_rels": pl_rels, "pc_rels": pc_rels}, (points_num, lines_num, circles_num)

    def get_sym2geo_rel(self, annot_each, target_det, points_num, lines_num, circles_num):
        
        sym_ids = target_det.get_field("ids")
        sym_labels = target_det.get_field("labels")
        sym2geo = annot_each["relations"]["sym2geo"]
        sym2sym = annot_each["relations"]["sym2sym"]
        
        text_symbol_geo_rel, head_symbol_geo_rel, text_symbol_class, head_symbol_class = self.get_text_symbol_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            sym2sym=sym2sym,
            points_num=points_num,
            lines_num=lines_num,
            circles_num=circles_num,
        )
        
        # sym_labels, sym_ids, sym2geo, num, sym_type
        angle_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="angle",
        )
        
        double_angle_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="double angle",
        )
        
        triple_angle_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="triple angle",
        )
    
        quad_angle_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="quad angle",
        )

        penta_angle_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="penta angle",
        )
    

        bar_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="bar",
        )

        double_bar_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="double bar",
        )
        
        triple_bar_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="triple bar",
        )

        quad_bar_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="quad bar",
        )

        parallel_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=lines_num,
            sym_type="parallel",
        )
        
        double_parallel_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=lines_num,
            sym_type="double parallel",
        )

        triple_parallel_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=lines_num,
            sym_type="triple parallel",
        )

        perpendicular_symbols_geo_rel = self.get_others_symbols_geo_rel(
            sym_labels=sym_labels,
            sym_ids=sym_ids,
            sym2geo=sym2geo,
            num=points_num + lines_num,
            sym_type="perpendicular",
        )
        
        return {
            "text_symbol_geo_rel": text_symbol_geo_rel,
            "head_symbol_geo_rel": head_symbol_geo_rel,
            "text_symbol_class": text_symbol_class,
            "head_symbol_class": head_symbol_class,
            "angle_symbols_geo_rel": angle_symbols_geo_rel,
            "double_angle_symbols_geo_rel": double_angle_symbols_geo_rel,
            "triple_angle_symbols_geo_rel": angle_symbols_geo_rel,
            "quad_angle_symbols_geo_rel": quad_angle_symbols_geo_rel,
            "penta_angle_symbols_geo_rel": penta_angle_symbols_geo_rel,
            "bar_symbols_geo_rel": bar_symbols_geo_rel,
            "double_bar_symbols_geo_rel": double_bar_symbols_geo_rel,
            "triple_bar_symbols_geo_rel": triple_bar_symbols_geo_rel,
            "quad_bar_symbols_geo_rel": quad_bar_symbols_geo_rel,
            "parallel_symbols_geo_rel": parallel_symbols_geo_rel,
            "double_parallel_symbols_geo_rel": double_parallel_symbols_geo_rel,
            "triple_parallel_symbols_geo_rel": triple_parallel_symbols_geo_rel,
            "perpendicular_symbols_geo_rel": perpendicular_symbols_geo_rel,
        }
    
    def get_text_symbol_geo_rel(self, sym_labels, sym_ids, sym2geo, sym2sym, points_num, lines_num, circles_num):
        
        # P + L + LPL + PLP + PCP
        # !!! I think there's a bug in the original code,
        #       the points_num for LPL, and lines_num for PLP should be zero if:
        #       no line available for LPL, and no point available for PLP.
        # Hence, we create lpl_num, plp_num, pcp_num.
        lpl_num = points_num if lines_num != 0 else 0
        plp_num = lines_num if points_num != 0 else 0
        pcp_num = circles_num if points_num != 0 else 0
        # column_len = points_num + lines_num + lpl_num + plp_num + pcp_num
        column_len = points_num + lines_num + circles_num
        row_len = sym_labels.tolist().count(self.CLASSES_SYM.index("text"))
   
        head_sym_num = 0
        head_symbol_geo_rel = None
        head_symbol_class = None
        if len(sym2sym) != 0:
            for rel in sym2sym:
                if len(rel[1]) == 1 and (sym_labels.tolist()[int(rel[1][0][1:])] == self.CLASSES_SYM.index("head")):
                    head_sym_num +=1
            if head_sym_num != 0:
                # head_symbol_geo_rel = torch.zeros((head_sym_num, points_num + lpl_num + plp_num + pcp_num), dtype=torch.long)
                head_symbol_geo_rel = torch.zeros((head_sym_num, points_num + lines_num + circles_num), dtype=torch.long)
                head_symbol_class = torch.zeros((head_sym_num, ), dtype=torch.long)
                column_len += head_sym_num
        
        text_sym_dict = {}
        head_sym_dict = {}
        ordinal_ids = 0
        ordinal_ids_head = 0
        for i, ids in enumerate(sym_ids):
            if sym_labels.tolist()[i] == self.CLASSES_SYM.index("text"):
                text_sym_dict[ids] = ordinal_ids
                ordinal_ids +=1
            if head_symbol_geo_rel != None:
                if sym_labels.tolist()[i] == self.CLASSES_SYM.index("head"):
                    head_sym_dict[ids] = ordinal_ids_head
                    ordinal_ids_head +=1
        
        if row_len != 0 and column_len != 0:
            text_symbol_geo_rel = torch.zeros((row_len, column_len), dtype=torch.long)
            text_symbol_class = torch.zeros((row_len, ), dtype=torch.long)
        else:
            text_symbol_geo_rel = None
            text_symbol_class = None
        for rel in sym2geo:
            sym = rel[0]
            geos = rel[1]
            
            if sym in text_sym_dict:
                if len(geos) == 1:  # P, L
                    if geos[0][0] == "p":
                        text_symbol_geo_rel[text_sym_dict[sym], int(geos[0][1:])] = 1.
                    elif geos[0][1] == "l":
                        offset = points_num
                        text_symbol_geo_rel[text_sym_dict[sym], offset + int(geos[0][1:])] = 1.
                    else:
                        raise ValueError
                elif len(geos) == 3:    # LPL, PLP, PCP
                    p_num = sum([1 if geo[0] == "p" else 0 for geo in geos])
                    l_num = sum([1 if geo[0] == "l" else 0 for geo in geos])
                    c_num = sum([1 if geo[0] == "c" else 0 for geo in geos])
                    if l_num == 2 and p_num == 1:   # LPL
                        # offset = points_num + lines_num
                        p_idx = [int(geo[1:]) for geo in geos if geo[0] == "p"]
                        l_idx = [int(geo[1:]) for geo in geos if geo[0] == "l"]
                        text_symbol_geo_rel[text_sym_dict[sym], p_idx[0]] = 1.
                        text_symbol_geo_rel[text_sym_dict[sym], l_idx] = 1.
                        text_symbol_class[text_sym_dict[sym]] = 1
                    elif l_num == 1 and p_num == 2: # PLP
                        # offset = points_num + lines_num + lpl_num
                        l_idx = [int(geo[1:]) for geo in geos if geo[0] == "l"]
                        p_idx = [int(geo[1:]) for geo in geos if geo[0] == "p"]
                        text_symbol_geo_rel[text_sym_dict[sym], l_idx[0]] = 1.
                        text_symbol_geo_rel[text_sym_dict[sym], p_idx] = 1.
                        text_symbol_class[text_sym_dict[sym]] = 2
                    elif c_num == 1 and p_num == 2: # PCP
                        # offset = points_num + lines_num + lpl_num + plp_num
                        c_idx = [int(geo[1:]) for geo in geos if geo[0] == "c"]
                        p_idx = [int(geo[1:]) for geo in geos if geo[0] == "p"]
                        text_symbol_geo_rel[text_sym_dict[sym], c_idx[0]] = 1.
                        text_symbol_geo_rel[text_sym_dict[sym], p_idx] = 1.
                        text_symbol_class[text_sym_dict[sym]] = 3
                    else:
                        raise ValueError
        
        if head_symbol_geo_rel != None:
            assert len(head_sym_dict) > 0
            for rel in sym2sym:
                if len(rel[1]) == 1:
                    head_sym = rel[1][0]
                    # offset = points_num + lines_num + lpl_num + plp_num + pcp_num
                    offset = points_num + lines_num + circles_num
                    text_symbol_geo_rel[text_sym_dict[rel[0]], offset + head_sym_dict[head_sym]] = 1.
                    text_symbol_class[text_sym_dict[rel[0]]] = 0
                    
                    for rel_sym2geo in sym2geo:
                        if rel_sym2geo[0] == head_sym:
                            p_num = sum([1 if geo[0] == "p" else 0 for geo in rel_sym2geo[1]])
                            l_num = sum([1 if geo[0] == "l" else 0 for geo in rel_sym2geo[1]])
                            c_num = sum([1 if geo[0] == "c" else 0 for geo in rel_sym2geo[1]])
                            
                            if l_num == 2 and p_num == 1:   # LPL
                                # offset = points_num
                                p_idx = [int(geo[1:]) for geo in rel_sym2geo[1] if geo[0] == "p"]
                                l_idx = [int(geo[1:]) for geo in rel_sym2geo[1] if geo[0] == "l"]
                                head_symbol_geo_rel[head_sym_dict[head_sym], p_idx[0]] = 1.
                                head_symbol_geo_rel[head_sym_dict[head_sym], l_idx] = 1.
                                head_symbol_class[head_sym_dict[head_sym]] = 1
                            elif l_num == 1 and p_num == 2: # PLP
                                # offset = points_num + lpl_num
                                l_idx = [int(geo[1:]) for geo in rel_sym2geo[1] if geo[0] == "l"]
                                p_idx = [int(geo[1:]) for geo in rel_sym2geo[1] if geo[0] == "p"]
                                head_symbol_geo_rel[head_sym_dict[head_sym], l_idx[0]] = 1.
                                head_symbol_geo_rel[head_sym_dict[head_sym], p_idx] = 1.
                                head_symbol_class[head_sym_dict[head_sym]] = 2
                            elif c_num == 1 and p_num == 2: # PCP
                                # offset = points_num + lpl_num + plp_num
                                c_idx = [int(geo[1:]) for geo in rel_sym2geo[1] if geo[0] == "c"]
                                p_idx = [int(geo[1:]) for geo in rel_sym2geo[1] if geo[0] == "p"]
                                head_symbol_geo_rel[head_sym_dict[head_sym], c_idx[0]] = 1.
                                head_symbol_geo_rel[head_sym_dict[head_sym], p_idx] = 1.
                                head_symbol_class[head_sym_dict[head_sym]] = 3
                            elif p_num == 1:
                                # offset = 0
                                p_idx = [int(geo[1:]) for geo in rel_sym2geo[1] if geo[0] == "p"]
                                head_symbol_geo_rel[head_sym_dict[head_sym], p_idx[0]] = 1.
                            else:
                                raise ValueError
                            
                            break
        
        return text_symbol_geo_rel, head_symbol_geo_rel, text_symbol_class, head_symbol_class

    def get_others_symbols_geo_rel(self, sym_labels, sym_ids, sym2geo, num, sym_type):
                
        sym_dict = {}
        ordinal_ids = 0
        for i, ids in enumerate(sym_ids):
            if sym_labels.tolist()[i] == self.CLASSES_SYM.index(sym_type):
                sym_dict[ids] = ordinal_ids
                ordinal_ids +=1
        
        if ordinal_ids == 0 or num == 0:
            return None
        
        symbol_geo_rel = torch.zeros((ordinal_ids, num), dtype=torch.long)
        for rel in sym2geo:
            if rel[0] in sym_dict:
                if "angle" in sym_type:     # LPL
                    p_idx = [int(geo[1:]) for geo in rel[1] if geo[0] == "p"]
                    l_idx = [int(geo[1:]) for geo in rel[1] if geo[0] == "l"]
                    # print("1p_idx: ", p_idx)
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    symbol_geo_rel[sym_dict[rel[0]], p_idx[0]] = 1.
                    symbol_geo_rel[sym_dict[rel[0]], l_idx] = 1.
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    # input()
                elif "bar" in sym_type:     # PLP
                    l_idx = [int(geo[1:]) for geo in rel[1] if geo[0] == "l"]
                    p_idx = [int(geo[1:]) for geo in rel[1] if geo[0] == "p"]
                    # print("2l_idx: ", l_idx)
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    if len(l_idx) != 0:
                        symbol_geo_rel[sym_dict[rel[0]], l_idx[0]] = 1.
                        symbol_geo_rel[sym_dict[rel[0]], p_idx] = 1.
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    # input()
                elif "parallel" in sym_type: # l
                    assert len(rel[1]) == 1
                    l_idx = int(rel[1][0][1:])
                    # print("3l_idx: ", l_idx)
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    symbol_geo_rel[sym_dict[rel[0]], l_idx] = 1.
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    # input()
                elif "perpendicular" in sym_type: # LPL
                    p_idx = [int(geo[1:]) for geo in rel[1] if geo[0] == "p"]
                    l_idx = [int(geo[1:]) for geo in rel[1] if geo[0] == "l"]
                    # print("4l_idx: ", p_idx)
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    symbol_geo_rel[sym_dict[rel[0]], p_idx[0]] = 1.
                    symbol_geo_rel[sym_dict[rel[0]], l_idx] = 1.
                    # print("symbol_geo_rel: ", symbol_geo_rel[sym_dict[rel[0]], ])
                    # input()
                else:
                    raise ValueError
        
        return symbol_geo_rel