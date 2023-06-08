# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import os
import codecs
import json
import torch

from PIL import Image

class UniGeoDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, ann_file, parse_file, transforms=None, is_train=False, cfg=None):
        
        self.img_root = root
        self.transforms = transforms
        
        self.ids = []
        self.is_train = is_train
        
        with codecs.open(MAIN_PATH / ann_file, "r") as file:
            self.contents = json.load(file)
        with codecs.open(MAIN_PATH / parse_file, "r") as file:
            self.parse_contents = json.load(file)
        for key in self.contents.keys():
            self.ids.append(key)
        
        if cfg.toy_data:
            self.ids = self.ids[:50]
        
        self.enable_geo_rel = cfg.enable_geo_rel
        self.cfg = cfg
        
    def __getitem__(self, index):
        
        img_id = self.ids[index]
        annot_each = self.contents[img_id]
        parse_each = self.parse_contents[img_id]
        
        # img_org = Image.open(os.path.join(str(MAIN_PATH / self.img_root), f"{img_id}.png")).convert("RGB")
        
        # TODO: 在这里到时候加上Diagram Description的内容
        # process the "parse_each"
        diagram_description = ""
        if self.cfg.enable_diagram_descirption:
            for rel_name, des in parse_each.items():
                if not self.enable_geo_rel:
                    if rel_name in ["points", "lines", "angle"]:
                        continue
                if type(des) == str:
                    diagram_description = diagram_description + des + " "
                elif (type(des) == list) and (len(des) != 0):
                    diagram_description = diagram_description + " ".join(des) + " "
                
        problem_type = annot_each["p_type"]
        problem = diagram_description + annot_each["problem"]
        program = annot_each["program"]
        numbers = annot_each["numbers"]
        choice_numbers = annot_each["choice_numbers"]
        label = annot_each["label"]
        
        # if self.transforms is not None:
        #     img, _, _, _ = self.transforms(img_org, is_train=False)
        img = None
        
        return problem_type, problem, program, numbers, choice_numbers, label, img, img_id
    
    def __len__(self):
        return len(self.ids)