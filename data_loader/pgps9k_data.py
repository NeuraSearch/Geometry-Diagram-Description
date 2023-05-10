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

class PGPS9KDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, ann_file, parse_file, transforms=None, is_train=False, cfg=None):
        
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
    
    def __getitem__(self, index):
        
        img_id = self.ids[index]    # here is not "xxx.png", it's "prob_xx"
        annot_each = self.contents[img_id]
        parse_each = self.parse_contents[img_id]
        
        diagram_description = ""
        for _, des in parse_each.items():
            if type(des) == str:
                diagram_description = diagram_description + des + " "
            elif (type(des) == list) and (len(des) != 0):
                diagram_description = diagram_description + " ".join(des) + " "
                
        problem_type = annot_each["type"]
        problem = diagram_description + annot_each["text"]
        program = annot_each["expression"]
        numbers = None
        choice_numbers = annot_each["choices"]
        label = annot_each["answer"]
        
        # if self.transforms is not None:
        #     img, _, _, _ = self.transforms(img_org, is_train=False)
        img = None
        
        return problem_type, problem, program, numbers, choice_numbers, label, img, img_id
    
    def __len__(self):
        return len(self.ids)
        