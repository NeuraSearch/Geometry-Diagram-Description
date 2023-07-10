# coding:utf-8

import os
import re
import json
import codecs
import pickle
import cv2 as cv
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from tqdm import tqdm

def process_image(img, min_side=224):
    # fill the diagram with a white background and resize it
    size = img.shape
    h, w = size[0], size[1]

    scale = max(w, h) / float(min_side)
    new_w, new_h = int(w/scale), int(h/scale)
    resize_img = cv.resize(img, (new_w, new_h))

    top, bottom, left, right = 0, min_side-new_h, 0, min_side-new_w
    pad_img = cv.copyMakeBorder(resize_img, int(top), int(bottom), int(left), int(right),
                                cv.BORDER_CONSTANT, value=[255,255,255])
    pad_img = pad_img / 255

    return pad_img

def split_elements(sequence):
    new_sequence = []
    for token in sequence:
        if 'N_' in token or 'NS_' in token or 'frac' in token:
            new_sequence.append(token)
        elif token.istitle():
            new_sequence.append(token)
        elif re.search(r'[A-Z]', token):
            # split geometry elements with a space: ABC -> A B C
            new_sequence.extend(token)
        else:
            new_sequence.append(token)

    return new_sequence

def process_english_text(ori_text):
    text = re.split(r'([=≠≈+-/△∠∥⊙☉⊥⟂≌≅▱∽⁀⌒;,:.•?])', ori_text)
    text = ' '.join(text)

    text = text.split()
    text = split_elements(text)
    text = ' '.join(text)

    # The initial version of the calculation problem (GeoQA) is in Chinese.
    # The translated English version still contains some Chinese tokens,
    # which should be replaced by English words.
    replace_dict ={'≠': 'not-equal', '≈': 'approximate', '△': 'triangle', '∠': 'angle', '∥': 'parallel',
                   '⊙': 'circle', '☉': 'circle', '⊥': 'perpendicular', '⟂': 'perpendicular', '≌': 'congruent', '≅': 'congruent',
                   '▱': 'parallelogram', '∽': 'similar', '⁀': 'arc', '⌒': 'arc'
                   }
    for k, v in replace_dict.items():
        text = text.replace(k, v)

    return text

def load_calculation_problems(train_data, val_dataset, test_dataset, subset_dict):
    os.makedirs("./images_unigeo_cal", exist_ok=True)
    
    datasets_list = [train_data, val_dataset, test_dataset]
    for i, dataset in enumerate(datasets_list):
        datas = {}
        for sample in tqdm(dataset):
            
            data_id = sample["id"]
            
            assert data_id not in datas
            
            # save image
            img = sample["image"]
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
            im = Image.fromarray(np.uint8(img)).convert("RGB")
            im.save(f"./images_unigeo_cal/{data_id}.png")
            
            problem = process_english_text(sample["English_problem"])
            problem = 'Calculation: ' + problem
            program = " ".join(sample["manual_program"])

            numbers = sample["numbers"]
            choice_numbers = sample["choice_nums"]
            label = sample["label"]
            
            p_type = subset_dict[data_id]
            
            datas[data_id] = {
                "p_type": p_type,
                "problem": problem,
                "program": program,
                "numbers": numbers,
                "choice_numbers": choice_numbers,
                "label": label,
            }

        if i == 0:
            name = "train"
        elif i == 1:
            name = "val"
        elif i == 2:
            name = "test"
        with codecs.open(f"UniGeo_CAL_{name}.json", "w", "utf-8") as file:
            json.dump(datas, file, indent=4)

def load_proving_problems(train_dataset, val_dataset, test_dataset, start_id=0):
    data_id = start_id
    os.makedirs("./images_unigeo_prv", exist_ok=True)
    
    datasets_list = [train_dataset, val_dataset, test_dataset]
    for i, dataset in enumerate(datasets_list):
        datas = {}
        for sample in tqdm(dataset):
            
            # save image
            img = sample["img"]
            img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
            im = Image.fromarray(np.uint8(img)).convert("RGB")
            im.save(f"./images_unigeo_prv/{data_id}.png")
            
            # problem = process_english_text(sample["English_problem"])
            problem = 'Proving: ' + sample["input_text"]
            program = " ".join(sample["proving_sequence"])

            # numbers = sample["numbers"]
            # choice_numbers = sample["choice_nums"]
            # label = sample["label"]
            
            p_type = sample["problem_type"]
            
            datas[data_id] = {
                "p_type": p_type,
                "problem": problem,
                "program": program,
                "numbers": None,
                "choice_numbers": None,
                "label": None,
            }
            
            data_id +=1
            
        if i == 0:
            name = "train"
        elif i == 1:
            name = "val"
        elif i == 2:
            name = "test"
        with codecs.open(f"UniGeo_PRV_{name}.json", "w", "utf-8") as file:
            json.dump(datas, file, indent=4)

if __name__ == "__main__":
    # with codecs.open("./UniGeo/calculation_train.pk", "rb") as file:
    #     train_dataset = pickle.load(file)

    # with codecs.open("./UniGeo/calculation_val.pk", "rb") as file:
    #     val_dataset = pickle.load(file)

    # with codecs.open("./UniGeo/calculation_test.pk", "rb") as file:
    #     test_dataset = pickle.load(file)

    # with open("./UniGeo/sub_dataset_dict.pk", 'rb') as file:
    #     subset_dict = pickle.load(file)
        
    # load_calculation_problems(train_dataset, val_dataset, test_dataset, subset_dict)

    with codecs.open("./UniGeo/proving_train.pk", "rb") as file:
        train_dataset = pickle.load(file)
        
    with codecs.open("./UniGeo/proving_val.pk", "rb") as file:
        val_dataset = pickle.load(file)

    with codecs.open("./UniGeo/proving_test.pk", "rb") as file:
        test_dataset = pickle.load(file)        

    load_proving_problems(train_dataset, val_dataset, test_dataset)