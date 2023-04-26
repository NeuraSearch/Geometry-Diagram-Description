# coding:utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

from . import transforms as T
from .pgdp_data import GEODataset

from train_utils import get_world_size
from image_structure import to_image_list

def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0):
    num_gpus = get_world_size()
    if is_train:
        images_per_batch = cfg.train_img_per_batch
        assert (
            images_per_batch % num_gpus == 0
        ), f"cfg.image_per_batch ({images_per_batch}) must be divisible by the number \
                of GPUs ({num_gpus}) used."

        images_per_gpu = images_per_batch // num_gpus
        shuffle = True
    else:
        images_per_batch = cfg.test_img_per_batch
        assert (
            images_per_batch % num_gpus == 0
        ), f"TEST.IMS_PER_BATCH ({images_per_batch}) must be divisible by the number \
                of GPUs ({num_gpus}) used."
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        start_iter = 0
    
    if images_per_gpu > 1:
        pass
        # Equivalent schedules with...
        # 1 GPU:
        #   BASE_LR: 0.0025
        #   MAX_ITER: 60000
        #   STEPS: [0, 30000, 40000]
        # 2 GPUs:
        #   BASE_LR: 0.005
        #   MAX_ITER: 30000
        #   STEPS: [0, 15000, 20000]
        # 4 GPUs:
        #   BASE_LR: 0.01
        #   MAX_ITER: 15000
        #   STEPS: [0, 7500, 10000]
        # 8 GPUs:
        #   BASE_LR: 0.02
        #   MAX_ITER: 7500
        #   STEPS: [0, 3750, 5000]
    
    cfg.train_img_per_batch = images_per_gpu
    cfg.test_img_per_batch = images_per_gpu
    
    normalize_transform = T.Normalize(
        mean=[200.0, 200.0, 200.0],
        std=[1., 1., 1.],
        to_bgr255=True
    )
    
    if is_train:
        min_size = cfg.min_size_train
        max_size = cfg.max_size_train
        flip_prob = 0.5
    else:
        min_size = cfg.min_size_test
        max_size = cfg.max_size_test
        flip_prob = 0.0
    
    transforms = T.Compose(
        [
            T.Resize(min_size, max_size, fpn_strides=cfg.seg_fpn_strides),
            T.RandomHorizontalFlip(flip_prob),
            T.ToTensor(),
            normalize_transform,
        ]
    )
    
    train_dataset = None
    eval_dataset = None
    test_dataset = None
    if is_train:
        train_dataset = GEODataset(
            root=cfg.train_img_path,
            ann_file=cfg.train_annot_path,
            transforms=transforms,
            is_train=True,
            cfg=cfg,
        )
        eval_dataset = GEODataset(
            root=cfg.eval_img_path,
            ann_file=cfg.eval_annot_path,
            transforms=transforms,
            is_train=False,
            cfg=cfg,
        )
    else:
        test_dataset = GEODataset(
            root=cfg.test_img_path,
            ann_file=cfg.test_annot_path,
            transforms=transforms,
            is_train=False,
            cfg=cfg,
        )
    
    return train_dataset, eval_dataset, test_dataset

def geo_data_collate_fn(datas_list):
    batch = list(zip(*datas_list))
    
    images = to_image_list(batch[0], 32)
    images_not_tensor = list(batch[1])
    targets_det = list(batch[2])
    targets_seg = list(batch[3])
    targets_geo = list(batch[4])
    targets_sym = list(batch[5])
    images_id = list(batch[6])
        
    return {
        "images": images,
        "images_not_tensor": images_not_tensor,
        "targets_det": targets_det,
        "targets_seg": targets_seg,
        "targets_geo": targets_geo,
        "targets_sym":targets_sym,
        "images_id": images_id,
    }