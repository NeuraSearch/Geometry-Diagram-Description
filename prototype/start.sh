#!/usr/bin/env bash

# set -xe

BASE_DIR=.
TRAIN_ARGS_YAML=${BASE_DIR}/"config.yaml"

ARGS="--train_args_yaml ${TRAIN_ARGS_YAML}"

# single gpu
CUDA_VISIBLE_DEVICES=0 python main.py ${ARGS}

# two gpus on one node
# torchrun --nproc_per_node=2 main.py ${ARGS}

# python -m torch.distributed.launch --nproc_per_node=2 --use_env main.py ${ARGS}

# train program generator
CUDA_VISIBLE_DEVICES=0 python main_gp.py ${ARGS}