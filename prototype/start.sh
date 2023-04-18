#!/usr/bin/env bash

# set -xe

BASE_DIR=.
TRAIN_ARGS_YAML=${BASE_DIR}/"config.yaml"

ARGS="--train_args_yaml ${TRAIN_ARGS_YAML}"

CUDA_VISIBLE_DEVICES=0 python main.py ${ARGS}

# torchrun --nproc_per_node=8 main.py ${ARGS}