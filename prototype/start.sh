#!/usr/bin/env bash

# set -xe

BASE_DIR=.
TRAIN_ARGS_YAML=${BASE_DIR}/"config.yaml"

ARGS="--train_args_yaml ${TRAIN_ARGS_YAML}"

python main.py ${ARGS}