# coding:utf-8

from argparse import ArgumentParser

def add_train_args(parser: ArgumentParser):
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--num_classes", type=list)