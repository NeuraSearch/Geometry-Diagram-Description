# coding:utf-8

import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import torch
import yaml
import codecs

from data_loader import make_data_loader, geo_data_collate_fn
from train_utils import create_aspect_ratio_groups, GroupBatchSampler

with codecs.open("config.yaml", "r", "utf-8") as file:
    config = yaml.safe_load(file)

args = {}
for _, config in config.items():
    for key, val in config.items():
        args[key] = val

from argparse import Namespace
args = Namespace(**args)
print(args)

train_dataset, eval_dataset, test_dataset = make_data_loader(args)

train_sampler = torch.utils.data.RandomSampler(train_dataset)
group_ids = create_aspect_ratio_groups(train_dataset, k=3)
train_batch_sampler = GroupBatchSampler(train_sampler, group_ids, 2)
train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_sampler=train_batch_sampler,
    collate_fn=geo_data_collate_fn
)

eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
eval_batch_sampler = torch.utils.data.BatchSampler(eval_sampler, 2, drop_last=False)
eval_data_loader = torch.utils.data.DataLoader(
    eval_dataset, batch_sampler=eval_batch_sampler,
    collate_fn=geo_data_collate_fn
)

# test_sampler = torch.utils.data.SequentialSampler(test_dataset)
# test_batch_sampler = torch.utils.data.BatchSampler(test_sampler, 2, drop_last=True)
# test_data_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_sampler=test_batch_sampler,
#     collate_fn=geo_data_collate_fn
# )

# test_data_loader = torch.utils.data.DataLoader(
#     test_dataset, batch_size=2, collate_fn=geo_data_collate_fn, shuffle=False
# )

for i, batch in enumerate(train_data_loader):
    print(i+1)
    print(batch)
    print()
    print()
    print()
    print("*"*100)
    print()
    print()
    print()