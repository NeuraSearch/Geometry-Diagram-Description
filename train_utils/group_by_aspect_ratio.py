# coding:utf-8

import math
import copy
import bisect
import numpy as np

from collections import defaultdict
from itertools import repeat, chain
from torch.utils.data.sampler import BatchSampler, Sampler

def _reapeat_to_at_least(iterable, n):
    # no need to random select from `iterable`, which obtained from RandomSampler,
    # as a result, we just duplicate the entire iterable
    repeat_times = math.ceil(n / len(iterable)) # >=1
    # repeat -> [ [1, 2, 3], [1, 2, 3], [1, 2, 3] ]
    # chain.from_iterable -> [1, 2, 3, 1, 2, 3, 1, 2, 3]
    repeats = chain.from_iterable(repeat(iterable, repeat_times))
    return list(repeats)

class GroupBatchSampler(BatchSampler):
    """
    Wraps another sampler to yield a mini-batch of indices.
    It enforces that the batch only contain elements from the same group.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.
    Arguments:
        sampler (Sampler): Base sampler.
        group_ids (list[int]): If the sampler produces indices in range [0, N),
            `group_ids` must be a list of `N` ints which contains the group id of each sample.
            The group ids must be a continuous set of integers starting from
            0, i.e. they must be in the range [0, num_groups).
        batch_size (int): Size of mini-batch.
    """
    def __init__(self, sampler, group_ids, batch_size):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        self.sampler = sampler
        self.group_ids = group_ids
        self.batch_size = batch_size

    def __iter__(self):
        buffer_per_group = defaultdict(list)
        samples_per_group = defaultdict(list)
        
        num_batches = 0
        for idx in self.sampler:
            group_id = self.group_ids[idx]
            buffer_per_group[group_id].append(idx)
            samples_per_group[group_id].append(idx)
            if len(buffer_per_group[group_id]) == self.batch_size:
                yield buffer_per_group[group_id]
                num_batches +=1
                del buffer_per_group[group_id]
            
            assert len(buffer_per_group[group_id]) < self.batch_size
        
        num_batches_remaining = len(self) - num_batches
        if num_batches_remaining > 0:
            for group_id, _ in sorted(buffer_per_group.items(),
                                      key=lambda x: len(x[1]), reverse=True):
                num_data_remaining = self.batch_size - len(buffer_per_group[group_id])
                
                samples_to_fill = _reapeat_to_at_least(samples_per_group[group_id], num_data_remaining)
                buffer_per_group[group_id].extend(samples_to_fill[:num_data_remaining])
                assert len(buffer_per_group[group_id]) == self.batch_size
                yield buffer_per_group[group_id]
                num_batches_remaining -=1
                if num_batches_remaining == 0:
                    break
                
        assert num_batches_remaining == 0 
        
    def __len__(self):
        return len(self.sampler) // self.batch_size

def compute_aspect_ratios(dataset, indices=None):
    # `dataset` should contains a function to return H and W
    assert hasattr(dataset, "get_height_and_width")
    
    if indices is None:
        indices = range(len(dataset))
    aspect_ratios=  []
    for i in range(indices):
        h, w = dataset.get_height_and_width(i)
        aspect_ratio = float(w) / float(h)
    return aspect_ratios

def _quantize(aspect_ratios, bins):
    bins = copy.deepcopy(bins)
    bins = sorted(bins)
    return list(map(lambda x: bisect.bisect_right(bins, x), aspect_ratios))
    
def create_aspect_ratio_groups(dataset, k=0, show_stat=True):
    """
        k: 2*k+1 is the number of bins.
        show_stat: show some statistical informtion.
    """
    # calculate W/H of each image in the dataset
    aspect_ratios = compute_aspect_ratios(dataset)
    # split [0.5, 1] into 2*k+1 pieces
    bins = (2 ** np.linspace(-1, 1, 2*k+1)).tolist() if k > 0 else [1.0]
    # divide each image index into different bins
    groups = _quantize(aspect_ratios, bins)
    
    if show_stat:
        counts = np.unique(groups, return_counts=True)[1]
        fbins = [0] + bins + [np.inf]
        print(f"Using {fbins} as bins for aspect ratio quantization")
        print(f"Count of instances per bin: {counts}")
    return groups