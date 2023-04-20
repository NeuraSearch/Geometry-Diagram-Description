# coding:utf-8

import os
import sys
import time
import datetime
import logging
import random
import numpy as np
import torch
import torch.distributed as dist

from collections import defaultdict, deque

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            # value, global_avg perform like placeholder,
            # future passing value could pass through fmt.format(value=1, global_avg=2)
            fmt = "{value:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0.0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count +=n
        self.total += value * n
    
    def synchronize_between_process(self):
        if not is_dist_avail_and_initialize():
            # single GPU no need to synchronize
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        # sum all the data from each process, and save in each process
        # NOTE: different to "all_gather", which gather all the data from each process, then copy to each process
        #   tensor_list = [torch.zeros(2) for _ in range(2)], [[0, 0], [0, 0]], rank0 and rank1
        #   tensor = torch.arrange(2) + 1 + 2 * rank, [[1, 2], [3, 4]], rank0 and rank1
        #   all_gather(tensor_list, tensor) -> tensor_list [[1, 2], [3, 4]] for both rank0 and rank1
        #   all_reduce(tensor) -> [[4, 6]], for both rank0 and rank1
        dist.all_reduce(t)
        t = t.tolist()
        # we don't synchronize the deque here,
        # so the avg is for each GPU, the global_avg is for all GPU
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @ property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()
    
    @property
    def global_avg(self):
        # before "synchronize_between_process", I think that avg == global_avg,
        # after "synchronize_between_process", total > sum(deque), count > len(deque)
        return self.total / self.count
    
    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        # NOTE: actually, default fmt only returns "global_avg" and "value"
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def all_gather(data):
    """Run all_gather on arbitrary picklable data (not necessary tensors)
    Args:
        data: any picklable object
    Returns:
        List[Data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    
    data_list = [None] * world_size
    # data: List[Dict, Dict, ...], each Dict is predictions of one data
    # data_list: [ List[Dict, Dict, ...], List[Dict, Dict, ...], ... ], 
    #   each list is a bunch of datas' predictions from a rank
    dist.all_gather_object(data_list, data)
    
    return data_list

def reduce_dict(input_dict, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        keys = []
        values = []
        # dict has no order, to make sure keys of input_dict are consistent across ranks
        for k in sorted(input_dict.keys()):
            keys.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        
        reduced_dict = {k: v for k, v in zip(keys, values)}
        return reduce_dict

def reduce_dict_nonelegant(input_dict, average=True):
    world_size = dist.get_world_size()
    
    gather_dict = [None for _ in range(world_size)]
    
    
    output = {key: 0.0 for key in input_dict.keys()}
    
    dist.all_gather(gather_dict, input_dict)
    for rank_inp in gather_dict[0]:
        for key, val in rank_inp.items():
            output[key] += val
    
    if average:
        output = {key: value / world_size for key, value in output.items()}
    
    return output

class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None):
        # a dict, whose value is SmoothedValue, and key is the metric name
        self.meters = defaultdict(SmoothedValue)
        self.delimeter = delimiter
        self.logger = logger
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor): # in case we don't call .item() before send to here
                v = v.item()
            assert isinstance(v, (int, float))
            self.meters[k].update(v)
        
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]    # return SmoothedValue
        elif attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"{name}: {str(meter)}"
            )
        return self.delimeter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_process()
    
    def add_meter(self, name, meter):
        # meter: SmoothedValue
        self.meters[name] = meter
    
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")  # data+forward+backward
        data_time = SmoothedValue(fmt="{avg:.4f}")  # data
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"    # ":2d" number of digits of the batch_size
        if torch.cuda.is_available():
            log_msg = self.delimeter.join([header,
                                           "[{0" + space_fmt + "}/{1}]",
                                           "eta: {eta}",
                                           "{meters}",
                                           "time: {time}",
                                           "data: {data}",
                                           "max mem: {memory:.0f} MB"])
        else:
            log_msg = self.delimeter.join([header,
                                           "[{0" + space_fmt + "}/{1}]",
                                           "eta: {eta}",
                                           "{meters}",
                                           "time: {time}",
                                           "data: {data}"])
        MB = 1024.0 * 1024.0    # by dafault, torch.cuda.max_memory_allocated() return 1 Byte = 1/1024 KB
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_second = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=eta_second))
                if torch.cuda.is_available():
                    if self.logger == None:
                        print(log_msg.format(i, len(iterable),
                                            eta=eta_string,
                                            meters=str(self),
                                            time=str(iter_time),
                                            data=str(data_time),
                                            memory=torch.cuda.max_memory_allocated() / MB))
                    else:
                        if is_main_process():   # actually redudant check, self.logger wll not be created on non-main process
                            self.logger.info(log_msg.format(
                                                i, len(iterable),
                                                eta=eta_string,
                                                meters=str(self),
                                                time=str(iter_time),
                                                data=str(data_time),
                                                memory=torch.cuda.max_memory_allocated() / MB
                                                )
                                            )
                else:
                    if self.logger == None:
                        print(log_msg.format(i, len(iterable),
                                            eta=eta_string,
                                            meters=str(self),
                                            time=str(iter_time),
                                            data=str(data_time)))
                    else:   # only single process in CPU mode
                        self.logger.info(log_msg.format(
                                            i, len(iterable),
                                            eta=eta_string,
                                            meters=str(self),
                                            time=str(iter_time),
                                            data=str(data_time),
                                            memory=torch.cuda.max_memory_allocated() / MB
                                            )
                                        )
            i +=1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")

def get_rank():
    if not is_dist_avail_and_initialize():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    # as long as the input order is: "weights_dict", "file", 
    # which is consistent with use "torch.save()"" directly
    if is_main_process():
        torch.save(*args, **kwargs)

def get_world_size():
    if not is_dist_avail_and_initialize():
        return 1
    return dist.get_world_size()

def is_dist_avail_and_initialize():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def setup_for_distributed(is_master):
    """
    This function disables when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # singel node: LOCAL_RANK == RANK
    # multiple nodes: 
    #             |    Node1  |   Node2    |
    # ____________| p1 |  p2  |  p3  |  p4 |
    # local_rank  | 0  |   1  |  0   |   1 |
    # rank        | 0  |   1  |  2   |   4 |
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ("RANK"))
        args.word_size = int(os.environ("WORLD_SIZE"))
        args.gpu = int(os.environ("LOCAL_RANK"))
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = -1  # use one GPU, we set rank==-1
        return
        
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier(device_ids=[args.rank])
    setup_for_distributed(args.rank==0)

def build_optmizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.lr
        weight_decay = cfg.weight_decay     # the weight regularization terms
        if "bias" in key:
            lr = lr * cfg.bias_factor
            weight_decay = cfg.weight_decay_bias
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
    
    if cfg.optimzation_method == "sgd":
        optimizer = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif cfg.optimization_method == "adam":
        optimizer = torch.optim.Adam(params, lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)
    
    return optimizer

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Args:
        warmup_iters (_type_): In one epoch, <= warmup_iters, the LR is amplified by the ration returned by f(x)
    """
    
    def f(x):
        if x >= warmup_iters:      
            return 1
        alpha = float(x) / warmup_iters  # min(1000, bsz)
        return warmup_factor * (1 - alpha) + alpha  # 1 / 1000.
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

def set_environment(seed=17, set_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)

def create_logger(name=None, silent=False, to_disk=True, log_dir=None):
    logging.getLogger('PIL').setLevel(logging.WARNING)  # we don't want the PIL log
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propogate = False
    
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        # ch.setFormatter(formatter)
        log.addHandler(ch)
    
    if to_disk:
        log_file = os.path.join(log_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.log"))
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        # fh.addFormatter(formatter)
        log.addHandler(fh)
    
    return log