# coding:utf-8

import sys
import time
import math
import torch

from distributed_utils import MetricLogger, SmoothedValue, is_main_process, warmup_lr_scheduler, reduce_dict

def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None, run=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    
    # different with "lr_scheduler" in outside func, which only updates each epoch
    # this is used for update each step
    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)  # one epoch, the number of steps to warmup lr
        
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    mloss = torch.zeros(1).to(device)
    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        # TODO: targets不一定是tensor
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # mixed-precision train, no action taken in CPU
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            
            losses = sum(loss for loss in loss_dict.values())
        
        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_reduced = sum(loss for loss in loss_dict_reduced.values())
        
        loss_value = loss_reduced.item()
        # mloss * i: total loss of previous steps
        mloss = (mloss * i + loss_value) / (i + 1)
        
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stop training")
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        
        # wandb is not None only on rank0
        if run != None:
            run.log({"loss_reduced": loss_reduced})
            run.log(loss_dict_reduced)
            
    return mloss, now_lr

@torch.no_grad()
def evaluate(model, data_loader, device):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    
    # TODO: define the metric here, e.g., accuracy
    
    # TODO: might not images, targets, revise according to real circumstance
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        
        # we have to wait until everything has been done,
        # then we start to count
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        
        model_time = time.time()
        outputs = model(image)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # NOTE: we don't need to call "torch.cuda.synchronize(device)" again,
        #   because the above line needs to wait until it obtains "outputs" from the model
        # This is also true for the time.time() in metric_logger.log_every,
        # because before the time, there is lots of python code waiting for "outputs" from the model
        model_time = time.time() - model_time
        
        # TODO: update metric if defined
        
        metric_logger.update(model_time=model_time)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats: ", metric_logger)
    
    # TODO: 自定义的metric也应该有同步函数, 将所有的预测整合到一起, 这样在下面做一次evaluate就行
    
    if is_main_process():
        # TODO: 上面自定义metric整合过以后, 在这里再做一次evaluate
        pass
    else:
        # TODO: 将预测结果设为None
        pass
    
    # TODO: 返回预测结果
    return