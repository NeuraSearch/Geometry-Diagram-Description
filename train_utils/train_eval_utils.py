# coding:utf-8

import os
import sys
import time
import math
import torch
import codecs
import json

from .distributed_utils import MetricLogger, SmoothedValue, is_main_process, warmup_lr_scheduler, reduce_dict, all_gather

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
    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = batch_data["images"]
        images_not_tensors = batch_data["images_not_tensors"]
        targets_det = batch_data["targets_det"]
        targets_seg = batch_data["targets_seg"]
        targets_geo = batch_data["targets_geo"]
        targets_sym = batch_data["targets_sym"]
        
        images = list(image.to(device) for image in images)
        targets_det = [target.to(device) for target in targets_det]
        targets_geo = [target.to(device) for target in targets_geo]
        targets_sym = [target.to(device) for target in targets_sym]
        # NOTE: targets_seg.masks will be used to get the targeted pixel, after that, it will be sent to GPU in model internally,
        #       images_not_tensors is for OCR, no need to put in GPU here.

        # mixed-precision train, no action taken in CPU
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # loss_dict.keys:
            #   "loss_cls", "loss_reg", "loss_centerness", "loss_binary_seg", "loss_var", "loss_dist", "loss_mean_reg"
            #   {"pl_loss": Tensor, "pc_loss": Tensor,
            #    "text_symbol_geo_rel_loss", "head_symbol_geo_rel_loss",
            #    "angle_symbols_geo_rel_loss", "bar_symbols_geo_rel_loss",
            #    "parallel_symbols_geo_rel_loss", "perpendicular_symbols_geo_rel_loss"}
            loss_dict = model(
                images=images,
                images_not_tensors=images_not_tensors,
                targets_det=targets_det,
                targets_seg=targets_seg,
                targets_geo=targets_geo,
                targets_sym=targets_sym,
            )
            
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

        # **loss_dict_reduced means we pass all the loss value as key=val mode.
        metric_logger.update(loss=loss_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)
        
        # wandb is not None only on rank0
        if run != None:
            run.log({"loss_reduced": loss_reduced})
            run.log(loss_dict_reduced)
            
    return mloss, now_lr

@torch.no_grad()
def evaluate(model, data_loader, device, save_dir, epoch=None):
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    
    # TODO: define the metric here, e.g., accuracy
    
    predictions = {}
    for batch_data in metric_logger.log_every(data_loader, 100, header):
        # "images": images,
        # "images_not_tensors": images_not_tensors,
        # "targets_det": targets_det,
        # "targets_seg": targets_seg,
        # "targets_geo": targets_geo,
        # "targets_sym":targets_sym,
        
        images = batch_data["images"]
        images_not_tensors = batch_data["images_not_tensors"]
        
        images = list(img.to(device) for img in images)
        
        # we have to wait until everything has been done,
        # then we start to count
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)
        
        model_time = time.time()
        outputs = model(images=images, images_not_tensors=images_not_tensors)
        
        """ *** Customized Part *** """
        parse_results, natural_language_results = outputs
        natural_language_results_with_id = {img_id: natural_language_results[idx] for idx, img_id in enumerate(batch_data["images_id"])}
        gathered_natural_language_results = all_gather(natural_language_results_with_id)[0]
        predictions.update(gathered_natural_language_results)
        """ *** *** *** *** *** *** """
        
        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        # NOTE: we don't need to call "torch.cuda.synchronize(device)" again,
        #   because the above line needs to wait until it obtains "outputs" from the model
        # This is also true for the time.time() in metric_logger.log_every,
        # because before the time, there is lots of python code waiting for "outputs" from the model
        model_time = time.time() - model_time
                
        metric_logger.update(model_time=model_time)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats: ", metric_logger)
    
    return predictions
    
    # TODO: 自定义的metric也应该有同步函数, 将所有的预测整合到一起, 这样在下面做一次evaluate就行
    
    if is_main_process():
        # TODO: 上面自定义metric整合过以后, 在这里再做一次evaluate
        pass
    else:
        # TODO: 将预测结果设为None
        pass
    
    # TODO: 返回预测结果
    return