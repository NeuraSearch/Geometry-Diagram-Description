# coding:utf-8

import os
import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import time
import json
import codecs
import torch
import wandb
import datetime
import numpy as np
from functools import partial
from transformers import T5Tokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from models import TransformerProgramGenerator
from train_utils import init_distributed_mode, \
        GroupBatchSampler, \
            train_one_epoch_program, evaluate_program, \
                save_on_master, set_environment, \
                    is_main_process, build_optmizer_for_t5, create_logger, \
                        load_from_url
from data_loader import make_data_loader_for_T5, unigeo_data_collate_fn

def main(args):
    # setup multiple GPUs training
    init_distributed_mode(args)
    print("Finish initializing distribution mode.")
    
    # set random seed
    set_environment(args.seed)
    
    # args.device: "cuda"
    # 在不同进程的时候, init_distributed_mode()通过`torch.cuda.set_device(args.gpu)`
    # 指定每个进程应该看到哪一个GPU, 因此在这里torch.device(args.device)虽然只知道"cuda",
    # 但因为只知道一个GPU, 所以就会分配给对应的GPU
    device = torch.device(args.device)
    
    # results save path
    now = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    results_file = f"results_{now}"
    save_dir = str(MAIN_PATH / f"{args.save_dir}/{now}")
    os.makedirs(save_dir, exist_ok=True)

    # create logger
    if is_main_process():
        logger = create_logger(log_dir=save_dir)
    else:
        logger = None

    # create logger
    if is_main_process():
        logger = create_logger(log_dir=save_dir)
    else:
        logger = None
    
    # don't worry about the "print()", it has been force to work on the main rank
    # by "setup_for_distributed()" after calling "init_distributed_mode()" above
    print(args)
    with codecs.open(os.path.join(save_dir, "config.json"), "w", "utf-8") as file:
        json.dump(vars(args), file, indent=2)
    
    # wandb
    if is_main_process() and args.wandb_key != None:
        wandb.login(key=args.wandb_key)
        run = wandb.init(name=f"{now}", project=f"{args.project_name}", config=args)
    else:
        run = None

    # create dataset
    print("Loading data")
    # we might update args.train_img_per_batch, args.test_img_per_batch
    train_dataset, eval_dataset, test_dataset = make_data_loader_for_T5(args, is_train=args.is_train)

    # create sampler
    print("Creating data loaders")
    if args.distributed:
        if args.is_train:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
            eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset)
        else:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        if args.is_train:
            train_sampler = torch.utils.data.RandomSampler(train_dataset)
            # train_sampler = torch.utils.data.SequentialSampler(train_dataset)
            eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
        else:
            test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    # create batch sampler
    if args.is_train:
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.t5_train_img_per_batch, drop_last=True)

    tokenizer = T5Tokenizer.from_pretrained(args.model_type)
    unigeo_data_collate_fn_func = partial(unigeo_data_collate_fn, model_type=tokenizer)
    if args.is_train:
        # create data loader
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=unigeo_data_collate_fn_func)
        data_loader_eval = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.t5_test_img_per_batch, 
            sampler=eval_sampler, num_workers=args.workers,
            collate_fn=unigeo_data_collate_fn_func)
    else:
        # NOTE: Actually, Since test dataset is sample sequentially,
        #   we don't need to create sampler for it. Instead, we set batch_size=N, shffule=False
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=unigeo_data_collate_fn_func)

    # create model
    print("creating model")
    model = TransformerProgramGenerator(args)
    # model = ToyModel()
    model.to(device)
    print("finsih creating model")

    # multiple GPU mode, we need to convert the bn to sync_BN
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    print("detect whether wrap model to DDP")
    # make model ddp
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # after DDP wrap, the model is "model.module", so better use "model_without_ddp" to avoid "module" prefix.
        model_without_ddp = model.module
    print("finish detection for whether wrap model to DDP")
    
    if is_main_process() and run != None:
        run.watch(model)

    # build optimizer, specified the weight_decay for bias
    optimizer = build_optmizer_for_t5(args, model)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, 1000, args.epoch * len(data_loader_train) // args.t5_train_img_per_batch)
    
    args.start_epoch = 0
    if args.resume:
        print(f"Restore model from {os.path.join(str(MAIN_PATH / args.save_dir), args.resume)}")
        checkpoint = torch.load(os.path.join(str(MAIN_PATH / args.save_dir), args.resume), map_location="cpu")
                
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    train_loss = []
    learning_rate = []
    
    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        mean_loss, lr = train_one_epoch_program(model, optimizer, data_loader_train,
                                        device, epoch, lr_scheduler, args.print_freq,
                                        warmup=args.warmpup, scaler=scaler, run=run, logger=logger)
    
        print(f"[Epoch: {epoch}] starts evaluation ...")
        predictions = evaluate_program(model, data_loader_eval, device=device, tokenizer=tokenizer, cfg=args, save_dir=save_dir, epoch=epoch, logger=logger)
        
        # only write in the main rank
        if args.rank in [-1, 0]:
            os.makedirs(save_dir, exist_ok=True)
            save_file_name = f"{epoch}_{results_file}.json"
            with codecs.open(os.path.join(save_dir, save_file_name), "w", "utf-8") as file:
                json.dump(predictions, file, indent=2)
                
            # only save weights in main rank
            save_files = {"model": model_without_ddp.state_dict(),
                          "optimizer": optimizer.state_dict(),
                          "lr_scheduler": lr_scheduler.state_dict(),
                          "args": args,
                          "epoch": epoch}
            if args.amp:
                save_files["scaler"] = scaler.state_dict()
            save_on_master(save_files,
                           os.path.join(save_dir, f"model_{epoch}.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_args_yaml", type=str)
    args = parser.parse_args()
    
    import yaml
    import codecs
    with codecs.open(args.train_args_yaml, "r", "utf-8") as file:
        config = yaml.safe_load(file)
    
    args = vars(args)   # Namespace to dict
    for _, conf in config.items():
        for key, val in conf.items():
            args[key] = val
    
    from argparse import Namespace
    args = Namespace(**args)    # dict to Namespace
    print(args.resume == None)
    args.test_only = args.is_train == False
    print(args.test_only)
    
    # port_id = 10000 + np.random.randint(0, 1000)
    # args.dist_url = 'tcp://127.0.0.1:29500'
    main(args)