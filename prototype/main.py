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

from models import DiagramDescribe
from train_args import add_train_args
from train_utils import init_distributed_mode, create_aspect_ratio_groups, \
        GroupBatchSampler, \
            train_one_epoch, evaluate, \
                save_on_master, set_environment, \
                    is_main_process, build_optmizer, create_logger
from data_loader import make_data_loader, geo_data_collate_fn

def main(args):
    # setup multiple GPUs training
    init_distributed_mode(args)
    
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
    
    # don't worry about the "print()", it has been force to work on the main rank
    # by "setup_for_distributed()" after calling "init_distributed_mode()" above
    print(args)
    with codecs.open(os.path.join(save_dir, "config.json"), "w", "utf-8") as file:
        json.dump(vars(args), file, indent=2)
    
    # wandb
    if is_main_process() and args.wandb_key != None:
        wandb.login(key=args.wandb_key)
        run = wandb.init(project=f"{args.project_name}_{now}", config=args)
    else:
        run = None

    # create dataset
    print("Loading data")
    train_dataset, eval_dataset, test_dataset = make_data_loader(args, is_train=args.is_train)

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
            eval_sampler = torch.utils.data.SequentialSampler(eval_dataset)
        else:
            test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    
    # create batch sampler
    if args.is_train:
        if args.aspect_ratio_group_factor >= 0:
            # category indices of images into different bins according to W/H
            group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
            train_batch_sampler = GroupBatchSampler(train_sampler, group_ids, args.train_img_per_batch)
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.train_img_per_batch, drop_last=True)
    
    if args.is_train:
        # create data loader
        data_loader_train = torch.utils.data.DataLoader(
            train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=geo_data_collate_fn)
        data_loader_eval = torch.utils.data.DataLoader(
            eval_dataset, batch_size=args.test_img_per_batch, 
            sampler=eval_sampler, num_workers=args.workers,
            collate_fn=geo_data_collate_fn)
    else:
        # NOTE: Actually, Since test dataset is sample sequentially,
        #   we don't need to create sampler for it. Instead, we set batch_size=N, shffule=False
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=geo_data_collate_fn)
    
    # create model
    print("creating model")
    model = DiagramDescribe(args)
    model.to(device)

    # multiple GPU mode, we need to convert the bn to sync_BN
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # make model ddp
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    if is_main_process() and run != None:
        run.watch(model)
    
    # build optimizer, specified the weight_decay for bias
    optimizer = build_optmizer(args, model)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # build lr_scheduler
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    
    args.start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if args.map and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if not args.is_train:
        predictions = evaluate(model, data_loader_test, device=device, logger=logger)
        os.makedirs(save_dir, exist_ok=True)
        save_file_name = f"not_train_{results_file}.json"
        with codecs.open(save_file_name, "w", "utf-8") as file:
            json.dump(predictions, file, indent=2)
        return

    train_loss = []
    learning_rate = []
    # TODO: 定义一些我们自己的metric记录列表
    
    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            # In distributed mode, calling the set_epoch() method at the beginning of each epoch 
            # before creating the DataLoader iterator is necessary to make shuffling work properly 
            # across multiple epochs. Otherwise, the same ordering will be always used.
            train_sampler.set_epoch(epoch)
        
        # mean_loss: gathered loss from all GPU mean.
        mean_loss, lr = train_one_epoch(model, optimizer, data_loader_train,
                                        device, epoch, args.print_freq,
                                        warmup=args.warmpup, scaler=scaler, run=run, logger=logger)

        # this external lr_scheduler adjusts lr every epoch
        # update learning rate, should call lr_scheduler.step() after optimizer.step() in latest version of Pytorch
        lr_scheduler.step()
        
        print(f"[Epoch: {epoch}] starts evaluation ...")
<<<<<<< HEAD
        predictions = evaluate(model, data_loader_eval, device=device, logger=logger)
=======
        predictions = evaluate(model, data_loader_eval, device=device)
>>>>>>> b522a840c121bd214f01d47a55b17bbb5338706d
        
        # only write in the main rank
        if args.rank in [-1, 0]:
            train_loss.append(mean_loss.item())
            learning_rate.append(lr)
            # TODO: 自建metric_list加入
            
            os.makedirs(save_dir, exist_ok=True)
            save_file_name = f"{epoch}_{results_file}.json"
            with codecs.open(save_file_name, "w", "utf-8") as file:
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
    
    if args.rank in [-1, 0] and args.plot:
        if len(train_loss) != 0 and len(learning_rate) != 0:
            # TODO: plot something, the plot_curve has bug !!!.
            pass
        pass

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
    
    main(args)