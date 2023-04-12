# coding:utf-8

import os
import sys
from pathlib import Path
MAIN_PATH = Path(__file__).absolute().parent.parent
sys.path.insert(0, str(MAIN_PATH))

import time
import json
import torch
import wandb
import datetime

from train_args import add_train_args
from train_utils import init_distributed_mode, create_aspect_ratio_groups, \
        GroupBatchSampler, \
            train_one_epoch, evaluate, \
                save_on_master, set_environment, \
                    is_main_process

def main(args):
    # setup multiple GPUs training
    init_distributed_mode(args)
    
    # set random seed
    set_environment(args.seed)
    
    # don't worry about the "print()", it has been force to work on the main rank
    # by "setup_for_distributed()" after calling "init_distributed_mode()" above
    print(args)
    with codecs.open(os.path.join(args.output_dir, "config.json"), "w", "utf-8") as file:
        json.dump(vars(args), file, indent=2)
    
    # args.device: "cuda"
    # 在不同进程的时候, init_distributed_mode()通过`torch.cuda.set_device(args.gpu)`
    # 指定每个进程应该看到哪一个GPU, 因此在这里torch.device(args.device)虽然只知道"cuda",
    # 但因为只知道一个GPU, 所以就会分配给对应的GPU
    device = torch.device(args.device)
    
    # results save path
    now = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    results_file = f"results_{now}.txt"
    
    # wandb
    if is_main_process() and args.wandb_key != None:
        wandb.login(key=args.wandb_key)
        run = wandb.init(project=f"{args.project_name}_{now}", config=args)
    else:
        run = None

    print("Loading data")
    data_transform = {"train": None, "test": None}
    # TODO: 下面创建数据集
    # train_dataset = 
    # test_dataset =
    
    # TODO: 构建sampler, 这个或许可以整合到上一步中
    print("Creating data loaders")
    if args.distributed:
        # TODO: 构建dataset之后, 去掉注释
        pass
        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        # test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        # TODO:
        pass
        # train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # test_sampler = torch.utils.data.SequentialSampler(train_dataset)
    
    # create batch sampler
    if args.aspect_ratio_group_factor >= 0:
        # category indices of images into different bins according to W/H
        # TODO: 构建dataset之后, 去掉注释
        pass
        # group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # train_batch_sampler = GroupBatchSampler(train_sampler, group_ids, args.batch_size)
    else:
        # TODO: 构建dataset之后, 去掉注释
        pass
        # train_batch_sampler = torch.utils.data.BatchSampler(
        #     train_sampler, args.batch_size, drop_last=True)
    
    # 实现train_dataset.collate_fn, 去掉注释
    # data_loader_train = torch.utils.data.DataLoader(
    #     train_dataset, batch_sampler=train_batch_sampler, num_workers=args.workers,
    #     collate_fn=train_dataset.collate_fn)
    
    # NOTE: Actually, Since test dataset is sample sequentially,
    #       we don't need to create sampler for it. Instead, we set batch_size=N, shffule=False
    # data_loader_test = torch._utils.data.DataLoader(
    #     test_dataset, batch_size=1,
    #     sampler=test_sampler, num_workers=args.workers,
    #     collate_fn=train_dataset.collate_fn)
    
    print("creating model")
    # TODO: create model
    # model = create_model(*args, **kwargs)
    # model.to(device)

    # multiple GPU mode, we need to convert the bn to sync_BN
    if args.distributed and args.sync_bn:
        # TODO: 去掉注释
        pass
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # TODO: 去掉注释
    # model_without_ddp = model
    if args.distributed:
        # TODO: 去掉注释
        pass
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        # model_without_ddp = model.module
    
    # TODO: 去掉注释
    # if is_main_process() and run != None:
    #     run.watch(model)
    
    # TODO: 去掉注释
    # params = [p for p in models.parameters() if p.requires_grad]
    # optimizer = torch.optim.SGD(
    #     params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # TODO: 去掉注释
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)
    
    if args.resume:
        # TODO: 去掉注释
        checkpoint = torch.load(args.resume, map_location="cpu")
        # model_without_ddp.load_state_dict(checkpoint["model"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        # args.start_epoch = checkpoint["epoch"] + 1
        if args.map and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # TODO: 去掉注释
        # evaluate(model, data_loader_test, device=device)
        pass
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
            # TODO: 去掉注释
            pass
            # train_sampler.set_epoch(epoch)
        
        # TODO: 去掉注释
        # mean_loss, lr = train_one_epoch(model, optimizer, data_loader_train,
        #                                 device, epoch, args.print_freq,
        #                                 warmup=True, scaler=scaler, run=run)

        # TODO: 去掉注释
        # lr_scheduler.step()
        
        # update learning rate, should call lr_scheduler.step() after optimizer.step() in latest version of Pytorch
        # TODO: 去掉注释
        # lr_scheduler.step()
        
        # TODO: 去掉注释
        # evaluate(model, data_loader_test, device=device)
        
        # only write in the main rank
        if args.rank in [-1, 0]:
            # TODO: 去掉注释
            # !!! 这里可能会报错, mean_loss就是通过loss_reduced.item()求得的
            # train_loss.append(mean_loss.item())
            # learning_rate.append(lr)
            # TODO: 自建metric_list加入
            
            # TODO: 写入信息
            with codecs.open(results_file, "a", "utf-8") as file:
                file.write(None)

        if args.output_dit:
            # only save weights in main rank
            # TODO: 去掉注释
            # save_files = {"model": model_without_ddp.state_dict(),
            #               "optimizer": optimizer.state_dict(),
            #               "lr_scheduler": lr_scheduler.state_dict(),
            #               "args": args,
            #               "epoch": epoch}
            if args.amp:
                # TODO: 去掉注释
                # save_files["scaler"] = scaler.state_dict()
                pass
            # TODO: 去掉注释
            # save_on_master(save_files,
            #                os.path.join(args.output_dir, f"model_{epoch}.pth"))
            pass
    
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
    add_train_args(parser)
    args = parser.parse_args()
    
    import yaml
    import codecs
    with codecs.open(args.train_args_yaml, "r", "utf-8") as file:
        config = yaml.safe_load(file)
    
    args = vars(args)
    for _, conf in config.items():
        for key, val in conf.items():
            assert key in args.keys()
            args[key] = val
    
    from argparse import Namespace
    args = Namespace(**args)
    print(args.data_path)
    print(args.num_classes)
    
    
    