import argparse
import datetime
import json
import random
import time
import numpy as np
import torch
import datasets
import util.misc as utils
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_dataset
from engine import evaluate_swig, train_one_epoch
from models import build_model
from pathlib import Path
import matplotlib.pyplot as plt

from datasets.swig import collater

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def print_sample(sample):
    print(sample['img'].shape)
    print(sample['annot'].shape)
    print(sample['img_name'])
    print(sample['verb_idx'])
    print(sample['verb_role_idx'])

    plt.figure()
    plt.imshow(sample['img'])
    plt.show()  # display it


def train(args:Namespace):
    print("git:\n  {}\n".format(utils.get_sha()))
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build dataset
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    args.num_noun_classes = dataset_train.num_nouns()

    # build model
    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    # optimizer & LR scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    output_dir = Path(args.output_dir)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    data_loader_train = DataLoader(dataset_train, num_workers=args.num_workers,
                                   collate_fn=collater, batch_sampler=batch_sampler_train)
    data_loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                                 drop_last=False, collate_fn=collater, sampler=sampler_val)

    # train model
    print("Start training")
    start_time = time.time()
    max_test_mean_acc = 42
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer,
                                      device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        # evaluate
        test_stats = evaluate_swig(model, criterion, data_loader_val, device, args.output_dir)

        # log & output
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # save checkpoint for every new max accuracy
            if log_stats['test_mean_acc_unscaled'] > max_test_mean_acc:
                max_test_mean_acc = log_stats['test_mean_acc_unscaled']
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({'model': model_without_ddp.state_dict(),
                                      'optimizer': optimizer.state_dict(),
                                      'lr_scheduler': lr_scheduler.state_dict(),
                                      'epoch': epoch,
                                      'args': args}, checkpoint_path)

        # write log
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def test_accuracy(args:Namespace):
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_test = build_dataset(image_set='test', args=args)

    # build model
    model, criterion = build_model(args)
    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]

    # optimizer & LR scheduler
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    output_dir = Path(args.output_dir)

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = DataLoader(dataset_test, num_workers=args.num_workers,
                                  drop_last=False, collate_fn=collater, sampler=sampler_test)

    checkpoint = torch.load(args.saved_model, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    data_loader = data_loader_test

    test_stats = evaluate_swig(model, criterion, data_loader, device, args.output_dir)
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}

    # write log
    if args.output_dir and utils.is_main_process():
        with (output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    return


if __name__ == '__main__':
    args = Namespace(
        lr=0.0001,
        lr_backbone=1e-05,
        lr_drop=100,
        weight_decay=0.0001,
        clip_max_norm=0.1,
        batch_size=2,
        epochs=40,
        backbone='resnet50',
        position_embedding='learned',
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=2048,
        hidden_dim=512,
        dropout=0.15,
        nheads=8,
        noun_loss_coef=1,
        verb_loss_coef=1,
        bbox_loss_coef=5,
        bbox_conf_loss_coef=5,
        giou_loss_coef=5,
        dataset_file='swig',
        swig_path='SWiG',
        dev=False,
        test=True,
        inference=False,
        output_dir='',
        device='cpu',
        seed=42,
        start_epoch=0,
        num_workers=4,
        saved_model='gsrtr_checkpoint.pth',
        world_size=1,
        dist_url='env://'
    )

    train(args)