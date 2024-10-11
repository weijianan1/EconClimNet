import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from economics import Economics, EconomicsAnalysis
from model import build as build_model
# from model_s import build as build_model
import util.misc as utils
from lr_scheduler import LinearWarmupCosineAnnealingLR
from engine import train_one_epoch, evaluate, analysis

import os
import logging
import wandb
from torch.optim.lr_scheduler import MultiStepLR

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--model_name', default='mlp', type=str)

    parser.add_argument('--root_dir', default='/mnt/sda/liuleili/weijianan/climate/economics', type=str)
    parser.add_argument('--output_mode', default='gdp', type=str)
    parser.add_argument('--warmup_epochs', default=60, type=int)
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--warmup_start_lr', default=1e-8, type=float)
    parser.add_argument('--eta_min', default=1e-8, type=float)

    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # input proj
    parser.add_argument('--input_dims', default=111, type=int)
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of stage1 decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    # * WanDB
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--project_name', default='climate')
    parser.add_argument('--group_name', default='economics')
    parser.add_argument('--run_name', default='test')

    parser.add_argument('--mask_rate', default=0.1, type=float)

    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--analysis_dir', default='/mnt/sda/liuleili/weijianan/climate/further_analysis/latitude', type=str)
    parser.add_argument('--save_path', default='/mnt/sda/liuleili/weijianan/climate/further_results', type=str)
    # parser.add_argument('--mode', default='gdp', type=str)

    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # # 添加
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    # torch.use_deterministic_algorithms(True)

    model = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = MultiStepLR(optimizer, milestones=[10000], gamma=0.2)

    dataset_train = Economics(root_dir=args.root_dir, image_set='train', output_mode=args.output_mode, mask_rate=args.mask_rate)
    dataset_val = Economics(root_dir=args.root_dir, image_set='test', output_mode=args.output_mode, mask_rate=args.mask_rate)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_size=2, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.analysis:
        for folder in os.listdir(args.analysis_dir):
            print(folder)
            dataset_val = EconomicsAnalysis(root_dir=args.analysis_dir, image_set=folder, mask_rate=args.mask_rate)
            if args.distributed:
                sampler_val = DistributedSampler(dataset_val, shuffle=False)
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            data_loader_val = DataLoader(dataset_val, batch_size=2, sampler=sampler_val, drop_last=False, num_workers=args.num_workers)
            test_stats = analysis(model, data_loader_val, device, args.save_path, args.output_mode, folder)
        return

    if args.eval:
        print(checkpoint['epoch'])
        test_stats = evaluate(model, data_loader_val, device)
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': checkpoint['epoch'] + 1}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "test_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        return

    # add argparse
    os.environ["WANDB_API_KEY"] = '6d9869849c60053e101ee6d2fd52e153138ba87b'
    if args.wandb and utils.get_rank() == 0:
        wandb.init(
            project=args.project_name,
            group=args.group_name,
            name=args.run_name,
            config=args
        )
        wandb.watch(model)

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer, 
    #     max_lr=args.lr,
    #     total_steps=args.epochs,
    #     # pct_start=0.05,
    #     pct_start=0.3,
    #     cycle_momentum=False,
    #     anneal_strategy='cos',
    #     last_epoch=-1,
    # )

    print("Start training")
    start_time = time.time()
    best_performance = 1.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, epoch, args.clip_max_norm)
        lr_scheduler.step()

        checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pth')
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args,
        }, checkpoint_path)

        test_stats = evaluate(model, data_loader_val, device)
        performance = test_stats['rmse']
        if args.wandb and utils.get_rank() == 0:
            # wandb.log({
            #     'rmse': test_stats['rmse'],
            #     'r2': test_stats['r2'],
            # })
            train_stats.update({'rmse': test_stats['rmse'], 'r2': test_stats['r2']})
            wandb.log(train_stats)

        if performance < best_performance:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_best.pth')
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)

            best_performance = performance

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch,
                    'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if epoch == args.epochs - 1 and os.path.exists(os.path.join(output_dir, 'checkpoint_best.pth')):
            checkpoint = torch.load(os.path.join(output_dir, 'checkpoint_best.pth'), map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = -1
                best_epoch = checkpoint['epoch'] + 1
            model.to(device)
            test_stats = evaluate(model, data_loader_val, device)

            if args.output_dir and utils.is_main_process():
                #  add eval in log for my convenience
                with (output_dir / "log.txt").open("a") as f:
                    f.write('Test result:' + json.dumps(test_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('GEN VLKT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

