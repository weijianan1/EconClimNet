import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch
import torch.nn.functional as F

import util.misc as utils
from metrics import mse

import wandb
from sklearn.metrics import r2_score


def train_one_epoch(model: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = len(data_loader) // 1
    weight_dict = {'loss_mse': 1.0}

    for inputs, targets, i_mask, o_mask in metric_logger.log_every(data_loader, print_freq, header):
        inputs = inputs.to(torch.float32).to(device).squeeze(1)
        targets = targets.to(torch.float32).to(device).squeeze(1)
        # i_mask = i_mask.to(torch.float32).to(device).squeeze(1)
        # o_mask = o_mask.to(torch.float32).to(device).squeeze(1)
        i_mask = i_mask.to(device).squeeze(1)
        o_mask = o_mask.to(device).squeeze(1)

        outputs = model(inputs, i_mask)
        losses = mse(outputs, targets, o_mask)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict({'loss_mse': losses})
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # if utils.get_rank() == 0: wandb.log({k: meter.global_avg for k, meter in metric_logger.meters.items()})
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = len(data_loader) // 1

    preds = []
    gts = []
    _masks = []
    _filenames = []
    for samples, targets, i_mask, o_mask, filenames in metric_logger.log_every(data_loader, print_freq, header):
        # print(samples.size(), targets.size(), masks.size())
        samples = samples.to(torch.float32).to(device).squeeze(1)
        targets = targets.to(torch.float32).to(device).squeeze(1)
        i_mask = i_mask.to(torch.float32).to(device).squeeze(1)
        o_mask = o_mask.to(torch.float32).to(device).squeeze(1)
        outputs = model(samples, i_mask)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(outputs)))))
        # For avoiding a runtime error, the copy is used
        gts.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(targets)))))
        _masks.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(o_mask)))))
        _filenames.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(filenames)))))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    preds = torch.stack(preds, dim=0)
    gts = torch.stack(gts, dim=0)
    _masks = torch.stack(_masks, dim=0)

    _rmse = torch.sqrt(mse(preds, gts, _masks)).item()

    _masks = _masks == 1
    gts_masked = gts[_masks[...,0]].cpu()
    preds_masked = preds[_masks[...,0]].cpu()

    r2 = r2_score(gts_masked, preds_masked)
    print('Evaluate', {'rmse': _rmse, 'r2': r2})

    return {'rmse': _rmse, 'r2': r2}


@torch.no_grad()
def analysis(model, data_loader, device, save_path, mode, type):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = len(data_loader) // 1

    preds = []
    _filenames = []
    _i_masks = []
    for samples, i_mask, filenames in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(torch.float32).to(device).squeeze(1)
        i_mask = i_mask.to(torch.float32).to(device).squeeze(1)
        outputs = model(samples, i_mask)

        preds.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(outputs)))))
        _filenames.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(filenames)))))
        _i_masks.extend(list(itertools.chain.from_iterable(utils.all_gather(copy.deepcopy(i_mask)))))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    # visualization
    import json
    import pandas as pd
    with open("/mnt/sda/liuleili/weijianan/climate/economics/grid_map.json", "r") as json_file:
        grid_map = json.load(json_file)
        grid_idx = []
        for iso in grid_map.keys():
            for id_1 in grid_map[iso]:
                grid_idx.append({'iso': iso, 'id_1': id_1})
        print('Grid number', len(grid_idx))

    result = []
    for pred, filename, i_mask in zip(preds, _filenames, _i_masks):
        for index, grid in enumerate(grid_idx):
            pred_gdp = pred[index, 0].item()
            result.append({
                'iso': grid['iso'],
                'id_1': grid['id_1'],
                'year': filename.strip('.npy'),
                'pred': pred_gdp,
                'avaliable': i_mask[index].item(),
            })

    if not os.path.exists(os.path.join(save_path, mode)):
        os.makedirs(os.path.join(save_path, mode))
    pd.DataFrame(result).to_csv(os.path.join(save_path, mode, f"{type}.csv"))






