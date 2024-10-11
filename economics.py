from pathlib import Path
from PIL import Image
import json
import numpy as np

import os
import torch
import torch.utils.data
import torchvision
from torchvision.transforms import transforms as T
import random

mode_mapping = {
    'gdp': 0,
    'ag': 1,
    'man': 2,
    'serv': 3,
}

class Economics(torch.utils.data.Dataset):

    def __init__(self, root_dir, image_set, output_mode, mask_rate=0.1):
        self.root_dir = root_dir
        self.image_set = image_set
        self.mask_rate = mask_rate
        self.filenames = os.listdir(os.path.join(root_dir, image_set))
        self.transforms, self.output_transforms = self.get_normalize()
        self.output_index = mode_mapping[output_mode]

    def get_normalize(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])

        output_transforms = T.Compose([
            T.ToTensor(),
        ])

        return transforms, output_transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.load(os.path.join(self.root_dir, self.image_set, filename), allow_pickle=True).item()
        inputs = self.transforms(data['input'])
        outputs = self.output_transforms(data['output'])[:, :, self.output_index:self.output_index+1]
        # outputs = self.output_transforms(data['output'])

        # print(inputs.shape, outputs.shape)
        i_mask = torch.logical_not(torch.isnan(inputs))
        i_mask = i_mask.sum(dim=-1) > 0
        o_mask = torch.logical_not(torch.isnan(outputs))

        # 基于区域进行mask，随机选取1554中ratio的区域设置为0
        inputs = torch.nan_to_num(inputs) + 1e-8
        outputs = torch.nan_to_num(outputs)

        if self.image_set == 'train':
            return inputs, outputs, i_mask, o_mask
        else:
            return inputs, outputs, i_mask, o_mask, filename


class EconomicsAnalysis(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_set, mask_rate=0.1):
        self.root_dir = root_dir
        self.image_set = image_set
        self.mask_rate = mask_rate
        self.filenames = os.listdir(os.path.join(root_dir, image_set))
        self.transforms, self.output_transforms = self.get_normalize()

    def get_normalize(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])

        output_transforms = T.Compose([
            T.ToTensor(),
        ])

        return transforms, output_transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = np.load(os.path.join(self.root_dir, self.image_set, filename), allow_pickle=True).item()
        inputs = self.transforms(data['input'])
        i_mask = torch.logical_not(torch.isnan(inputs))
        i_mask = i_mask.sum(dim=-1) > 0

        inputs = torch.nan_to_num(inputs) + 1e-8

        return inputs, i_mask, filename



