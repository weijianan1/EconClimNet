import torch
import torch.nn.functional as F
from torch import nn
import json

from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

import copy
import numpy as np
from transformer import build_transformer
from pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
    get_iso_region_sincos_pos_embed_from_grid,
)


class EconClimNet(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer
        hidden_dim = transformer.d_model

        self.pos_embed = nn.Parameter(torch.zeros(1, 1554, hidden_dim), requires_grad=True)
        self.input_proj = nn.Sequential(
            nn.Linear(111, hidden_dim),
            nn.GELU()
        )

        self.iso_mask = torch.from_numpy(np.load('data/iso_mask.npy'))

        self.head = nn.ModuleList()
        for _ in range(2):
            self.head.append(nn.Linear(hidden_dim, hidden_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(hidden_dim, 1))
        self.head = nn.Sequential(*self.head)

        self.initialize_weights()

    def initialize_weights(self):
        iso2reg = torch.from_numpy(np.load('data/iso2reg.npy'))
        pos_embed = get_iso_region_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], [np.arange(1554), np.arange(iso2reg[-1]+1)], iso2reg)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x: torch.Tensor, mask=None):
        x = self.input_proj(x)
        iso_mask = self.iso_mask.to(x.device).float()
        iso_mask = iso_mask == 1.0

        out_transformers = self.transformer(x, ~iso_mask, None, self.pos_embed)
        preds = self.head(out_transformers)

        return preds


def build(args):
    if args.model_name == 'EconClimNet':
        transformer = build_transformer(args)
        model = EconClimNet(transformer)

    return model



