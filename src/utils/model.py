'''
Author: your name
Date: 2021-04-23 16:33:42
LastEditTime: 2021-04-23 16:33:41
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /steganography_platform_pl/src/utils/model.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .srm_kernels import all_normalized_hpf_list

__all__ = ['ABS', 'TLU', 'HPF', 'SPPLayer']

# absult value operation
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, input):
        output = torch.abs(input)
        return output

# Truncation operation
class TLU(nn.Module):
    def __init__(self, threshold):
        super(TLU, self).__init__()

        self.threshold = threshold

    def forward(self, input):
        output = torch.clamp(input, min=-self.threshold, max=self.threshold)

        return output

# Pre-processing Module
class HPF(nn.Module):
    def __init__(self):
        super(HPF, self).__init__()

        # Load 30 SRM Filters
        all_hpf_list_5x5 = []

        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

            all_hpf_list_5x5.append(hpf_item)

        hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)

        self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
        self.hpf.weight = hpf_weight

        # Truncation, threshold = 3
        self.tlu = TLU(3.0)

    def forward(self, input):

        output = self.hpf(input)
        output = self.tlu(output)

        return output

# https://gist.github.com/erogol/a324cc054a3cdc30e278461da9f1a05e
class SPPLayer(nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)

            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, dim=-1)
        return x
