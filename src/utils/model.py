import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .srm_kernels import all_normalized_hpf_list

__all__ = ['ABS', 'SRM', 'SPP', 'TLU']
class ABS(nn.Module):
    def __init__(self):
        super(ABS, self).__init__()

    def forward(self, x):
        return torch.abs(x)

class TLU(nn.Module):
    def __init__(self, threshold=3.0):
        super(TLU, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        return torch.clamp(x, min=-self.threshold, max=self.threshold)

class SRM(nn.Module):
    def __init__(self, is_update=False):
        super(SRM, self).__init__()
        # Load 30 SRM Filters
        all_hpf_list_5x5 = []
        for hpf_item in all_normalized_hpf_list:
            if hpf_item.shape[0] == 3:
                hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
            all_hpf_list_5x5.append(hpf_item)
        all_hpf_list_5x5 = torch.tensor(all_hpf_list_5x5).unsqueeze(1)
        # conv
        self.conv = nn.Conv2d(1, 30, 5, 1, 2, )
        self.conv.weight = nn.Parameter(all_hpf_list_5x5, requires_grad=False)

    def forward(self, x):
        out = self.conv(x)
        return out

# https://gist.github.com/erogol/a324cc054a3cdc30e278461da9f1a05e
class SPP(nn.Module):
    def __init__(self, num_levels):
        super(SPP, self).__init__()

        self.num_levels = num_levels

    def forward(self, x):
        B, C, H, W = x.shape
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = H // (2 ** i)

            tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                  stride=kernel_size).reshape((B, -1))
            pooling_layers.append(tensor)
        x = torch.cat(pooling_layers, axis=-1)
        return x
