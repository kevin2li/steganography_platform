import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['initialization', 'partition_parameters']
def initialization(model: nn.Module):
    # Common practise for initialization.
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(layer.weight, mode='fan_out',
                                        nonlinearity='relu')
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(layer.weight, val=1.0)
            torch.nn.init.constant_(layer.bias, val=0.0)
        elif isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, val=0.0)

def partition_parameters(model: nn.Module, weight_decay):
    params_wd, params_rest = [], []
    for m in model.parameters():
        if m.requires_grad:
            (params_wd if m.dim()!=1 else params_rest).append(m)
    param_groups = [{'params': params_wd, 'weight_decay': weight_decay},
                    {'params': params_rest}]
    return param_groups

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
