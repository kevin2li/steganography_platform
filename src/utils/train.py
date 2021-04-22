import numbers
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['initialization', 'partition_parameters', 'transfer_parameters', 'AverageMeter', 'EarlyStopping']

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

def transfer_parameters(own_model: nn.Module, checkpoint):
    """
    Args:
        model: nn.Module, the model be transfered to
        checkpoint: trained model parameters
    """
    own_state_dict = own_model.state_dict()
    for k, v in checkpoint.items():
        if k in own_state_dict:
            if own_state_dict[k].shape == v.shape:
                own_state_dict[k] = v
    own_model.load_state_dict(own_state_dict)

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

# modified from paddle.callbacks.EarlyStopping
class EarlyStopping():
    def __init__(self, monitor='loss', mode='auto', patience=5, min_delta=0, baseline=None):
        super(EarlyStopping, self).__init__()
        self.monitor = monitor  # (loss, acc)
        self.wait_epoch = 0
        self.patience = patience
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.is_stop_traing = False
        self.stopped_epoch = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        # When mode == 'auto', the mode should be inferred by `self.monitor`
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        if self.baseline is not None:
            self.best_value = self.baseline
        else:
            self.best_value = np.inf if self.monitor_op == np.less else -np.inf

    def __call__(self, logs):
        if logs is None or self.monitor not in logs:
            warnings.warn('Monitor of EarlyStopping should be loss or metric name.')
            return
        current = logs[self.monitor]
        if isinstance(current, (list, tuple)):
            current = current[0]
        elif isinstance(current, numbers.Number):
            current = current
        else:
            return

        if self.monitor_op(current - self.min_delta, self.best_value):
            self.best_value = current
            self.wait_epoch = 0
        else:
            self.wait_epoch += 1
            print(f"INFO: Early stopping counter {self.wait_epoch} of {self.patience}")
        if self.wait_epoch >= self.patience:
            self.is_stop_traing = True
            print('INFO: Early stopping.')
