import pytorch_lightning as pl
import torch
import torch.optim as optim
import torchmetrics
from src.utils import initialization, partition_parameters
from icecream import ic
from . import yednet, zhunet, xunet
from .yednet import *
from .zhunet import *
from .xunet import *

__all__ = ['ZhuNet', 'YedNet', 'XuNet']

# class ZhuNet(ZhuNet, pl.LightningModule):
#     def __init__(self, loss_fn):
#         super(ZhuNet, self).__init__()
#         self.loss_fn = loss_fn
#         self.acc_metric = torchmetrics.Accuracy()
#         initialization(self)

#     def training_step(self, batch, batch_idx):
#         data, target = batch
#         N, C, H, W = data.shape
#         data = data.reshape(N*C, 1, H, W)
#         target = target.reshape(-1)
#         logits = self.forward(data)
#         loss = self.loss_fn(logits, target)
#         acc = self.acc_metric(logits, target)

#         return {'loss': loss, 'acc': acc}

#     def validation_step(self, batch, batch_idx):
#         data, target = batch
#         N, C, H, W = data.shape
#         data = data.reshape(N*C, 1, H, W)
#         target = target.reshape(-1)
#         logits = self.forward(data)
#         loss = self.loss_fn(logits, target)
#         return {'loss': loss}

#     def configure_optimizers(self):
#         params_wd, params_rest = [], []
#         for m in self.parameters():
#             if m.requires_grad:
#                 (params_wd if m.dim()!=1 else params_rest).append(m)
#         param_groups = [{'params': params_wd, 'weight_decay': 5e-4},
#                         {'params': params_rest}]
#         return optim.Adam(param_groups, lr=1e-3)
