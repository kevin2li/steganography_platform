# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from icecream import ic
from visualdl import LogWriter

from src.config import args
from src.datasetmgr import getDataLoader
from src.models import YedNet, ZhuNet, XuNet
from src.utils import initialization, partition_parameters, transfer_parameters
from trainer import Trainer
import pytorch_lightning as pl
# %%
class ZhuNet(ZhuNet, pl.LightningModule):
    def __init__(self, loss_fn, **kwargs):
        super(ZhuNet, self).__init__()
        self.loss_fn = loss_fn
        self.args = kwargs
        self.acc_metric = torchmetrics.Accuracy()
        initialization(self)

    def training_step(self, batch, batch_idx):
        data, target = batch
        N, C, H, W = data.shape
        data = data.reshape(N*C, 1, H, W)
        target = target.reshape(-1)
        logits = self.forward(data)
        batch_loss = self.loss_fn(logits, target)
        batch_acc = self.acc_metric(logits, target)
        avg_acc = self.acc_metric.compute()
        return {'loss': batch_loss, 'batch_acc': batch_acc, 'avg_acc': avg_acc}

    def validation_step(self, batch, batch_idx):
        data, target = batch
        N, C, H, W = data.shape
        data = data.reshape(N*C, 1, H, W)
        target = target.reshape(-1)
        logits = self.forward(data)
        loss = self.loss_fn(logits, target)
        return {'loss': loss}

    def configure_optimizers(self):
        params_wd, params_rest = [], []
        for m in self.parameters():
            if m.requires_grad:
                (params_wd if m.dim()!=1 else params_rest).append(m)
        param_groups = [{'params': params_wd, 'weight_decay': self.args['weight_decay']},
                        {'params': params_rest}]
        return optim.SGD(param_groups, lr=1e-3)

# %%
loss_fn = nn.CrossEntropyLoss()
model = ZhuNet(loss_fn=loss_fn, **args)

trainer = pl.Trainer(gpus=1, max_epochs=3)
train_loader, test_loader = getDataLoader(args)
trainer.fit(model, train_loader, test_loader)