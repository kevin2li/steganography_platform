'''
Author: your name
Date: 2021-04-23 20:33:20
LastEditTime: 2021-04-26 13:42:23
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /steganography_platform_pl/main.py
'''
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
model = ZhuNet().to(device)
initialization(model)
param_groups = partition_parameters(model, args['weight_decay'])

# 加载预训练模型
path = 'trained_model/zhunet_wow.ptparams'
params = torch.load(path)
model.load_state_dict(params)


optimizer = optim.SGD(lr=args['lr'], params=param_groups)
loss_fn = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
acc_metric = torchmetrics.Accuracy().to(device)
log_writer = LogWriter(logdir=args['save_dir'])
train_loader, test_loader = getDataLoader(args)

# %%
trainer = Trainer(model, loss_fn, optimizer, log_writer, acc_metric, lr_scheduler, **args)
trainer.print_config(input_size=(None, 1, 256, 256)) # (40, 1024, 1, 30) for cnn, (32, 30, 1024) for rnn
# trainer.fit(train_loader, test_loader)

# %%
trainer.evaluate(test_loader)
# %%
