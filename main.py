# %%
import torch.nn as nn

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import torchvision.transforms as T
from icecream import ic
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from visualdl import LogWriter

from src.config import args
from src.datasetmgr import getDataLoader
from src.models import YedNet, ZhuNet
from src.utils import initialization, partition_parameters, transfer_parameters
from trainer import Trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# transforms = T.Compose([
#     T.ToTensor()
# ])

# train_dataset = CIFAR10(root='data', train=True, download=False, transform=transforms)
# test_dataset = CIFAR10(root='data', train=False, download=False, transform=transforms)
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
# %%
model = ZhuNet().to(device)
initialization(model)
param_groups = partition_parameters(model, args['weight_decay'])

# 加载预训练模型(cifar10)
param_path = 'experiment/1/best/checkpoint.ptparams'
checkpoint = torch.load(param_path)
transfer_parameters(model, checkpoint)

optimizer = optim.SGD(lr=args['lr'], params=param_groups)
loss_fn = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
acc_metric = torchmetrics.Accuracy().to(device)
log_writer = LogWriter(logdir=args['save_dir'])
train_loader, test_loader = getDataLoader(args)

# %%
trainer = Trainer(model, loss_fn, optimizer, log_writer, acc_metric, lr_scheduler, **args)
# trainer.print_config(input_size=(None, 1, 256, 256)) # (40, 1024, 1, 30) for cnn, (32, 30, 1024) for rnn
trainer.print_config(input_size=(None, 3, 32, 32)) # (40, 1024, 1, 30) for cnn, (32, 30, 1024) for rnn
trainer.fit(train_loader, test_loader)

# %%
