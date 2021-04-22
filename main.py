# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from visualdl import LogWriter

from src.config import args
from src.datasetmgr import getDataLoader
from src.models import YedNet, ZhuNet
from src.utils import initialization
from trainer import Trainer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# %%
model = ZhuNet().to(device)
initialization(model)
optimizer = optim.SGD(lr=args['lr'], params=model.parameters())
loss_fn = nn.CrossEntropyLoss()
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma=args['gamma'])
acc_metric = torchmetrics.Accuracy()
log_writer = LogWriter(logdir=args['save_dir'])
train_loader, test_loader = getDataLoader(args)

# %%
trainer = Trainer(model, loss_fn, optimizer, log_writer, acc_metric, lr_scheduler, **args)
trainer.print_config(input_size=(None, 1, 256, 256)) # (40, 1024, 1, 30) for cnn, (32, 30, 1024) for rnn
trainer.fit(train_loader, test_loader)

# %%
