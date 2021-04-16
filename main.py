# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from src.models import ZhuNet, YedNet
from src.datasetmgr import getDataLoader
from src.config import args

# %%
loss_fn = nn.CrossEntropyLoss()
model = ZhuNet(loss_fn=loss_fn)
train_loader, test_loader = getDataLoader(args)
# %%
trainer = pl.Trainer(
    max_epochs=5,
    gradient_clip_val=1.0,
    fast_dev_run=1
)
trainer.fit(model, train_loader, test_loader)
# %%
model
# %%

