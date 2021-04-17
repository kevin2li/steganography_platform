import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import ABS, TLU, SRM, SPP

__all__ = ['ZhuNet']

class ZhuNet(nn.Module):
    def __init__(self):
        super(ZhuNet, self).__init__()
        self.srm = SRM()
        self.spp = SPP(3)
        self.sepconv_block1 = nn.Sequential(
            nn.Conv2d(30, 60, 3, padding=1, groups=30),
            ABS(),
            nn.Conv2d(60, 30, 1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.sepconv_block2 = nn.Sequential(
            nn.Conv2d(30, 60, 3, padding=1, groups=30),
            ABS(),
            nn.Conv2d(60, 30, 1),
            nn.BatchNorm2d(30),
        )
        self.basic_block1 = nn.Sequential(
            nn.Conv2d(30, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((128, 128))
        )
        self.basic_block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64))
        )
        self.basic_block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.basic_block4 = nn.Sequential(
            nn.Conv2d(64, 128, 3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classfier = nn.Sequential(
            nn.Linear(2688, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
        )

    def forward(self, x):
        x_0 = self.srm(x)
        x = self.sepconv_block1(x_0)
        x = self.sepconv_block2(x)
        x = x + x_0
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        x = self.basic_block3(x)
        x = self.basic_block4(x)
        x = self.spp(x)
        x = x.squeeze()
        out = self.classfier(x)
        return out
