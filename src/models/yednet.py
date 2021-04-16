import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import ABS, TLU, SRM, SPP

__all__ = ['YedNet']

class YedNet(nn.ModuleDict):
    def __init__(self):
        super(YedNet, self).__init__()
        self.srm = SRM()
        self.group1 = nn.Sequential(
            nn.Conv2d(30, 30, 5, 1, 2),
            ABS(),
            nn.BatchNorm2d(30),
            TLU(3.0)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(30, 30, 5, 1, 2),
            nn.BatchNorm2d(30),
            TLU(3.0),
            nn.AdaptiveAvgPool2d((128, 128))
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(30, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((64, 64))
        )
        self.group4 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.group5 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classfier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.srm(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        x = x.squeeze()
        out = self.classfier(x)
        return out