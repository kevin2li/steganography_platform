import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import ABS, TLU, HPF, SPPLayer

__all__ = ['XuNet']

#TODO not finished
class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()
        self.hpf = HPF()
        self.group1 = nn.Sequential(
            nn.Conv2d(30, 30, 5, 1, 2),
            ABS(),
            nn.BatchNorm2d(30),
            nn.Tanh(),
            nn.AvgPool2d(5, stride=2)
        )
        self.group2 = nn.Sequential(
            nn.Conv2d(8, 16, 5, 1, 2),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(5, stride=2)
        )
        self.group3 = nn.Sequential(
            nn.Conv2d(16, 32, 1, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(5, stride=2)
        )
        self.group4 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(5, stride=2)
        )
        self.group5 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classfier = nn.Sequential(
            nn.Linear(128, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)
        x = self.group5(x)
        out = self.classfier(x)
        return out