import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils import ABS, SPPLayer, HPF, TLU
from icecream import ic
import numpy as np
__all__ = ['ZhuNet']

# class ZhuNet(nn.Module):
#     def __init__(self):
#         super(ZhuNet, self).__init__()
#         self.srm = SRM()
#         self.spp = SPP(3)
#         self.sepconv_block1 = nn.Sequential(
#             nn.Conv2d(30, 60, 3, padding=1, groups=30),
#             ABS(),
#             nn.Conv2d(60, 30, 1),
#             nn.BatchNorm2d(30),
#             nn.ReLU()
#         )
#         self.sepconv_block2 = nn.Sequential(
#             nn.Conv2d(30, 60, 3, padding=1, groups=30),
#             nn.Conv2d(60, 30, 1),
#             nn.BatchNorm2d(30),
#         )
#         self.basic_block1 = nn.Sequential(
#             nn.Conv2d(30, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
#         )
#         self.basic_block2 = nn.Sequential(
#             nn.Conv2d(32, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
#         )
#         self.basic_block3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
#         )
#         self.basic_block4 = nn.Sequential(
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#         )
#         self.classfier = nn.Sequential(
#             nn.Linear(2688, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 2),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         x_0 = self.srm(x)
#         x = self.sepconv_block1(x_0)
#         x = self.sepconv_block2(x)
#         x = x + x_0
#         x = self.basic_block1(x)
#         x = self.basic_block2(x)
#         x = self.basic_block3(x)
#         x = self.basic_block4(x)
#         ic(x.shape)
#         x = self.spp(x)
#         out = self.classfier(x)
#         return out


class ZhuNet(nn.Module):
    def __init__(self):
        super(ZhuNet, self).__init__()
        self.layer1 = HPF()
        self.layer2 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1, groups=30),
            ABS(),
            nn.Conv2d(60, 30, kernel_size=1),
            nn.BatchNorm2d(30),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(30, 60, kernel_size=3, stride=1, padding=1, groups=30),
            nn.Conv2d(60, 30, kernel_size=1),
            nn.BatchNorm2d(30)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(30, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer8 = SPPLayer(3)
        self.fc = nn.Sequential(
            nn.Linear(128*21, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out1 = self.layer1(x)
        
        # sepconv
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out3 = out3 + out1

        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        out9 = self.fc(out8)

        return out9