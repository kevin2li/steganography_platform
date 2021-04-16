import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from .dataset import ImageDataset

__all__ = ['ImageDataset', 'getDataLoader']

train_transforms = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

eval_transforms = T.Compose([
    T.ToTensor()
])

def getDataLoader(args):
    train_dataset = ImageDataset(args['dataset_path'], mode='train', transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True, num_workers=2)

    test_dataset = ImageDataset(args['dataset_path'], mode='test', transforms=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True, num_workers=2)
    return train_loader, test_loader

