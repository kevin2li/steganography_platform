import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from .dataset import ImageDataset

__all__ = ['ImageDataset', 'getDataLoader']

class AugData():
    def __call__(self, data):
        # Rotation
        rot = np.random.randint(0, 3)
        data = np.rot90(data, rot, axes=[1, 2]).copy()

        # Mirroring
        if np.random.random() < 0.5:
            data = np.flip(data, axis=2).copy()

        return data

class ToTensor():
    def __call__(self, data):
        data = data.astype(np.float32)
        # data = np.expand_dims(data, 1)
        # data = data / 255.0
        return torch.from_numpy(data)

train_transforms = T.Compose([
    AugData(),
    ToTensor(),
])

eval_transforms = T.Compose([
    ToTensor()
])

def getDataLoader(args):
    train_dataset = ImageDataset(args['dataset_path'], mode='train', transforms=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, pin_memory=True, num_workers=4)

    test_dataset = ImageDataset(args['dataset_path'], mode='test', transforms=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, test_loader

