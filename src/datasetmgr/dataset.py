import glob
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

__all__ = ['ImageDataset']

class ImageDataset(Dataset):
    def __init__(self, dataset_dir, transforms, mode='train', train_test_ratio=0.8):
        super(ImageDataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.transforms = transforms
        self.train_test_ratio = train_test_ratio
        self.stego_path_list = []
        for _ in dataset_dir[:-1]:
            self.stego_path_list.extend(glob.glob(os.path.join(_,'*.png')))
        self.cover_path_list = glob.glob(os.path.join(dataset_dir[-1],'*.png'))
        np.random.seed(2021)
        np.random.shuffle(self.cover_path_list)
        np.random.seed(2021)
        np.random.shuffle(self.stego_path_list)
        n = int(train_test_ratio * len(self.cover_path_list))
        if self.mode == 'train':
            self.cover_path_list = self.cover_path_list[:n]
            self.stego_path_list = self.stego_path_list[:n]
        else:
            self.cover_path_list = self.cover_path_list[n:]
            self.stego_path_list = self.stego_path_list[n:]

    def __getitem__(self, idx):
        cover_img = Image.open(self.cover_path_list[idx])
        stego_img = Image.open(self.stego_path_list[idx])

        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)

        data = torch.cat([cover_img, stego_img])
        label = torch.tensor([0, 1], dtype=torch.int64)

        return data, label

    def __len__(self):
        return len(self.cover_path_list)
