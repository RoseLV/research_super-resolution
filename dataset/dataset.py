from pathlib import Path

import numpy as np
import tifffile as tif
import torch
from einops import rearrange
from torch.utils.data import Dataset


class PPTDataset(Dataset):
    def __init__(self, root_dir: str, start_yr: int, end_yr: int, transform=None):
        imglist = []
        for yr in range(start_yr, end_yr + 1):
            imglist.append(tif.imread(Path(root_dir) / f"PPT_{yr}.tiff"))
        imgs = np.concatenate(imglist, axis=0)
        imgs = rearrange(imgs, "n h w -> n 1 h w")
        self.imgs = (torch.tensor(imgs) / 255.0).clip(min=0.0, max=1.0)
        self.imgs = self.imgs[:, :, 50:50+128, 150:150+128]
        self.transform = transform

    def __len__(self):
        return self.imgs.size()[0]

    def __getitem__(self, index):
        img = self.imgs[index, ...]
        if self.transform:
            img = self.transform(img)
        return img
