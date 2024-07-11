from pathlib import Path

import numpy as np
import tifffile as tif
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf
from einops import rearrange
from torch.utils.data import Dataset


class PPTSRDataset(Dataset):
    def __init__(self, root_dir: str, start_yr: int, end_yr: int):
        imglist = []
        for yr in range(start_yr, end_yr + 1):
            imglist.append(tif.imread(Path(root_dir) / f"PPT_{yr}.tiff"))
        imgs = np.concatenate(imglist, axis=0)
        imgs = rearrange(imgs, "n h w -> n 1 h w")
        imgs = tf.crop(torch.tensor(imgs), 80, 260, 64, 64)
        self.imgs = (imgs / 255.0).clip(min=0.0, max=1.0) ** 0.15

    def __len__(self):
        return self.imgs.size()[0]

    def __getitem__(self, index):
        img = self.imgs[index, ...]
        # TODO rescale to [-1, 1]
        return {
            "hr": img,
            "lr": F.interpolate(img.unsqueeze(0), scale_factor=(0.5, 0.5), mode='bilinear').squeeze(0),
        }
