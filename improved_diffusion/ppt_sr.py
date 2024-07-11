from pathlib import Path

import numpy as np
import tifffile as tif
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset


class PPTSRDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        start_yr: int,
        end_yr: int,
        large_size: int,
        small_size: int,
        norm: str,
    ):
        imglist = []
        for yr in range(start_yr, end_yr + 1):
            imglist.append(tif.imread(Path(root_dir) / f"PPT_{yr}.tiff"))
        imgs = np.concatenate(imglist, axis=0)
        imgs = rearrange(imgs, "n h w -> n 1 h w")
        assert imgs.shape[-1] == large_size
        imgs = torch.tensor(imgs)

        self.small_size = small_size
        if norm == "gamma":
            self.imgs = (imgs / 255.0).clip(min=0.0, max=1.0) ** 0.15
        else:
            raise Exception(f"Unsupported norm {norm}")

    def __len__(self):
        return self.imgs.size()[0]

    def __getitem__(self, index):
        img = self.imgs[index, ...]
        return {
            "hr": img,
            "lr": F.interpolate(
                img.unsqueeze(0),
                size=(self.small_size, self.small_size),
                mode="bilinear",
            ).squeeze(0),
        }
