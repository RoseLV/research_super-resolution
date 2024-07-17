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
        topo_file: str = "",
    ):
        imglist = []
        for yr in range(start_yr, end_yr + 1):
            imglist.append(tif.imread(Path(root_dir) / f"PPT_{yr}.tiff"))
        hrs = np.concatenate(imglist, axis=0)
        hrs = rearrange(hrs, "n h w -> n 1 h w")
        assert hrs.shape[-1] == large_size
        hrs = torch.tensor(hrs)

        self.small_size = small_size

        lrs = F.interpolate(hrs, size=(small_size, small_size), mode="bilinear")
        lrs = F.interpolate(lrs, size=(large_size, large_size), mode="nearest")

        if len(topo_file) > 0:
            topo = tif.imread(topo_file).astype(np.float32)
            topo = (topo - np.min(topo)) / (1e-6 + np.max(topo) - np.min(topo))
            topo = rearrange(topo, "h w -> 1 h w")
            self.topo = torch.tensor(topo)
        else:
            self.topo = None

        if norm == "gamma":
            self.hrs = (hrs / 255.0).clip(min=0.0, max=1.0) ** 0.15
            self.lrs = (lrs / 255.0).clip(min=0.0, max=1.0) ** 0.15
        else:
            raise Exception(f"Unsupported norm {norm}")

    def __len__(self):
        return self.hrs.size()[0]

    def __getitem__(self, index):
        hr = self.hrs[index, ...]
        lr = self.lrs[index, ...]

        if self.topo is not None:
            lr = torch.cat([lr, self.topo], dim=0)

        return {"hr": hr, "lr": lr}
