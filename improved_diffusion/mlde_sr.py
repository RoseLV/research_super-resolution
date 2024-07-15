import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from einops import rearrange
from torch.utils.data import Dataset


class MLDEDataset(Dataset):
    def __init__(self, nc_file, norm: str):
        ds = xr.open_dataset(nc_file, engine="netcdf4")
        hrs = np.array(ds["target_pr"][0, ...], np.float32)
        hrs = hrs / np.float32(np.max(hrs))
        self.hrs = rearrange(hrs, "n h w -> n 1 h w")
        if norm == "gamma":
            self.hrs = self.hrs**0.15
        else:
            raise Exception(f"Unsupported norm {norm}")

        conds = []
        for v in list(ds.data_vars):
            if v == "target_pr":
                continue
            if len(ds[v].shape) != 4:
                continue
            arr = np.array(ds[v][0, ...], dtype=np.float32)
            arr = rearrange(arr, "n h w -> n 1 h w")
            conds.append(arr)

        conds = np.concatenate(conds, axis=1)
        tmp = rearrange(conds, "n c h w -> c (n h w)")
        maxes = np.max(tmp, axis=1)
        self.conds = conds / rearrange(maxes, "c -> 1 c 1 1")

    def __len__(self):
        return self.hrs.shape[0]

    def __getitem__(self, index):
        hr = self.hrs[index, ...]
        cond = self.conds[index, ...]
        return {"hr": hr, "lr": cond}


class MLDESingleDataset(Dataset):
    def __init__(self, nc_file, norm: str):
        ds = xr.open_dataset(nc_file, engine="netcdf4")
        hrs = np.array(ds["target_pr"][0, ...], np.float32)
        hrs = hrs / np.float32(np.max(hrs))
        self.hrs = rearrange(hrs, "n h w -> n 1 h w")
        self.hrs = torch.tensor(self.hrs)
        if norm == "gamma":
            self.hrs = self.hrs**0.15
        else:
            raise Exception(f"Unsupported norm {norm}")

        lrs = F.interpolate(self.hrs, size=(10, 10), mode="bilinear")
        self.lrs = F.interpolate(lrs, size=(64, 64), mode="nearest")

    def __len__(self):
        return self.hrs.shape[0]

    def __getitem__(self, index):
        hr = self.hrs[index, ...]
        lr = self.lrs[index, ...]
        return {"hr": hr, "lr": lr}
