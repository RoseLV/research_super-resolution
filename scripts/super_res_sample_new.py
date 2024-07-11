"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from improved_diffusion.ppt_sr import PPTSRDataset
from improved_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    sr_create_model_and_diffusion,
    sr_model_and_diffusion_defaults,
)


def main():
    args = create_argparser().parse_args()

    print("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to("cuda")
    model.eval()

    print("creating data loader...")
    val_ds = PPTSRDataset(args.data_dir, 2021, 2021)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=16, shuffle=False
    )

    print("creating samples...")
    all_samples = []
    all_hrs = []
    all_lrs = []
    for i in tqdm(range(args.num_samples // args.batch_size)):
        batch = next(iter(val_loader))
        model_kwargs = {"low_res": batch["lr"].to("cuda")}

        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 1, args.large_size, args.large_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        all_samples.append(sample.cpu().numpy())
        all_hrs.append(batch["hr"].cpu().numpy())
        all_lrs.append(batch["lr"].cpu().numpy())

    hr = np.concatenate(all_hrs, axis=0)
    lr = np.concatenate(all_lrs, axis=0)
    sample = np.concatenate(all_samples, axis=0)
    np.savez("temp2021.npz", hr=hr, lr=lr, sample=sample)


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=16,
        batch_size=16,
        use_ddim=False,
        data_dir="",
        model_path="",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
