"""
Train a super-resolution model.
"""

import argparse
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch import loggers as pl_loggers
from torch.utils.data import DataLoader

from improved_diffusion.ddpm import DDPM
from improved_diffusion.ppt_sr import PPTSRDataset
from improved_diffusion.resample import create_named_schedule_sampler
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
    model.to("cuda")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs", version=args.version)
    model_dir = Path(tb_logger.log_dir) / "checkpoints"

    tb_logger.log_hyperparams(args)

    ddpm = DDPM(
        model,
        diffusion,
        args.lr,
        args.ema_rate,
        schedule_sampler,
        args.weight_decay,
        str(model_dir),
    )

    print("creating data loader...")
    train_ds = PPTSRDataset(
        args.data_dir, 1986, 2018, args.large_size, args.small_size, args.norm
    )
    val_ds = PPTSRDataset(
        args.data_dir, 2019, 2020, args.large_size, args.small_size, args.norm
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=16, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=16, shuffle=False
    )

    checkpoint = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        every_n_epochs=args.save_interval,
        save_top_k=3,
    )
    callbacks = [checkpoint]

    print("creating trainer...")
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=1000,
        log_every_n_steps=args.log_interval,
        strategy="auto",
        callbacks=callbacks,
        logger=tb_logger,
    )

    print("begin training")
    trainer.fit(ddpm, train_dataloaders=train_loader, val_dataloaders=val_loader)


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        ema_rate="0.9999",
        log_interval=1,
        save_interval=10,
        version="iDDPM",
        norm="gamma",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
