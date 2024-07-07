import lightning as pl
import torch

from vae import autoenc


def setup_autoenc_training(encoder, decoder, model_dir):
    autoencoder = autoenc.AutoencoderKL(encoder, decoder)

    num_gpus = torch.cuda.device_count()
    accelerator = "gpu" if (num_gpus > 0) else "cpu"
    devices = torch.cuda.device_count() if (accelerator == "gpu") else 1

    checkpoint = pl.pytorch.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename="{epoch}-{val_rec_loss:.4f}",
        monitor="val_rec_loss",
        every_n_epochs=5,
        save_top_k=3,
    )
    callbacks = [checkpoint]

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1000,
        log_every_n_steps=20,
        strategy="dp" if num_gpus > 1 else "auto",
        callbacks=callbacks,
    )

    return (autoencoder, trainer)
