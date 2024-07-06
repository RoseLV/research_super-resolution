import numpy as np
import tifffile as tif
import torch
import tyro
from einops import rearrange

from vae import encoder, training


def setup_model(model_dir=None):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    (autoencoder, trainer) = training.setup_autoenc_training(
        encoder=enc, decoder=dec, model_dir=model_dir
    )
    return (autoencoder, trainer)


def train(
    batch_size: int = 64,
    model_dir: str = "./model",
):
    print("Loading data...")
    arr = tif.imread("/home/linhan/data/PPT_hr_yearly/PPT_2013.tiff")
    arr = rearrange(arr, "n h w -> n 1 h w") / 255.0
    arr = arr.astype(np.float32)
    trainset = torch.utils.data.TensorDataset(torch.tensor(arr).clip(min=0.0, max=1.0))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")
    trainer.fit(model, train_dataloaders=trainloader)


if __name__ == "__main__":
    tyro.cli(train)
