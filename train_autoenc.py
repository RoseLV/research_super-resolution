import tyro
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset.dataset import PPTDataset
from vae import encoder, training


def setup_model(model_dir=None):
    enc = encoder.SimpleConvEncoder()
    dec = encoder.SimpleConvDecoder()
    (autoencoder, trainer) = training.setup_autoenc_training(
        encoder=enc, decoder=dec, model_dir=model_dir
    )
    return (autoencoder, trainer)


def train(
    data_dir: str,
    batch_size: int = 256,
    model_dir: str = "./model",
):
    print("Loading data...")
    train_data = PPTDataset(root_dir=data_dir, start_yr=1986, end_yr=2019)
    val_data = PPTDataset(root_dir=data_dir, start_yr=2020, end_yr=2022)
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=16
    )
    val_loader = DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=16
    )

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    tyro.cli(train)
