import tyro
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

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
    # arr = tif.imread("/home/linhan/data/PPT_hr_yearly/PPT_2013.tiff")
    # arr = rearrange(arr, "n h w -> n 1 h w") / 255.0
    # arr = arr.astype(np.float32)
    # trainset = torch.utils.data.TensorDataset(torch.tensor(arr).clip(min=0.0, max=1.0) - 0.5)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_data = MNIST(root="./data", train=True, transform=transform, download=True)
    test_data = MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, num_workers=8)

    print("Setting up model...")
    (model, trainer) = setup_model(model_dir=model_dir)

    print("Starting training...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    tyro.cli(train)
