import numpy as np
import torch.nn as nn

from vae.resnet import ResBlock


class SimpleConvEncoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=3):
        sequence = []
        channels = [in_dim] + (2 ** np.arange(1, levels + 1)).clip(max=4).tolist()

        for i in range(levels):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])
            res_kernel_size = (3, 3)
            res_block = ResBlock(
                in_channels,
                out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1},
            )
            sequence.append(res_block)
            downsample = nn.Conv2d(
                out_channels, out_channels, kernel_size=(2, 2), stride=(2, 2)
            )
            sequence.append(downsample)
            in_channels = out_channels

        super().__init__(*sequence)


class SimpleConvDecoder(nn.Sequential):
    def __init__(self, in_dim=1, levels=3):
        sequence = []
        channels = [in_dim] + (2 ** np.arange(1, levels + 1)).clip(max=4).tolist()

        for i in reversed(list(range(levels))):
            in_channels = int(channels[i + 1])
            out_channels = int(channels[i])
            upsample = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=(2, 2), stride=(2, 2)
            )
            sequence.append(upsample)
            res_kernel_size = (3, 3)
            res_block = ResBlock(
                in_channels,
                out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 1},
            )
            sequence.append(res_block)
            in_channels = out_channels

        super().__init__(*sequence)


if __name__ == "__main__":
    import torch

    encoder = SimpleConvEncoder()
    print(encoder)
    input = torch.rand(16, 1, 128, 128)
    output = encoder(input)
    print(output.size())

    decoder = SimpleConvDecoder()
    print(decoder)
    print(decoder(output).size())
