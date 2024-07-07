import torch.nn as nn

from vae.resnet import ResBlock


class SimpleConvEncoder(nn.Sequential):
    def __init__(self):
        sequence = []

        channels = [1, 128, 256, 512, 512]
        down_layers = [1, 2, 3, 4]
        for i in range(len(channels) - 1):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])
            res_kernel_size = (3, 3)
            res_block = ResBlock(
                in_channels,
                out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 32},
            )
            sequence.append(res_block)
            if i in down_layers:
                downsample = nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=2, padding=1
                )
                sequence.append(downsample)
            in_channels = out_channels
        olayer = nn.Conv2d(channels[-1], 4, kernel_size=3, stride=1, padding=1)
        sequence.append(olayer)
        super().__init__(*sequence)


class SimpleConvDecoder(nn.Sequential):
    def __init__(self):
        sequence = []

        channels = [4, 512, 512, 256, 128]
        up_layers = [1, 2, 3, 4]
        for i in range(len(channels) - 1):
            in_channels = int(channels[i])
            out_channels = int(channels[i + 1])
            if i in up_layers:
                upsample = nn.ConvTranspose2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                )
                sequence.append(upsample)
            res_kernel_size = (3, 3)
            res_block = ResBlock(
                in_channels,
                out_channels,
                kernel_size=res_kernel_size,
                norm_kwargs={"num_groups": 32},
            )
            sequence.append(res_block)
            in_channels = out_channels

        olayer = nn.Conv2d(channels[-1], 1, kernel_size=3, stride=1, padding=1)
        sequence.append(olayer)
        super().__init__(*sequence)


if __name__ == "__main__":
    import torch

    encoder = SimpleConvEncoder()
    print(encoder)
    input = torch.rand(4, 1, 128, 128)
    output = encoder(input)
    print(output.size())

    decoder = SimpleConvDecoder()
    print(decoder)
    print(decoder(output).size())
