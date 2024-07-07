from torch import nn


def normalization(channels, norm_type: str = "group", num_groups: int = 32):
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups=num_groups if num_groups < channels else 1, num_channels=channels)
    elif (not norm_type) or (norm_type.lower() == "none"):
        return nn.Identity()
    else:
        raise NotImplementedError(norm_type)


def activation(act_type: str = "swish"):
    if act_type == "swish":
        return nn.SiLU()
    elif act_type == "gelu":
        return nn.GELU()
    elif act_type == "relu":
        return nn.ReLU()
    elif act_type == "tanh":
        return nn.Tanh()
    elif not act_type:
        return nn.Identity()
    else:
        raise NotImplementedError(act_type)
