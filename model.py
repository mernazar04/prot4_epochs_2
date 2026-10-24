import torch
import torch.nn as nn
from diffusers import UNet2DModel


class DriftNet(nn.Module):
    def __init__(self, image_size: int = 28, base_channels: int = 32):
        super().__init__()
        self.unet = UNet2DModel(
            sample_size=image_size,
            in_channels=1,
            out_channels=1,
            layers_per_block=1,
            block_out_channels=(base_channels, base_channels * 2, base_channels * 2),
            down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        timesteps = (t * 1000.0).float()
        return self.unet(x_t, timesteps).sample


def make_forward_backward_nets(image_size: int = 28, base_channels: int = 32):
    forward_net = DriftNet(image_size, base_channels)
    backward_net = DriftNet(image_size, base_channels)
    return forward_net, backward_net