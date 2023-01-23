from typing import Tuple

import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    kernel_size: int = 3
    stride: int = 1
    padding: int = 1

    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNET(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Tuple[int] = [64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                ))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            dx = skip_connection.shape[2] - x.shape[2]
            dy = skip_connection.shape[3] - x.shape[3]

            left_pad = dx // 2
            right_pad = dx - left_pad

            top_pad = dy // 2
            bottom_pad = dy - top_pad

            x = nn.functional.pad(x,
                                  (left_pad, right_pad, top_pad, bottom_pad),
                                  mode='constant')

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print(x)
    else:
        print("MPS device not found.")
