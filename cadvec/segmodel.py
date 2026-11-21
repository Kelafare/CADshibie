"""小型 U-Net 语义分割模型（dense segmentation 路线）。

输入白底 RGB（1000×1000），输出 37 类像素概率（0–35 语义 + 36 背景）。
结构为 4 级编码器 + 双线性上采样解码器，约 1.9M 参数，可在 6GB 显存上以
512×512 裁剪、batch 8 训练。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 37
BACKGROUND_INDEX = 36


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = NUM_CLASSES,
                 base: int = 32, depth: int = 4):
        super().__init__()
        channels = [base * (2 ** level) for level in range(depth)]
        self.encoders = nn.ModuleList()
        previous = in_channels
        for channel in channels:
            self.encoders.append(ConvBlock(previous, channel))
            previous = channel
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(channels[-1], channels[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = channels[-1] * 2
        for channel in reversed(channels):
            self.upconvs.append(nn.Conv2d(previous, channel, 3, padding=1))
            self.decoders.append(ConvBlock(channel * 2, channel))
            previous = channel
        self.head = nn.Conv2d(channels[0], classes, 1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = upconv(x)
            x = decoder(torch.cat([x, skip], dim=1))
        return self.head(x)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
