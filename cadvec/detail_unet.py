"""原分辨率 ROI 细节头：RGB + 512 粗概率 → native 墨迹。

用于路线 A 的可行性验证：实例身份/类别仍由现有 512 ROI 模型提供，细节头
只负责把粗概率修正到原分辨率几何。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .roi_ink import _conv_block


class DetailUNet(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.enc1 = _conv_block(4, base)
        self.enc2 = _conv_block(base, base * 2)
        self.enc3 = _conv_block(base * 2, base * 4)
        self.enc4 = _conv_block(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(inputs)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)
