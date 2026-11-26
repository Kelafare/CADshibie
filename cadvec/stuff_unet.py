"""全图 Stuff 语义分割的小型 U-Net（5 类，30–34）。

与 Thing ROI 模型不同：输入是原图的 512 重叠分块，输出每类 Stuff 墨迹概率；
推理时拼接回整图，每类按 instance=-1 输出。用于验证全图结构支路是否可行。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .roi_ink import _conv_block


class StuffUNet(nn.Module):
    def __init__(self, classes: int = 5, base: int = 32):
        super().__init__()
        self.enc1 = _conv_block(3, base)
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
        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def multiclass_dice(logits: torch.Tensor, target: torch.Tensor,
                    eps: float = 1.0) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()
