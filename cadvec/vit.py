"""ViT 对象分类器：以对象假设为中心的局部图像裁剪 → 语义类别。

模型是一个小型纯 ViT（无卷积主干），输入 4 通道：

1-3: 白底合成后的 RGB 裁剪；
4:   对象假设自身的描边掩码（告诉模型「要判断的是哪几条线」）。

这样同一个符号（门弧、窗线、家具轮廓）即使与背景线条颜色相同，模型也能
凭形状、上下文与线宽区分。推理时批量送 GPU。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import STUFF_CLASS_MAX

CLASS_COUNT = 36
PATCH_SIZE = 16
from .crops import CROP_SIZE

#: 类别 → 是否 Stuff。
STUFF_CLASSES = set(range(30, STUFF_CLASS_MAX + 1))


class StrokeViT(nn.Module):
    """小型 ViT：14×14 patch，宽 192，6 层，6 头（约 2.4M 参数）。"""

    def __init__(self, in_channels: int = 4, dim: int = 192, depth: int = 6,
                 heads: int = 6, classes: int = CLASS_COUNT, image: int = CROP_SIZE,
                 patch: int = PATCH_SIZE, objectness: bool = False):
        super().__init__()
        tokens = (image // patch) ** 2
        self.patch = nn.Conv2d(in_channels, dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, tokens + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.05, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, classes)
        self.objectness = objectness
        self.iou_head = nn.Linear(dim, 1) if objectness else None

    def forward(self, images: torch.Tensor):
        tokens = self.patch(images)  # B, C, 14, 14
        tokens = tokens.flatten(2).transpose(1, 2)
        cls = self.cls.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos
        tokens = self.encoder(tokens)
        features = self.norm(tokens[:, 0])
        logits = self.head(features)
        if self.iou_head is None:
            return logits
        return logits, self.iou_head(features).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


class VitClassifier:
    """GPU/CPU 批量推理封装。"""

    def __init__(self, model: StrokeViT, device: torch.device, size: int = CROP_SIZE):
        self.model = model.to(device).eval()
        self.device = device
        self.size = size

    @classmethod
    def load(cls, weights_path: str | Path, device: str | None = None) -> "VitClassifier | None":
        path = Path(weights_path)
        if not path.exists():
            return None
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location="cpu", weights_only=False)
        model = StrokeViT(**state.get("model_kwargs", {}))
        model.load_state_dict(state["state_dict"], strict=False)
        return cls(model, torch.device(device), state.get("crop_size", CROP_SIZE))

    @property
    def has_objectness(self) -> bool:
        return self.model.iou_head is not None

    @torch.no_grad()
    def classify(
        self, crops: list[np.ndarray], batch_size: int = 96,
    ) -> np.ndarray:
        return self.classify_with_iou(crops, batch_size)[0]

    @torch.no_grad()
    def classify_with_iou(
        self, crops: list[np.ndarray], batch_size: int = 96,
    ) -> tuple[np.ndarray, np.ndarray]:
        """返回 (类别概率 B×36, 预测 IoU B)；无 objectness 头时 IoU 全为 1。"""
        if not crops:
            return (np.zeros((0, CLASS_COUNT), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))
        probabilities = []
        ious = []
        for start in range(0, len(crops), batch_size):
            block = torch.from_numpy(
                np.stack(crops[start:start + batch_size])
            ).to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda"):
                output = self.model(block)
            if isinstance(output, tuple):
                logits, iou_logit = output
                ious.append(torch.sigmoid(iou_logit.float()).cpu().numpy())
            else:
                logits = output
                ious.append(np.ones(block.shape[0], dtype=np.float32))
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0), np.concatenate(ious, axis=0)


def class_weights(labels: np.ndarray, power: float = 0.5) -> np.ndarray:
    counts = np.bincount(labels, minlength=CLASS_COUNT).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / counts) ** power
    weights /= weights.mean()
    return weights.astype(np.float32)
