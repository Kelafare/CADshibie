"""图像特征提取：ResNet18（ImageNet 预训练）layer3 特征在笔画中点采样。

参考 CADTransformer 的混合表示：图元 token = 几何特征 + 栅格图像 CNN 特征。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 256

_CACHE: dict[str, object] = {}


def _backbone():
    if "model" in _CACHE:
        return _CACHE["model"]
    from torchvision.models import ResNet18_Weights, resnet18

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    trunk = nn.Sequential(*list(model.children())[:7])  # conv1..layer3
    trunk.eval()
    _CACHE["model"] = trunk
    return trunk


def load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    composite = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


@torch.no_grad()
def node_image_features(image: np.ndarray, points: np.ndarray,
                        device: str = "cuda") -> np.ndarray:
    """``points`` 为画布坐标 N×2（0-100），返回 N×256 特征。"""
    if len(points) == 0:
        return np.zeros((0, FEATURE_DIM), dtype=np.float16)
    height, width = image.shape[:2]
    trunk = _backbone().to(device)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    tensor = ((tensor.to(device) - mean) / std)
    feature_map = trunk(tensor)  # 1×256×h×w
    feature_height, feature_width = feature_map.shape[-2:]
    x = points[:, 0] / 100.0 * width
    y = points[:, 1] / 100.0 * height
    grid_x = (x / max(width - 1, 1)) * 2.0 - 1.0
    grid_y = (y / max(height - 1, 1)) * 2.0 - 1.0
    grid = torch.from_numpy(np.stack([grid_x, grid_y], axis=1))[None, None].float().to(device)
    sampled = F.grid_sample(feature_map, grid, align_corners=True)  # 1×256×1×N
    sampled = sampled[0, :, 0, :].transpose(0, 1).cpu().numpy()
    return sampled.astype(np.float16)
