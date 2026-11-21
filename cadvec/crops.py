"""对象裁剪渲染（不依赖 torch，训练与推理共用）。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import COMMAND_CIRCLE, COMMAND_ELLIPSE

CROP_SIZE = 224


def _affine(bbox: tuple[float, float, float, float], side: float, size: int):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    scale = size / max(side, 1e-6)
    matrix = np.array([
        [scale, 0.0, (side / 2.0 - cx) * scale],
        [0.0, scale, (side / 2.0 - cy) * scale],
    ], dtype=np.float32)
    return matrix


def crop_box(bbox: tuple[float, float, float, float], padding: float = 0.20) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    width = max(x1 - x0, 0.5)
    height = max(y1 - y0, 0.5)
    side = max(width, height) * (1.0 + 2.0 * padding)
    return side, side


def render_crop(
    image: np.ndarray,
    bbox: tuple[float, float, float, float],
    strokes: list[dict],
    scale_x: float,
    scale_y: float,
    *,
    size: int = CROP_SIZE,
    padding: float = 0.20,
    flip: int = 0,
) -> np.ndarray:
    """返回 4×size×size 的 float32 张量数据（CHW，0-1）。

    ``image`` 是白底合成后的 RGB（原始像素分辨率），``bbox`` 用画布坐标。
    ``strokes`` 是假设/真值图元的列表（``points`` 为画布坐标）。
    """
    side, _ = crop_box(bbox, padding)
    matrix = _affine(bbox, side, size)  # 画布坐标 → 输出 224 坐标
    # 原图是像素坐标：把画布坐标的仿射矩阵换成像素输入的等价形式。
    height, width = image.shape[:2]
    full = matrix.copy()
    full[0, 0] /= max(scale_x, 1e-6)
    full[1, 1] /= max(scale_y, 1e-6)
    if side * min(scale_x, scale_y) / size > 2.0:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
    crop = cv2.warpAffine(image, full, (size, size), flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    mask = np.zeros((size, size), dtype=np.uint8)
    for stroke in strokes:
        points = np.asarray(stroke["points"], dtype=np.float32)
        pixels = np.round(
            np.column_stack([
                points[:, 0] * matrix[0, 0] + matrix[0, 2],
                points[:, 1] * matrix[1, 1] + matrix[1, 2],
            ])
        ).astype(np.int32)
        closed = int(stroke["command"]) in (COMMAND_CIRCLE, COMMAND_ELLIPSE)
        cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], closed, 1, thickness=2)

    if flip == 1:
        crop = crop[:, ::-1]
        mask = mask[:, ::-1]
    elif flip == 2:
        crop = crop[::-1]
        mask = mask[::-1]

    tensor = np.concatenate([
        crop.astype(np.float32) / 255.0,
        mask.astype(np.float32)[..., None],
    ], axis=2)
    return np.ascontiguousarray(tensor.transpose(2, 0, 1))


def prepare_image(path: str | Path) -> tuple[np.ndarray, float, float]:
    """读 PNG → 白底 RGB 图 + 像素/画布单位缩放。"""
    from PIL import Image
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    composite = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
    composite = np.clip(composite, 0, 255).astype(np.uint8)
    height, width = composite.shape[:2]
    return composite, width / 100.0, height / 100.0


