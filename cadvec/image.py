"""PNG 读取与按描边颜色分层。

CAD 平面图的 PNG 通常由有限几种描边颜色组成，颜色本身对应图层/类别。
先把前景按颜色聚成若干层，后续每层独立做骨架和几何拟合。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import InferenceConfig


@dataclass
class ColorLayer:
    layer_id: int
    rgb: list[int]
    mask: np.ndarray
    pixel_count: int


def load_rgb(path: str | Path) -> tuple[np.ndarray, int, int]:
    """读取 PNG 为 RGB uint8；透明区域合成到白底。"""
    with Image.open(path) as image:
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
    height, width = array.shape[:2]
    return array, width, height


def _quantize(rgb: np.ndarray, step: int) -> np.ndarray:
    step = max(1, int(step))
    return np.clip((rgb.astype(np.int16) // step) * step + step // 2, 0, 255).astype(np.uint8)


def _dominant_color(pixels: np.ndarray) -> list[int]:
    """取层内最有代表性的精确颜色（量化众数桶的中位数）。"""
    if pixels.size == 0:
        return [0, 0, 0]
    quantized = _quantize(pixels, 8)
    keys, counts = np.unique(quantized, axis=0, return_counts=True)
    winner = keys[int(np.argmax(counts))]
    mask = np.all(quantized == winner, axis=1)
    selected = pixels[mask]
    return [int(value) for value in np.median(selected, axis=0)]


def segment_colors(rgb: np.ndarray, config: InferenceConfig) -> list[ColorLayer]:
    """把前景像素聚成颜色层。"""
    height, width = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)

    # 背景取全图最高频的量化色；CAD 图几乎总是白底，但先按数据说话。
    quantized_all = _quantize(flat, config.color_quant_step)
    all_keys, all_counts = np.unique(quantized_all, axis=0, return_counts=True)
    background = all_keys[int(np.argmax(all_counts))].astype(np.int16)

    distance_bg = np.linalg.norm(flat.astype(np.int16) - background, axis=1)
    foreground = distance_bg > config.background_distance
    if foreground.sum() < config.min_layer_pixels:
        return []

    foreground_pixels = flat[foreground]
    quantized = _quantize(foreground_pixels, config.color_quant_step)
    keys, counts = np.unique(quantized, axis=0, return_counts=True)
    order = np.argsort(counts)[::-1]
    keys, counts = keys[order], counts[order]

    min_pixels = max(
        int(config.min_layer_pixels),
        int(foreground_pixels.shape[0] * config.min_layer_fraction),
    )

    # 贪心合并相近的量化色。
    cluster_centers: list[np.ndarray] = []
    cluster_keys: list[list[int]] = []
    for key, count in zip(keys, counts):
        if count < min_pixels:
            continue
        center = key.astype(np.float32)
        best_index, best_distance = -1, None
        for index, existing in enumerate(cluster_centers):
            distance = float(np.linalg.norm(center - existing))
            if best_distance is None or distance < best_distance:
                best_index, best_distance = index, distance
        if best_index >= 0 and best_distance is not None and best_distance <= config.color_merge_distance:
            cluster_keys[best_index].append(int(count))
            total = sum(cluster_keys[best_index])
            cluster_centers[best_index] = (
                cluster_centers[best_index] * (total - count) + center * count
            ) / max(total, 1)
        else:
            cluster_centers.append(center)
            cluster_keys.append([int(count)])

    if not cluster_centers:
        return []

    centers = np.stack(cluster_centers)
    foreground_rgb = foreground_pixels.astype(np.float32)
    distances = np.linalg.norm(
        foreground_rgb[:, None, :] - centers[None, :, :], axis=2,
    )
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(distances.shape[0]), nearest]

    layers: list[ColorLayer] = []
    flat_indices = np.flatnonzero(foreground)
    for cluster_index in range(len(centers)):
        selected = (nearest == cluster_index) & (nearest_distance <= config.color_assign_distance)
        if selected.sum() < min_pixels:
            continue
        mask = np.zeros(height * width, dtype=bool)
        mask[flat_indices[selected]] = True
        mask = mask.reshape(height, width)
        layer_rgb = _dominant_color(flat[flat_indices[selected]])
        layers.append(ColorLayer(
            layer_id=0, rgb=layer_rgb, mask=mask, pixel_count=int(selected.sum()),
        ))

    layers.sort(key=lambda layer: layer.pixel_count, reverse=True)
    for index, layer in enumerate(layers, start=1):
        layer.layer_id = index
    return layers
