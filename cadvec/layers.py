"""Alpha-aware stroke color recovery and color-layer segmentation (v2).

CAD 平面图的 PNG 是透明底 + 抗锯齿细线。直接把 RGBA 合成到白底会得到大量
「描边色 × 透明度」的混合色，颜色先验因此失效。这里改为：

1. 每个像素的渲染色都位于「白 → 描边色」射线上：``c = 255 + t*(s - 255)``；
2. 对调色板中的每种描边色计算像素到该射线的距离，取最近射线作为像素颜色；
3. 灰度色（黑/灰）无法用色相区分，按亮度分成若干桶。

调色板来自训练集 SVG 的 ``stroke`` 精确颜色，见 ``tools/build_palette.py``。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

#: 无彩色（黑/灰）像素按亮度分段，避免不同灰阶互相吞并。
ACHROMATIC_BUCKETS = ((0, 45), (45, 110), (110, 200), (200, 256))


@dataclass
class Layer:
    layer_id: int
    rgb: list[int]
    mask: np.ndarray
    pixel_count: int
    achromatic: bool = False


def load_rgba(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """返回 (RGB uint8, alpha float32 in [0,1])；不经白底合成。"""
    with Image.open(path) as image:
        image = image.convert("RGBA")
        array = np.asarray(image, dtype=np.uint8)
    return array[..., :3], array[..., 3].astype(np.float32) / 255.0


def nearest_ray(
    pixels: np.ndarray, palette: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把像素（N×3）匹配到最近的「白→调色板色」射线。

    返回 ``(index, distance, t)``，``index`` 为调色板下标，``t`` 为射线参数
    （≈ 覆盖率）。
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    directions = palette.astype(np.float32) - 255.0
    norms = np.einsum("ij,ij->i", directions, directions)
    valid = norms > 16.0  # 纯白/近白没有方向信息
    safe = np.where(valid, norms, 1.0)

    # 分块计算，控制 N×K 的内存峰值。
    index = np.zeros(pixels.shape[0], dtype=np.int16)
    distance = np.full(pixels.shape[0], np.inf, dtype=np.float32)
    coverage = np.zeros(pixels.shape[0], dtype=np.float32)
    chunk = max(1, 8_000_000 // max(len(palette), 1))
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start:start + chunk]
        centered = block - 255.0
        t = np.clip((centered @ directions.T) / safe, 0.0, 1.0)
        residual = centered[:, None, :] - t[..., None] * directions[None, :, :]
        dist = np.einsum("ijk,ijk->ij", residual, residual)
        dist[:, ~valid] = np.inf
        best = np.argmin(dist, axis=1)
        rows = np.arange(block.shape[0])
        index[start:start + chunk] = best
        distance[start:start + chunk] = np.sqrt(dist[rows, best])
        coverage[start:start + chunk] = t[rows, best]
    return index, distance, coverage


def _bucket_values(values: np.ndarray) -> np.ndarray:
    buckets = np.zeros(values.shape, dtype=np.int32)
    for index, (low, high) in enumerate(ACHROMATIC_BUCKETS):
        buckets[(values >= low) & (values < high)] = index
    return buckets


def layer_key(
    rgb: np.ndarray, index: np.ndarray, coverage: np.ndarray, palette: np.ndarray,
) -> np.ndarray:
    """把调色板下标映射为「层键」。

    无彩色像素（黑/各种灰）无法靠色相区分，改按「反混合后的亮度」分桶：
    ``base = 255 + (gray - 255) / t``。这样黑、深灰、浅灰不会互相吞并。
    """
    pixels = rgb.reshape(-1, 3)
    spread = pixels.max(axis=1) - pixels.min(axis=1)
    mean = pixels.mean(axis=1)
    achromatic = (spread <= np.maximum(10.0, 0.10 * mean)) & (mean < 245.0)
    base_gray = 255.0 + (mean - 255.0) / np.maximum(coverage.reshape(-1), 0.06)
    buckets = _bucket_values(np.clip(base_gray, 0, 255))

    keys = index.reshape(-1).astype(np.int32)
    chroma_key = keys.copy()
    achromatic_key = len(palette) + buckets
    keys = np.where(achromatic, achromatic_key, chroma_key)
    keys = np.where((index.reshape(-1) < 0) & ~achromatic, -1, keys)
    return keys.reshape(rgb.shape[:2])


def segment_layers(
    rgb: np.ndarray,
    alpha: np.ndarray,
    palette: np.ndarray,
    *,
    max_color_distance: float = 26.0,
    min_coverage: float = 0.04,
    min_pixels: int = 40,
    min_fraction: float = 0.00002,
) -> list[Layer]:
    """按精确描边色把前景像素分成若干颜色层。"""
    height, width = alpha.shape
    pixels = rgb.reshape(-1, 3)
    foreground_flat = alpha.reshape(-1) > 0.05
    index_flat = np.full(pixels.shape[0], -1, dtype=np.int16)
    distance_flat = np.full(pixels.shape[0], np.inf, dtype=np.float32)
    coverage_flat = np.zeros(pixels.shape[0], dtype=np.float32)
    if foreground_flat.any():
        matched = np.flatnonzero(foreground_flat)
        idx, dist, cov = nearest_ray(pixels[matched], palette)
        index_flat[matched] = idx
        distance_flat[matched] = dist
        coverage_flat[matched] = cov
    index = index_flat.reshape(height, width)
    distance = distance_flat.reshape(height, width)
    coverage = coverage_flat.reshape(height, width)
    keys = layer_key(rgb, index, coverage, palette)

    achromatic = keys >= len(palette)
    foreground = (
        (alpha > 0.05) & (coverage > min_coverage) & (keys >= 0)
        & ((distance < max_color_distance) | achromatic)
    )
    if not foreground.any():
        return []

    minimum = max(int(min_pixels), int(foreground.sum() * min_fraction))
    labels = np.unique(keys[foreground])
    layers: list[Layer] = []
    for key in labels:
        mask = foreground & (keys == key)
        count = int(mask.sum())
        if count < minimum:
            continue
        if key < len(palette):
            color = [int(v) for v in palette[key]]
        else:
            low, high = ACHROMATIC_BUCKETS[int(key - len(palette))]
            value = (low + high) // 2
            color = [value, value, value]
        layers.append(Layer(0, color, mask, count, achromatic=key >= len(palette)))

    layers.sort(key=lambda layer: layer.pixel_count, reverse=True)
    for layer_id, layer in enumerate(layers, start=1):
        layer.layer_id = layer_id
    return layers


def layer_color_stats(layers: list[Layer]) -> list[dict]:
    return [
        {"layer_id": layer.layer_id, "rgb": layer.rgb, "pixels": layer.pixel_count}
        for layer in layers
    ]


def save_palette(colors: list[list[int]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 2, "colors": colors}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    return path


def load_palette(path: str | Path) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    colors = payload.get("colors") or [[0, 0, 0]]
    return np.asarray(colors, dtype=np.float32)
