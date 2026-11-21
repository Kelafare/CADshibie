"""像素级语义标签与实例后处理（dense segmentation 路线）。

标签与评分口径一致：
- Thing（0–29）只有带实例号的图元计分；实例号缺失的归入忽略类 35；
- Stuff（30–34）按类别计分；
- 忽略类 35 与背景 255 分开：35 是“画了但不计分”的线，255 是白底。
"""
from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np

from .config import IGNORE_CLASS, STUFF_CLASS_MAX, STUFF_CLASS_MIN, THING_CLASS_MAX

BACKGROUND = 255
CLASS_COUNT = 36


def stroke_class(primitive) -> int:
    semantic = primitive.semantic_id
    if semantic == IGNORE_CLASS:
        return IGNORE_CLASS
    if semantic <= THING_CLASS_MAX and primitive.instance_id < 0:
        return IGNORE_CLASS
    return semantic


def rasterize_label_map(primitives, canvas_width: float = 100, canvas_height: float = 100,
                        size: int = 1000) -> np.ndarray:
    """把 GT 图元栅格化为类别图（uint8，255=背景）。"""
    label = np.full((size, size), BACKGROUND, dtype=np.uint8)
    sx = size / max(canvas_width, 1e-6)
    sy = size / max(canvas_height, 1e-6)
    for primitive in primitives:
        class_id = stroke_class(primitive)
        points = np.asarray(primitive.points, dtype=np.float64)
        pixels = np.round(points * (sx, sy)).astype(np.int32)
        if primitive.command in (0, 1):
            cv2.polylines(label, [pixels.reshape(-1, 1, 2)], False,
                          int(class_id), thickness=1)
        else:
            low = pixels.min(axis=0)
            high = pixels.max(axis=0)
            center = ((low + high) // 2).tolist()
            axes = [max(1, int((high[0] - low[0]) // 2)), max(1, int((high[1] - low[1]) // 2))]
            cv2.ellipse(label, tuple(center), tuple(axes), 0, 0, 360,
                        int(class_id), thickness=1)
    return label


def class_instance_map(label: np.ndarray, class_id: int,
                       min_pixels: int = 12) -> tuple[np.ndarray, int]:
    """单个类别的连通域实例图（0=背景，1..n=实例）。"""
    mask = (label == class_id).astype(np.uint8)
    if mask.sum() < min_pixels:
        return np.zeros_like(mask, dtype=np.int32), 0
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    count, instances = cv2.connectedComponents(mask, connectivity=8)
    return instances.astype(np.int32), count - 1


def vectorize_mask(mask: np.ndarray, scale_x: float, scale_y: float,
                   min_length: float = 0.4,
                   offset: tuple[int, int] = (0, 0)) -> list[list[tuple[float, float]]]:
    """把实例掩码细化成骨架折线（4 点采样，画布坐标）。

    ``offset`` 是掩码在原图中的左上角（像素），用于在裁剪后还原坐标。
    """
    from skimage.morphology import skeletonize

    from .skeleton import sample_polyline, trace_paths

    if not mask.any():
        return []
    skeleton = skeletonize(mask.astype(bool))
    paths = trace_paths(skeleton, min_pixels=5)
    polylines = []
    for path in paths:
        shifted = [(row + offset[1], column + offset[0]) for row, column in path]
        points = sample_polyline(shifted, 4, scale_x, scale_y)
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        if length >= min_length:
            polylines.append(points)
    return polylines


def instance_segments(label: np.ndarray, scale: float = 10.0):
    """把类别图转成 (class_id, instance_index, 像素掩码) 序列。

    Thing：每个连通域一个实例；Stuff：同类合并为 -1。
    """
    segments = []
    for class_id in range(CLASS_COUNT):
        if class_id == IGNORE_CLASS:
            continue
        mask = label == class_id
        if not mask.any():
            continue
        if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX:
            segments.append((class_id, -1, mask))
            continue
        instances, count = class_instance_map(label, class_id)
        for index in range(1, count + 1):
            instance_mask = instances == index
            if instance_mask.sum() >= 12:
                segments.append((class_id, index - 1, instance_mask))
    return segments
