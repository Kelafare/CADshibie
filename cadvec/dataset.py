"""训练样本构建：对象假设 ↔ 真值实例匹配。

对训练/本地测试集的每张 PNG 跑一遍几何流水线，得到对象假设；再读同名 SVG
把真值实例栅格化，按覆盖率/IoU 给假设打标签：

* ``IoU > 0.5``：假设与真值实例基本重合 → 该实例类别；
* ``覆盖率 > 0.6``：假设是某个真值实例的组成部分 → 该类别；
* ``覆盖率 < 0.15``：与任何计分真值都无关 → 忽略类 35；
* 介于两者之间：歧义样本，直接丢弃。

这样训练分布与推理时**同一套几何流程**产生的候选一致，避免候选域偏移。
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import (
    COMMAND_CIRCLE,
    COMMAND_ELLIPSE,
    IGNORE_CLASS,
    STUFF_CLASS_MAX,
    STUFF_CLASS_MIN,
    THING_CLASS_MAX,
)
from .layers import Layer, load_palette, load_rgba, segment_layers
from .objects import Hypothesis, build_hypotheses, rasterize_hypothesis
from .parallel import align_ribbon_endpoints, parallel_ribbons, snap_axis_lines
from .strokes import Stroke, extract_strokes

RASTER = 400


@dataclass
class Scene:
    width: int
    height: int
    scale_x: float
    scale_y: float
    strokes: list[Stroke]
    ribbons: list
    hypotheses: list[Hypothesis]
    layers: list[Layer] = field(default_factory=list)


_SCENE_LOCAL = __import__("threading").local()


def component_scene(
    path: str | Path, palette: np.ndarray, config,
    canvas_width: int = 100, canvas_height: int = 100,
    group_parallel: bool = False,
) -> Scene:
    """以「颜色层骨架连通分量」为对象假设，直接输出骨架折线。

    与 Hough+合并的图元假设相比，骨架折线保留了原始描边的形状与端点；
    实测 4 点采样后仍有约 18% 的真值实例达到 IoU>0.5（Hough 方案约 4%）。
    """
    from scipy import ndimage
    from skimage.morphology import skeletonize

    from .skeleton import polyline_length, sample_polyline, trace_paths

    # hybrid 引擎会在同一张图上依次调用 v2/GAT/segdet；按线程缓存最近一次
    # 场景，避免重复做颜色分层与骨架追踪（稠密图上可省 2/3 时间）。
    cache_key = (str(path), int(canvas_width), int(canvas_height))
    cached = getattr(_SCENE_LOCAL, "scene", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    rgb, alpha = load_rgba(path)
    height, width = alpha.shape
    layers = segment_layers(rgb, alpha, palette)
    scale_x = width / max(canvas_width, 1)
    scale_y = height / max(canvas_height, 1)
    strokes: list[Stroke] = []
    hypotheses: list[Hypothesis] = []

    for layer in layers:
        skeleton = skeletonize(layer.mask)
        if skeleton.sum() < 12:
            continue
        labels, count = ndimage.label(
            ndimage.binary_dilation(skeleton, iterations=2),
            structure=np.ones((3, 3)),
        )
        sizes = np.bincount(labels.ravel())
        for label, box in enumerate(ndimage.find_objects(labels), start=1):
            if box is None or sizes[label] < 24:
                continue
            local = skeleton[box] & (labels[box] == label)
            paths = trace_paths(local, min_pixels=4)
            if not paths:
                continue
            indices = []
            for traced in paths:
                shifted = [
                    (row + box[0].start, column + box[1].start)
                    for row, column in traced
                ]
                points = sample_polyline(shifted, 4, scale_x, scale_y)
                length = polyline_length([
                    (x * scale_x, y * scale_y) for x, y in points
                ]) / ((scale_x + scale_y) / 2.0)
                if length < 0.35:
                    continue
                stroke = Stroke(
                    command=0, points=points, length=length,
                    p0=points[0], p1=points[-1], width=1.0,
                    layer_id=layer.layer_id, rgb=tuple(layer.rgb),
                )
                indices.append(len(strokes))
                strokes.append(stroke)
            if not indices:
                continue
            xs = [point[0] for index in indices for point in strokes[index].points]
            ys = [point[1] for index in indices for point in strokes[index].points]
            hypotheses.append(Hypothesis(
                indices=indices, bbox=(min(xs), min(ys), max(xs), max(ys)),
                layer_ids={layer.layer_id},
            ))

    # 平行线优化可选开关：实验表明把平行分量并成实例会降低 IoU，
    # 默认关闭；保留实现供笔画级流水线与后续实验使用。
    ribbons = []
    if group_parallel:
        try:
            from .parallel import parallel_ribbons
            ribbons = parallel_ribbons(strokes, 1.0, max_gap_px=3.5)
        except Exception:  # noqa: BLE001
            ribbons = []
    owner = {}
    for index, hypothesis in enumerate(hypotheses):
        for stroke_index in hypothesis.indices:
            owner[stroke_index] = index
    parent = list(range(len(hypotheses)))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first, second):
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for ribbon in ribbons:
        if len(ribbon.members) > 12:
            continue
        ordered = sorted(ribbon.members, key=lambda i: _offset_canvas(strokes[i]))
        for previous, current in zip(ordered, ordered[1:]):
            if _offset_gap(strokes[previous], strokes[current]) > 3.5:
                continue
            first_owner = owner.get(previous)
            second_owner = owner.get(current)
            if first_owner is None or second_owner is None or first_owner == second_owner:
                continue
            first_box = hypotheses[first_owner].bbox
            second_box = hypotheses[second_owner].bbox
            gap = max(
                first_box[0] - second_box[2], second_box[0] - first_box[2],
                first_box[1] - second_box[3], second_box[1] - first_box[3],
                0.0,
            )
            if gap <= 1.2:
                union(first_owner, second_owner)

    merged: dict[int, list[int]] = {}
    for index, hypothesis in enumerate(hypotheses):
        merged.setdefault(find(index), []).extend(hypothesis.indices)
    hypotheses = []
    for indices in merged.values():
        xs = [point[0] for index in indices for point in strokes[index].points]
        ys = [point[1] for index in indices for point in strokes[index].points]
        hypotheses.append(Hypothesis(
            indices=sorted(indices), bbox=(min(xs), min(ys), max(xs), max(ys)),
        ))
    hypotheses.sort(key=lambda item: -sum(strokes[i].length for i in item.indices))
    scene = Scene(width, height, scale_x, scale_y, strokes, [], hypotheses, layers)
    _SCENE_LOCAL.scene = (cache_key, scene)
    return scene


def _offset_canvas(stroke: Stroke) -> float:
    import math as _math
    dx = stroke.p1[0] - stroke.p0[0]
    dy = stroke.p1[1] - stroke.p0[1]
    norm = _math.hypot(dx, dy) or 1.0
    return -dy / norm * stroke.p0[0] + dx / norm * stroke.p0[1]


def _offset_gap(first: Stroke, second: Stroke) -> float:
    import math as _math
    dx = first.p1[0] - first.p0[0]
    dy = first.p1[1] - first.p0[1]
    norm = _math.hypot(dx, dy) or 1.0
    nx, ny = -dy / norm, dx / norm
    return abs((second.p0[0] - first.p0[0]) * nx + (second.p0[1] - first.p0[1]) * ny)


def scene_from_png(path: str | Path, palette: np.ndarray, config,
                   canvas_width: int = 100, canvas_height: int = 100) -> Scene:
    rgb, alpha = load_rgba(path)
    height, width = alpha.shape
    layers = segment_layers(rgb, alpha, palette)
    scale_x = width / max(canvas_width, 1)
    scale_y = height / max(canvas_height, 1)
    mean_scale = (scale_x + scale_y) / 2.0
    strokes: list[Stroke] = []
    for layer in layers:
        for stroke in extract_strokes(layer.mask, layer, config):
            stroke.points = [(x / scale_x, y / scale_y) for x, y in stroke.points]
            stroke.p0 = (stroke.p0[0] / scale_x, stroke.p0[1] / scale_y)
            stroke.p1 = (stroke.p1[0] / scale_x, stroke.p1[1] / scale_y)
            stroke.length /= mean_scale
            stroke.width /= mean_scale
            if stroke.shape:
                shape = stroke.shape
                shape["cx"] /= scale_x
                shape["cy"] /= scale_y
                shape["a"] /= mean_scale
                shape["b"] /= mean_scale
            strokes.append(stroke)
    snap_axis_lines(strokes, 1.0)
    ribbons = parallel_ribbons(strokes, 1.0)
    align_ribbon_endpoints(strokes, ribbons)
    hypotheses = build_hypotheses(strokes, ribbons)
    return Scene(width, height, scale_x, scale_y, strokes, ribbons, hypotheses, layers)


def gt_segment_masks(primitives, canvas_width: float, canvas_height: float,
                     raster: int = RASTER) -> dict[tuple[int, int], np.ndarray]:
    """按评测口径栅格化真值：Thing 按实例、Stuff 按类别合并，忽略类 35 跳过。"""
    masks: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros((raster, raster), dtype=np.uint8)
    )
    sx = raster / max(canvas_width, 1e-6)
    sy = raster / max(canvas_height, 1e-6)
    for primitive in primitives:
        semantic = primitive.semantic_id
        if semantic == IGNORE_CLASS:
            continue
        if semantic <= THING_CLASS_MAX:
            if primitive.instance_id < 0:
                continue
            key = (semantic, primitive.instance_id)
        elif semantic <= STUFF_CLASS_MAX:
            key = (semantic, -1)
        else:
            continue
        points = np.array(
            [(x * sx, y * sy) for x, y in primitive.points], dtype=np.float32
        )
        if primitive.command in (0, 1):
            cv2.polylines(masks[key], [np.round(points).astype(np.int32)],
                          False, 1, thickness=1)
        else:
            low = np.floor(points.min(axis=0)).astype(np.int32)
            high = np.ceil(points.max(axis=0)).astype(np.int32)
            cv2.ellipse(masks[key], (int((low[0] + high[0]) // 2), int((low[1] + high[1]) // 2)),
                        (max(1, int((high[0] - low[0]) // 2)), max(1, int((high[1] - low[1]) // 2))),
                        0, 0, 360, 1, thickness=1)
    return dict(masks)


def gt_samples(primitives, max_samples: int = 48) -> list[dict]:
    """真值实例作为正样本：模型直接学习完整对象的形状。"""
    groups: dict[tuple[int, int], list] = defaultdict(list)
    for primitive in primitives:
        semantic = primitive.semantic_id
        if semantic == IGNORE_CLASS:
            continue
        if semantic <= THING_CLASS_MAX:
            if primitive.instance_id < 0:
                continue
            key = (semantic, primitive.instance_id)
        elif semantic <= STUFF_CLASS_MAX:
            key = (semantic, -1)
        else:
            continue
        groups[key].append(primitive)
    samples = []
    for (semantic, _), primitives_group in groups.items():
        if not primitives_group:
            continue
        xs = [point[0] for primitive in primitives_group for point in primitive.points]
        ys = [point[1] for primitive in primitives_group for point in primitive.points]
        samples.append({
            "bbox": [round(min(xs), 4), round(min(ys), 4),
                     round(max(xs), 4), round(max(ys), 4)],
            "label": int(semantic),
            "iou": 1.0,
            "cover": 1.0,
            "kind": "gt",
            "strokes": [
                {
                    "command": int(primitive.command),
                    "points": [[round(x, 4), round(y, 4)] for x, y in primitive.points],
                    "length": round(primitive.length, 4),
                    "rgb": [int(v) for v in primitive.rgb],
                    "layer": int(primitive.layer_id),
                    "width": round(primitive.width, 4),
                }
                for primitive in primitives_group[:64]
            ],
            "length": round(sum(primitive.length for primitive in primitives_group), 4),
            "parallel_pairs": 0,
        })
    samples.sort(key=lambda item: -item["length"])
    return samples[:max_samples]


def label_hypotheses(
    scene: Scene, gt_masks: dict[tuple[int, int], np.ndarray],
    canvas_width: int = 100, canvas_height: int = 100, raster: int = RASTER,
) -> list[dict]:
    """给每个假设打标签，返回可序列化的样本列表。"""
    samples = []
    for hypothesis in scene.hypotheses:
        if not hypothesis.indices:
            continue
        mask = rasterize_hypothesis(
            scene.strokes, hypothesis.indices, canvas_width, canvas_height, raster,
        )
        if mask.sum() < 6:
            continue
        best_iou = 0.0
        best_class = IGNORE_CLASS
        cover_class = IGNORE_CLASS
        cover_value = 0.0
        area = int(mask.sum())
        for (semantic, _), gt in gt_masks.items():
            intersection = int(np.count_nonzero(mask & gt))
            if intersection == 0:
                continue
            iou = intersection / max(area + int(gt.sum()) - intersection, 1)
            cover = intersection / max(area, 1)
            if iou > best_iou:
                best_iou = iou
                best_class = semantic
            if cover > cover_value:
                cover_value = cover
                cover_class = semantic

        if best_iou > 0.5:
            label = best_class
        elif cover_value > 0.60:
            label = cover_class
        elif cover_value < 0.15:
            label = IGNORE_CLASS
        else:
            continue
        total_length = sum(scene.strokes[index].length for index in hypothesis.indices)
        samples.append({
            "bbox": [round(value, 4) for value in hypothesis.bbox],
            "label": int(label),
            "kind": "hyp",
            "iou": round(float(best_iou), 4),
            "cover": round(float(cover_value), 4),
            "strokes": [
                {
                    "command": int(scene.strokes[index].command),
                    "points": [
                        [round(x, 4), round(y, 4)]
                        for x, y in (
                            # 与推理 _stroke_payload 一致：Line 使用骨架四点，
                            # 不能退化成两个端点，否则训练/推理的第四通道掩码不同。
                            scene.strokes[index].points
                            if scene.strokes[index].command == 0
                            else scene.strokes[index].resample(32)
                        )
                    ],
                    "length": round(scene.strokes[index].length, 4),
                    "rgb": [int(v) for v in scene.strokes[index].rgb],
                    "layer": int(scene.strokes[index].layer_id),
                    "width": round(scene.strokes[index].width, 4),
                }
                for index in hypothesis.indices[:80]
            ],
            "length": round(total_length, 4),
            "parallel_pairs": int(hypothesis.parallel_pairs),
        })
    return samples
