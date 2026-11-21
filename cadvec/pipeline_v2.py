"""v2 单图推理：颜色层 → 直线/曲线 → 平行线优化 → 对象假设 → ViT 分类。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import (
    COMMAND_ARC,
    COMMAND_CIRCLE,
    COMMAND_ELLIPSE,
    COMMAND_LINE,
    IGNORE_CLASS,
    STUFF_CLASS_MAX,
    STUFF_CLASS_MIN,
    THING_CLASS_MAX,
    InferenceConfig,
)
from .dataset import component_scene
from .layers import load_palette
from .schema import assemble_output, empty_output, validate_output
from .strokes import Stroke
from .vit import CROP_SIZE, VitClassifier  # noqa: F401


@dataclass
class V2Result:
    data: dict[str, Any]
    hypotheses: int = 0
    classified: int = 0
    seconds: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)


def load_color_prior(path: str | Path | None) -> dict[tuple[int, int, int], np.ndarray]:
    """读入「描边色 → 类别分布」表，返回归一化后的 0-34 类先验。"""
    if path is None or not Path(path).exists():
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    table: dict[tuple[int, int, int], np.ndarray] = {}
    for key, entry in payload.items():
        try:
            rgb = tuple(int(part) for part in key.split(","))
        except ValueError:
            continue
        counts = entry.get("counts", {})
        vector = np.full(35, 0.5, dtype=np.float64)
        for label, count in counts.items():
            label = int(label)
            if 0 <= label <= 34:
                vector[label] += float(count)
        vector /= vector.sum()
        table[rgb] = vector
    return table


def load_class_policy(path: str | Path | None) -> dict:
    """读取逐类阈值策略（tools/calibrate_thresholds.py 生成）。"""
    if path is None or not Path(path).exists():
        return {}
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    thresholds = np.full(36, -1.0, dtype=np.float32)
    for key, value in (payload.get("thresholds") or {}).items():
        try:
            thresholds[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        "thresholds": thresholds,
        "suppress": {int(value) for value in payload.get("suppress", [])},
    }


def _mask_nms(
    hypotheses, semantic: np.ndarray, confidence: np.ndarray,
    strokes: list[Stroke], canvas_width: int, canvas_height: int,
    iou_threshold: float,
) -> list[int]:
    """同类别按掩码 IoU 去重，保留置信度高的假设。"""
    from .objects import rasterize_hypothesis

    kept: list[int] = []
    if iou_threshold <= 0:
        return list(range(len(hypotheses)))
    for class_id in np.unique(semantic):
        if class_id == IGNORE_CLASS:
            continue
        indices = [index for index in range(len(hypotheses)) if semantic[index] == class_id]
        if len(indices) < 2:
            kept.extend(indices)
            continue
        masks = {
            index: rasterize_hypothesis(
                strokes, hypotheses[index].indices,
                canvas_width, canvas_height, 160,
            )
            for index in indices
        }
        order = sorted(indices, key=lambda index: -confidence[index])
        survivors: list[int] = []
        for index in order:
            mask = masks[index]
            duplicate = False
            for other in survivors:
                other_mask = masks[other]
                intersection = int(np.count_nonzero(mask & other_mask))
                if not intersection:
                    continue
                union = int(np.count_nonzero(mask | other_mask))
                if intersection / max(union, 1) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                survivors.append(index)
        kept.extend(survivors)
    return sorted(kept)


def apply_color_prior(
    probabilities: np.ndarray, hypotheses, strokes: list[Stroke],
    table: dict[tuple[int, int, int], np.ndarray], weight: float,
) -> np.ndarray:
    """在非忽略类上乘以颜色先验，保留 35 类概率不变。"""
    if weight <= 0.0 or not table:
        return probabilities
    result = probabilities.astype(np.float64).copy()
    for index, hypothesis in enumerate(hypotheses):
        if not hypothesis.indices:
            continue
        rgb = tuple(int(value) for value in strokes[hypothesis.indices[0]].rgb)
        prior = table.get(rgb)
        if prior is None:
            # 灰阶桶色可能不在表里，退化到最接近的灰度。
            if max(rgb) - min(rgb) <= 8:
                gray = int(round(sum(rgb) / 3.0))
                candidates = [
                    (key, value) for key, value in table.items()
                    if max(key) - min(key) <= 8
                ]
                if candidates:
                    prior = min(candidates, key=lambda item: abs(sum(item[0]) / 3 - gray))[1]
        if prior is None:
            continue
        result[index, :35] *= np.power(prior + 1e-6, weight)
        total = result[index].sum()
        if total > 0:
            result[index] /= total
    return result.astype(np.float32)


def _stroke_payload(stroke: Stroke) -> dict:
    if stroke.command == COMMAND_LINE and len(stroke.points) >= 2:
        # 骨架折线本身就可能带拐点，直接使用采样点。
        points = stroke.points
    elif stroke.command == COMMAND_LINE:
        points = [stroke.p0, stroke.p1]
    else:
        points = stroke.resample(32)
    return {
        "command": int(stroke.command),
        "points": [[float(x), float(y)] for x, y in points],
        "length": float(stroke.length),
        "rgb": [int(v) for v in stroke.rgb],
        "layer": int(stroke.layer_id),
        "width": float(stroke.width),
    }


def _bbox_gap(first, second) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0.0)
    dy = max(first[1] - second[3], second[1] - first[3], 0.0)
    return math.hypot(dx, dy)


def _merge_same_class(hypotheses: list[int], probabilities: np.ndarray,
                      boxes: list[tuple], threshold: float) -> list[list[int]]:
    """同类别、空间相接的假设合并为一个实例（并查集，避免 O(n³)）。"""
    if not hypotheses:
        return []
    local = {index: position for position, index in enumerate(hypotheses)}
    parent = list(range(len(hypotheses)))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for position, first in enumerate(hypotheses):
        first_box = boxes[first]
        for second in hypotheses[position + 1:]:
            if _bbox_gap(first_box, boxes[second]) <= 0.8:
                union(position, local[second])

    groups: dict[int, list[int]] = {}
    for position, index in enumerate(hypotheses):
        groups.setdefault(find(position), []).append(index)
    return list(groups.values())


def _union_box(boxes: list[tuple]) -> tuple:
    return (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )


def _primitive_from_stroke(stroke: Stroke) -> dict:
    if stroke.command == COMMAND_LINE and len(stroke.points) >= 4:
        points = stroke.points[:4]
    elif stroke.command == COMMAND_LINE:
        points = [
            (
                stroke.p0[0] + (stroke.p1[0] - stroke.p0[0]) * fraction,
                stroke.p0[1] + (stroke.p1[1] - stroke.p0[1]) * fraction,
            )
            for fraction in (0.0, 1 / 3, 2 / 3, 1.0)
        ]
    else:
        points = stroke.resample(4)
    return {
        "command": int(stroke.command),
        "points": points,
        "length": float(stroke.length),
        "width": float(stroke.width),
        "rgb": [int(v) for v in stroke.rgb],
        "layer": int(stroke.layer_id),
    }


class _Primitive:
    """schema.assemble_output 需要的最小图元接口。"""

    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, payload: dict, semantic: int, instance: int):
        self.command = payload["command"]
        self.points = payload["points"]
        self.length = payload["length"]
        self.semantic_id = semantic
        self.instance_id = instance
        self.rgb = payload["rgb"]
        self.layer_id = payload["layer"]
        self.width = payload["width"]


def run_image_v2(
    png_path: str | Path,
    config: InferenceConfig,
    palette_path: str | Path,
    classifier: VitClassifier | None,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    confidence_threshold: float = 0.30,
    max_hypotheses: int = 240,
    color_prior_path: str | Path | None = None,
    color_prior_weight: float = 0.6,
    class_policy_path: str | Path | None = None,
    nms_iou: float = 0.6,
    objectness_threshold: float = 0.0,
    max_instances: int = 0,
) -> V2Result:
    import time
    started = time.perf_counter()
    palette = load_palette(palette_path)
    scene = component_scene(png_path, palette, config, canvas_width, canvas_height)
    strokes = scene.strokes
    hypotheses = [
        hypothesis for hypothesis in scene.hypotheses
        if sum(strokes[i].length for i in hypothesis.indices) >= 1.2
    ]
    hypotheses.sort(key=lambda item: -sum(strokes[i].length for i in item.indices))
    hypotheses = hypotheses[:max_hypotheses]
    if not hypotheses:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return V2Result(data, 0, 0, time.perf_counter() - started)

    probabilities = np.zeros((len(hypotheses), 36), dtype=np.float32)
    objectness = np.ones((len(hypotheses),), dtype=np.float32)
    if classifier is not None:
        from .crops import prepare_image, render_crop
        image, scale_x, scale_y = prepare_image(png_path)
        crops = [
            render_crop(
                image, hypothesis.bbox,
                [_stroke_payload(strokes[index]) for index in hypothesis.indices],
                scale_x, scale_y, size=CROP_SIZE,
            )
            for hypothesis in hypotheses
        ]
        probabilities, objectness = classifier.classify_with_iou(crops)

    if color_prior_weight > 0.0:
        probabilities = apply_color_prior(
            probabilities, hypotheses, strokes,
            load_color_prior(color_prior_path), color_prior_weight,
        )

    semantic = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    if classifier is None:
        semantic[:] = IGNORE_CLASS
        confidence[:] = 0.0

    # objectness/IoU 头：预测与 GT 实例的 IoU，低于阈值直接视为背景。
    if objectness_threshold > 0.0 and classifier is not None and classifier.has_objectness:
        semantic = np.where(objectness >= objectness_threshold, semantic, IGNORE_CLASS)

    # 逐类阈值 + 低精度类抑制；policy 缺省时退化为全局阈值。
    policy = load_class_policy(class_policy_path)
    if policy:
        thresholds = policy["thresholds"]
        suppressed = policy["suppress"]
        for index in range(len(hypotheses)):
            class_id = int(semantic[index])
            if class_id in suppressed:
                semantic[index] = IGNORE_CLASS
                continue
            required = max(confidence_threshold, float(thresholds[class_id]))
            if confidence[index] < required:
                semantic[index] = IGNORE_CLASS
    else:
        semantic = np.where(confidence >= confidence_threshold, semantic, IGNORE_CLASS)

    if nms_iou > 0:
        kept = _mask_nms(
            hypotheses, semantic, confidence, strokes,
            canvas_width, canvas_height, nms_iou,
        )
        kept_set = set(kept)
        hypotheses = [hypotheses[index] for index in kept]
        probabilities = probabilities[kept]
        semantic = semantic[kept]
        confidence = confidence[kept]

    boxes = [hypothesis.bbox for hypothesis in hypotheses]
    emitted: list[_Primitive] = []
    thing_instances: list[tuple[float, int, list[dict]]] = []
    for class_id in range(36):
        indices = [index for index in range(len(hypotheses)) if semantic[index] == class_id]
        if not indices:
            continue
        if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX:
            points_payload = [
                _primitive_from_stroke(strokes[index])
                for hypothesis in indices for index in hypotheses[hypothesis].indices
            ]
            for payload in _dedupe_payloads(points_payload):
                emitted.append(_Primitive(payload, class_id, -1))
            continue
        if class_id > THING_CLASS_MAX:
            continue
        groups = _merge_same_class(indices, probabilities, boxes, confidence_threshold)
        for group in groups:
            score = max(float(confidence[index]) for index in group)
            payloads = [
                _primitive_from_stroke(strokes[index])
                for hypothesis in group for index in hypotheses[hypothesis].indices
            ]
            payloads = _dedupe_payloads(payloads)
            if payloads:
                thing_instances.append((score, class_id, payloads))

    # 每图只保留置信度最高的 K 个 Thing 实例。GT 每图中位只有 3 个计分段，
    # 密集图里大量低分实例是纯 FP；按分数截断比全局阈值更贴合评分结构。
    if max_instances > 0 and len(thing_instances) > max_instances:
        thing_instances.sort(key=lambda item: -item[0])
        dropped = len(thing_instances) - max_instances
        thing_instances = thing_instances[:max_instances]
    else:
        dropped = 0
    instance_id = 0
    for _score, class_id, payloads in thing_instances:
        for payload in payloads:
            emitted.append(_Primitive(payload, class_id, instance_id))
        instance_id += 1

    data = assemble_output(emitted, width=canvas_width, height=canvas_height)
    validate_output(data)
    elapsed = time.perf_counter() - started
    counts = {"line": 0, "arc": 0, "circle": 0, "ellipse": 0}
    for primitive in emitted:
        counts[("line", "arc", "circle", "ellipse")[primitive.command]] += 1
    return V2Result(
        data=data, hypotheses=len(hypotheses), classified=int((semantic != 35).sum()),
        seconds=elapsed, stats={
            "primitive_count": len(emitted),
            "hypotheses": len(hypotheses),
            "objects": instance_id,
            "dropped_instances": dropped,
            "seconds": round(elapsed, 3),
            "command_counts": counts,
        },
    )


def _dedupe_payloads(payloads: list[dict]) -> list[dict]:
    """同一实例里完全重复的线段只保留一条（同一份 SVG 常重复绘制）。"""
    kept = []
    seen = set()
    for payload in payloads:
        points = payload["points"]
        key = (
            payload["command"],
            tuple(round(value, 1) for point in points for value in point),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(payload)
    return kept
