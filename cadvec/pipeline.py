"""单张 PNG 的完整推理流程。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import (
    COMMAND_ARC,
    COMMAND_CIRCLE,
    COMMAND_ELLIPSE,
    COMMAND_LINE,
    InferenceConfig,
)
from .image import load_rgb, segment_colors
from .instances import assign_instances
from .prior import ColorPrior, PrimitiveClassifier, apply_semantics
from .schema import assemble_output, empty_output, validate_output
from .vectorize import Primitive, vectorize_all


@dataclass
class PipelineResult:
    data: dict[str, Any]
    primitives: list[Primitive]
    layer_count: int
    stats: dict[str, Any] = field(default_factory=dict)


def _midpoint(primitive: Primitive) -> tuple[float, float]:
    xs = [point[0] for point in primitive.points]
    ys = [point[1] for point in primitive.points]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def _is_duplicate(first: Primitive, second: Primitive, tolerance: float = 0.6) -> bool:
    if first.layer_id != second.layer_id or first.command != second.command:
        return False
    if first.command == COMMAND_LINE:
        first_mid, second_mid = _midpoint(first), _midpoint(second)
        if math.dist(first_mid, second_mid) > tolerance:
            return False
        first_angle = math.atan2(
            first.points[-1][1] - first.points[0][1],
            first.points[-1][0] - first.points[0][0],
        )
        second_angle = math.atan2(
            second.points[-1][1] - second.points[0][1],
            second.points[-1][0] - second.points[0][0],
        )
        difference = abs(first_angle - second_angle) % math.pi
        difference = min(difference, math.pi - difference)
        return difference < math.radians(2.0)
    if first.command in (COMMAND_CIRCLE, COMMAND_ELLIPSE, COMMAND_ARC):
        first_mid, second_mid = _midpoint(first), _midpoint(second)
        if math.dist(first_mid, second_mid) > tolerance:
            return False
        return abs(first.length - second.length) <= max(0.5, 0.08 * max(first.length, second.length))
    return False


def deduplicate(primitives: list[Primitive], tolerance: float = 0.6) -> list[Primitive]:
    kept: list[Primitive] = []
    for primitive in sorted(primitives, key=lambda item: item.length, reverse=True):
        if not any(_is_duplicate(primitive, existing, tolerance) for existing in kept):
            kept.append(primitive)
    kept.sort(key=lambda item: (
        item.layer_id, item.command,
        min(point[0] for point in item.points),
        min(point[1] for point in item.points),
    ))
    return kept


def _filter_tiny(
    primitives: list[Primitive], canvas_width: int, canvas_height: int,
    min_length: float = 0.0,
) -> list[Primitive]:
    limit = max(0.3, min(canvas_width, canvas_height) * 0.005, min_length)
    result = []
    for primitive in primitives:
        if primitive.length < limit:
            continue
        if not any(
            -5 <= x <= canvas_width + 5 and -5 <= y <= canvas_height + 5
            for x, y in primitive.points
        ):
            continue
        result.append(primitive)
    return result


def run_image(
    png_path: str | Path,
    config: InferenceConfig,
    prior: ColorPrior,
    classifier: PrimitiveClassifier,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    refiner: Any | None = None,
) -> PipelineResult:
    """读取一张 PNG，返回已校验的 12 字段预测。

    ``refiner`` 是可选的外部语义复核器（见 ``cadvec/external.py``），
    默认 None；传入时必须实现 ``refine(png_path, primitives) -> int``。
    """
    rgb, _, _ = load_rgb(png_path)
    layers = segment_colors(rgb, config)
    if not layers:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return PipelineResult(data, [], 0, {"reason": "no_foreground"})

    primitives = vectorize_all(
        layers, config,
        canvas_width=canvas_width, canvas_height=canvas_height,
    )
    primitives = _filter_tiny(
        primitives, canvas_width, canvas_height, config.min_primitive_length,
    )
    primitives = deduplicate(primitives)
    apply_semantics(
        primitives, prior, classifier,
        min_confidence=config.classifier_min_confidence,
    )
    refined = 0
    if refiner is not None:
        # 外部复核只改语义；实例聚合在其后执行，保证同一实例语义一致。
        refined = int(refiner.refine(png_path, primitives) or 0)
    assign_instances(primitives, config)

    data = assemble_output(primitives, width=canvas_width, height=canvas_height)
    validate_output(data)
    return PipelineResult(
        data=data,
        primitives=primitives,
        layer_count=len(layers),
        stats={
            "primitive_count": len(primitives),
            "layer_count": len(layers),
            "thing_count": sum(1 for item in primitives if 0 <= item.semantic_id <= 29),
            "ignored_count": sum(1 for item in primitives if item.semantic_id == 35),
            "external_refined": refined,
            "command_counts": {
                "line": sum(1 for item in primitives if item.command == COMMAND_LINE),
                "arc": sum(1 for item in primitives if item.command == COMMAND_ARC),
                "circle": sum(1 for item in primitives if item.command == COMMAND_CIRCLE),
                "ellipse": sum(1 for item in primitives if item.command == COMMAND_ELLIPSE),
            },
        },
    )
