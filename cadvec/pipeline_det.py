"""检测器 + 模板拟合：YOLO 给出完整实例框，模板检索+对齐补全几何。"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy import ndimage

from .config import IGNORE_CLASS, InferenceConfig, STUFF_CLASS_MAX, STUFF_CLASS_MIN
from .dataset import component_scene
from .layers import load_palette
from .pipeline_tpl import TEMPLATE_CACHE, _library
from .schema import assemble_output, empty_output, validate_output

YOLO_CACHE: dict[str, Any] = {}
COLOR_CACHE: dict[str, dict[int, list[list[int]]]] = {}
COLOR_PATH = Path(__file__).resolve().parents[1] / "weights" / "class_colors.json"


def _class_colors() -> dict[int, list[list[int]]]:
    key = str(COLOR_PATH)
    table = COLOR_CACHE.get(key)
    if table is None:
        table = {}
        if COLOR_PATH.exists():
            import json
            try:
                payload = json.loads(COLOR_PATH.read_text(encoding="utf-8"))
                for class_id, entries in payload.items():
                    table[int(class_id)] = [entry[0] for entry in entries[:8]]
            except Exception:  # noqa: BLE001
                table = {}
        COLOR_CACHE[key] = table
    return table


def _color_matches(rgb, palette: list[list[int]], tolerance: float = 40.0) -> bool:
    for reference in palette:
        if sum((int(a) - int(b)) ** 2 for a, b in zip(rgb, reference)) <= tolerance ** 2:
            return True
    return False


def _yolo(path: str | Path):
    key = str(Path(path).resolve())
    model = YOLO_CACHE.get(key)
    if model is None:
        import os
        os.environ.setdefault("YOLO_OFFLINE", "1")
        import ultralytics.utils.checks as ultra_checks
        # 离线评测：禁用 ultralytics 的联网检测/字体下载。
        ultra_checks.check_online = lambda *args, **kwargs: False
        try:
            import ultralytics.utils as ultra_utils
            ultra_utils.is_online = lambda *args, **kwargs: False
        except Exception:  # noqa: BLE001
            pass
        from ultralytics import YOLO
        model = YOLO(str(path))
        YOLO_CACHE[key] = model
    return model


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, payload: dict, instance_id: int):
        self.command = int(payload["command"])
        self.points = payload["points"]
        self.length = float(payload["length"])
        self.semantic_id = int(payload["semantic"])
        self.instance_id = int(instance_id)
        self.rgb = [int(v) for v in payload["rgb"]]
        self.layer_id = int(payload["layer"])
        self.width = float(payload["width"])


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    union = int(np.count_nonzero(first | second))
    return intersection / max(union, 1)


def run_image_det(
    png_path: str | Path,
    yolo_weights: str | Path,
    template_path: str | Path,
    config: InferenceConfig,
    palette_path: str | Path,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    det_conf: float = 0.25,
    det_iou: float = 0.5,
    max_distance_px: float = 3.0,
    max_instances: int = 12,
    class_correction: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    library = _library(template_path)
    palette = load_palette(palette_path)
    scene = component_scene(png_path, palette, config, canvas_width, canvas_height)
    if library is None:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return {"data": data, "stats": {"primitive_count": 0, "instances": 0,
                                        "seconds": round(time.perf_counter() - started, 3),
                                        "command_counts": {"line": 0, "arc": 0,
                                                           "circle": 0, "ellipse": 0}}}
    stroke_mask = np.zeros_like(scene.layers[0].mask, dtype=bool) if scene.layers else None
    if stroke_mask is None:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return {"data": data, "stats": {"primitive_count": 0, "instances": 0,
                                        "seconds": round(time.perf_counter() - started, 3),
                                        "command_counts": {"line": 0, "arc": 0,
                                                           "circle": 0, "ellipse": 0}}}
    for layer in scene.layers:
        stroke_mask |= layer.mask
    distance_transform = ndimage.distance_transform_edt(~stroke_mask)
    height, width = stroke_mask.shape
    canvas_to_pixel = width / canvas_width

    model = _yolo(yolo_weights)
    result = model.predict(source=str(png_path), imgsz=640, conf=det_conf,
                           iou=det_iou, verbose=False, device=0)[0]
    boxes = result.boxes
    detections = []
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), class_id, confidence in zip(xyxy, classes, confidences):
            detections.append((
                float(x0) / width * canvas_width, float(y0) / height * canvas_height,
                float(x1) / width * canvas_width, float(y1) / height * canvas_height,
                int(class_id), float(confidence),
            ))

    # 候选笔画：检测框（外扩 20%）内的笔画点。
    def strokes_in_box(box, class_id: int | None = None):
        x0, y0, x1, y1 = box
        margin_x = 0.2 * (x1 - x0) + 0.5
        margin_y = 0.2 * (y1 - y0) + 0.5
        colors = _class_colors().get(class_id, []) if class_id is not None else []
        filtered = []
        all_points = []
        for stroke in scene.strokes:
            sx0, sy0, sx1, sy1 = stroke.bbox()
            if sx1 < x0 - margin_x or sx0 > x1 + margin_x:
                continue
            if sy1 < y0 - margin_y or sy0 > y1 + margin_y:
                continue
            all_points.extend(stroke.points)
            if colors and _color_matches(stroke.rgb, colors):
                filtered.extend(stroke.points)
        chosen = filtered if len(filtered) >= 8 else all_points
        return np.asarray(chosen, dtype=np.float64) if chosen else np.zeros((0, 2))

    fitted = []
    stuff_records = []
    for x0, y0, x1, y1, class_id, confidence in detections:
        box = (x0, y0, x1, y1)
        points = strokes_in_box(box, class_id)
        if class_id >= STUFF_CLASS_MIN:
            if len(points) >= 8:
                stuff_records.append((class_id, points, confidence))
            continue
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        scale_override = max(x1 - x0, y1 - y0)
        if len(points) < 6:
            corners = np.asarray([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1],
            ], dtype=np.float64)
            points = corners
        fit = library.fit(points, distance_transform, canvas_to_pixel,
                          class_filter=class_id, top_k=6,
                          center=center, scale_override=scale_override)
        if class_correction and (fit is None or fit.score > max_distance_px + 0.5):
            alternative = library.fit(points, distance_transform, canvas_to_pixel,
                                      class_filter=None, top_k=10,
                                      center=center, scale_override=scale_override)
            if alternative is not None and alternative.class_id < STUFF_CLASS_MIN:
                if fit is None or alternative.score < fit.score - 0.3:
                    fit = alternative
        if fit is not None and fit.class_id < STUFF_CLASS_MIN and fit.score <= max_distance_px:
            fitted.append((confidence * (1.0 - min(fit.score / max(max_distance_px, 1e-6), 1.0)),
                           fit, class_id))

    fitted.sort(key=lambda item: -item[0])
    kept = []
    kept_masks = []
    for score, fit, _ in fitted:
        mask = np.zeros((200, 200), dtype=np.uint8)
        for payload in library.instantiate(fit, np.zeros((0, 2))):
            points = np.round(np.asarray(payload["points"]) * 2.0).astype(np.int32)
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, thickness=1)
        mask = mask.astype(bool)
        if any(_mask_iou(mask, existing) > 0.5 for existing in kept_masks):
            continue
        kept.append(fit)
        kept_masks.append(mask)
        if max_instances > 0 and len(kept) >= max_instances:
            break

    emitted = []
    instance_id = 0
    for fit in kept:
        for payload in library.instantiate(fit, np.zeros((0, 2))):
            emitted.append(_Primitive(payload, instance_id))
        instance_id += 1
    # Stuff（墙/栏杆等）由像素管线负责；检测引擎只输出 Thing。

    data = assemble_output(emitted, width=canvas_width, height=canvas_height)
    validate_output(data)
    return {
        "data": data,
        "stats": {
            "primitive_count": len(emitted),
            "instances": instance_id,
            "detections": len(detections),
            "seconds": round(time.perf_counter() - started, 3),
            "command_counts": {"line": len(emitted), "arc": 0, "circle": 0, "ellipse": 0},
        },
    }
