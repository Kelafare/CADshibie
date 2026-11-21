"""YOLO-seg 实例分割 + 图像笔画矢量化。

实例分割提供“完整实例候选区域”；输出时不用分割掩码本身（它是凸包），
而是把区域内的图像笔画矢量化，保证几何与 GT 的 4 点折线一致。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import (
    IGNORE_CLASS,
    InferenceConfig,
    STUFF_CLASS_MAX,
    STUFF_CLASS_MIN,
)
from .dataset import component_scene
from .layers import load_palette
from .schema import assemble_output, empty_output, validate_output
from .templating import TemplateLibrary

YOLO_SEG_CACHE: dict[str, Any] = {}
TEMPLATE_CACHE: dict[str, Any] = {}


def _library(path):
    if path is None:
        return None
    key = str(Path(path).resolve())
    library = TEMPLATE_CACHE.get(key)
    if library is None:
        library = TemplateLibrary.load(path)
        if library is not None:
            TEMPLATE_CACHE[key] = library
    return library


def _model(path: str | Path):
    key = str(Path(path).resolve())
    model = YOLO_SEG_CACHE.get(key)
    if model is None:
        import os
        os.environ.setdefault("YOLO_OFFLINE", "1")
        from ultralytics import YOLO
        model = YOLO(str(path))
        YOLO_SEG_CACHE[key] = model
    return model


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, stroke=None, semantic=35, instance=-1, payload=None):
        if payload is not None:
            self.command = int(payload["command"])
            self.points = payload["points"]
            self.length = float(payload["length"])
            self.semantic_id = int(payload["semantic"])
            self.instance_id = int(instance)
            self.rgb = [int(v) for v in payload["rgb"]]
            self.layer_id = int(payload["layer"])
            self.width = float(payload["width"])
            return
        self.command = 0
        self.points = stroke.points[:4]
        self.length = float(stroke.length)
        self.semantic_id = int(semantic)
        self.instance_id = int(instance)
        self.rgb = [int(v) for v in stroke.rgb]
        self.layer_id = int(stroke.layer_id)
        self.width = float(stroke.width)


def run_image_segdet(
    png_path: str | Path,
    weights: str | Path,
    config: InferenceConfig,
    palette_path: str | Path,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    det_conf: float = 0.25,
    det_iou: float = 0.5,
    max_instances: int = 12,
    inside_fraction: float = 0.5,
    templates_path: str | Path | None = None,
    template_max_score: float = 3.0,
) -> dict[str, Any]:
    started = time.perf_counter()
    palette = load_palette(palette_path)
    scene = component_scene(png_path, palette, config, canvas_width, canvas_height)
    if not scene.strokes or not scene.layers:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return {"data": data, "stats": {"primitive_count": 0, "instances": 0,
                                        "seconds": round(time.perf_counter() - started, 3),
                                        "command_counts": {"line": 0, "arc": 0,
                                                           "circle": 0, "ellipse": 0}}}
    height, width = scene.layers[0].mask.shape
    from scipy import ndimage
    stroke_mask = np.zeros((height, width), dtype=bool)
    for layer in scene.layers:
        stroke_mask |= layer.mask
    distance_transform = ndimage.distance_transform_edt(~stroke_mask)
    canvas_to_pixel = width / max(canvas_width, 1)
    library = _library(templates_path)
    model = _model(weights)
    result = model.predict(source=str(png_path), imgsz=640, conf=det_conf,
                           iou=det_iou, verbose=False, device=0, retina_masks=False)[0]
    detections = []
    if result.masks is not None and result.boxes is not None and len(result.masks):
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()
        polygons = result.masks.xy
        for polygon, class_id, confidence in zip(polygons, classes, confidences):
            if polygon is None or len(polygon) < 3:
                continue
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
            detections.append((float(confidence), int(class_id), mask.astype(bool)))
    detections.sort(key=lambda item: -item[0])

    scale_x = width / max(canvas_width, 1)
    scale_y = height / max(canvas_height, 1)
    claimed = [False] * len(scene.strokes)
    groups = []
    stuff_points = []
    for confidence, class_id, mask in detections:
        if class_id == IGNORE_CLASS:
            continue
        members = []
        for index, stroke in enumerate(scene.strokes):
            if claimed[index]:
                continue
            points = stroke.points
            inside = 0
            for x, y in points:
                column = int(min(max(x * scale_x, 0), width - 1))
                row = int(min(max(y * scale_y, 0), height - 1))
                if mask[row, column]:
                    inside += 1
            if inside / max(len(points), 1) >= inside_fraction:
                members.append(index)
        if not members:
            continue
        if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX:
            for index in members:
                claimed[index] = True
            stuff_points.extend((class_id, index) for index in members)
            continue
        for index in members:
            claimed[index] = True
        groups.append((confidence, class_id, members))

    if max_instances > 0:
        groups = groups[:max_instances]
    emitted = []
    instance_id = 0
    for confidence, class_id, members in groups:
        fit = None
        if library is not None:
            points = np.asarray([point for index in members for point in scene.strokes[index].points])
            local_mask = np.zeros((height, width), dtype=np.uint8)
            for index in members:
                pixels = np.round(
                    np.asarray(scene.strokes[index].points) * canvas_to_pixel
                ).astype(np.int32)
                cv2.polylines(local_mask, [pixels.reshape(-1, 1, 2)], False, 1,
                              thickness=2)
            bounds = np.asarray([point for index in members for point in scene.strokes[index].points])
            if len(points) >= 8 and len(bounds):
                box_scale = max(float(np.ptp(bounds[:, 0])), float(np.ptp(bounds[:, 1])), 0.5)
                center = (float(bounds[:, 0].mean()), float(bounds[:, 1].mean()))
                fit = library.fit(
                    points, distance_transform, canvas_to_pixel, top_k=8,
                    class_filter=class_id, center=center, scale_override=box_scale,
                    stroke_mask=local_mask.astype(bool),
                )
        if fit is not None and fit.class_id <= STUFF_CLASS_MAX and fit.score <= template_max_score:
            for payload in library.instantiate(fit, np.zeros((0, 2))):
                emitted.append(_Primitive(payload=payload, instance=instance_id))
        else:
            for index in members:
                emitted.append(_Primitive(scene.strokes[index], class_id, instance_id))
        instance_id += 1
    for class_id, index in stuff_points:
        emitted.append(_Primitive(scene.strokes[index], int(class_id), -1))

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
