"""模板拟合推理：GAT 候选分组 → 模板检索 → 距离变换对齐 → 完整实例输出。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from .config import IGNORE_CLASS, InferenceConfig, STUFF_CLASS_MAX, STUFF_CLASS_MIN
from .dataset import component_scene
from .image_features import load_rgb, node_image_features
from .layers import load_palette
from .pipeline_gat import MAX_NODES, MODEL_CACHE, _load_model, _union_groups
from .pipeline_v2 import load_class_policy, load_color_prior
from .primitive_graph import build_graph, polar_edge_features
from .schema import assemble_output, empty_output, validate_output
from .templating import TemplateLibrary

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

TEMPLATE_CACHE: dict[str, TemplateLibrary] = {}


def _library(path: str | Path) -> TemplateLibrary | None:
    key = str(Path(path).resolve())
    library = TEMPLATE_CACHE.get(key)
    if library is None:
        library = TemplateLibrary.load(path)
        if library is not None:
            TEMPLATE_CACHE[key] = library
    return library


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


def _analyze(png_path, weights_path, device_name, config, palette_path, *,
             canvas_width, canvas_height, confidence, edge_threshold,
             color_prior_path, color_prior_weight, class_policy_path):
    device = torch.device(device_name)
    model = _load_model(weights_path, device)
    palette = load_palette(palette_path)
    scene = component_scene(png_path, palette, config, canvas_width, canvas_height)
    strokes = scene.strokes
    keep = list(range(len(strokes)))
    if len(strokes) > MAX_NODES:
        keep = sorted(range(len(strokes)), key=lambda index: -strokes[index].length)[:MAX_NODES]
        keep.sort()
        strokes = [strokes[index] for index in keep]
    if not strokes:
        return scene, [], np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32), {}, []
    features, edge_index, edge_features, adjacency = build_graph(strokes)
    edge_features = polar_edge_features(features, edge_index, edge_features)
    if int(model.encoder[0].in_features) > features.shape[1]:
        image = load_rgb(png_path)
        centers = features[:, :2].astype(np.float64) * 100.0
        extra = node_image_features(image, centers, device=str(device))
        features = np.concatenate([features, extra.astype(np.float32)], axis=1)
    with torch.no_grad():
        logits, edge_logits = model(
            torch.from_numpy(features).float().to(device),
            torch.from_numpy(adjacency).to(device),
            torch.from_numpy(edge_index).long().to(device),
            torch.from_numpy(edge_features).float().to(device),
        )
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        edge_prob = torch.sigmoid(edge_logits.float()).cpu().numpy()
    if color_prior_weight > 0:
        table = load_color_prior(color_prior_path)
        if table:
            adjusted = probabilities.astype(np.float64).copy()
            for index, stroke in enumerate(strokes):
                rgb = tuple(int(v) for v in stroke.rgb)
                prior = table.get(rgb)
                if prior is None and max(rgb) - min(rgb) <= 8:
                    gray = sum(rgb) / 3.0
                    candidates = [(k, v) for k, v in table.items() if max(k) - min(k) <= 8]
                    if candidates:
                        prior = min(candidates, key=lambda item: abs(sum(item[0]) / 3.0 - gray))[1]
                if prior is None:
                    continue
                adjusted[index, :35] *= np.power(prior + 1e-6, color_prior_weight)
                total = adjusted[index].sum()
                if total > 0:
                    adjusted[index] /= total
            probabilities = adjusted.astype(np.float32)
    semantic = probabilities.argmax(axis=1)
    node_confidence = probabilities.max(axis=1)
    policy = load_class_policy(class_policy_path)
    if policy:
        thresholds = policy["thresholds"]
        suppressed = policy["suppress"]
        for index in range(len(strokes)):
            class_id = int(semantic[index])
            if class_id in suppressed or node_confidence[index] < max(
                confidence, float(thresholds[class_id])
            ):
                semantic[index] = IGNORE_CLASS
    else:
        semantic = np.where(node_confidence >= confidence, semantic, IGNORE_CLASS)
    groups = _union_groups(len(strokes), edge_index, edge_prob, semantic, edge_threshold)
    return scene, strokes, semantic, node_confidence, groups, keep


def run_image_tpl(
    png_path: str | Path,
    gat_weights: str | Path,
    template_path: str | Path,
    device_name: str,
    config: InferenceConfig,
    palette_path: str | Path,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    confidence: float = 0.3,
    edge_threshold: float = 0.45,
    color_prior_path: str | Path | None = None,
    color_prior_weight: float = 0.5,
    class_policy_path: str | Path | None = None,
    max_instances: int = 12,
    max_distance_px: float = 3.0,
    class_correction: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    library = _library(template_path)
    scene, strokes, semantic, node_confidence, groups, keep = _analyze(
        png_path, gat_weights, device_name, config, palette_path,
        canvas_width=canvas_width, canvas_height=canvas_height,
        confidence=confidence, edge_threshold=edge_threshold,
        color_prior_path=color_prior_path, color_prior_weight=color_prior_weight,
        class_policy_path=class_policy_path,
    )
    if not strokes or library is None:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return {"data": data, "stats": {"primitive_count": 0, "instances": 0,
                                        "seconds": round(time.perf_counter() - started, 3),
                                        "command_counts": {"line": 0, "arc": 0,
                                                           "circle": 0, "ellipse": 0}}}

    # 图像笔画距离变换（像素）。
    stroke_mask = np.zeros_like(scene.layers[0].mask, dtype=bool) if scene.layers else None
    if stroke_mask is not None:
        for layer in scene.layers:
            stroke_mask |= layer.mask
        distance_transform = ndimage.distance_transform_edt(~stroke_mask)
    else:
        distance_transform = np.zeros((10, 10), dtype=np.float32)
    canvas_to_pixel = stroke_mask.shape[1] / canvas_width if stroke_mask is not None else 10.0

    # 候选对象：颜色层连通分量（对象级），映射到 GAT 使用的笔画下标。
    keep_position = {old: new for new, old in enumerate(keep)}
    candidates = []
    for hypothesis in scene.hypotheses:
        mapped = [keep_position[index] for index in hypothesis.indices if index in keep_position]
        if mapped:
            candidates.append(mapped)
    # 补充 GAT 合并出的跨分量组（但只保留多笔画组）。
    for indices in groups.values():
        if len(indices) >= 2:
            candidates.append(list(indices))

    records = []
    for indices in candidates:
        classes = semantic[indices]
        classes = classes[classes != IGNORE_CLASS]
        if len(classes) == 0:
            continue
        class_id = int(np.bincount(classes).argmax())
        members = [index for index in indices if semantic[index] == class_id]
        points = np.asarray([point for index in members for point in strokes[index].points])
        score = float(node_confidence[members].mean())
        if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX:
            for index in members:
                records.append(("stuff", class_id, strokes[index].points, strokes[index],
                                score, None))
            continue
        if score < 0.55:
            continue
        fit = library.fit(points, distance_transform, canvas_to_pixel,
                          class_filter=class_id, top_k=6)
        if class_correction and (fit is None or fit.score > max_distance_px + 0.5):
            alternative = library.fit(points, distance_transform, canvas_to_pixel,
                                      class_filter=None, top_k=10)
            if alternative is not None and (
                fit is None or alternative.score < fit.score - 0.2
            ):
                fit = alternative
        if fit is not None and fit.class_id >= STUFF_CLASS_MIN:
            fit = None
        if fit is not None and fit.score <= max_distance_px:
            records.append(("fit", fit.class_id, None, None, score, fit))

    records.sort(key=lambda item: -item[4])
    things = [item for item in records if item[0] != "stuff"]
    stuff = [item for item in records if item[0] == "stuff"]
    # 拟合实例按掩码 IoU 去重，保留置信度高的。
    kept = []
    kept_masks = []
    for item in things:
        fit = item[5]
        mask = np.zeros((200, 200), dtype=np.uint8)
        import cv2
        for payload in library.instantiate(fit, np.zeros((0, 2))):
            points = np.round(np.asarray(payload["points"]) * 2.0).astype(np.int32)
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, thickness=1)
        mask = mask.astype(bool)
        duplicate = False
        for existing in kept_masks:
            intersection = int(np.count_nonzero(mask & existing))
            if not intersection:
                continue
            union = int(np.count_nonzero(mask | existing))
            if intersection / max(union, 1) > 0.5:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(item)
        kept_masks.append(mask)
    things = kept
    if max_instances > 0:
        things = things[:max_instances]
    emitted = []
    instance_id = 0
    for kind, class_id, _points, stroke, _score, fit in things:
        if kind == "fit":
            for payload in library.instantiate(fit, np.zeros((0, 2))):
                emitted.append(_Primitive(payload, instance_id))
        else:
            emitted.append(_Primitive({
                "command": 0, "points": stroke.points[:4],
                "length": float(stroke.length), "width": float(stroke.width),
                "rgb": list(stroke.rgb), "layer": int(stroke.layer_id),
                "semantic": int(class_id),
            }, instance_id))
        instance_id += 1
    for kind, class_id, _points, stroke, _score, _fit in stuff:
        emitted.append(_Primitive({
            "command": 0, "points": stroke.points[:4],
            "length": float(stroke.length), "width": float(stroke.width),
            "rgb": list(stroke.rgb), "layer": int(stroke.layer_id),
            "semantic": int(class_id),
        }, -1))

    data = assemble_output(emitted, width=canvas_width, height=canvas_height)
    validate_output(data)
    return {
        "data": data,
        "stats": {
            "primitive_count": len(emitted),
            "instances": instance_id,
            "seconds": round(time.perf_counter() - started, 3),
            "command_counts": {"line": len(emitted), "arc": 0, "circle": 0, "ellipse": 0},
        },
    }
