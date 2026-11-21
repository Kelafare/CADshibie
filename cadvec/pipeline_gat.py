"""图元级全景检测推理：图元图 → GAT → 语义+邻接 → 实例 → 12 字段 JSON。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import IGNORE_CLASS, InferenceConfig, STUFF_CLASS_MAX, STUFF_CLASS_MIN
from .dataset import component_scene
from .gat_model import PrimitiveGAT
from .layers import load_palette
from .pipeline_v2 import load_color_prior
from .primitive_graph import build_graph, polar_edge_features
from .schema import assemble_output, empty_output, validate_output
from .strokes import Stroke

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

MODEL_CACHE: dict[tuple[str, str], Any] = {}
MAX_NODES = 1200


def _load_model(weights_path: str | Path, device: str):
    key = (str(Path(weights_path).resolve()), str(device))
    model = MODEL_CACHE.get(key)
    if model is not None:
        return model
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = PrimitiveGAT(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    MODEL_CACHE[key] = model
    return model


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, stroke: Stroke, semantic: int, instance: int):
        self.command = 0
        self.points = stroke.points[:4]
        self.length = float(stroke.length)
        self.semantic_id = int(semantic)
        self.instance_id = int(instance)
        self.rgb = [int(value) for value in stroke.rgb]
        self.layer_id = int(stroke.layer_id)
        self.width = float(stroke.width)


def _union_groups(count: int, edge_index: np.ndarray, edge_prob: np.ndarray,
                  semantic: np.ndarray, threshold: float) -> dict[int, list[int]]:
    parent = list(range(count))
    sizes = [1] * count

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for position in range(edge_index.shape[1]):
        if edge_prob[position] < threshold:
            continue
        first = int(edge_index[0, position])
        second = int(edge_index[1, position])
        if semantic[first] == IGNORE_CLASS or semantic[first] != semantic[second]:
            continue
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            continue
        if sizes[first_root] + sizes[second_root] > 120:
            continue
        if sizes[first_root] < sizes[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        sizes[first_root] += sizes[second_root]
    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return groups


def run_image_gat(
    png_path: str | Path,
    weights_path: str | Path,
    device_name: str,
    config: InferenceConfig,
    palette_path: str | Path,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    confidence: float = 0.30,
    edge_threshold: float = 0.5,
    max_instances: int = 20,
    min_group: int = 1,
    color_prior_path: str | Path | None = None,
    color_prior_weight: float = 0.0,
    class_policy_path: str | Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    device = torch.device(device_name)
    model = _load_model(weights_path, device)
    palette = load_palette(palette_path)
    scene = component_scene(png_path, palette, config, canvas_width, canvas_height)
    strokes = scene.strokes
    if len(strokes) > MAX_NODES:
        keep = sorted(range(len(strokes)), key=lambda index: -strokes[index].length)[:MAX_NODES]
        keep.sort()
        strokes = [strokes[index] for index in keep]
    if not strokes:
        data = empty_output(canvas_width, canvas_height)
        validate_output(data)
        return {"data": data, "stats": {"primitive_count": 0, "instances": 0,
                                        "seconds": round(time.perf_counter() - started, 3),
                                        "command_counts": {"line": 0, "arc": 0,
                                                           "circle": 0, "ellipse": 0}}}

    features, edge_index, edge_features, adjacency = build_graph(strokes)
    edge_features = polar_edge_features(features, edge_index, edge_features)
    in_dim = int(model.encoder[0].in_features)
    if in_dim > features.shape[1]:
        from .image_features import load_rgb, node_image_features
        image = load_rgb(png_path)
        centers = features[:, :2].astype(np.float64) * 100.0
        extra = node_image_features(image, centers, device=str(device))
        features = np.concatenate([features, extra.astype(np.float32)], axis=1)
    tensor_features = torch.from_numpy(features).float().to(device)
    tensor_adjacency = torch.from_numpy(adjacency).to(device)
    tensor_edges = torch.from_numpy(edge_index).long().to(device)
    tensor_edge_features = torch.from_numpy(edge_features).float().to(device)
    with torch.no_grad():
        logits, edge_logits = model(tensor_features, tensor_adjacency,
                                    tensor_edges, tensor_edge_features)
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        edge_prob = torch.sigmoid(edge_logits.float()).cpu().numpy()

    if color_prior_weight > 0.0:
        prior_table = load_color_prior(color_prior_path)
        if prior_table:
            adjusted = probabilities.astype(np.float64).copy()
            for index, stroke in enumerate(strokes):
                rgb = tuple(int(value) for value in stroke.rgb)
                prior = prior_table.get(rgb)
                if prior is None and max(rgb) - min(rgb) <= 8:
                    gray = sum(rgb) / 3.0
                    candidates = [(key, value) for key, value in prior_table.items()
                                  if max(key) - min(key) <= 8]
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

    from .pipeline_v2 import load_class_policy
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
    emitted_primitives: list[_Primitive] = []
    thing_records = []
    stuff_records = []
    for indices in groups.values():
        classes = semantic[indices]
        classes = classes[classes != IGNORE_CLASS]
        if len(classes) == 0:
            continue
        class_id = int(np.bincount(classes).argmax())
        members = [index for index in indices if semantic[index] == class_id]
        if len(members) < min_group:
            continue
        score = float(node_confidence[members].mean())
        if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX:
            stuff_records.extend(members)
        else:
            thing_records.append((score, class_id, members))

    thing_records.sort(key=lambda item: -item[0])
    if max_instances > 0:
        thing_records = thing_records[:max_instances]

    instance_id = 0
    for _score, class_id, members in thing_records:
        for index in members:
            emitted_primitives.append(_Primitive(strokes[index], class_id, instance_id))
        instance_id += 1
    for index in stuff_records:
        emitted_primitives.append(_Primitive(strokes[index], int(semantic[index]), -1))

    data = assemble_output(emitted_primitives, width=canvas_width, height=canvas_height)
    validate_output(data)
    return {
        "data": data,
        "stats": {
            "primitive_count": len(emitted_primitives),
            "instances": instance_id,
            "seconds": round(time.perf_counter() - started, 3),
            "command_counts": {"line": len(emitted_primitives), "arc": 0,
                               "circle": 0, "ellipse": 0},
        },
    }
