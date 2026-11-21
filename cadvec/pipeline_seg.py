"""dense segmentation 推理：像素分类 → 实例掩码 → 骨架折线 → 12 字段 JSON。"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import IGNORE_CLASS, InferenceConfig, STUFF_CLASS_MAX, STUFF_CLASS_MIN
from .schema import assemble_output, empty_output, validate_output
from .segmentation import BACKGROUND, instance_segments, vectorize_mask

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


_MODEL_CACHE: dict[tuple[str, str], object] = {}
_MODEL_LOCK = None


def _load_model(weights_path: str | Path, device: str):
    from .segmodel import UNet

    key = (str(Path(weights_path).resolve()), str(device))
    model = _MODEL_CACHE.get(key)
    if model is not None:
        return model
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = UNet(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    _MODEL_CACHE[key] = model
    return model


def _load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    composite = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


def _axis_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def predict_probabilities(model, image: np.ndarray, device, tile: int = 512,
                          overlap: int = 64, batch: int = 2) -> np.ndarray:
    """分块推理并累加 softmax 概率（H×W×37）。"""
    height, width = image.shape[:2]
    classes = model.head.out_channels
    if height <= 768 and width <= 768:
        tensor = image_to_tensor(image)[None].to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(tensor)
        return torch.softmax(logits.float(), dim=1)[0].cpu().numpy().transpose(1, 2, 0)

    stride = max(8, tile - overlap)
    accumulator = np.zeros((height, width, classes), dtype=np.float32)
    counter = np.zeros((height, width, 1), dtype=np.float32)
    jobs = []
    for y0 in _axis_positions(height, tile, stride):
        for x0 in _axis_positions(width, tile, stride):
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            patch = image[y0:y1, x0:x1]
            if patch.shape[0] < tile or patch.shape[1] < tile:
                padded = np.full((tile, tile, 3), 255, dtype=np.uint8)
                padded[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded
            jobs.append((image_to_tensor(patch), y0, x0, y1 - y0, x1 - x0))
    for start in range(0, len(jobs), batch):
        block = jobs[start:start + batch]
        tensor = torch.stack([item[0] for item in block]).to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(tensor)
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
        for index, (_, y0, x0, valid_h, valid_w) in enumerate(block):
            tile_prob = probabilities[index][:, :valid_h, :valid_w].transpose(1, 2, 0)
            accumulator[y0:y0 + valid_h, x0:x0 + valid_w] += tile_prob
            counter[y0:y0 + valid_h, x0:x0 + valid_w] += 1.0
    return accumulator / np.maximum(counter, 1.0)


def image_to_tensor(image: np.ndarray):
    return torch.from_numpy(
        (image.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)
    )


def _polylines_to_primitives(polylines, class_id: int, instance_id: int,
                             rgb, layer_id: int) -> list[dict]:
    primitives = []
    for points in polylines:
        length = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
        primitives.append({
            "command": 0,
            "points": points,
            "length": length,
            "width": 0.1,
            "rgb": list(rgb),
            "layer": layer_id,
            "semantic": class_id,
            "instance": instance_id,
        })
    return primitives


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, payload: dict):
        self.command = payload["command"]
        self.points = payload["points"]
        self.length = payload["length"]
        self.semantic_id = payload["semantic"]
        self.instance_id = payload["instance"]
        self.rgb = payload["rgb"]
        self.layer_id = payload["layer"]
        self.width = payload["width"]


def run_image_seg(
    png_path: str | Path,
    weights_path: str | Path,
    device_name: str,
    config: InferenceConfig,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
    threshold: float = 0.30,
    min_pixels: int = 12,
    max_instances: int = 12,
) -> dict[str, Any]:
    """单图 dense 分割推理，返回 12 字段 JSON。"""
    started = time.perf_counter()
    device = torch.device(device_name)
    model = _load_model(weights_path, device)
    image = _load_rgb(png_path)
    height, width = image.shape[:2]
    probabilities = predict_probabilities(model, image, device)
    confidence = probabilities.max(axis=2)
    label = probabilities.argmax(axis=2).astype(np.uint8)
    label[confidence < threshold] = BACKGROUND

    scale_x = width / max(canvas_width, 1)
    scale_y = height / max(canvas_height, 1)
    segments = instance_segments(label)
    records = []
    for class_id, instance_index, mask in segments:
        if int(mask.sum()) < min_pixels:
            continue
        rows, columns = np.nonzero(mask)
        if rows.size == 0:
            continue
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(columns.min()), int(columns.max()) + 1
        polylines = vectorize_mask(
            mask[y0:y1, x0:x1], scale_x, scale_y, offset=(x0, y0),
        )
        if not polylines:
            continue
        score = float(confidence[mask].mean()) if mask.any() else 0.0
        records.append((score, class_id, instance_index, polylines))
    if max_instances > 0:
        things = [record for record in records if record[1] < STUFF_CLASS_MIN]
        stuff = [record for record in records if record[1] >= STUFF_CLASS_MIN]
        things.sort(key=lambda record: -record[0])
        records = things[:max_instances] + stuff

    emitted = []
    next_instance = 0
    for score, class_id, instance_index, polylines in records:
        if STUFF_CLASS_MIN <= class_id <= STUFF_CLASS_MAX:
            instance = -1
        else:
            instance = next_instance
            next_instance += 1
        emitted.extend(_polylines_to_primitives(polylines, class_id, instance, (0, 0, 0), 1))

    data = assemble_output([_Primitive(item) for item in emitted],
                           width=canvas_width, height=canvas_height)
    validate_output(data)
    return {
        "data": data,
        "stats": {
            "primitive_count": len(emitted),
            "instances": next_instance,
            "seconds": round(time.perf_counter() - started, 3),
            "command_counts": {"line": len(emitted), "arc": 0, "circle": 0, "ellipse": 0},
        },
    }
