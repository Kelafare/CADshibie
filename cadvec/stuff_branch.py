"""全图 Stuff 语义分支：512 重叠分块推理，按类输出 instance=-1 的四点段。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .layers import load_rgba
from .roi_ink import vectorize_mask
from .schema import assemble_output
from .stuff_unet import StuffUNet

CLASSES = tuple(range(30, 35))
RASTER = 1000
TILE = 512
STRIDE = 384


def _starts(size: int, tile: int, stride: int) -> list[int]:
    values = list(range(0, max(size - tile, 0) + 1, stride))
    if not values or values[-1] != size - tile:
        values.append(size - tile)
    return values


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")


def _to_primitives(data: dict):
    primitives = []
    for index in range(len(data["commands"])):
        item = _Primitive()
        item.command = int(data["commands"][index])
        values = data["args"][index]
        item.points = [(float(values[k]), float(values[k + 1]))
                       for k in range(0, 8, 2)]
        item.length = float(data["lengths"][index])
        item.semantic_id = int(data["semanticIds"][index])
        item.instance_id = int(data["instanceIds"][index])
        item.rgb = [int(v) for v in data["rgb"][index]]
        item.layer_id = int(data["layerIds"][index])
        item.width = float(data["widths"][index])
        primitives.append(item)
    return primitives


def append_stuff(data: dict, stuff_items, width: int, height: int) -> dict:
    replaced = {int(class_id) for class_id, _ in stuff_items}
    primitives = [item for item in _to_primitives(data)
                  if item.semantic_id not in replaced]
    for class_id, segments in stuff_items:
        for points, length in segments:
            item = _Primitive()
            item.command = 0
            item.points = [(float(x), float(y)) for x, y in points]
            item.length = float(length)
            item.semantic_id = int(class_id)
            item.instance_id = -1
            item.rgb = [0, 0, 0]
            item.layer_id = 0
            item.width = 1.0
            primitives.append(item)
    return assemble_output(primitives, width=width, height=height)


class StuffBranch:
    def __init__(self, model: StuffUNet, device: torch.device,
                 thresholds: dict[int, float], tile: int = TILE,
                 stride: int = STRIDE):
        self.model = model
        self.device = device
        self.thresholds = thresholds
        self.tile = int(tile)
        self.stride = int(stride)

    @classmethod
    def load(cls, weights: str | Path, thresholds_path: str | Path | None = None,
             device: str | None = "auto") -> "StuffBranch | None":
        if not str(weights).strip():
            return None
        path = Path(weights)
        if not path.exists():
            return None
        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = StuffUNet(**checkpoint.get("model_kwargs", {}))
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device).eval()
        thresholds = {}
        if thresholds_path and Path(thresholds_path).exists():
            try:
                payload = json.loads(Path(thresholds_path).read_text(encoding="utf-8"))
                thresholds = {
                    int(key): (dict(value) if isinstance(value, dict)
                               else float(value))
                    for key, value in payload.items()
                }
            except (OSError, ValueError):
                thresholds = {}
        return cls(model, torch.device(device), thresholds)

    @torch.no_grad()
    def probability(self, png_path: str | Path) -> np.ndarray:
        rgb, alpha = load_rgba(str(png_path))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        if image.shape[:2] != (RASTER, RASTER):
            image = cv2.resize(image, (RASTER, RASTER),
                               interpolation=cv2.INTER_AREA)
        positions = [(top, left)
                     for top in _starts(RASTER, self.tile, self.stride)
                     for left in _starts(RASTER, self.tile, self.stride)]
        accum = np.zeros((5, RASTER, RASTER), dtype=np.float32)
        counts = np.zeros((RASTER, RASTER), dtype=np.float32)
        batch, meta = [], []

        def flush():
            nonlocal batch, meta
            if not batch:
                return
            tensor = torch.from_numpy(np.stack(batch)).to(self.device)
            tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
            probability = torch.sigmoid(self.model(tensor)).float().cpu().numpy()
            for (top, left), value in zip(meta, probability):
                accum[:, top:top + self.tile, left:left + self.tile] += value
                counts[top:top + self.tile, left:left + self.tile] += 1.0
            batch, meta = [], []

        for top, left in positions:
            batch.append(image[top:top + self.tile, left:left + self.tile])
            meta.append((top, left))
            if len(batch) >= 9:
                flush()
        flush()
        accum /= np.maximum(counts, 1.0)[None]
        return accum

    def infer(self, png_path: str | Path) -> list[tuple[int, list]]:
        accum = self.probability(png_path)
        results = []
        for index, class_id in enumerate(CLASSES):
            spec = self.thresholds.get(class_id, 0.5)
            if isinstance(spec, dict):
                threshold = float(spec.get("threshold", 0.5))
                min_area = int(spec.get("min_area", 0))
            else:
                threshold = float(spec)
                min_area = 0
            probability = accum[index]
            if float(probability.max()) < threshold:
                continue
            if int((probability > threshold).sum()) < min_area:
                continue
            segments = vectorize_mask(
                probability, 0, 0, RASTER, RASTER, 10.0, 10.0, threshold)
            if segments:
                results.append((class_id, segments))
        return results
