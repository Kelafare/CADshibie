"""按类别合并两个引擎的预测（hybrid engine）。

默认：v2/v3 像素管线作为主引擎，GAT 图元引擎只接管类 8（盲窗）——两个
验证半集都显示 GAT 在 class 8 上 PQ 更高，其余类别 v3 更好。
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from .schema import assemble_output, validate_output

RASTER = 400

_FIELDS = (
    "commands", "args", "lengths", "semanticIds", "instanceIds",
    "rgb", "layerIds", "widths",
)


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, data: dict[str, Any], index: int, instance_offset: int = 0):
        self.command = int(data["commands"][index])
        points = data["args"][index]
        self.points = [
            (float(points[position]), float(points[position + 1]))
            for position in range(0, 8, 2)
        ]
        self.length = float(data["lengths"][index])
        self.semantic_id = int(data["semanticIds"][index])
        self.instance_id = int(data["instanceIds"][index])
        if self.instance_id >= 0:
            self.instance_id += instance_offset
        self.rgb = [int(value) for value in data["rgb"][index]]
        self.layer_id = int(data["layerIds"][index])
        self.width = float(data["widths"][index])


def _class_instances(data: dict[str, Any], class_id: int,
                     instance_offset: int = 0):
    """把目标类的图元按实例分组并栅格化。"""
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(data["commands"])):
        if int(data["semanticIds"][index]) == class_id:
            groups[int(data["instanceIds"][index])].append(index)
    instances = []
    for instance, indices in groups.items():
        mask = np.zeros((RASTER, RASTER), dtype=np.uint8)
        for index in indices:
            points = np.round(
                np.asarray(data["args"][index]) * (RASTER / 100.0)
            ).astype(np.int32)
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, thickness=1)
        primitives = [
            _Primitive(data, index,
                       instance_offset if instance >= 0 else 0)
            for index in indices
        ]
        instances.append((mask.astype(bool), primitives))
    return instances


def _items(data: dict[str, Any], prefix: str) -> list[dict]:
    items = []
    for index in range(len(data["commands"])):
        points = data["args"][index]
        items.append({
            "command": int(data["commands"][index]),
            "points": [(float(points[k]), float(points[k + 1])) for k in range(0, 8, 2)],
            "length": float(data["lengths"][index]),
            "semantic": int(data["semanticIds"][index]),
            "instance": int(data["instanceIds"][index]),
            "key": (prefix, int(data["instanceIds"][index])),
            "rgb": [int(v) for v in data["rgb"][index]],
            "layer": int(data["layerIds"][index]),
            "width": float(data["widths"][index]),
        })
    return items


def _assemble(items: list[dict], width: int, height: int) -> dict[str, Any]:
    mapping: dict[tuple, int] = {}
    next_instance = 0

    def renumber(key):
        nonlocal next_instance
        if key[1] < 0:
            return -1
        if key not in mapping:
            mapping[key] = next_instance
            next_instance += 1
        return mapping[key]

    primitives = []
    for item in items:
        instance = renumber(item["key"])
        payload = {
            "commands": [item["command"]],
            "args": [[value for point in item["points"] for value in point]],
            "lengths": [item["length"]],
            "semanticIds": [item["semantic"]],
            "instanceIds": [instance],
            "rgb": [item["rgb"]],
            "layerIds": [item["layer"]],
            "widths": [item["width"]],
        }
        primitives.append(_Primitive(payload, 0))
    data = assemble_output(primitives, width=width, height=height)
    validate_output(data)
    return data


def merge_final(
    primary: dict[str, Any],
    gat: dict[str, Any] | None = None,
    segdet: dict[str, Any] | None = None,
    *,
    gat_classes: set[int] = frozenset({8}),
    segdet_classes: set[int] = frozenset(),
    suppress: set[int] = frozenset(),
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """最终融合：v3 ∪ GAT(class 8) → segdet 接管指定类别 → 抑制低精度类。"""
    width = int(primary.get("width", 100))
    height = int(primary.get("height", 100))
    items = _items(primary, "p")
    if gat is not None and gat_classes:
        import cv2

        def instance_masks(items):
            groups: dict[tuple, list[dict]] = {}
            for item in items:
                if item["semantic"] in gat_classes:
                    groups.setdefault(item["key"], []).append(item)
            masks = []
            for key, group in groups.items():
                mask = np.zeros((RASTER, RASTER), np.uint8)
                for item in group:
                    pixels = np.round(
                        np.asarray(item["points"]) * (RASTER / 100.0)
                    ).astype(np.int32)
                    cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], False, 1,
                                  thickness=1)
                masks.append((key, mask.astype(bool), group))
            return masks

        kept = instance_masks(items)
        kept_masks = [mask for _, mask, _ in kept]
        for key, mask, group in instance_masks(_items(gat, "g")):
            duplicate = False
            for existing in kept_masks:
                intersection = int(np.count_nonzero(mask & existing))
                if not intersection:
                    continue
                union = int(np.count_nonzero(mask | existing))
                if intersection / max(union, 1) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept_masks.append(mask)
                items.extend(group)
    if segdet is not None and segdet_classes:
        items = [item for item in items if item["semantic"] not in segdet_classes]
        items.extend(item for item in _items(segdet, "s")
                     if item["semantic"] in segdet_classes)
    if suppress:
        items = [item for item in items if item["semantic"] not in suppress]
    return _assemble(items, width, height)


def merge_predictions(primary: dict[str, Any], secondary: dict[str, Any],
                      classes: set[int], iou_threshold: float = 0.5) -> dict[str, Any]:
    """主引擎保留非目标类的图元；目标类取两个引擎的并集，按实例掩码
    IoU 去重（保留主引擎实例）。"""
    if not classes:
        return primary
    primitives = [
        _Primitive(primary, index)
        for index in range(len(primary["commands"]))
        if int(primary["semanticIds"][index]) not in classes
    ]
    kept_masks = []
    for class_id in sorted(classes):
        for mask, items in _class_instances(primary, class_id):
            kept_masks.append(mask)
            primitives.extend(items)
        for mask, items in _class_instances(secondary, class_id, instance_offset=1_000_000):
            duplicate = False
            for existing in kept_masks:
                intersection = int(np.count_nonzero(mask & existing))
                if not intersection:
                    continue
                union = int(np.count_nonzero(mask | existing))
                if intersection / max(union, 1) > iou_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept_masks.append(mask)
                primitives.extend(items)
    data = assemble_output(primitives,
                           width=int(primary.get("width", 100)),
                           height=int(primary.get("height", 100)))
    validate_output(data)
    return data
