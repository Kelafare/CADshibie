"""原子级 atom-anchor 归属：逐图元选择 YOLO 锚点，按锚点生成实例。"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import assemble_output


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")


def _bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_iou(first, second):
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / max(area, 1e-9)


def _inside(points, box, samples=8):
    x0, y0, x1, y1 = box
    hits = total = 0
    for index in range(len(points) - 1):
        ax, ay = points[index]
        bx, by = points[index + 1]
        for step in range(samples):
            ratio = step / samples
            x = ax + (bx - ax) * ratio
            y = ay + (by - ay) * ratio
            total += 1
            if x0 <= x <= x1 and y0 <= y <= y1:
                hits += 1
    return hits / max(total, 1)


def _features(atom, anchor):
    points = atom["points"]
    box = atom["box"]
    anchor_box = anchor[2]
    ax0, ay0, ax1, ay1 = anchor_box
    aw = max(ax1 - ax0, 1e-6)
    ah = max(ay1 - ay0, 1e-6)
    gx = (box[0] + box[2]) / 2.0
    gy = (box[1] + box[3]) / 2.0
    cx = (ax0 + ax1) / 2.0
    cy = (ay0 + ay1) / 2.0
    gw = max(box[2] - box[0], 1e-6)
    gh = max(box[3] - box[1], 1e-6)
    direction = np.asarray(points[-1]) - np.asarray(points[0])
    norm = float(np.linalg.norm(direction))
    angle = abs(direction[0]) / max(norm, 1e-6) if norm else 0.0
    return {
        "inside": _inside(points, anchor_box),
        "dx": (gx - cx) / aw, "dy": (gy - cy) / ah,
        "dist": float(np.hypot((gx - cx) / aw, (gy - cy) / ah)),
        "w_ratio": gw / aw, "h_ratio": gh / ah,
        "area_ratio": (gw * gh) / (aw * ah),
        "length_ratio": atom["length"] / max(np.hypot(aw, ah), 1e-6),
        "axis_align": angle,
        "bbox_iou": _bbox_iou(box, anchor_box),
        "anchor_conf": float(anchor[1]),
        "atom_length": atom["length"],
    }


def _probability(spec, features):
    z = float(spec["intercept"])
    for name, coef, mean, scale in zip(spec["features"], spec["coef"],
                                       spec["mean"], spec["scale"]):
        z += float(coef) * (float(features.get(name, 0.0)) - float(mean)) / max(
            float(scale), 1e-9)
    return 1.0 / (1.0 + np.exp(-max(min(z, 50.0), -50.0)))


def assign_atoms(data: dict, detections: list[dict], classes: set[int],
                 width: int, height: int, model: dict,
                 keep_unassigned: bool = False) -> dict:
    if not detections or not classes or not model:
        return data
    anchors = [(int(item["class"]), float(item.get("conf", 1.0)),
                tuple(float(v) for v in item["box"]))
               for item in detections if int(item["class"]) in classes]
    if not anchors:
        return data
    threshold = float(model.get("threshold", 0.5))
    canvas_w = max(float(data.get("width", 100)), 1e-6)
    canvas_h = max(float(data.get("height", 100)), 1e-6)
    sx, sy = width / canvas_w, height / canvas_h
    primitives = []
    for semantic, instance, command, args, length, rgb, layer, line_width in zip(
            data["semanticIds"], data["instanceIds"], data["commands"],
            data["args"], data["lengths"], data["rgb"], data["layerIds"],
            data["widths"]):
        semantic = int(semantic)
        instance = int(instance)
        item = _Primitive()
        item.command = int(command)
        item.points = [(float(args[k]), float(args[k + 1]))
                       for k in range(0, 8, 2)]
        item.length = float(length)
        item.rgb = [int(v) for v in rgb]
        item.layer_id = int(layer)
        item.width = float(line_width)
        if semantic in classes and instance >= 0:
            points_px = [(x * sx, y * sy) for x, y in item.points]
            atom = {"points": points_px, "box": _bbox(points_px),
                    "length": float(length) * sx}
            best = (0.0, None)
            for anchor_index, anchor in enumerate(anchors):
                if anchor[0] != semantic:
                    continue
                probability = _probability(model, _features(atom, anchor))
                if probability >= threshold and probability > best[0]:
                    best = (probability, anchor_index)
            if best[1] is None and not keep_unassigned:
                continue
            if best[1] is not None:
                item.semantic_id = semantic
                item.instance_id = 2_000_000 + best[1]
            else:
                item.semantic_id = semantic
                item.instance_id = instance
        else:
            item.semantic_id = semantic
            item.instance_id = instance
        primitives.append(item)
    return assemble_output(primitives, width=int(data.get("width", 100)),
                           height=int(data.get("height", 100)))
