"""B-lite：用 YOLO 框锚点对已有实例做整实例重归属。

只改变实例归属，不改变原子几何；用于合并同类重复实例、把实例挂到正确
锚点上。未匹配锚点的实例保持原样。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .schema import assemble_output


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")


def _inside_ratio(args, box, width, height, samples=8):
    x0, y0, x1, y1 = box
    if args and isinstance(args[0], (list, tuple)):
        points = [(float(p[0]), float(p[1])) for p in args]
    else:
        points = [(args[i], args[i + 1]) for i in range(0, 8, 2)]
    hits = total = 0
    for index in range(3):
        ax, ay = points[index]
        bx, by = points[index + 1]
        for step in range(samples):
            ratio = step / samples
            x = (ax + (bx - ax) * ratio) / 100.0 * width
            y = (ay + (by - ay) * ratio) / 100.0 * height
            total += 1
            if x0 <= x <= x1 and y0 <= y <= y1:
                hits += 1
    return hits / max(total, 1)


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


def _features(group, anchor):
    points = group["points"]
    box = group["box"]
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
    dx = (gx - cx) / aw
    dy = (gy - cy) / ah
    ratios = [_inside_ratio(atom, anchor_box, group["width"],
                            group["height"]) for atom in group["atoms"]]
    return {
        "inside_mean": float(np.mean(ratios)) if ratios else 0.0,
        "inside_max": float(np.max(ratios)) if ratios else 0.0,
        "dx": dx, "dy": dy, "dist": float(np.hypot(dx, dy)),
        "w_ratio": gw / aw, "h_ratio": gh / ah,
        "area_ratio": (gw * gh) / (aw * ah),
        "length_ratio": group["length"] / max(np.hypot(aw, ah), 1e-6),
        "n_atoms": float(len(group["atoms"])),
        "anchor_conf": float(anchor[1]),
        "bbox_iou": _bbox_iou(box, anchor_box),
    }


def _probability(spec, features):
    z = float(spec["intercept"])
    for name, coef, mean, scale in zip(spec["features"], spec["coef"],
                                       spec["mean"], spec["scale"]):
        z += float(coef) * (float(features.get(name, 0.0)) - float(mean)) / max(
            float(scale), 1e-9)
    return 1.0 / (1.0 + np.exp(-max(min(z, 50.0), -50.0)))


def reassign(data: dict, detections: list[dict], classes: set[int],
             width: int, height: int, inside: float = 0.6,
             affinity: dict | None = None) -> dict:
    if not detections or not classes:
        return data
    anchors = [(int(item["class"]), float(item.get("conf", 1.0)),
                tuple(float(v) for v in item["box"]))
               for item in detections if int(item["class"]) in classes]
    if not anchors:
        return data
    groups = defaultdict(list)
    for index, (semantic, instance) in enumerate(zip(
            data["semanticIds"], data["instanceIds"])):
        semantic = int(semantic)
        instance = int(instance)
        if semantic in classes and instance >= 0:
            groups[(semantic, instance)].append(index)
    assignment = {}
    canvas_w = max(float(data.get("width", 100)), 1e-6)
    canvas_h = max(float(data.get("height", 100)), 1e-6)
    sx, sy = width / canvas_w, height / canvas_h
    threshold = float(affinity.get("threshold", 0.5)) if affinity else 0.0
    for (semantic, _instance), indices in groups.items():
        best = (0.0, None)
        if affinity:
            points = []
            atoms = []
            for index in indices:
                values = data["args"][index]
                atom = [(float(values[k]) * sx, float(values[k + 1]) * sy)
                        for k in range(0, 8, 2)]
                atoms.append(atom)
                points.extend(atom)
            group = {
                "points": points, "atoms": atoms, "box": _bbox(points),
                "length": float(sum(data["lengths"][index]
                                    for index in indices)) * sx,
                "width": width, "height": height,
            }
            for anchor_index, anchor in enumerate(anchors):
                if anchor[0] != semantic:
                    continue
                probability = _probability(affinity, _features(group, anchor))
                if probability >= threshold and probability > best[0]:
                    best = (probability, anchor_index)
        else:
            for anchor_index, (class_id, confidence, box) in enumerate(anchors):
                if class_id != semantic:
                    continue
                ratios = [_inside_ratio(data["args"][index], box, width, height)
                          for index in indices]
                ratio = float(np.mean(ratios))
                score = ratio + 0.1 * confidence
                if ratio >= inside and score > best[0]:
                    best = (score, anchor_index)
        if best[1] is not None:
            class_id, _confidence, _box = anchors[best[1]]
            for index in indices:
                assignment[index] = (class_id, 1_000_000 + best[1])

    primitives = []
    for index, (semantic, instance, command, args, length, rgb, layer,
                line_width) in enumerate(zip(
            data["semanticIds"], data["instanceIds"], data["commands"],
            data["args"], data["lengths"], data["rgb"], data["layerIds"],
            data["widths"])):
        item = _Primitive()
        item.command = int(command)
        item.points = [(float(args[k]), float(args[k + 1]))
                       for k in range(0, 8, 2)]
        item.length = float(length)
        if index in assignment:
            item.semantic_id, item.instance_id = assignment[index]
        else:
            item.semantic_id = int(semantic)
            item.instance_id = int(instance)
        item.rgb = [int(v) for v in rgb]
        item.layer_id = int(layer)
        item.width = float(line_width)
        primitives.append(item)
    return assemble_output(primitives, width=int(data.get("width", 100)),
                           height=int(data.get("height", 100)))
