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


def reassign(data: dict, detections: list[dict], classes: set[int],
             width: int, height: int, inside: float = 0.6) -> dict:
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
    for (semantic, _instance), indices in groups.items():
        best = (0.0, None)
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
