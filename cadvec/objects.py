"""从图元聚合出「对象假设」，供语义分类与实例输出使用。

聚合规则（按优先级）：

1. 平行线带（``parallel.Ribbon``）里的直线属于同一对象；
2. 同色层、端点距离 ≤ ``endpoint_gap`` 的图元相连——门/窗符号的弧与边框
   端点相接；
3. 彩色图元与无彩色图元之间用更小的容差相连，避免黑色墙线把整张图连成
   一个巨型连通域；
4. 圆弧的两个端点分别连接到附近直线。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import COMMAND_ARC, COMMAND_CIRCLE, COMMAND_ELLIPSE, COMMAND_LINE
from .parallel import Ribbon
from .strokes import Stroke

#: 无彩色层（黑/灰）在调色板中的下标从 len(palette) 开始。
ACHROMATIC_LAYER = -1


@dataclass
class Hypothesis:
    indices: list[int] = field(default_factory=list)
    node: int = 0
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    layer_ids: set[int] = field(default_factory=set)
    colored: bool = False
    parallel_pairs: int = 0
    semantic: int = 35
    confidence: float = 0.0

    @property
    def size(self) -> float:
        return max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


def _endpoints(stroke: Stroke) -> list[tuple[float, float]]:
    if stroke.command == COMMAND_LINE:
        return [stroke.p0, stroke.p1]
    if stroke.command == COMMAND_ARC:
        shape = stroke.shape
        a = shape["a"]
        start = shape["start"]
        span = shape["coverage"]
        return [
            (shape["cx"] + a * math.cos(start), shape["cy"] + a * math.sin(start)),
            (shape["cx"] + a * math.cos(start + span), shape["cy"] + a * math.sin(start + span)),
        ]
    return []


def _bbox(strokes: list[Stroke], indices: list[int]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for index in indices:
        for x, y in strokes[index].points:
            xs.append(x)
            ys.append(y)
    return min(xs), min(ys), max(xs), max(ys)


def _near(first: Stroke, second: Stroke, gap: float) -> bool:
    first_points = _endpoints(first)
    second_points = _endpoints(second)
    return any(
        math.dist(a, b) <= gap for a in first_points for b in second_points
    )


def _overlaps(first: tuple[float, float, float, float],
              second: tuple[float, float, float, float], margin: float = 0.5) -> bool:
    return not (
        first[2] + margin < second[0] or second[2] + margin < first[0]
        or first[3] + margin < second[1] or second[3] + margin < first[1]
    )


def build_hypotheses(
    strokes: list[Stroke],
    ribbons: list[Ribbon],
    *,
    endpoint_gap: float = 1.4,
    cross_layer_gap: float = 1.0,
    max_ribbon_members: int = 12,
) -> list[Hypothesis]:
    """按连通关系聚合图元；返回至少包含一个图元的对象假设。"""
    count = len(strokes)
    if not count:
        return []
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    # 平行线带只连接「相邻」成员，而不是整带互相连接：停车位一排、填充线
    # 一整片平行线都是绘图图案，把它们全并成一个对象会吞掉整张图。
    for ribbon in ribbons:
        if len(ribbon.members) > max_ribbon_members:
            continue
        ordered = sorted(
            ribbon.members,
            key=lambda index: _offset(strokes[index]),
        )
        for previous, current in zip(ordered, ordered[1:]):
            if _offset_difference(strokes[previous], strokes[current]) <= 3.5:
                union(previous, current)

    boxes = [_bbox(strokes, [index]) for index in range(count)]
    # 端点空间哈希：把 O(n²) 的两两比较限制在相邻网格内。
    cell = max(endpoint_gap * 2.0, 0.5)
    grid: dict[tuple[int, int, int], list[int]] = {}
    for index in range(count):
        for x, y in _endpoints(strokes[index]):
            key = (strokes[index].layer_id,
                   int(math.floor(x / cell)), int(math.floor(y / cell)))
            grid.setdefault(key, []).append(index)

    checked: set[tuple[int, int]] = set()
    for first in range(count):
        for x, y in _endpoints(strokes[first]):
            base_x = int(math.floor(x / cell))
            base_y = int(math.floor(y / cell))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for second in grid.get(
                        (strokes[first].layer_id, base_x + dx, base_y + dy), (),
                    ):
                        if second <= first:
                            continue
                        key = (first, second)
                        if key in checked:
                            continue
                        checked.add(key)
                        if find(first) == find(second):
                            continue
                        if not _overlaps(boxes[first], boxes[second]):
                            continue
                        if not _near(strokes[first], strokes[second], endpoint_gap):
                            continue
                        # 两条长直线端点相接往往只是墙线相交，不合并。
                        if (_both_long(strokes[first], strokes[second])
                                and not _same_ribbon(ribbons, first, second)):
                            continue
                        union(first, second)

    # 跨色连接（彩色符号 + 黑色边框）单独处理，容差更小。
    cross_cell = max(cross_layer_gap * 2.0, 0.5)
    cross_grid: dict[tuple[int, int], list[int]] = {}
    for index in range(count):
        for x, y in _endpoints(strokes[index]):
            key = (int(math.floor(x / cross_cell)), int(math.floor(y / cross_cell)))
            cross_grid.setdefault(key, []).append(index)
    for first in range(count):
        if strokes[first].length <= 4.0:
            continue
        for x, y in _endpoints(strokes[first]):
            base_x = int(math.floor(x / cross_cell))
            base_y = int(math.floor(y / cross_cell))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for second in cross_grid.get((base_x + dx, base_y + dy), ()):
                        if second == first:
                            continue
                        if strokes[second].layer_id == strokes[first].layer_id:
                            continue
                        if strokes[second].length > 4.0:
                            continue
                        if find(first) == find(second):
                            continue
                        if not _overlaps(boxes[first], boxes[second]):
                            continue
                        if _near(strokes[first], strokes[second], cross_layer_gap):
                            union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)

    pair_lookup: dict[int, int] = {}
    for ribbon in ribbons:
        for index in ribbon.members:
            pair_lookup[index] = pair_lookup.get(index, 0) + ribbon.pairs

    hypotheses: list[Hypothesis] = []
    for node, indices in groups.items():
        box = _bbox(strokes, indices)
        layers = {strokes[index].layer_id for index in indices}
        hypotheses.append(Hypothesis(
            indices=sorted(indices), node=node, bbox=box, layer_ids=layers,
            colored=any(strokes[index].shape.get("colorful", True) for index in indices),
            parallel_pairs=sum(pair_lookup.get(index, 0) for index in indices),
        ))
    hypotheses.sort(key=lambda item: -sum(strokes[i].length for i in item.indices))
    return hypotheses


def _both_long(first: Stroke, second: Stroke, limit: float = 6.0) -> bool:
    return first.length > limit and second.length > limit


def _direction(stroke: Stroke) -> tuple[float, float]:
    dx = stroke.p1[0] - stroke.p0[0]
    dy = stroke.p1[1] - stroke.p0[1]
    norm = math.hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


def _offset(stroke: Stroke) -> float:
    direction = _direction(stroke)
    return -direction[1] * stroke.p0[0] + direction[0] * stroke.p0[1]


def _offset_difference(first: Stroke, second: Stroke) -> float:
    first_direction = _direction(first)
    second_direction = _direction(second)
    if abs(first_direction[0] * second_direction[0]
           + first_direction[1] * second_direction[1]) < 0.98:
        return float("inf")
    normal = (-first_direction[1], first_direction[0])
    return abs(
        (second.p0[0] - first.p0[0]) * normal[0]
        + (second.p0[1] - first.p0[1]) * normal[1]
    )


def _same_ribbon(ribbons: list[Ribbon], first: int, second: int) -> bool:
    return any(first in ribbon.members and second in ribbon.members for ribbon in ribbons)


def rasterize_hypothesis(
    strokes: list[Stroke], indices: list[int],
    canvas_width: float, canvas_height: float, raster: int,
) -> np.ndarray:
    """把假设画进全局画布栅格（与真值栅格同一坐标系，用于匹配打标签）。"""
    import cv2
    mask = np.zeros((raster, raster), dtype=np.uint8)
    sx = raster / max(canvas_width, 1e-6)
    sy = raster / max(canvas_height, 1e-6)
    for index in indices:
        stroke = strokes[index]
        points = stroke.resample(64 if stroke.command != 0 else 2)
        pixels = np.array(
            [(round(x * sx), round(y * sy)) for x, y in points], dtype=np.int32
        )
        closed = stroke.command in (COMMAND_CIRCLE, COMMAND_ELLIPSE)
        cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], closed, 1, thickness=1)
    return mask.astype(bool)


def render_hypothesis(
    strokes: list[Stroke], indices: list[int],
    canvas_width: float, canvas_height: float, resolution: int,
    padding: float = 0.15, thickness: int = 1,
) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    """把假设中的图元画成二值掩码，返回 (mask, 画布 bbox, 像素/画布单位)。"""
    xs: list[float] = []
    ys: list[float] = []
    if not indices:
        return np.zeros((resolution, resolution), dtype=np.float32), (0, 0, 0, 0), 1.0
    for index in indices:
        for x, y in strokes[index].points:
            xs.append(x)
            ys.append(y)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    width = max(x1 - x0, 1e-3)
    height = max(y1 - y0, 1e-3)
    margin_x = max(width * padding, 1.0)
    margin_y = max(height * padding, 1.0)
    # 保持方形视野，避免缩放变形。
    side = max(width + 2 * margin_x, height + 2 * margin_y)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = cx - side / 2, cy - side / 2
    x1, y1 = cx + side / 2, cy + side / 2
    scale = resolution / side

    def to_pixel(x: float, y: float) -> tuple[int, int]:
        return (
            int(round((x - x0) * scale)),
            int(round((y - y0) * scale)),
        )

    import cv2
    mask = np.zeros((resolution, resolution), dtype=np.uint8)
    for index in indices:
        stroke = strokes[index]
        points = stroke.resample(64 if stroke.command != 0 else 2)
        pixels = np.array([to_pixel(x, y) for x, y in points], dtype=np.int32)
        closed = stroke.command in (COMMAND_CIRCLE, COMMAND_ELLIPSE)
        cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], closed, 1,
                      thickness=max(1, thickness))
    return mask.astype(np.float32), (x0, y0, x1, y1), scale
