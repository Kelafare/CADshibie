"""骨架折线追踪：把一个颜色层的骨架拆成若干可输出的折线。

与 Hough + 合并不同，这里直接沿骨架像素行走，能最大程度保留原始描边的
形状与端点。输出每个折线只保留 4 个等弧长采样点（与赛题 GT 的采样方式
一致），因此既符合 12 字段格式，又不会因为过多顶点而偏离 GT 表示。
"""
from __future__ import annotations

import math

import numpy as np

_NEIGHBORS = (
    (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1),
)


def _neighbors(point: tuple[int, int], pixels: set) -> list[tuple[int, int]]:
    row, column = point
    return [
        (row + dy, column + dx)
        for dy, dx in _NEIGHBORS
        if (row + dy, column + dx) in pixels
    ]


def trace_paths(skeleton: np.ndarray, min_pixels: int = 6) -> list[list[tuple[int, int]]]:
    """把骨架拆成折线（像素坐标 (row, column)）。"""
    rows, columns = np.nonzero(skeleton)
    if rows.size == 0:
        return []
    pixels = set(zip(rows.tolist(), columns.tolist()))
    neighbors = {point: _neighbors(point, pixels) for point in pixels}
    nodes = {point for point in pixels if len(neighbors[point]) != 2}
    visited: set[frozenset] = set()
    paths: list[list[tuple[int, int]]] = []

    def walk(start, second):
        path = [start, second]
        visited.add(frozenset((start, second)))
        previous, current = start, second
        while current not in nodes:
            candidates = [item for item in neighbors[current]
                          if item != previous and frozenset((current, item)) not in visited]
            if not candidates:
                break
            following = candidates[0]
            visited.add(frozenset((current, following)))
            path.append(following)
            previous, current = current, following
        return path

    for node in nodes:
        for neighbor in neighbors[node]:
            if frozenset((node, neighbor)) in visited:
                continue
            paths.append(walk(node, neighbor))

    # 剩下的边构成纯环（所有像素度数为 2）。
    for point in pixels:
        for neighbor in neighbors[point]:
            edge = frozenset((point, neighbor))
            if edge in visited:
                continue
            path = walk(point, neighbor)
            if path and path[-1] != path[0] and len(path) > 2:
                path.append(path[0])
            paths.append(path)
    return [path for path in paths if len(path) >= min_pixels]


def sample_polyline(path: list[tuple[int, int]], count: int = 4,
                    scale_x: float = 10.0, scale_y: float = 10.0,
                    ) -> list[tuple[float, float]]:
    """按弧长等距采样，并换算到画布坐标（像素中心 +0.5）。"""
    points = np.asarray([(column + 0.5, row + 0.5) for row, column in path], dtype=np.float64)
    if len(points) == 1:
        point = points[0]
        return [(point[0] / scale_x, point[1] / scale_y)] * count
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(deltas)])
    total = float(cumulative[-1])
    if total <= 1e-9:
        return [(points[0][0] / scale_x, points[0][1] / scale_y)] * count
    samples = []
    for index in range(count):
        target = total * index / (count - 1)
        position = int(np.searchsorted(cumulative, target, side="right") - 1)
        position = min(max(position, 0), len(points) - 2)
        segment = deltas[position]
        ratio = 0.0 if segment <= 1e-9 else (target - cumulative[position]) / segment
        x = points[position][0] + (points[position + 1][0] - points[position][0]) * ratio
        y = points[position][1] + (points[position + 1][1] - points[position][1]) * ratio
        samples.append((x / scale_x, y / scale_y))
    return samples


def polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(
        math.dist(first, second) for first, second in zip(points, points[1:])
    )
