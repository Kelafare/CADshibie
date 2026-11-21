"""图元图（primitive graph）：节点=笔画，边=近邻/端点连接，用于 GAT 全景检测。

节点特征全部归一化到画布尺度；边特征供注意力偏置使用。训练与推理共用
本模块，保证分布一致。
"""
from __future__ import annotations

import math

import numpy as np

from .strokes import Stroke

NODE_FEATURES = 14
#: 基础 4 维 + 极坐标/距离编码 10 维（参考 PolarSym 的方向/径向约束）。
BASE_EDGE_FEATURES = 4
EDGE_FEATURES = 14


def _midpoint(stroke: Stroke) -> tuple[float, float]:
    xs = [point[0] for point in stroke.points]
    ys = [point[1] for point in stroke.points]
    return (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0


def node_features(stroke: Stroke, density: float = 0.0) -> np.ndarray:
    x, y = _midpoint(stroke)
    x0, y0, x1, y1 = stroke.bbox()
    direction = stroke.angle if stroke.is_line else 0.0
    r, g, b = (float(value) / 255.0 for value in stroke.rgb[:3])
    achromatic = 1.0 if max(stroke.rgb[:3]) - min(stroke.rgb[:3]) <= 8 else 0.0
    curvature = 0.0
    if not stroke.is_line:
        radius = float(stroke.shape.get("a", 1.0)) if stroke.shape else 1.0
        curvature = min(1.0, 1.0 / max(radius, 1e-3))
    return np.asarray([
        x / 100.0, y / 100.0,
        min(stroke.length / 10.0, 4.0),
        math.cos(2.0 * direction), math.sin(2.0 * direction),
        r, g, b,
        min(stroke.width, 2.0),
        curvature,
        (x1 - x0) / 100.0, (y1 - y0) / 100.0,
        achromatic,
        min(density / 20.0, 2.0),
    ], dtype=np.float32)


def build_graph(strokes: list[Stroke], k: int = 6, endpoint_gap: float = 1.2):
    """返回 (node_features, edge_index(2×E), edge_features(E×4), adjacency_dense)。"""
    count = len(strokes)
    if count == 0:
        return (np.zeros((0, NODE_FEATURES), np.float32),
                np.zeros((2, 0), np.int64), np.zeros((0, EDGE_FEATURES), np.float32),
                np.zeros((0, 0), bool))
    centers = np.asarray([_midpoint(stroke) for stroke in strokes], dtype=np.float32)
    # 局部密度：每个节点 2 个单位内的邻居数。
    delta = centers[:, None, :] - centers[None, :, :]
    distances = np.sqrt((delta ** 2).sum(axis=2))
    density = (distances < 2.0).sum(axis=1) - 1
    features = np.stack([
        node_features(stroke, float(density[index]))
        for index, stroke in enumerate(strokes)
    ])
    edges: dict[tuple[int, int], np.ndarray] = {}
    order = np.argsort(distances, axis=1)
    for first in range(count):
        for second in order[first][1:k + 1]:
            second = int(second)
            key = (min(first, second), max(first, second))
            if key in edges:
                continue
            edges[key] = _edge_feature(strokes[first], strokes[second], distances)
    # 端点邻近（同色或短笔画）补充跨分量连接。
    for first in range(count):
        near = np.nonzero(distances[first] < endpoint_gap)[0]
        for second in near:
            second = int(second)
            if second == first:
                continue
            key = (min(first, second), max(first, second))
            if key in edges:
                continue
            if strokes[first].layer_id != strokes[second].layer_id:
                if min(strokes[first].length, strokes[second].length) > 4.0:
                    continue
            edges[key] = _edge_feature(strokes[first], strokes[second], distances)
    edge_keys = list(edges.keys())
    edge_index = np.asarray(edge_keys, dtype=np.int64).T if edge_keys else np.zeros((2, 0), np.int64)
    edge_features = (np.stack([edges[key] for key in edge_keys])
                     if edge_keys else np.zeros((0, EDGE_FEATURES), np.float32))
    adjacency = np.zeros((count, count), dtype=bool)
    for first, second in edge_keys:
        adjacency[first, second] = adjacency[second, first] = True
    return features, edge_index, edge_features, adjacency


def polar_edge_features(features: np.ndarray, edge_index: np.ndarray,
                        edge_features: np.ndarray) -> np.ndarray:
    """在基础边特征上拼接极坐标方向与多尺度距离编码。"""
    if edge_index.shape[1] == 0:
        return np.zeros((0, EDGE_FEATURES), dtype=np.float32)
    centers = np.asarray(features[:, :2], dtype=np.float64)
    delta = centers[edge_index[1]] - centers[edge_index[0]]
    distance = np.linalg.norm(delta, axis=1)
    angle = np.arctan2(delta[:, 1], delta[:, 0])
    extra = np.stack([
        np.sin(angle), np.cos(angle),
        np.sin(2 * angle), np.cos(2 * angle),
        np.minimum(distance * 5.0, 4.0),
        np.log1p(distance * 10.0),
        np.sin(distance * 2 * np.pi / 0.5), np.cos(distance * 2 * np.pi / 0.5),
        np.sin(distance * 2 * np.pi / 1.5), np.cos(distance * 2 * np.pi / 1.5),
    ], axis=1).astype(np.float32)
    return np.concatenate([edge_features.astype(np.float32), extra], axis=1)


def _edge_feature(first: Stroke, second: Stroke, distances: np.ndarray) -> np.ndarray:
    # 用中点距与端点距的组合。
    a = np.asarray(first.points, dtype=np.float64)
    b = np.asarray(second.points, dtype=np.float64)
    endpoint = float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))
    mid_a = a.mean(axis=0)
    mid_b = b.mean(axis=0)
    middle = float(np.linalg.norm(mid_a - mid_b))
    angle = abs(first.angle - second.angle) % math.pi
    angle = min(angle, math.pi - angle)
    same_color = 1.0 if tuple(first.rgb) == tuple(second.rgb) else 0.0
    return np.asarray([
        min(endpoint / 2.0, 2.0), min(middle / 10.0, 2.0),
        math.sin(angle), same_color,
    ], dtype=np.float32)
