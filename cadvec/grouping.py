"""学习式笔画分组：判断两条笔画是否属于同一个 GT 实例。

分组是当前 PQ 的主要瓶颈：颜色层连通分量往往把穿过的尺寸线/墙线也并进
对象，或者把一个符号拆成多个分量。这里用成对特征训练一个梯度提升模型，
推理时对邻近笔画对预测「同实例」概率，再做并查集聚类。

训练标签来自真值：一条笔画与某个 GT 实例掩码的重叠像素 ≥ 50% 即视为属于
该实例；两条笔画标签相同则为正样本。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .strokes import Stroke

#: 特征顺序（训练与推理必须一致）。
FEATURE_NAMES = (
    "endpoint_distance",
    "segment_distance",
    "angle_parallel",
    "same_color",
    "same_component",
    "log_length_ratio",
    "bbox_gap",
    "projection_overlap",
    "endpoint_junction",
    "mean_length",
    "component_size",
    "parallel_spacing",
)


def _sampled_points(stroke: Stroke, count: int = 8) -> np.ndarray:
    if stroke.command == 0 and len(stroke.points) >= 2:
        points = np.asarray(stroke.points, dtype=np.float64)
    else:
        points = np.asarray(stroke.resample(count), dtype=np.float64)
    return points


def _direction(points: np.ndarray) -> tuple[float, float]:
    delta = points[-1] - points[0]
    norm = float(np.linalg.norm(delta)) or 1.0
    return float(delta[0] / norm), float(delta[1] / norm)


def _segment_distance(first: np.ndarray, second: np.ndarray) -> float:
    """两条折线之间的最小点距（采样近似）。"""
    return float(
        np.sqrt(
            ((first[:, None, :] - second[None, :, :]) ** 2).sum(axis=2)
        ).min()
    )


def pair_features(
    first: Stroke, second: Stroke,
    *, same_component: bool = False, component_size: int = 1,
) -> np.ndarray:
    a = _sampled_points(first)
    b = _sampled_points(second)
    endpoint_distance = float(
        min(
            np.linalg.norm(a[0] - b[0]), np.linalg.norm(a[0] - b[-1]),
            np.linalg.norm(a[-1] - b[0]), np.linalg.norm(a[-1] - b[-1]),
        )
    )
    segment_distance = _segment_distance(a, b)
    da = _direction(a)
    db = _direction(b)
    cosine = abs(da[0] * db[0] + da[1] * db[1])
    angle_parallel = 1.0 - cosine
    same_color = 1.0 if tuple(first.rgb) == tuple(second.rgb) else 0.0
    log_length_ratio = math.log(max(first.length, 1e-3) / max(second.length, 1e-3))
    box_a = first.bbox()
    box_b = second.bbox()
    bbox_gap = math.hypot(
        max(box_a[0] - box_b[2], box_b[0] - box_a[2], 0.0),
        max(box_a[1] - box_b[3], box_b[1] - box_a[3], 0.0),
    )
    # 沿 A 主方向的投影重叠比例。
    direction = np.array(da)
    projection_a = np.sort(a @ direction)
    projection_b = np.sort(b @ direction)
    low = max(projection_a[0], projection_b[0])
    high = min(projection_a[-1], projection_b[-1])
    overlap = max(0.0, high - low)
    span = max(projection_a[-1] - projection_a[0], 1e-6)
    projection_overlap = overlap / span
    # 端点对端点 / 端点对中段（T 型连接）。
    endpoints_a = a[[0, -1]]
    endpoints_b = b[[0, -1]]
    endpoint_junction = float(
        min(
            np.linalg.norm(endpoints_a[:, None, :] - endpoints_b[None, :, :], axis=2).min(),
            np.linalg.norm(endpoints_a[:, None, :] - b[None, :, :], axis=2).min(),
            np.linalg.norm(endpoints_b[:, None, :] - a[None, :, :], axis=2).min(),
        )
    )
    mean_length = 0.5 * (first.length + second.length)
    normal = np.array([-da[1], da[0]])
    spacing = abs(float((b[0] - a[0]) @ normal)) if cosine > 0.9 else 0.0
    return np.asarray([
        endpoint_distance, segment_distance, angle_parallel, same_color,
        1.0 if same_component else 0.0, log_length_ratio, bbox_gap,
        projection_overlap, endpoint_junction, mean_length,
        float(component_size), spacing,
    ], dtype=np.float64)


def candidate_pairs(strokes: list[Stroke], radius: float = 2.5) -> list[tuple[int, int]]:
    """只用包围盒邻近的笔画对，控制推理复杂度。"""
    boxes = [stroke.bbox() for stroke in strokes]
    cell = max(radius, 0.5)
    grid: dict[tuple[int, int], list[int]] = {}
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        for gx in range(int(math.floor(x0 / cell)), int(math.floor(x1 / cell)) + 1):
            for gy in range(int(math.floor(y0 / cell)), int(math.floor(y1 / cell)) + 1):
                grid.setdefault((gx, gy), []).append(index)
    pairs: set[tuple[int, int]] = set()
    for indices in grid.values():
        for position, first in enumerate(indices):
            for second in indices[position + 1:]:
                key = (min(first, second), max(first, second))
                if key in pairs:
                    continue
                a, b = boxes[first], boxes[second]
                gap = math.hypot(
                    max(a[0] - b[2], b[0] - a[2], 0.0),
                    max(a[1] - b[3], b[1] - a[3], 0.0),
                )
                if gap <= radius:
                    pairs.add(key)
    return sorted(pairs)


class StrokeGrouper:
    """加载成对链接模型并做并查集聚类。"""

    def __init__(self, model, link_threshold: float = 0.5, max_group: int = 80):
        self.model = model
        self.link_threshold = link_threshold
        self.max_group = max_group

    @classmethod
    def load(cls, path: str | Path | None) -> "StrokeGrouper | None":
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        try:
            import joblib
            payload = joblib.load(path)
        except Exception:  # noqa: BLE001
            return None
        return cls(payload["model"], payload.get("link_threshold", 0.5))

    def group(self, strokes: list[Stroke], components: dict[int, set[int]] | None = None
              ) -> list[list[int]]:
        count = len(strokes)
        if count == 0:
            return []
        parent = list(range(count))
        sizes = [1] * count

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(first: int, second: int) -> None:
            first_root, second_root = find(first), find(second)
            if first_root == second_root:
                return
            if sizes[first_root] + sizes[second_root] > self.max_group:
                return
            if sizes[first_root] < sizes[second_root]:
                first_root, second_root = second_root, first_root
            parent[second_root] = first_root
            sizes[first_root] += sizes[second_root]

        owner = {}
        for component_id, indices in (components or {}).items():
            for index in indices:
                owner[index] = component_id
        pairs = candidate_pairs(strokes)
        if len(pairs) > 20000:
            # 控制单图开销：保留距离最近的若干对。
            strokes_by_distance = sorted(
                pairs,
                key=lambda pair: math.hypot(
                    strokes[pair[0]].bbox()[0] - strokes[pair[1]].bbox()[0],
                    strokes[pair[0]].bbox()[1] - strokes[pair[1]].bbox()[1],
                ),
            )
            pairs = strokes_by_distance[:20000]
        if not pairs:
            return [[index] for index in range(count)]
        features = np.asarray([
            pair_features(
                strokes[first], strokes[second],
                same_component=owner.get(first) == owner.get(second) and first in owner,
                component_size=len(components.get(owner.get(first), ())) if components else 1,
            )
            for first, second in pairs
        ])
        probabilities = self.model.predict_proba(features)[:, 1]
        for (first, second), probability in zip(pairs, probabilities):
            if probability >= self.link_threshold:
                union(first, second)
        groups: dict[int, list[int]] = {}
        for index in range(count):
            groups.setdefault(find(index), []).append(index)
        return list(groups.values())
