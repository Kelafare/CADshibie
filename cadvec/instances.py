"""Thing 实例聚合。

赛题要求：类别 0-29 的图元必须带非负实例号，同一对象的图元共用实例号，
不同对象不得共用；类别 30-34（Stuff）和 35（忽略）使用 -1。

做法：同类别中的 Line/Arc 按**端点邻近**做连通分量。相比包围盒相交，
端点连接能避免把交叉但独立的两个对象（例如两根相交的墙线）合成一个实例；
每条闭合圆/椭圆本身就是一个实例。实例号在同一张图内唯一，跨图可重复。
"""
from __future__ import annotations

import math

from .config import COMMAND_ARC, COMMAND_LINE, THING_CLASS_MAX, THING_CLASS_MIN, InferenceConfig
from .vectorize import Primitive


def _bbox(primitive: Primitive) -> tuple[float, float, float, float]:
    xs = [point[0] for point in primitive.points]
    ys = [point[1] for point in primitive.points]
    return min(xs), min(ys), max(xs), max(ys)


def _endpoints(primitive: Primitive) -> list[tuple[float, float]]:
    if primitive.command in (COMMAND_LINE, COMMAND_ARC) and len(primitive.points) >= 2:
        return [primitive.points[0], primitive.points[-1]]
    return []


def _endpoints_near(first: Primitive, second: Primitive, gap: float) -> bool:
    first_points = _endpoints(first)
    second_points = _endpoints(second)
    if not first_points or not second_points:
        return False
    return any(
        math.dist(first_point, second_point) <= gap
        for first_point in first_points
        for second_point in second_points
    )


class _UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def assign_instances(primitives: list[Primitive], config: InferenceConfig) -> None:
    for primitive in primitives:
        primitive.instance_id = -1

    next_instance = 0
    for semantic in range(THING_CLASS_MIN, THING_CLASS_MAX + 1):
        indices = [
            index for index, primitive in enumerate(primitives)
            if primitive.semantic_id == semantic
        ]
        if not indices:
            continue
        union_find = _UnionFind(len(indices))
        for first in range(len(indices)):
            for second in range(first + 1, len(indices)):
                if _endpoints_near(
                    primitives[indices[first]], primitives[indices[second]],
                    config.instance_gap,
                ):
                    union_find.union(first, second)

        components: dict[int, list[int]] = {}
        for local_index, global_index in enumerate(indices):
            root = union_find.find(local_index)
            components.setdefault(root, []).append(global_index)

        # 按位置排序，保证输出稳定且同一张图的实例号可复现。
        ordered = sorted(
            components.values(),
            key=lambda group: (
                min(_bbox(primitives[index])[0] for index in group),
                min(_bbox(primitives[index])[1] for index in group),
            ),
        )
        for group in ordered:
            for index in group:
                primitives[index].instance_id = next_instance
            next_instance += 1
