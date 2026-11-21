"""Parallel-line optimisation.

工程图里绝大多数可识别对象都由**成组的平行线**构成：墙体/窗口是两条平行
的长线，家具轮廓是矩形边，停车位是一排短平行线。因此把「平行线对」作为
一等公民：

1. ``ParallelRibbon``：一组方向一致、垂距稳定、投影区间重叠的直线；
2. 共线碎片在门洞/窗洞处断开时，记录 ``gap`` 关系而不是强行合并；
3. 平行关系用于对象聚合（``objects.py``）与几何规整（对齐角度/端点）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .strokes import Stroke


@dataclass
class Ribbon:
    members: list[int] = field(default_factory=list)
    angle: float = 0.0
    #: 平均垂距（像素）。
    spacing: float = 0.0
    #: 投影方向的覆盖长度。
    extent: float = 0.0
    pairs: int = 0


def _line_geometry(stroke: Stroke, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first = np.asarray(stroke.p0, dtype=np.float64) * scale
    second = np.asarray(stroke.p1, dtype=np.float64) * scale
    delta = second - first
    length = float(np.linalg.norm(delta))
    direction = delta / max(length, 1e-9)
    angle = math.atan2(direction[1], direction[0]) % math.pi
    normal = np.array([-direction[1], direction[0]])
    offset = float(np.dot(first, normal))
    projection = float(np.dot((first + second) / 2.0, direction))
    half = length / 2.0
    return direction, np.array([angle, offset, projection]), half, delta


def _overlap(first_half: float, first_projection: float,
             second_half: float, second_projection: float) -> float:
    low = max(first_projection - first_half, second_projection - second_half)
    high = min(first_projection + first_half, second_projection + second_half)
    return max(0.0, high - low)


def parallel_ribbons(
    strokes: list[Stroke], scale: float,
    *,
    max_angle_deg: float = 3.0,
    min_gap_px: float = 0.15,
    max_gap_px: float = 3.5,
    min_overlap_fraction: float = 0.30,
) -> list[Ribbon]:
    """把方向一致、垂距合理、投影重叠的直线聚成平行线带。"""
    indices = [index for index, stroke in enumerate(strokes) if stroke.is_line and stroke.length > 0]
    if len(indices) < 2:
        return []
    geometry = {}
    for index in indices:
        direction, params, half, _ = _line_geometry(strokes[index], scale)
        geometry[index] = (params, half, direction)

    parent = {index: index for index in indices}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    limit = math.radians(max_angle_deg)
    pair_count: dict[int, int] = {index: 0 for index in indices}
    for position, first in enumerate(indices):
        first_params, first_half, _ = geometry[first]
        for second in indices[position + 1:]:
            second_params, second_half, _ = geometry[second]
            delta_angle = abs(first_params[0] - second_params[0]) % math.pi
            delta_angle = min(delta_angle, math.pi - delta_angle)
            if delta_angle > limit:
                continue
            gap = abs(first_params[1] - second_params[1])
            if not (min_gap_px <= gap <= max_gap_px):
                continue
            overlap = _overlap(first_half, first_params[2], second_half, second_params[2])
            if overlap < min_overlap_fraction * min(first_half, second_half) * 2.0:
                continue
            union(first, second)
            pair_count[first] += 1
            pair_count[second] += 1

    groups: dict[int, list[int]] = {}
    for index in indices:
        groups.setdefault(find(index), []).append(index)

    # 传递闭包会把整张图的平行线串成一条带（例如所有水平尺寸线），
    # 因此按垂距排序后重新切分：相邻两条垂距超过 max_gap_px 就断开。
    split_groups: list[list[int]] = []
    for members in groups.values():
        ordered = sorted(members, key=lambda item: geometry[item][0][1])
        current = [ordered[0]]
        for index in ordered[1:]:
            if abs(geometry[index][0][1] - geometry[current[-1]][0][1]) > max_gap_px:
                if len(current) >= 2:
                    split_groups.append(current)
                current = [index]
            else:
                current.append(index)
        if len(current) >= 2:
            split_groups.append(current)

    ribbons: list[Ribbon] = []
    for members in split_groups:
        angles = [geometry[index][0][0] for index in members]
        offsets = [geometry[index][0][1] for index in members]
        projections = [geometry[index][0][2] for index in members]
        halves = [geometry[index][1] for index in members]
        angle = float(np.median(angles))
        spacing = float(np.median(np.abs(np.subtract.outer(offsets, offsets))))
        low = max(projection - half for projection, half in zip(projections, halves))
        high = min(projection + half for projection, half in zip(projections, halves))
        extent = max(0.0, high - low)
        if extent <= 0:
            extent = max(2.0 * half for half in halves)
        ribbons.append(Ribbon(
            members=sorted(members), angle=angle, spacing=spacing,
            extent=extent, pairs=sum(pair_count[index] for index in members) // 2,
        ))
    ribbons.sort(key=lambda ribbon: len(ribbon.members), reverse=True)
    return ribbons


def align_ribbon_endpoints(
    strokes: list[Stroke], ribbons: list[Ribbon], tolerance: float = 2.0,
) -> int:
    """把同一条平行线带的成员端点对齐。

    窗、家具轮廓、停车位图案一般都是「平行线 + 两端对齐」；把端点沿投影
    方向对齐到带的中位数位置，可显著减少评估栅格上的差分。
    """
    changed = 0
    for ribbon in ribbons:
        members = [strokes[index] for index in ribbon.members]
        if len(members) < 2:
            continue
        direction = np.array([math.cos(ribbon.angle), math.sin(ribbon.angle)])
        starts = [float(np.dot(member.p0, direction)) if
                  np.dot(np.asarray(member.p1) - member.p0, direction) >= 0
                  else float(np.dot(member.p1, direction)) for member in members]
        ends = [float(np.dot(member.p1, direction)) if
                np.dot(np.asarray(member.p1) - member.p0, direction) >= 0
                else float(np.dot(member.p0, direction)) for member in members]
        median_start = float(np.median(starts))
        median_end = float(np.median(ends))
        for member, start, end in zip(members, starts, ends):
            if abs(start - median_start) > tolerance or abs(end - median_end) > tolerance:
                continue
            if abs(end - start) <= 1e-6:
                continue
            p0 = np.asarray(member.p0)
            p1 = np.asarray(member.p1)
            forward = np.dot(p1 - p0, direction) >= 0
            low, high = (p0, p1) if forward else (p1, p0)
            low = low + direction * (median_start - start)
            high = high + direction * (median_end - end)
            member.p0 = (float(low[0]), float(low[1]))
            member.p1 = (float(high[0]), float(high[1]))
            member.length = float(np.linalg.norm(high - low))
            changed += 1
    return changed


def snap_axis_lines(strokes: list[Stroke], scale: float, tolerance_deg: float = 1.2) -> int:
    """把接近水平/竖直的直线吸附到精确方向，减少评估栅格的亚像素抖动。"""
    changed = 0
    limit = math.radians(tolerance_deg)
    for stroke in strokes:
        if not stroke.is_line:
            continue
        first = np.asarray(stroke.p0) * scale
        second = np.asarray(stroke.p1) * scale
        angle = math.atan2(second[1] - first[1], second[0] - first[0])
        for target in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            if abs(math.atan2(math.sin(angle - target), math.cos(angle - target))) <= limit:
                direction = np.array([math.cos(target), math.sin(target)])
                center = (first + second) / 2.0
                half = float(np.linalg.norm(second - first)) / 2.0
                first_snapped = center - direction * half
                second_snapped = center + direction * half
                stroke.p0 = (float(first_snapped[0]), float(first_snapped[1]))
                stroke.p1 = (float(second_snapped[0]), float(second_snapped[1]))
                changed += 1
                break
    return changed
