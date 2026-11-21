"""SVG 标注解析（本地训练/验证用，不参与 dev 推理）。

训练集与 test_final 的 SVG 带 ``semantic-id`` / ``instance-id``。本模块把它们
转换成与赛题 README 一致的图元结构，用于：

1. 生成 GT JSON（本地 PQ 评测）；
2. 统计颜色 → 语义先验；
3. 训练可选的图元分类器。

输入 SVG 属于赛事数据，本模块只在显式的训练/评测命令中调用；``infer.py``
只读取 PNG，不会读取任何 SVG 真值。
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import (
    COMMAND_ARC,
    COMMAND_CIRCLE,
    COMMAND_ELLIPSE,
    COMMAND_LINE,
    IGNORE_CLASS,
)

NUMBER = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
TOKEN_RE = re.compile(rf"[MmLlHhVvAaZzCcQqSsTt]|{NUMBER}")
COLOR_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
COLOR_RGB_RE = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")


@dataclass
class SvgPrimitive:
    command: int
    points: list[tuple[float, float]]
    length: float
    semantic_id: int
    instance_id: int
    rgb: list[int]
    layer_id: int
    width: float


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_color(value: str | None) -> list[int] | None:
    if not value:
        return None
    value = value.strip()
    if value.lower() in {"none", "transparent"}:
        return None
    match = COLOR_HEX_RE.match(value)
    if match:
        hex_value = match.group(1)
        if len(hex_value) == 3:
            return [int(channel * 2, 16) for channel in hex_value]
        return [int(hex_value[index:index + 2], 16) for index in (0, 2, 4)]
    match = COLOR_RGB_RE.match(value)
    if match:
        return [max(0, min(255, int(float(match.group(index))))) for index in (1, 2, 3)]
    named = {
        "black": [0, 0, 0], "white": [255, 255, 255], "red": [255, 0, 0],
        "green": [0, 128, 0], "blue": [0, 0, 255], "yellow": [255, 255, 0],
        "cyan": [0, 255, 255], "magenta": [255, 0, 255],
    }
    return named.get(value.lower())


def parse_viewbox(root: ET.Element) -> tuple[int, int]:
    view_box = root.get("viewBox")
    if view_box:
        parts = [float(value) for value in view_box.replace(",", " ").split()]
        if len(parts) == 4:
            return max(1, int(round(parts[2]))), max(1, int(round(parts[3])))
    width = root.get("width")
    height = root.get("height")
    if width and height:
        return max(1, int(float(re.sub(r"[^\d.]", "", width)))), max(1, int(float(re.sub(r"[^\d.]", "", height))))
    raise ValueError("SVG 缺少 viewBox 和 width/height")


def _ids(element: ET.Element) -> tuple[int, int]:
    semantic = element.get("semantic-id", element.get("semanticId"))
    instance = element.get("instance-id", element.get("instanceId"))
    semantic_id = int(semantic) - 1 if semantic is not None else IGNORE_CLASS
    instance_id = int(instance) if instance is not None else -1
    return semantic_id, instance_id


def _style(element: ET.Element) -> tuple[list[int], float]:
    stroke = element.get("stroke")
    fill = element.get("fill")
    color = parse_color(stroke) or parse_color(fill) or [0, 0, 0]
    try:
        width = float(element.get("stroke-width", "0.1"))
    except (TypeError, ValueError):
        width = 0.1
    return color, max(0.0, width)


def _tokenize(path_data: str) -> list[str]:
    return TOKEN_RE.findall(path_data)


def _arc_points(
    start: tuple[float, float], rx: float, ry: float, phi_deg: float,
    large_arc: int, sweep: int, end: tuple[float, float], samples: int = 16,
) -> list[tuple[float, float]]:
    """SVG endpoint 参数化转中心参数化并采样。"""
    x1, y1 = start
    x2, y2 = end
    if rx == 0 or ry == 0:
        return [end]
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi_deg % 360.0)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx2, dy2 = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_phi * dx2 + sin_phi * dy2
    y1p = -sin_phi * dx2 + cos_phi * dy2
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale
    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    factor = 0.0 if denominator == 0 else math.sqrt(max(0.0, numerator / denominator))
    if large_arc == sweep:
        factor = -factor
    cxp = factor * rx * y1p / ry
    cyp = -factor * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0
    theta1 = math.atan2((y1p - cyp) / ry, (x1p - cxp) / rx)
    theta2 = math.atan2((-y1p - cyp) / ry, (-x1p - cxp) / rx)
    delta = theta2 - theta1
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi
    points: list[tuple[float, float]] = []
    for index in range(1, samples + 1):
        theta = theta1 + delta * index / samples
        x = cx + rx * math.cos(theta) * cos_phi - ry * math.sin(theta) * sin_phi
        y = cy + rx * math.cos(theta) * sin_phi + ry * math.sin(theta) * cos_phi
        points.append((x, y))
    return points


def _bezier_point(
    p0: tuple[float, float], p1: tuple[float, float],
    p2: tuple[float, float], p3: tuple[float, float] | None, t: float,
) -> tuple[float, float]:
    if p3 is None:  # quadratic
        mt = 1 - t
        return (
            mt * mt * p0[0] + 2 * mt * t * p1[0] + t * t * p2[0],
            mt * mt * p0[1] + 2 * mt * t * p1[1] + t * t * p2[1],
        )
    mt = 1 - t
    return (
        mt ** 3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t ** 3 * p3[0],
        mt ** 3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t ** 3 * p3[1],
    )


def parse_path(path_data: str) -> tuple[list[list[tuple[float, float]]], int]:
    """解析 path 为若干子路径折线，并返回首段图元类型。

    ``command`` 与参考脚本一致：首段是 Arc 记 1，否则记 0。曲线按折线近似。
    """
    tokens = _tokenize(path_data)
    index = 0
    command = ""
    current = (0.0, 0.0)
    subpath_start = (0.0, 0.0)
    subpaths: list[list[tuple[float, float]]] = []
    current_subpath: list[tuple[float, float]] = []
    first_kind: str | None = None
    previous_control: tuple[float, float] | None = None
    previous_quad_control: tuple[float, float] | None = None

    def flush() -> None:
        nonlocal current_subpath
        if len(current_subpath) >= 2:
            subpaths.append(current_subpath)
        current_subpath = []

    def take(count: int) -> list[float]:
        nonlocal index
        values = [float(tokens[index + offset]) for offset in range(count)]
        index += count
        return values

    while index < len(tokens):
        token = tokens[index]
        if token.isalpha():
            command = token
            index += 1
            if command in "Zz":
                if current_subpath and current_subpath[-1] != subpath_start:
                    current_subpath.append(subpath_start)
                flush()
                current = subpath_start
                previous_control = previous_quad_control = None
                continue
        if command in "Mm":
            x, y = take(2)
            if command == "m":
                x, y = current[0] + x, current[1] + y
            if current_subpath:
                flush()
            current = (x, y)
            subpath_start = current
            current_subpath = [current]
            command = "L" if command == "M" else "l"
            previous_control = previous_quad_control = None
        elif command in "Ll":
            x, y = take(2)
            if command == "l":
                x, y = current[0] + x, current[1] + y
            if first_kind is None:
                first_kind = "line"
            current = (x, y)
            current_subpath.append(current)
            previous_control = previous_quad_control = None
        elif command in "Hh":
            (x,) = take(1)
            if command == "h":
                x += current[0]
            if first_kind is None:
                first_kind = "line"
            current = (x, current[1])
            current_subpath.append(current)
            previous_control = previous_quad_control = None
        elif command in "Vv":
            (y,) = take(1)
            if command == "v":
                y += current[1]
            if first_kind is None:
                first_kind = "line"
            current = (current[0], y)
            current_subpath.append(current)
            previous_control = previous_quad_control = None
        elif command in "Aa":
            values = take(7)
            rx, ry, rotation, large_arc, sweep, x, y = values
            if command == "a":
                x, y = current[0] + x, current[1] + y
            if first_kind is None:
                first_kind = "arc"
            current_subpath.extend(_arc_points(
                current, rx, ry, rotation, int(large_arc), int(sweep), (x, y),
            ))
            current = (x, y)
            previous_control = previous_quad_control = None
        elif command in "Cc":
            values = take(6)
            x1, y1, x2, y2, x, y = values
            if command == "c":
                x1, y1 = current[0] + x1, current[1] + y1
                x2, y2 = current[0] + x2, current[1] + y2
                x, y = current[0] + x, current[1] + y
            for step in range(1, 13):
                current_subpath.append(_bezier_point(
                    current, (x1, y1), (x2, y2), (x, y), step / 12.0,
                ))
            previous_control = (x2, y2)
            previous_quad_control = None
            current = (x, y)
        elif command in "Qq":
            values = take(4)
            x1, y1, x, y = values
            if command == "q":
                x1, y1 = current[0] + x1, current[1] + y1
                x, y = current[0] + x, current[1] + y
            for step in range(1, 13):
                current_subpath.append(_bezier_point(
                    current, (x1, y1), (x, y), None, step / 12.0,
                ))
            previous_quad_control = (x1, y1)
            previous_control = None
            current = (x, y)
        elif command in "Ss":
            values = take(4)
            x2, y2, x, y = values
            if command == "s":
                x2, y2 = current[0] + x2, current[1] + y2
                x, y = current[0] + x, current[1] + y
            control1 = (
                2 * current[0] - previous_control[0] if previous_control
                else current[0],
                2 * current[1] - previous_control[1] if previous_control
                else current[1],
            )
            for step in range(1, 13):
                current_subpath.append(_bezier_point(
                    current, control1, (x2, y2), (x, y), step / 12.0,
                ))
            previous_control = (x2, y2)
            current = (x, y)
        elif command in "Tt":
            values = take(2)
            x, y = values
            if command == "t":
                x, y = current[0] + x, current[1] + y
            control = (
                2 * current[0] - previous_quad_control[0] if previous_quad_control
                else current[0],
                2 * current[1] - previous_quad_control[1] if previous_quad_control
                else current[1],
            )
            for step in range(1, 13):
                current_subpath.append(_bezier_point(
                    current, control, (x, y), None, step / 12.0,
                ))
            previous_quad_control = control
            current = (x, y)
        else:
            # 未知命令：跳过当前 token，避免死循环。
            index += 1
    flush()
    return subpaths, COMMAND_ARC if first_kind == "arc" else COMMAND_LINE


def _subpath_lengths(subpaths: list[list[tuple[float, float]]]) -> tuple[list[float], float]:
    lengths = []
    for points in subpaths:
        total = 0.0
        for first, second in zip(points, points[1:]):
            total += math.hypot(second[0] - first[0], second[1] - first[1])
        lengths.append(total)
    return lengths, sum(lengths)


def sample_subpaths(
    subpaths: list[list[tuple[float, float]]], count: int = 4,
) -> list[tuple[float, float]]:
    """按总弧长在子路径上等距采样 ``count`` 个点。"""
    lengths, total = _subpath_lengths(subpaths)
    if not subpaths or total <= 0:
        return []
    samples: list[tuple[float, float]] = []
    for index in range(count):
        target = total * index / (count - 1)
        samples.append(_point_at_distance(subpaths, lengths, target))
    return samples


def _point_at_distance(
    subpaths: list[list[tuple[float, float]]], lengths: list[float], target: float,
) -> tuple[float, float]:
    if target <= 0:
        return subpaths[0][0]
    remaining = target
    for subpath, length in zip(subpaths, lengths):
        if remaining <= length or subpath is subpaths[-1]:
            return _point_at_fraction(subpath, 1.0 if length <= 0 else remaining / length)
        remaining -= length
    return subpaths[-1][-1]


def _point_at_fraction(points: list[tuple[float, float]], fraction: float) -> tuple[float, float]:
    if len(points) == 1:
        return points[0]
    lengths = [
        math.hypot(second[0] - first[0], second[1] - first[1])
        for first, second in zip(points, points[1:])
    ]
    total = sum(lengths)
    if total <= 0:
        return points[0]
    target = total * fraction
    travelled = 0.0
    for (first, second), length in zip(zip(points, points[1:]), lengths):
        if travelled + length >= target or (first, second) == (points[-2], points[-1]):
            ratio = 0.0 if length <= 0 else (target - travelled) / length
            return (
                first[0] + (second[0] - first[0]) * ratio,
                first[1] + (second[1] - first[1]) * ratio,
            )
        travelled += length
    return points[-1]


def _circle_primitives(
    element: ET.Element, cx: float, cy: float, rx: float, ry: float,
    command: int, semantic_id: int, instance_id: int,
    rgb: list[int], layer_id: int, width: float,
) -> SvgPrimitive:
    points = [
        (cx + rx * math.cos(angle), cy + ry * math.sin(angle))
        for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
    ]
    if command == COMMAND_CIRCLE:
        length = 2 * math.pi * rx
    else:
        major, minor = max(rx, ry), min(rx, ry)
        length = 2 * math.pi * minor + 4 * (major - minor)
    return SvgPrimitive(command, points, length, semantic_id, instance_id, rgb, layer_id, width)


def iter_primitives(svg_path: str | Path) -> Iterator[SvgPrimitive]:
    root = ET.parse(svg_path).getroot()
    layer_counter = 0

    def walk(element: ET.Element, current_layer: int) -> Iterator[SvgPrimitive]:
        nonlocal layer_counter
        tag = _local(element.tag)
        layer = current_layer
        if tag == "g":
            layer_counter += 1
            layer = layer_counter
        if tag in {"path", "circle", "ellipse"}:
            semantic_id, instance_id = _ids(element)
            rgb, width = _style(element)
            if tag == "path":
                subpaths, command = parse_path(element.get("d", ""))
                samples = sample_subpaths(subpaths)
                _, total = _subpath_lengths(subpaths)
                if len(samples) == 4 and total > 0:
                    yield SvgPrimitive(
                        command, samples, total, semantic_id, instance_id,
                        rgb, layer, width,
                    )
            elif tag == "circle":
                try:
                    cx = float(element.get("cx", "0"))
                    cy = float(element.get("cy", "0"))
                    radius = float(element.get("r", "0"))
                except (TypeError, ValueError):
                    return
                if radius > 0:
                    yield _circle_primitives(
                        element, cx, cy, radius, radius, COMMAND_CIRCLE,
                        semantic_id, instance_id, rgb, layer, width,
                    )
            elif tag == "ellipse":
                try:
                    cx = float(element.get("cx", "0"))
                    cy = float(element.get("cy", "0"))
                    rx = float(element.get("rx", "0"))
                    ry = float(element.get("ry", "0"))
                except (TypeError, ValueError):
                    return
                if rx > 0 and ry > 0:
                    yield _circle_primitives(
                        element, cx, cy, rx, ry, COMMAND_ELLIPSE,
                        semantic_id, instance_id, rgb, layer, width,
                    )
        for child in list(element):
            yield from walk(child, layer)

    yield from walk(root, 1)


def svg_to_output(svg_path: str | Path) -> dict:
    """把一个标注 SVG 转成 GT JSON（12 字段）。"""
    from .schema import assemble_output

    root = ET.parse(svg_path).getroot()
    width, height = parse_viewbox(root)
    primitives = list(iter_primitives(svg_path))
    return assemble_output(primitives, width=width, height=height)
