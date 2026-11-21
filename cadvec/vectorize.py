"""从位图颜色层提取 Line / Arc / Circle / Ellipse 图元。

实现不依赖 OpenCV：
- ``skimage.measure.find_contours`` + 代数圆拟合 / ``EllipseModel`` 找圆与椭圆；
- ``skimage.morphology.skeletonize`` + 概率 Hough 找直线；
- 按角度覆盖区分整圆与圆弧；
- 合并共线短线段和内外轮廓产生的重复圆。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from skimage.measure import EllipseModel, approximate_polygon, find_contours
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
from scipy import ndimage

from .config import (
    COMMAND_ARC,
    COMMAND_CIRCLE,
    COMMAND_ELLIPSE,
    COMMAND_LINE,
    InferenceConfig,
)
from .image import ColorLayer


@dataclass
class Primitive:
    command: int
    points: list[tuple[float, float]]
    length: float
    rgb: list[int]
    layer_id: int
    width: float
    #: 由分类器填写的语义与实例，几何阶段保持未知。
    semantic_id: int = 35
    instance_id: int = -1
    confidence: float = 0.0


@dataclass
class _Shape:
    kind: str  # circle / arc / ellipse
    cx: float
    cy: float
    a: float
    b: float
    theta: float
    coverage: float
    start: float
    end: float
    residual: float
    #: 命中的轮廓数量。粗描边圆/椭圆通常有内外两条轮廓，孤立孔洞只有一条。
    support: int = 1


def _fit_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    if len(x) < 3:
        return 0.0, 0.0, 0.0, 1.0
    matrix = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    target = x * x + y * y
    try:
        solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0, 1.0
    cx, cy = float(solution[0]), float(solution[1])
    radius_squared = float(solution[2] + cx * cx + cy * cy)
    if radius_squared <= 0:
        return 0.0, 0.0, 0.0, 1.0
    radius = math.sqrt(radius_squared)
    distances = np.hypot(x - cx, y - cy)
    residual = float(np.std(distances - radius) / max(radius, 1e-6))
    return cx, cy, radius, residual


def _angular_span(points: np.ndarray, cx: float, cy: float) -> tuple[float, float, float]:
    """返回 (覆盖角, 起点角, 终点角)，覆盖角是除最大缺口外的角度范围。"""
    angles = np.sort(np.mod(np.arctan2(points[:, 1] - cy, points[:, 0] - cx), 2 * math.pi))
    if len(angles) < 2:
        return 0.0, 0.0, 0.0
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * math.pi]))
    gap_index = int(np.argmax(gaps))
    coverage = float(2 * math.pi - gaps[gap_index])
    start = float(angles[(gap_index + 1) % len(angles)])
    return coverage, start, start + coverage


def _is_rectangular(points: np.ndarray) -> bool:
    """闭合轮廓若能用 4-6 个近直角顶点近似，视为多边形而不是圆/椭圆。"""
    if len(points) < 8:
        return False
    try:
        polygon = approximate_polygon(points, tolerance=2.0)
    except Exception:  # noqa: BLE001 - 拟合失败时交给圆/椭圆残差判断
        return False
    if not 4 <= len(polygon) <= 6:
        return False
    closed = np.linalg.norm(points[0] - points[-1]) < 2.0
    if not closed:
        return False
    right_angles = 0
    for index in range(len(polygon)):
        previous = polygon[index - 1] - polygon[index]
        following = polygon[(index + 1) % len(polygon)] - polygon[index]
        norm = np.linalg.norm(previous) * np.linalg.norm(following)
        if norm <= 1e-9:
            continue
        cosine = float(np.dot(previous, following) / norm)
        if abs(cosine) < 0.35:  # 约 70-110 度
            right_angles += 1
    return right_angles >= max(3, len(polygon) - 1)


def _circle_shapes(points: np.ndarray, config: InferenceConfig) -> list[_Shape]:
    if len(points) < 8:
        return []
    cx, cy, radius, residual = _fit_circle(points)
    if radius < config.min_radius_px or radius > 4 * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1.0):
        return []
    if residual > config.circle_residual:
        return []
    if _is_rectangular(points):
        return []
    coverage, start, end = _angular_span(points, cx, cy)
    closed = np.linalg.norm(points[0] - points[-1]) < 2.0
    if closed and coverage >= 2 * math.pi * config.full_circle_coverage:
        return [_Shape("circle", cx, cy, radius, radius, 0.0, 2 * math.pi, start, end, residual)]
    if coverage >= 2 * math.pi * config.min_arc_coverage:
        return [_Shape("arc", cx, cy, radius, radius, 0.0, coverage, start, end, residual)]
    return []


def _ellipse_shapes(points: np.ndarray, config: InferenceConfig) -> list[_Shape]:
    if len(points) < 12:
        return []
    closed = np.linalg.norm(points[0] - points[-1]) < 2.0
    if not closed:
        return []
    if _is_rectangular(points):
        return []
    model = EllipseModel()
    try:
        if not model.estimate(points):
            return []
    except Exception:  # noqa: BLE001 - 退化点集
        return []
    cx, cy, a, b, theta = [float(value) for value in model.params]
    if a <= 0 or b <= 0:
        return []
    major, minor = max(a, b), min(a, b)
    if minor < config.min_radius_px:
        return []
    if major / max(minor, 1e-6) > config.max_ellipse_aspect:
        return []
    residuals = model.residuals(points)
    rms = float(math.sqrt(np.mean(np.square(residuals))))
    if rms / max(major, 1e-6) > config.ellipse_residual:
        return []
    if abs(a - b) / max(a, b) < 0.12:
        # 接近正圆时交给 _circle_shapes，避免同一轮廓产生重复候选。
        return []
    # 采样一次完整椭圆判断轮廓是否真的覆盖整圈。
    coverage, _, _ = _angular_span(points, cx, cy)
    if coverage < 2 * math.pi * config.full_circle_coverage:
        return []
    return [_Shape(
        "ellipse", cx, cy, abs(a), abs(b), theta,
        2 * math.pi, 0.0, 2 * math.pi, rms / max(major, 1e-6),
    )]


def _merge_shapes(shapes: list[_Shape], config: InferenceConfig) -> list[_Shape]:
    """合并同一圆/椭圆的内外轮廓重复项。"""
    merged: list[_Shape] = []
    for shape in shapes:
        match = None
        for index, existing in enumerate(merged):
            center_distance = math.hypot(shape.cx - existing.cx, shape.cy - existing.cy)
            radius_scale = max(shape.a, shape.b, existing.a, existing.b, 1.0)
            if center_distance > max(2.0, 0.12 * radius_scale):
                continue
            if abs(shape.a - existing.a) > max(4.0, 0.30 * radius_scale):
                continue
            if abs(shape.b - existing.b) > max(4.0, 0.30 * radius_scale):
                continue
            match = index
            break
        if match is None:
            merged.append(shape)
            continue
        existing = merged[match]
        weight_new = max(shape.coverage, 1e-6)
        weight_old = max(existing.coverage, 1e-6)
        total = weight_new + weight_old
        kind = "circle" if "circle" in (shape.kind, existing.kind) else (
            "ellipse" if "ellipse" in (shape.kind, existing.kind) else "arc"
        )
        if {shape.kind, existing.kind} == {"circle", "arc"}:
            kind = "circle" if max(shape.coverage, existing.coverage) >= (
                2 * math.pi * config.full_circle_coverage
            ) else "arc"
        merged[match] = _Shape(
            kind,
            (shape.cx * weight_new + existing.cx * weight_old) / total,
            (shape.cy * weight_new + existing.cy * weight_old) / total,
            (shape.a * weight_new + existing.a * weight_old) / total,
            (shape.b * weight_new + existing.b * weight_old) / total,
            existing.theta,
            max(shape.coverage, existing.coverage),
            min(shape.start, existing.start) if kind == "arc" else 0.0,
            max(shape.end, existing.end) if kind == "arc" else 2 * math.pi,
            (shape.residual * weight_new + existing.residual * weight_old) / total,
            support=shape.support + existing.support,
        )
    return merged


def _sample_shape(shape: _Shape, count: int) -> list[tuple[float, float]]:
    """沿图元采样 ``count`` 个点，用于绘制/宽度估计。"""
    if shape.kind == "circle":
        return [
            (shape.cx + shape.a * math.cos(2 * math.pi * index / count),
             shape.cy + shape.a * math.sin(2 * math.pi * index / count))
            for index in range(count + 1)
        ]
    if shape.kind == "ellipse":
        return [
            _ellipse_point(shape, 2 * math.pi * index / count)
            for index in range(count + 1)
        ]
    return [
        (shape.cx + shape.a * math.cos(shape.start + shape.coverage * index / count),
         shape.cy + shape.a * math.sin(shape.start + shape.coverage * index / count))
        for index in range(count + 1)
    ]


def _shape_points(shape: _Shape, count: int = 4) -> list[tuple[float, float]]:
    """输出用的 4 个点：Line/Arc 沿弧长，Circle/Ellipse 取 0/90/180/270 度。"""
    if shape.kind == "circle":
        return [
            (shape.cx + shape.a * math.cos(angle), shape.cy + shape.a * math.sin(angle))
            for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
        ]
    if shape.kind == "ellipse":
        return [
            _ellipse_point(shape, angle)
            for angle in (0, math.pi / 2, math.pi, 3 * math.pi / 2)
        ]
    points = []
    for index in range(count):
        angle = shape.start + shape.coverage * index / (count - 1)
        points.append((shape.cx + shape.a * math.cos(angle), shape.cy + shape.a * math.sin(angle)))
    return points


def _shape_length(shape: _Shape) -> float:
    if shape.kind == "circle":
        return 2 * math.pi * shape.a
    if shape.kind == "ellipse":
        # 128 点折线近似，避免使用近似公式带来的系统误差。
        samples = [
            _ellipse_point(shape, 2 * math.pi * index / 128)
            for index in range(129)
        ]
        return sum(
            math.hypot(second[0] - first[0], second[1] - first[1])
            for first, second in zip(samples, samples[1:])
        )
    return shape.a * shape.coverage


def _ellipse_point(shape: _Shape, angle: float) -> tuple[float, float]:
    cosine, sine = math.cos(shape.theta), math.sin(shape.theta)
    x = shape.a * math.cos(angle)
    y = shape.b * math.sin(angle)
    return shape.cx + x * cosine - y * sine, shape.cy + x * sine + y * cosine


def _skeleton_coords(skeleton: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(skeleton)
    return np.column_stack([columns, rows]).astype(np.float64)


def _center_is_empty(mask: np.ndarray, shape: _Shape, max_fraction: float = 0.35) -> bool:
    """整圆/椭圆内部应大体为空，避免把实心色块或线交点当成圆。"""
    height, width = mask.shape
    cosine, sine = math.cos(shape.theta), math.sin(shape.theta)
    occupied = 0
    total = 0
    for factor in (0.0, 0.2, 0.4):
        for index in range(8):
            angle = 2 * math.pi * index / 8
            local_x = shape.a * factor * math.cos(angle)
            local_y = shape.b * factor * math.sin(angle)
            x = shape.cx + local_x * cosine - local_y * sine
            y = shape.cy + local_x * sine + local_y * cosine
            row, column = int(round(y)), int(round(x))
            if 0 <= row < height and 0 <= column < width:
                total += 1
                occupied += int(bool(mask[row, column]))
    return total > 0 and occupied / total <= max_fraction


def _refine_circular_shape(
    shape: _Shape, points: np.ndarray, mask: np.ndarray, config: InferenceConfig,
) -> _Shape | None:
    """用骨架点重新拟合圆/弧，并用角度覆盖区分整圆与圆弧。

    轮廓上的圆候选可能来自粗圆弧描边的闭合外轮廓（形状像香蕉），仅凭
    轮廓点会误判成整圆；骨架是描边中心线，能直接反映真实角度覆盖。
    """
    if len(points) < 10:
        return None
    radius = (shape.a + shape.b) / 2.0
    distances = np.abs(
        np.hypot(points[:, 0] - shape.cx, points[:, 1] - shape.cy) - radius
    )
    selected = points[distances <= config.skeleton_support_tolerance]
    if len(selected) < 10:
        return None
    cx, cy, fitted_radius, residual = _fit_circle(selected)
    if fitted_radius < config.min_radius_px:
        return None
    if residual > max(config.circle_residual, 0.18):
        return None
    coverage, start, end = _angular_span(selected, cx, cy)
    if coverage < 2 * math.pi * config.min_arc_coverage:
        return None
    if coverage >= 2 * math.pi * config.full_circle_coverage:
        # 整圆应由内外两条轮廓共同支持；孤立孔洞（如线交叉围出的空白）
        # 只有一条轮廓，不能当作实线圆。
        if shape.support < 2:
            return None
        circle = _Shape(
            "circle", cx, cy, fitted_radius, fitted_radius,
            0.0, 2 * math.pi, start, end, residual, support=shape.support,
        )
        if not _center_is_empty(mask, circle):
            return None
        return circle
    return _Shape(
        "arc", cx, cy, fitted_radius, fitted_radius,
        0.0, coverage, start, end, residual, support=shape.support,
    )


def _ellipse_distance(points: np.ndarray, shape: _Shape) -> np.ndarray:
    cosine, sine = math.cos(shape.theta), math.sin(shape.theta)
    dx = points[:, 0] - shape.cx
    dy = points[:, 1] - shape.cy
    u = dx * cosine + dy * sine
    v = -dx * sine + dy * cosine
    value = np.sqrt(
        (u / max(abs(shape.a), 1e-6)) ** 2
        + (v / max(abs(shape.b), 1e-6)) ** 2
    )
    return np.abs(value - 1.0) * max(min(abs(shape.a), abs(shape.b)), 1e-6)


def _shape_distance(
    shape: _Shape, x: float, y: float, angle_slack: float = 0.0,
) -> float:
    """点到圆/圆弧/椭圆中心线的距离；圆弧的角范围外返回无穷大。"""
    if shape.kind in ("circle", "arc"):
        dx, dy = x - shape.cx, y - shape.cy
        radius = math.hypot(dx, dy)
        distance = abs(radius - shape.a)
        if shape.kind == "arc":
            angle = math.atan2(dy, dx) % (2 * math.pi)
            delta = (angle - shape.start) % (2 * math.pi)
            slack = math.atan2(max(distance, 1.0), max(shape.a, 1e-6))
            if delta > shape.coverage + slack + angle_slack:
                return float("inf")
        return float(distance)
    return float(_ellipse_distance(np.array([[x, y]], dtype=np.float64), shape)[0])


def _hugs_curve(
    first: tuple[float, float],
    second: tuple[float, float],
    shapes: list[_Shape],
    tolerance: float,
) -> bool:
    """线段是否只是已检出曲线的残留描边。

    Hough 偶尔会在圆/圆弧边缘留下短碎片；若线段上大部分采样点都紧贴同一条
    曲线，它就不是独立直线，而是拟合误差造成的重复图元。
    """
    if not shapes:
        return False
    fractions = (0.0, 0.25, 0.5, 0.75, 1.0)
    samples = [
        (
            first[0] + (second[0] - first[0]) * fraction,
            first[1] + (second[1] - first[1]) * fraction,
        )
        for fraction in fractions
    ]
    required = max(3, int(math.ceil(0.6 * len(samples))))
    length = math.dist(first, second)
    for shape in shapes:
        # 圆弧拟合可能只覆盖真实圆弧的一段，留出与线段长度相当的角容差，
        # 这样圆弧两端的残留碎片也能被识别；直线若只与圆弧相切，多数
        # 采样点仍会远离圆弧，不会误删。
        angle_slack = length / max(shape.a, 1e-6) if shape.kind == "arc" else 0.0
        near = sum(
            1 for x, y in samples
            if _shape_distance(shape, x, y, angle_slack) <= tolerance
        )
        if near >= required:
            return True
    return False


def _refine_ellipse_shape(
    shape: _Shape, points: np.ndarray, mask: np.ndarray, config: InferenceConfig,
) -> _Shape | None:
    if shape.support < 2:
        return None
    if len(points) < 12:
        return None
    distances = _ellipse_distance(points, shape)
    selected = points[distances <= config.skeleton_support_tolerance]
    if len(selected) < 12:
        return None
    coverage, _, _ = _angular_span(selected, shape.cx, shape.cy)
    if coverage < 2 * math.pi * config.full_circle_coverage:
        return None
    model = EllipseModel()
    try:
        if not model.estimate(selected):
            return None
    except Exception:  # noqa: BLE001 - 退化点集
        return None
    cx, cy, a, b, theta = [float(value) for value in model.params]
    if a <= 0 or b <= 0:
        return None
    major, minor = max(a, b), min(a, b)
    if minor < config.min_radius_px or major / max(minor, 1e-6) > config.max_ellipse_aspect:
        return None
    if abs(a - b) / max(a, b) < 0.12:
        # 接近正圆：交给圆候选，避免重复输出。
        return None
    residual_values = model.residuals(selected)
    rms = float(math.sqrt(np.mean(np.square(residual_values))))
    if rms / max(major, 1e-6) > config.ellipse_residual:
        return None
    ellipse = _Shape(
        "ellipse", cx, cy, abs(a), abs(b), theta,
        2 * math.pi, 0.0, 2 * math.pi, rms / max(major, 1e-6),
        support=shape.support,
    )
    if not _center_is_empty(mask, ellipse):
        return None
    return ellipse


def _refine_shapes(
    shapes: list[_Shape], skeleton: np.ndarray, mask: np.ndarray,
    config: InferenceConfig,
) -> list[_Shape]:
    points = _skeleton_coords(skeleton)
    refined: list[_Shape] = []
    for shape in shapes:
        if shape.kind == "ellipse":
            candidate = _refine_ellipse_shape(shape, points, mask, config)
        else:
            candidate = _refine_circular_shape(shape, points, mask, config)
        if candidate is not None:
            refined.append(candidate)
    return _deduplicate_closed(refined)


def _deduplicate_closed(shapes: list[_Shape]) -> list[_Shape]:
    """精修后的圆心/半径可能轻微漂移，去掉重复的整圆/椭圆。"""
    kept: list[_Shape] = []
    for shape in shapes:
        if shape.kind not in ("circle", "ellipse"):
            kept.append(shape)
            continue
        duplicate = None
        for index, existing in enumerate(kept):
            if existing.kind not in ("circle", "ellipse"):
                continue
            scale = max(shape.a, shape.b, existing.a, existing.b, 1.0)
            if math.hypot(shape.cx - existing.cx, shape.cy - existing.cy) > max(1.0, 0.06 * scale):
                continue
            if abs(shape.a - existing.a) > max(2.0, 0.15 * scale):
                continue
            if abs(shape.b - existing.b) > max(2.0, 0.15 * scale):
                continue
            duplicate = index
            break
        if duplicate is None:
            kept.append(shape)
        elif shape.support > kept[duplicate].support:
            kept[duplicate] = shape
    return kept


def _line_segments(skeleton: np.ndarray, config: InferenceConfig) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """概率 Hough 对骨架采样敏感，固定多组种子取并集，避免偶发漏线。"""
    if skeleton.sum() < 4:
        return []
    found: list = []
    for offset in range(4):
        try:
            found.extend(probabilistic_hough_line(
                skeleton, threshold=config.hough_threshold,
                line_length=config.hough_min_length, line_gap=config.hough_line_gap,
                rng=config.seed + offset,
            ))
        except TypeError:  # 旧版 skimage 没有 rng 参数
            found.extend(probabilistic_hough_line(
                skeleton, threshold=config.hough_threshold,
                line_length=config.hough_min_length, line_gap=config.hough_line_gap,
            ))
            break
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for first, second in found:
        segment = (
            (float(first[0]), float(first[1])),
            (float(second[0]), float(second[1])),
        )
        if math.dist(*segment) < config.hough_min_length * 0.75:
            continue
        key = (round(segment[0][0], 3), round(segment[0][1], 3),
               round(segment[1][0], 3), round(segment[1][1], 3))
        if key in seen:
            continue
        seen.add(key)
        segments.append(segment)
    return segments


def _angle_difference(first: float, second: float) -> float:
    difference = abs(first - second) % math.pi
    return min(difference, math.pi - difference)


def _cluster_line(cluster: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回聚类点的中心、主方向与投影范围。"""
    points = np.array([point for segment in cluster for point in segment], dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    direction = vectors[0]
    projection = centered @ direction
    return center, direction, projection


def _cluster_fit(
    cluster: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """缓存聚类拟合结果，避免两两比较时重复 SVD。"""
    points = np.array([point for segment in cluster for point in segment], dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    try:
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vectors[0]
    return center, direction, centered @ direction, points


def _fits_mergeable(first, second, config: InferenceConfig) -> bool:
    """两个线段聚类是否共线且间隙足够小。"""
    if first is None or second is None:
        return False
    center, direction, projection, _ = first
    _, other_direction, _, points = second
    if _angle_difference(
        math.atan2(direction[1], direction[0]),
        math.atan2(other_direction[1], other_direction[0]),
    ) > math.radians(config.merge_angle_deg):
        return False
    normal = (-direction[1], direction[0])
    distances = np.abs((points - center) @ normal)
    if float(distances.max()) > config.merge_distance_px:
        return False
    segment_projection = (points - center) @ direction
    # 与整个聚类的投影范围比较，而不是只比较第一条线段。
    gap = max(
        float(segment_projection.min() - projection.max()),
        float(projection.min() - segment_projection.max()),
    )
    return gap <= config.merge_gap_px


def _merge_lines(segments, config: InferenceConfig):
    """把 Hough 碎片聚成完整线段。

    Hough 有时把一条长线拆成「长段 + 桥接小段」；如果按长度贪心一次分配，
    桥接小段可能最后才被处理，导致长段与另一侧碎片错失合并。这里反复做
    两两合并直到收敛，并缓存每个聚类的拟合结果控制开销。
    """
    clusters = [[segment] for segment in segments]
    fits = [_cluster_fit(cluster) for cluster in clusters]
    changed = True
    while changed:
        changed = False
        first_index = 0
        while first_index < len(clusters):
            if fits[first_index] is None:
                first_index += 1
                continue
            second_index = first_index + 1
            while second_index < len(clusters):
                if _fits_mergeable(
                    fits[first_index], fits[second_index], config,
                ):
                    clusters[first_index].extend(clusters[second_index])
                    clusters.pop(second_index)
                    fits.pop(second_index)
                    fits[first_index] = _cluster_fit(clusters[first_index])
                    changed = True
                    continue
                second_index += 1
            first_index += 1

    merged = []
    for cluster in clusters:
        try:
            center, direction, projection = _cluster_line(cluster)
        except np.linalg.LinAlgError:
            continue
        first = center + direction * projection.min()
        second = center + direction * projection.max()
        if math.dist(first, second) >= config.hough_min_length * 0.75:
            merged.append((
                (float(first[0]), float(first[1])),
                (float(second[0]), float(second[1])),
            ))
    return merged


def _line_width(
    mask: np.ndarray, skeleton: np.ndarray, distance: np.ndarray | None = None,
) -> float:
    if distance is None:
        distance = ndimage.distance_transform_edt(mask)
    values = distance[skeleton]
    if values.size == 0:
        return 1.0
    return max(0.5, float(2.0 * np.median(values)))


def _curve_width(
    mask: np.ndarray, shape: _Shape, distance: np.ndarray | None = None,
) -> float:
    height, width = mask.shape
    samples = _sample_shape(shape, 48)
    if distance is None:
        distance = ndimage.distance_transform_edt(mask)
    values = []
    for x, y in samples:
        column, row = int(round(x)), int(round(y))
        if 0 <= row < height and 0 <= column < width:
            values.append(distance[row, column])
    if not values:
        return 1.0
    width = 2.0 * float(np.median(values))
    # 采样点落在轮廓边缘时距离为 0，用整体厚度兜底。
    if width < 0.5:
        width = float(2.0 * np.percentile(distance[mask], 60)) if mask.any() else 1.0
    return max(0.5, width)


def _resize_mask(mask: np.ndarray, max_size: int) -> tuple[np.ndarray, float]:
    height, width = mask.shape
    longest = max(height, width)
    if longest <= max_size:
        return mask, 1.0
    scale = max_size / longest
    new_size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    resized = np.asarray(image.resize(new_size, Image.NEAREST)) > 0
    return resized, scale


def vectorize_layer(
    layer: ColorLayer,
    config: InferenceConfig,
    *,
    detection_scale: float = 1.0,
    canvas_width: int = 100,
    canvas_height: int = 100,
) -> list[Primitive]:
    """提取一个颜色层中的全部图元（检测坐标 -> 画布坐标）。"""
    mask = layer.mask
    if mask.sum() < config.min_layer_pixels:
        return []
    skeleton = skeletonize(mask)
    if skeleton.sum() == 0:
        return []

    height, width = mask.shape
    to_canvas_x = canvas_width / max(width * detection_scale, 1e-6)
    to_canvas_y = canvas_height / max(height * detection_scale, 1e-6)
    # skimage 的像素坐标以像素中心为整数；SVG 坐标以图像左上角为 0，
    # 因此像素索引需加 0.5 再换算到画布，避免整体约半像素的系统偏移。
    offset_x = 0.5 * to_canvas_x
    offset_y = 0.5 * to_canvas_y

    shapes: list[_Shape] = []
    for contour in find_contours(mask.astype(np.float64), 0.5):
        points = contour[:, ::-1]  # (row, col) -> (x, y)
        if len(points) < 8:
            continue
        shapes.extend(_circle_shapes(points, config))
        shapes.extend(_ellipse_shapes(points, config))
    shapes = _merge_shapes(shapes, config)
    shapes = _refine_shapes(shapes, skeleton, mask, config)

    primitives: list[Primitive] = []
    explained = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(explained)

    distance_cache: dict[str, np.ndarray] = {}

    def distance_map() -> np.ndarray:
        # 每层只计算一次距离变换；逐图元重算会占掉三成以上运行时间。
        if "value" not in distance_cache:
            distance_cache["value"] = ndimage.distance_transform_edt(mask)
        return distance_cache["value"]

    def append_shape(shape: _Shape) -> None:
        samples = _sample_shape(shape, 64)
        width_px = _curve_width(mask, shape, distance_map())
        draw.line([(x, y) for x, y in samples], fill=1,
                  width=max(2, int(round(width_px))))
        if shape.kind == "circle":
            command = COMMAND_CIRCLE
        elif shape.kind == "ellipse":
            command = COMMAND_ELLIPSE
        else:
            command = COMMAND_ARC
        points = _shape_points(shape)
        primitives.append(Primitive(
            command=command,
            points=[
                (x * to_canvas_x + offset_x, y * to_canvas_y + offset_y)
                for x, y in points
            ],
            length=_shape_length(shape) * (to_canvas_x + to_canvas_y) / 2.0,
            rgb=list(layer.rgb),
            layer_id=layer.layer_id,
            width=width_px * (to_canvas_x + to_canvas_y) / 2.0,
            confidence=1.0 - min(1.0, shape.residual),
        ))

    for shape in shapes:
        append_shape(shape)

    residual_skeleton = skeleton & ~(np.asarray(explained, dtype=bool))
    line_width = _line_width(mask, skeleton, distance_map())
    for first, second in _merge_lines(_line_segments(residual_skeleton, config), config):
        length = math.dist(first, second)
        if length <= 1e-6:
            continue
        # 已检出曲线边缘的残留骨架会变成短碎片，按“贴合曲线”过滤。
        if _hugs_curve(first, second, shapes, max(3.0, 1.5 * line_width)):
            continue
        points = [
            (
                first[0] + (second[0] - first[0]) * fraction,
                first[1] + (second[1] - first[1]) * fraction,
            )
            for fraction in (0.0, 1 / 3, 2 / 3, 1.0)
        ]
        primitives.append(Primitive(
            command=COMMAND_LINE,
            points=[
                (x * to_canvas_x + offset_x, y * to_canvas_y + offset_y)
                for x, y in points
            ],
            length=length * (to_canvas_x + to_canvas_y) / 2.0,
            rgb=list(layer.rgb),
            layer_id=layer.layer_id,
            width=line_width * (to_canvas_x + to_canvas_y) / 2.0,
            confidence=1.0,
        ))
    return primitives


def vectorize_all(
    layers: list[ColorLayer],
    config: InferenceConfig,
    *,
    canvas_width: int = 100,
    canvas_height: int = 100,
) -> list[Primitive]:
    primitives: list[Primitive] = []
    for layer in layers:
        original_mask = layer.mask
        resized_mask, _ = _resize_mask(original_mask, config.max_detection_size)
        layer.mask = resized_mask
        try:
            primitives.extend(vectorize_layer(
                layer, config,
                canvas_width=canvas_width, canvas_height=canvas_height,
            ))
        finally:
            layer.mask = original_mask
    return primitives
