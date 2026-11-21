"""Per-layer geometric stroke extraction (v2).

输入一个颜色层掩码，输出 Line / Arc / Circle / Ellipse 图元。与旧实现相比：

* 直线用 OpenCV 的 ``HoughLinesP``（C 实现，比 skimage 概率 Hough 快一个
  数量级），再做共线聚类与端点精修；
* 圆/椭圆/圆弧仍用 ``find_contours`` + 代数拟合，但按层独立处理，避免
  不同颜色的线在骨架里互相粘连；
* 所有坐标保留亚像素（像素中心 + 0.5）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

from .config import COMMAND_ARC, COMMAND_CIRCLE, COMMAND_ELLIPSE, COMMAND_LINE


@dataclass
class Stroke:
    command: int
    #: 画布坐标下的 4 个采样点。
    points: list[tuple[float, float]]
    length: float
    #: 像素坐标下的端点（直线）或圆心参数（曲线），供并行/分组使用。
    p0: tuple[float, float] = (0.0, 0.0)
    p1: tuple[float, float] = (0.0, 0.0)
    width: float = 1.0
    layer_id: int = 0
    rgb: tuple[int, int, int] = (0, 0, 0)
    shape: dict = field(default_factory=dict)

    @property
    def is_line(self) -> bool:
        return self.command == COMMAND_LINE

    @property
    def angle(self) -> float:
        return math.atan2(self.p1[1] - self.p0[1], self.p1[0] - self.p0[0])

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [point[0] for point in self.points]
        ys = [point[1] for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    def resample(self, count: int = 4) -> list[tuple[float, float]]:
        if self.command in (COMMAND_CIRCLE, COMMAND_ELLIPSE) or self.command == COMMAND_ARC:
            return _curve_points(self.shape, count)
        return [
            (
                self.p0[0] + (self.p1[0] - self.p0[0]) * index / (count - 1),
                self.p0[1] + (self.p1[1] - self.p0[1]) * index / (count - 1),
            )
            for index in range(count)
        ]


def _curve_points(shape: dict, count: int) -> list[tuple[float, float]]:
    cx, cy = shape["cx"], shape["cy"]
    a, b, theta = shape["a"], shape["b"], shape.get("theta", 0.0)
    if shape["kind"] in ("circle", "ellipse"):
        angles = [2 * math.pi * index / count for index in range(count)]
        if shape["kind"] == "circle":
            return [(cx + a * math.cos(angle), cy + a * math.sin(angle)) for angle in angles]
        cosine, sine = math.cos(theta), math.sin(theta)
        points = []
        for angle in angles:
            lx, ly = a * math.cos(angle), b * math.sin(angle)
            points.append((cx + lx * cosine - ly * sine, cy + lx * sine + ly * cosine))
        return points
    start, span = shape["start"], shape["coverage"]
    return [
        (cx + a * math.cos(start + span * index / (count - 1)),
         cy + a * math.sin(start + span * index / (count - 1)))
        for index in range(count)
    ]


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
    squared = float(solution[2] + cx * cx + cy * cy)
    if squared <= 0:
        return 0.0, 0.0, 0.0, 1.0
    radius = math.sqrt(squared)
    distances = np.hypot(x - cx, y - cy)
    return cx, cy, radius, float(np.std(distances - radius) / max(radius, 1e-6))


def _angular_span(points: np.ndarray, cx: float, cy: float) -> tuple[float, float]:
    angles = np.sort(np.mod(np.arctan2(points[:, 1] - cy, points[:, 0] - cx), 2 * math.pi))
    if len(angles) < 2:
        return 0.0, 0.0
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2 * math.pi]))
    gap_index = int(np.argmax(gaps))
    coverage = float(2 * math.pi - gaps[gap_index])
    return coverage, float(angles[(gap_index + 1) % len(angles)])


def _rectangular(points: np.ndarray) -> bool:
    if len(points) < 8:
        return False
    if np.linalg.norm(points[0] - points[-1]) > 2.0:
        return False
    try:
        polygon = approximate_polygon(points, tolerance=2.0)
    except Exception:  # noqa: BLE001
        return False
    if not 4 <= len(polygon) <= 6:
        return False
    right = 0
    for index in range(len(polygon)):
        previous = polygon[index - 1] - polygon[index]
        following = polygon[(index + 1) % len(polygon)] - polygon[index]
        norm = np.linalg.norm(previous) * np.linalg.norm(following)
        if norm <= 1e-9:
            continue
        if abs(float(np.dot(previous, following) / norm)) < 0.35:
            right += 1
    return right >= max(3, len(polygon) - 1)


def _ellipse_distance(points: np.ndarray, shape: dict) -> np.ndarray:
    cosine, sine = math.cos(shape["theta"]), math.sin(shape["theta"])
    dx = points[:, 0] - shape["cx"]
    dy = points[:, 1] - shape["cy"]
    u = dx * cosine + dy * sine
    v = -dx * sine + dy * cosine
    value = np.sqrt(
        (u / max(abs(shape["a"]), 1e-6)) ** 2
        + (v / max(abs(shape["b"]), 1e-6)) ** 2
    )
    return np.abs(value - 1.0) * max(min(abs(shape["a"]), abs(shape["b"])), 1e-6)


def _center_empty(mask: np.ndarray, shape: dict, max_fraction: float = 0.15) -> bool:
    """圆/椭圆内部应几乎为空：真实圆环内部是空白，阴影线区域内部是满的。"""
    height, width = mask.shape
    cosine, sine = math.cos(shape["theta"]), math.sin(shape["theta"])
    occupied = total = 0
    for u in np.linspace(-0.7, 0.7, 9):
        for v in np.linspace(-0.7, 0.7, 9):
            if u * u + v * v > 0.5:
                continue
            lx = shape["a"] * u
            ly = shape["b"] * v
            x = shape["cx"] + lx * cosine - ly * sine
            y = shape["cy"] + lx * sine + ly * cosine
            row, column = int(round(y)), int(round(x))
            if 0 <= row < height and 0 <= column < width:
                total += 1
                occupied += int(bool(mask[row, column]))
    return total == 0 or occupied / total <= max_fraction


def _refine_curve(
    shape: dict, skeleton_points: np.ndarray, mask: np.ndarray, tolerance: float = 3.0,
) -> dict | None:
    """用骨架点重新拟合，确认曲线真实存在并区分整圆/圆弧。"""
    if shape["kind"] == "ellipse":
        distances = _ellipse_distance(skeleton_points, shape)
    else:
        distances = np.abs(
            np.hypot(skeleton_points[:, 0] - shape["cx"],
                     skeleton_points[:, 1] - shape["cy"]) - shape["a"]
        )
    selected = skeleton_points[distances <= tolerance]
    if len(selected) < 12:
        return None
    coverage, start = _angular_span(selected, shape["cx"], shape["cy"])
    if coverage < 2 * math.pi * 0.12:
        return None

    if shape["kind"] == "ellipse":
        from skimage.measure import EllipseModel
        model = EllipseModel()
        try:
            if not model.estimate(selected):
                return None
        except Exception:  # noqa: BLE001
            return None
        cx, cy, a, b, theta = [float(value) for value in model.params]
        major, minor = max(abs(a), abs(b)), min(abs(a), abs(b))
        if minor < 8.0 or major / max(minor, 1e-6) > 12.0:
            return None
        if abs(abs(a) - abs(b)) / max(major, 1e-6) < 0.12:
            return None
        rms = float(np.sqrt(np.mean(np.square(model.residuals(selected)))))
        if rms / max(major, 1e-6) > 0.10:
            return None
        candidate = dict(shape, kind="ellipse", cx=cx, cy=cy, a=abs(a), b=abs(b),
                         theta=theta, start=0.0, coverage=2 * math.pi,
                         residual=rms / max(major, 1e-6))
        if coverage < 2 * math.pi * 0.80 or not _center_empty(mask, candidate):
            return None
        return candidate

    cx, cy, radius, residual = _fit_circle(selected)
    if radius < 8.0 or residual > 0.10:
        return None
    if coverage >= 2 * math.pi * 0.82:
        if shape.get("support", 1) < 2 or not _center_empty(
            mask, dict(shape, cx=cx, cy=cy, a=radius, b=radius, theta=0.0)
        ):
            return None
        return dict(shape, kind="circle", cx=cx, cy=cy, a=radius, b=radius,
                    theta=0.0, start=0.0, coverage=2 * math.pi, residual=residual)
    return dict(shape, kind="arc", cx=cx, cy=cy, a=radius, b=radius,
                theta=0.0, start=start, coverage=coverage, residual=residual)


def _curve_candidates(
    curved: np.ndarray, mask: np.ndarray, config,
) -> list[dict]:
    """在「去掉直线后的骨架」上找圆弧/圆/椭圆。

    直接对轮廓拟合会被相连的直线网络污染（门窗符号的弧总与边框相连），
    所以先从骨架中扣除 Hough 直线，剩余的曲线像素按连通域分组，再逐组
    拟合圆或椭圆。这样嵌在符号里的门弧、圆弧楼梯都能被单独识别。
    """
    if curved.sum() < 12:
        return []
    labels, count = ndimage.label(curved, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    shapes: list[dict] = []
    for label, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None or sizes[label] < 12:
            continue
        rows, columns = np.nonzero(labels[box] == label)
        points = np.column_stack([
            columns + box[1].start, rows + box[0].start,
        ]).astype(np.float64)
        extent = np.ptp(points, axis=0)
        if min(extent) < 6.0:
            continue
        cx, cy, radius, residual = _fit_circle(points)
        if radius >= 8.0 and residual <= 0.10:
            coverage, start = _angular_span(points, cx, cy)
            if coverage < 2 * math.pi * 0.12:
                continue
            if coverage >= 2 * math.pi * 0.82:
                circle = dict(
                    kind="circle", cx=cx, cy=cy, a=radius, b=radius, theta=0.0,
                    start=0.0, coverage=2 * math.pi, residual=residual, support=2,
                )
                if not _center_empty(mask, circle):
                    continue
                shapes.append(circle)
            else:
                shapes.append(dict(
                    kind="arc", cx=cx, cy=cy, a=radius, b=radius, theta=0.0,
                    start=start, coverage=coverage, residual=residual, support=1,
                ))
            continue
        if len(points) < 30 or max(extent) / max(min(extent), 1e-6) > 12.0:
            continue
        from skimage.measure import EllipseModel
        model = EllipseModel()
        try:
            if not model.estimate(points):
                continue
        except Exception:  # noqa: BLE001
            continue
        ex, ey, a, b, theta = [float(value) for value in model.params]
        major, minor = max(abs(a), abs(b)), min(abs(a), abs(b))
        if minor < 8.0 or major / max(minor, 1e-6) > 12.0:
            continue
        if abs(abs(a) - abs(b)) / max(major, 1e-6) < 0.12:
            continue
        cosine, sine = math.cos(theta), math.sin(theta)
        dx, dy = points[:, 0] - ex, points[:, 1] - ey
        u = (dx * cosine + dy * sine) / max(abs(a), 1e-6)
        v = (-dx * sine + dy * cosine) / max(abs(b), 1e-6)
        rms = float(np.mean(np.abs(np.sqrt(u * u + v * v) - 1.0))) * major
        if rms / max(major, 1e-6) > 0.10:
            continue
        coverage, _ = _angular_span(points, ex, ey)
        if coverage < 2 * math.pi * 0.75:
            continue
        ellipse = dict(
            kind="ellipse", cx=ex, cy=ey, a=abs(a), b=abs(b), theta=theta,
            start=0.0, coverage=2 * math.pi, residual=rms / max(major, 1e-6),
            support=2,
        )
        if not _center_empty(mask, ellipse):
            continue
        shapes.append(ellipse)
    return _dedupe_shapes(shapes)


def _dedupe_shapes(shapes: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for shape in shapes:
        for index, existing in enumerate(kept):
            scale = max(shape["a"], shape["b"], existing["a"], existing["b"], 1.0)
            if (math.hypot(shape["cx"] - existing["cx"], shape["cy"] - existing["cy"])
                    <= max(1.5, 0.06 * scale)
                    and abs(shape["a"] - existing["a"]) <= max(3.0, 0.15 * scale)):
                if shape.get("support", 1) > existing.get("support", 1):
                    kept[index] = shape
                break
        else:
            kept.append(shape)
    return kept


def _shape_primitive(shape: dict, layer) -> Stroke:
    kind = shape["kind"]
    command = {"circle": COMMAND_CIRCLE, "ellipse": COMMAND_ELLIPSE, "arc": COMMAND_ARC}[kind]
    if kind == "circle":
        length = 2 * math.pi * shape["a"]
    elif kind == "ellipse":
        samples = _curve_points(shape, 129)
        length = sum(
            math.dist(first, second) for first, second in zip(samples, samples[1:])
        )
    else:
        length = shape["a"] * shape["coverage"]
    points = _curve_points(shape, 4)
    return Stroke(
        command=command, points=points, length=length,
        p0=(shape["cx"], shape["cy"]), p1=(shape["a"], shape["b"]),
        layer_id=layer.layer_id, rgb=tuple(layer.rgb), shape=shape,
    )


def _line_hough(skeleton: np.ndarray, config) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """HoughLinesP 提取直线，按方向/偏移去重。"""
    image = (skeleton.astype(np.uint8) * 255)
    segments = cv2.HoughLinesP(
        image, 1, np.pi / 720, threshold=max(8, config.hough_threshold),
        minLineLength=config.hough_min_length, maxLineGap=config.hough_line_gap,
    )
    if segments is None:
        return []
    found = []
    for x1, y1, x2, y2 in segments.reshape(-1, 4):
        found.append(((float(x1), float(y1)), (float(x2), float(y2))))
    # HoughLinesP 对同一条线会输出大量重复段；先按「方向+垂距+投影」分桶，
    # 每桶只留最长的一条，把后续 O(n²) 聚类的规模压下来。
    buckets: dict[tuple[int, int, int], tuple[float, tuple, tuple]] = {}
    for segment in found:
        first, second = np.asarray(segment[0]), np.asarray(segment[1])
        delta = second - first
        length = float(np.linalg.norm(delta))
        if length <= 1e-6:
            continue
        direction = delta / length
        angle = math.atan2(direction[1], direction[0]) % math.pi
        normal = (-direction[1], direction[0])
        offset = float(np.dot(first, normal))
        middle = (first + second) / 2.0
        projection = float(np.dot(middle, direction))
        key = (
            int(round(angle / (math.pi / 90))),
            int(round(offset / 2.0)),
            int(round(projection / 8.0)),
        )
        current = buckets.get(key)
        if current is None or length > current[0]:
            buckets[key] = (length, segment[0], segment[1])
    return [(value[1], value[2]) for value in buckets.values()]


def _merge_collinear(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    config,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """方向一致且偏移很小、间隙很小的线段合并；间隙按「线宽倍数」限制。"""
    if not segments:
        return []
    # 先按（方向, 垂距）分桶：只有同桶内的线段才可能共线，避免整层 O(n²)。
    buckets: dict[tuple[int, int], list[int]] = {}
    prepared: list[tuple[list, tuple] | None] = []
    for segment in segments:
        points = np.array(segment, dtype=np.float64)
        center = points.mean(axis=0)
        delta = points[1] - points[0]
        norm = np.linalg.norm(delta)
        if norm <= 1e-6:
            prepared.append(None)
            continue
        direction = delta / norm
        angle = math.atan2(direction[1], direction[0]) % math.pi
        offset = float(np.dot(center, (-direction[1], direction[0])))
        key = (int(round(angle / (math.pi / 90))), int(round(offset / config.merge_distance_px)))
        buckets.setdefault(key, []).append(len(prepared))
        prepared.append(([segment], (center, direction, points @ direction)))

    clusters: list[list] = []
    fits: list = []
    for key, indices in buckets.items():
        local_clusters = [prepared[index][0] for index in indices]
        local_fits = [prepared[index][1] for index in indices]
        changed = True
        while changed:
            changed = False
            first = 0
            while first < len(local_clusters):
                second = first + 1
                while second < len(local_clusters):
                    if _mergeable(local_fits[first], local_fits[second], config):
                        local_clusters[first].extend(local_clusters[second])
                        local_clusters.pop(second)
                        local_fits.pop(second)
                        local_fits[first] = _cluster_fit(local_clusters[first])
                        changed = True
                        continue
                    second += 1
                first += 1
        clusters.extend(local_clusters)
        fits.extend(local_fits)
    merged = []
    for cluster in clusters:
        fit = _cluster_fit(cluster)
        if fit is None:
            continue
        center, direction, projection = fit
        first_point = center + direction * projection.min()
        second_point = center + direction * projection.max()
        if math.dist(first_point, second_point) >= config.hough_min_length * 0.75:
            merged.append((
                (float(first_point[0]), float(first_point[1])),
                (float(second_point[0]), float(second_point[1])),
            ))
    return merged


def _cluster_fit(cluster):
    points = np.array([point for segment in cluster for point in segment], dtype=np.float64)
    center = points.mean(axis=0)
    centered = points - center
    try:
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    direction = vectors[0]
    return center, direction, centered @ direction


def _mergeable(first, second, config) -> bool:
    if first is None or second is None:
        return False
    center, direction, projection = first
    _, other_direction, other_projection = second
    delta = abs(math.atan2(direction[1], direction[0]) - math.atan2(
        other_direction[1], other_direction[0])) % math.pi
    delta = min(delta, math.pi - delta)
    if delta > math.radians(config.merge_angle_deg):
        return False
    points = np.array([center + direction * value for value in projection])
    other_points = np.array([second[0] + other_direction * value for value in other_projection])
    normal = np.array([-direction[1], direction[0]])
    if float(np.abs((other_points - center) @ normal).max()) > config.merge_distance_px:
        return False
    gap = max(
        float(other_projection.min() - projection.max()),
        float(projection.min() - other_projection.max()),
    )
    return gap <= config.merge_gap_px


def extract_strokes(
    mask: np.ndarray, layer, config, *, detect_curves: bool = True,
) -> list[Stroke]:
    """从一个颜色层提取全部图元（像素坐标，未换算）。"""
    strokes: list[Stroke] = []
    if mask.sum() < config.min_layer_pixels:
        return strokes
    skeleton = skeletonize(mask)
    if skeleton.sum() == 0:
        return strokes

    line_width = _line_width(mask, skeleton)
    line_segments = _merge_collinear(_line_hough(skeleton, config), config)
    for first, second in line_segments:
        length = math.dist(first, second)
        if length < config.hough_min_length * 0.75:
            continue
        strokes.append(Stroke(
            command=COMMAND_LINE,
            points=[first, second],
            length=length,
            p0=first, p1=second,
            width=line_width,
            layer_id=layer.layer_id,
            rgb=tuple(layer.rgb),
        ))

    if detect_curves:
        # 先把直线从骨架里扣除，剩下的连通域就是圆弧/圆/椭圆。
        explained = np.zeros(mask.shape, dtype=np.uint8)
        for first, second in line_segments:
            cv2.line(
                explained,
                (int(round(first[0])), int(round(first[1]))),
                (int(round(second[0])), int(round(second[1]))),
                1, thickness=3,
            )
        curved = skeleton & ~explained.astype(bool)
        for shape in _curve_candidates(curved, mask, config):
            strokes.append(_shape_primitive(shape, layer))
    return strokes


def _line_width(mask: np.ndarray, skeleton: np.ndarray) -> float:
    distance = ndimage.distance_transform_edt(mask)
    values = distance[skeleton]
    if values.size == 0:
        return 1.0
    return max(0.5, float(2.0 * np.median(values)))
