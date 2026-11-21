"""符号模板库：检索 + 刚体对齐（template fitting）。

模板来自训练集 GT 实例；推理时对每个候选对象：
1. 用旋转/尺度不变的形状描述子检索同类别（或全类）模板；
2. 用质心+尺度+90° 旋转做初始对齐；
3. 在图像的笔画距离变换上搜索平移/尺度，最小化模板点到笔画的平均距离；
4. 输出变换后的完整模板图元（保证实例完整性）。
"""
from __future__ import annotations

import math
import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt as ndimage_distance

GRID = 16
ROTATIONS = 4
FEATURE_DIM = GRID * GRID


def _normalize(points: np.ndarray, rotation: int = 0):
    center = points.mean(axis=0)
    centered = points - center
    scale = max(
        float(centered[:, 0].max() - centered[:, 0].min()),
        float(centered[:, 1].max() - centered[:, 1].min()), 1e-3,
    )
    normalized = centered / scale
    for _ in range(rotation % 4):
        normalized = normalized[:, [1, 0]] * np.array([-1.0, 1.0])
    return normalized, scale, center


def descriptor(points: np.ndarray, rotation: int = 0) -> np.ndarray:
    normalized, _, _ = _normalize(points, rotation)
    xs = np.clip(((normalized[:, 0] + 0.5) * (GRID - 1)).round().astype(int), 0, GRID - 1)
    ys = np.clip(((normalized[:, 1] + 0.5) * (GRID - 1)).round().astype(int), 0, GRID - 1)
    grid = np.zeros((GRID, GRID), dtype=np.float32)
    np.add.at(grid, (ys, xs), 1.0)
    return (grid / max(grid.sum(), 1.0)).ravel()


def descriptors(points: np.ndarray) -> np.ndarray:
    return np.stack([descriptor(points, rotation) for rotation in range(ROTATIONS)])


def rotate(points: np.ndarray, rotation: int) -> np.ndarray:
    result = points.copy()
    for _ in range(rotation % 4):
        result = result[:, [1, 0]] * np.array([-1.0, 1.0])
    return result


@dataclass
class Fit:
    template_index: int
    class_id: int
    rotation: int
    score: float
    transform: tuple[float, float, float, float, float]  # scale, dx, dy, cx, cy


class TemplateLibrary:
    def __init__(self, payload: dict):
        self.templates = payload["templates"]
        self.grid = payload.get("grid", GRID)
        self.classes = np.asarray([item["class"] for item in self.templates], dtype=np.int64)
        self.points = []
        for item in self.templates:
            flat = [point for primitive in item["primitives"] for point in primitive["points"]]
            self.points.append(np.asarray(flat, dtype=np.float64) if flat else np.zeros((0, 2)))
        self.primitives = [item["primitives"] for item in self.templates]
        # 描述子必须由实际存储的图元重新计算，保证检索与实例化使用同一
        # 点集（构建时可能截断到 48 个图元，导致存储描述子与点集不一致）。
        self.descriptors = np.asarray([
            np.stack([descriptor(points, rotation) for rotation in range(ROTATIONS)])
            if len(points) >= 3 else np.zeros((ROTATIONS, FEATURE_DIM), dtype=np.float32)
            for points in self.points
        ], dtype=np.float32)

    @classmethod
    def load(cls, path: str | Path) -> "TemplateLibrary | None":
        path = Path(path)
        if not path.exists():
            return None
        try:
            payload = pickle.loads(zlib.decompress(path.read_bytes()))
        except Exception:  # noqa: BLE001
            return None
        return cls(payload)

    def retrieve(self, points: np.ndarray, top_k: int = 8,
                 class_filter: int | None = None) -> list[tuple[float, int, int]]:
        """返回 [(distance, template_index, rotation)]。"""
        if len(points) < 3:
            return []
        query = descriptors(points)  # 4×D
        distances = np.full(len(self.templates), np.inf, dtype=np.float64)
        rotations = np.zeros(len(self.templates), dtype=np.int64)
        # 查询固定在自身规范方向；模板分别取 4 个旋转版本，返回真正需要
        # 施加到模板上的旋转 k（修复此前 q[k] vs t[k] 的常量距离退化）。
        candidate = query[0]
        for rotation in range(ROTATIONS):
            template = self.descriptors[:, rotation, :]
            norm_query = float(candidate @ candidate)
            norm_template = np.einsum("ij,ij->i", template, template)
            distance = np.sqrt(np.maximum(
                norm_query + norm_template - 2.0 * (template @ candidate), 0.0,
            ))
            better = distance < distances
            distances[better] = distance[better]
            rotations[better] = rotation

        if class_filter is not None:
            mask = self.classes == class_filter
            distances = np.where(mask, distances, np.inf)
        order = np.argsort(distances)[:top_k]
        return [(float(distances[index]), int(index), int(rotations[index]))
                for index in order if math.isfinite(distances[index])]

    def match_box_pose(self, box, class_id: int, stroke_mask: np.ndarray,
                       canvas_to_pixel: float, max_templates: int = 160,
                       scales: tuple[float, ...] = (0.9, 1.0, 1.1),
                       max_candidates: int = 6) -> list[dict]:
        """框内滑窗模板匹配（位姿感知）。

        对每个候选模板/旋转/尺度把模板渲染成小 patch，用 TM_CCORR 在 ROI 内
        一次性扫描所有平移，取 F1 峰值，得到 (template, rotation, scale,
        center) 候选；随后交给 fit_template 做距离变换精修。
        """
        import cv2

        x0, y0, x1, y1 = box
        width = max(x1 - x0, 0.5)
        height = max(y1 - y0, 0.5)
        mx = 0.15 * width + 0.5
        my = 0.15 * height + 0.5
        rx0, ry0 = x0 - mx, y0 - my
        rx1, ry1 = x1 + mx, y1 + my
        px0 = max(0, int(rx0 * canvas_to_pixel))
        py0 = max(0, int(ry0 * canvas_to_pixel))
        px1 = min(stroke_mask.shape[1], int(rx1 * canvas_to_pixel) + 1)
        py1 = min(stroke_mask.shape[0], int(ry1 * canvas_to_pixel) + 1)
        if px1 <= px0 or py1 <= py0:
            return []
        roi = stroke_mask[py0:py1, px0:px1].astype(np.uint8)
        roi_area = int(roi.sum())
        if roi_area < 6:
            return []
        # ROI 左上角（画布坐标）。
        origin_x = px0 / canvas_to_pixel
        origin_y = py0 / canvas_to_pixel
        scale = max(width, height)
        box_aspect = width / max(height, 1e-6)
        indices = np.nonzero(self.classes == class_id)[0]
        if len(indices) > max_templates:
            order = np.argsort(-np.asarray([self.templates[i]["count"] for i in indices]))
            indices = indices[order[:max_templates]]
        candidates = []
        for index in indices:
            points = self.points[index]
            if len(points) < 3:
                continue
            _, template_scale, template_center = _normalize(points, 0)
            bbox = points.max(axis=0) - points.min(axis=0)
            aspect = float(bbox[0] / max(bbox[1], 1e-6))
            if aspect > 3.0 * box_aspect or aspect < box_aspect / 3.0:
                continue
            base_factor = scale / template_scale
            for rotation in range(ROTATIONS):
                rotated = rotate(points - template_center, rotation)
                for factor in scales:
                    transformed = rotated * (base_factor * factor)
                    low = transformed.min(axis=0)
                    high = transformed.max(axis=0)
                    patch_w = int(np.ceil((high[0] - low[0]) * canvas_to_pixel)) + 4
                    patch_h = int(np.ceil((high[1] - low[1]) * canvas_to_pixel)) + 4
                    if patch_w >= roi.shape[1] or patch_h >= roi.shape[0]:
                        continue
                    if patch_w < 3 or patch_h < 3:
                        continue
                    patch = np.zeros((patch_h, patch_w), dtype=np.uint8)
                    offset = np.array([2.0, 2.0]) - low * canvas_to_pixel
                    pixels = np.round(
                        transformed * canvas_to_pixel + offset
                    ).astype(np.int32)
                    cv2.polylines(patch, [pixels.reshape(-1, 1, 2)], False, 1,
                                  thickness=1)
                    patch_area = int(patch.sum())
                    if patch_area < 4:
                        continue
                    correlation = cv2.matchTemplate(
                        roi.astype(np.float32), patch.astype(np.float32),
                        cv2.TM_CCORR,
                    )
                    precision = correlation / patch_area
                    recall = correlation / max(roi_area, 1)
                    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-6)
                    peak = np.unravel_index(int(np.argmax(f1)), f1.shape)
                    score = float(f1[peak])
                    if score <= 0.0:
                        continue
                    # patch 内模板质心位置 = 2 - low*ctp（模板以质心为原点渲染）。
                    center_px = np.array([
                        peak[1] + 2.0 - low[0] * canvas_to_pixel,
                        peak[0] + 2.0 - low[1] * canvas_to_pixel,
                    ])
                    center_canvas = center_px / canvas_to_pixel + np.array([origin_x, origin_y])
                    candidates.append({
                        "f1": score,
                        "template": int(index),
                        "rotation": int(rotation),
                        "scale": float(base_factor * factor),
                        "center": (float(center_canvas[0]), float(center_canvas[1])),
                    })
        candidates.sort(key=lambda item: -item["f1"])
        return candidates[:max_candidates]

    def match_box(self, box, class_id: int, stroke_mask: np.ndarray,
                  canvas_to_pixel: float, max_templates: int = 300,
                  max_candidates: int = 8) -> list[tuple[float, int, int]]:
        """在检测框内用 F1（模板掩码 vs 局部笔画掩码）匹配模板。

        这是经典二值相关思路：只对 ROI 内的局部笔画计分，避免墙线污染；
        返回 [(f1, template_index, rotation)]，按 F1 降序。
        """
        import cv2

        x0, y0, x1, y1 = box
        width = max(x1 - x0, 0.5)
        height = max(y1 - y0, 0.5)
        margin_x = 0.15 * width + 0.5
        margin_y = 0.15 * height + 0.5
        px0 = max(0, int((x0 - margin_x) * canvas_to_pixel))
        px1 = min(stroke_mask.shape[1], int((x1 + margin_x) * canvas_to_pixel) + 1)
        py0 = max(0, int((y0 - margin_y) * canvas_to_pixel))
        py1 = min(stroke_mask.shape[0], int((y1 + margin_y) * canvas_to_pixel) + 1)
        if px1 <= px0 or py1 <= py0:
            return []
        local = stroke_mask[py0:py1, px0:px1].astype(np.uint8)
        local_area = int(local.sum())
        if local_area < 6:
            return []
        scale = max(width, height)
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        box_aspect = width / max(height, 1e-6)
        results = []
        indices = np.nonzero(self.classes == class_id)[0]
        if len(indices) > max_templates:
            order = np.argsort(-np.asarray([self.templates[i]["count"] for i in indices]))
            indices = indices[order[:max_templates]]
        for index in indices:
            points = self.points[index]
            if len(points) < 3:
                continue
            _, template_scale, template_center = _normalize(points, 0)
            bbox = points.max(axis=0) - points.min(axis=0)
            aspect = float(bbox[0] / max(bbox[1], 1e-6))
            if aspect > 3.0 * box_aspect or aspect < box_aspect / 3.0:
                continue
            factor = scale / template_scale
            for rotation in range(ROTATIONS):
                rotated = rotate(points - template_center, rotation)
                transformed = rotated * factor + np.array([center_x, center_y])
                pixels = np.round(
                    np.column_stack([
                        transformed[:, 0] * canvas_to_pixel - px0,
                        transformed[:, 1] * canvas_to_pixel - py0,
                    ])
                ).astype(np.int32)
                template_mask = np.zeros_like(local)
                cv2.polylines(template_mask, [pixels.reshape(-1, 1, 2)], False, 1,
                              thickness=1)
                area = int(template_mask.sum())
                if area < 4:
                    continue
                intersection = int(np.count_nonzero(
                    template_mask.astype(bool) & local.astype(bool)
                ))
                if not intersection:
                    continue
                precision = intersection / area
                recall = intersection / local_area
                f1 = 2 * precision * recall / max(precision + recall, 1e-9)
                results.append((f1, int(index), int(rotation)))
        results.sort(key=lambda item: -item[0])
        return results[:max_candidates]

    def fit_template(self, index: int, rotation: int, center: tuple[float, float],
                     scale: float, distance_transform: np.ndarray,
                     canvas_to_pixel: float, stroke_mask: np.ndarray | None = None,
                     search_units: float = 1.0, search_step: float = 0.5,
                     scale_factors: tuple[float, ...] = (0.9, 1.0, 1.1),
                     reverse_weight: float = 0.5) -> "Fit | None":
        """对指定模板/旋转做局部位姿精修（距离变换 + 反向距离）。"""
        import cv2
        points = self.points[index]
        if len(points) < 3:
            return None
        _, template_scale, template_center = _normalize(points, 0)
        rotated = rotate(points - template_center, rotation)
        base_scale = scale / template_scale
        height, width = distance_transform.shape
        best = None
        for scale_factor in scale_factors:
            scaled = rotated * (base_scale * scale_factor)
            for dx in np.arange(-search_units, search_units + 1e-6, search_step):
                for dy in np.arange(-search_units, search_units + 1e-6, search_step):
                    transformed = scaled + np.asarray(center) + np.array([dx, dy])
                    columns = np.clip((transformed[:, 0] * canvas_to_pixel).round().astype(int), 0, width - 1)
                    rows = np.clip((transformed[:, 1] * canvas_to_pixel).round().astype(int), 0, height - 1)
                    forward = float(distance_transform[rows, columns].mean())
                    if stroke_mask is not None:
                        x0 = max(0, int(columns.min()) - 6); x1 = min(width, int(columns.max()) + 7)
                        y0 = max(0, int(rows.min()) - 6); y1 = min(height, int(rows.max()) + 7)
                        if x1 > x0 and y1 > y0:
                            tm = np.zeros((y1 - y0, x1 - x0), np.uint8)
                            local = np.round(np.column_stack([columns - x0, rows - y0])).astype(np.int32)
                            cv2.polylines(tm, [local.reshape(-1, 1, 2)], False, 1, thickness=1)
                            im = stroke_mask[y0:y1, x0:x1]
                            if tm.any() and im.any():
                                tdt = ndimage_distance(tm)
                                forward += reverse_weight * float(np.minimum(tdt[im], 3.0).mean())
                    if best is None or forward < best[0]:
                        best = (forward, base_scale * scale_factor, float(dx), float(dy))
        if best is None:
            return None
        score, fitted_scale, dx, dy = best
        return Fit(template_index=index, class_id=int(self.classes[index]),
                   rotation=int(rotation), score=float(score),
                   transform=(fitted_scale, dx, dy, float(center[0]), float(center[1])))

    def fit(self, points: np.ndarray, distance_transform: np.ndarray,
            canvas_to_pixel: float, top_k: int = 6,
            class_filter: int | None = None,
            search_units: float = 1.5, search_step: float = 0.5,
            scale_factors: tuple[float, ...] = (0.85, 1.0, 1.15),
            center: tuple[float, float] | None = None,
            scale_override: float | None = None,
            stroke_mask: np.ndarray | None = None,
            reverse_weight: float = 0.6,
            reverse_tolerance: float = 3.0) -> Fit | None:
        """检索 + 双向距离拟合。

        正向：模板点到图像笔画的距离（距离变换）；
        反向：ROI 内图像笔画点到模板的距离，惩罚“模板只覆盖局部笔画”的错误
        匹配；另外对越界点施加惩罚。
        """
        candidates = self.retrieve(points, top_k=top_k, class_filter=class_filter)
        if not candidates:
            return None
        _, candidate_scale, candidate_center = _normalize(points, 0)
        if center is not None:
            candidate_center = np.asarray(center, dtype=np.float64)
        if scale_override is not None:
            candidate_scale = float(scale_override)
        height, width = distance_transform.shape
        import cv2

        def reverse_score(transformed: np.ndarray) -> float:
            if stroke_mask is None:
                return 0.0
            columns = transformed[:, 0] * canvas_to_pixel
            rows = transformed[:, 1] * canvas_to_pixel
            x0 = max(0, int(columns.min()) - 6)
            x1 = min(width, int(columns.max()) + 7)
            y0 = max(0, int(rows.min()) - 6)
            y1 = min(height, int(rows.max()) + 7)
            if x1 <= x0 or y1 <= y0:
                return reverse_tolerance
            template_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            local = np.round(np.column_stack([columns - x0, rows - y0])).astype(np.int32)
            cv2.polylines(template_mask, [local.reshape(-1, 1, 2)], False, 1,
                          thickness=1)
            image_mask = stroke_mask[y0:y1, x0:x1]
            if not template_mask.any() or not image_mask.any():
                return reverse_tolerance
            template_dt = ndimage_distance(template_mask)
            values = template_dt[image_mask]
            if values.size == 0:
                return reverse_tolerance
            return float(np.minimum(values, reverse_tolerance).mean())

        best: Fit | None = None
        for distance, index, rotation in candidates:
            template_points = self.points[index]
            if len(template_points) < 3:
                continue
            _, template_scale, template_center = _normalize(template_points, 0)
            rotated = rotate(template_points - template_center, rotation)
            base_scale = candidate_scale / template_scale
            local_best = None
            for scale_factor in scale_factors:
                scaled = rotated * (base_scale * scale_factor)
                for dx in np.arange(-search_units, search_units + 1e-6, search_step):
                    for dy in np.arange(-search_units, search_units + 1e-6, search_step):
                        transformed = scaled + candidate_center + np.array([dx, dy])
                        columns = np.clip(
                            (transformed[:, 0] * canvas_to_pixel).round().astype(int),
                            0, width - 1,
                        )
                        rows = np.clip(
                            (transformed[:, 1] * canvas_to_pixel).round().astype(int),
                            0, height - 1,
                        )
                        forward = float(distance_transform[rows, columns].mean())
                        outside = int(((transformed[:, 0] < 0) | (transformed[:, 0] > 100)
                                       | (transformed[:, 1] < 0) | (transformed[:, 1] > 100)).sum())
                        forward += 2.0 * outside / max(len(transformed), 1)
                        total = forward + 0.5 * distance
                        if local_best is None or total < local_best[0]:
                            local_best = (total, scale_factor, float(dx), float(dy), transformed)
            if local_best is None:
                continue
            total, scale_factor, dx, dy, transformed = local_best
            total += reverse_weight * reverse_score(transformed)
            if best is None or total < best.score:
                best = Fit(
                    template_index=index, class_id=int(self.classes[index]),
                    rotation=int(rotation), score=float(total),
                    transform=(base_scale * scale_factor, dx, dy,
                               float(candidate_center[0]), float(candidate_center[1])),
                )
        return best

    def instantiate(self, fit: Fit, points: np.ndarray) -> list[dict]:
        """把模板图元按拟合变换映射到画布坐标。"""
        template_points = self.points[fit.template_index]
        _, template_scale, template_center = _normalize(template_points, 0)
        rotated = rotate(template_points - template_center, fit.rotation)
        scale, dx, dy, center_x, center_y = fit.transform
        # 用模板逐图元变换，保持颜色与线宽。
        results = []
        template_vertices = [
            np.asarray([point for point in primitive["points"]], dtype=np.float64)
            for primitive in self.primitives[fit.template_index]
        ]
        for primitive, vertices in zip(self.primitives[fit.template_index], template_vertices):
            local = rotate(vertices - template_center, fit.rotation) * scale
            local += np.array([center_x + dx, center_y + dy])
            length = sum(math.dist(a, b) for a, b in zip(local, local[1:]))
            results.append({
                # 保留模板原始图元类型（Arc/Circle/Ellipse 不降级为 Line）。
                "command": int(primitive.get("command", 0)),
                "points": [(float(x), float(y)) for x, y in local],
                "length": float(length),
                "width": float(primitive["width"]),
                "rgb": list(primitive["rgb"]),
                "layer": 1,
                "semantic": fit.class_id,
            })
        return results
