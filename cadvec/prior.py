"""语义分类：颜色先验 + 可选小型分类器。

优先级：
1. ``weights/primitive_classifier.joblib``（若存在，由 ``tools/train_classifier.py`` 生成）；
2. ``weights/color_prior.json``（由 ``tools/build_prior.py`` 从训练集 SVG 统计）；
3. 赛题默认忽略类 35。

颜色先验的依据：训练 SVG 的 ``stroke`` 颜色会原样出现在 PNG 中，同一数据集的
CAD 图层颜色通常与类别稳定对应。先验不读取 dev/test 的任何真值。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import IGNORE_CLASS, InferenceConfig
from .vectorize import Primitive


@dataclass
class ColorEntry:
    rgb: list[int]
    semantic_id: int
    confidence: float
    count: int
    counts: dict[int, int]


class ColorPrior:
    def __init__(
        self,
        entries: list[ColorEntry] | None = None,
        class_frequencies: dict[int, float] | None = None,
        *,
        default_semantic: int = IGNORE_CLASS,
        max_distance: float = 70.0,
    ):
        self.entries = entries or []
        self.class_frequencies = class_frequencies or {}
        self.default_semantic = default_semantic
        self.max_distance = max_distance
        #: 同一张图里大量图元共享少数几种描边色，缓存最近邻结果。
        self._cache: dict[tuple[int, int, int], tuple[int, float]] = {}
        self._matrix: np.ndarray | None = None

    @classmethod
    def load(cls, directory: str | Path, config: InferenceConfig) -> "ColorPrior":
        path = Path(directory) / "color_prior.json"
        if not path.exists():
            return cls(default_semantic=config.fallback_semantic,
                       max_distance=config.prior_max_distance)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls(default_semantic=config.fallback_semantic,
                       max_distance=config.prior_max_distance)
        entries = []
        for item in payload.get("colors", []):
            if not isinstance(item, dict):
                continue
            try:
                rgb = [int(value) for value in item["rgb"]]
                semantic = int(item["semantic"])
                count = int(item.get("count", 0))
                confidence = float(item.get("confidence", 0.0))
                counts = {int(key): int(value) for key, value in (item.get("counts") or {}).items()}
            except (KeyError, TypeError, ValueError):
                continue
            entries.append(ColorEntry(rgb, semantic, confidence, count, counts))
        frequencies = {}
        for key, value in (payload.get("class_frequencies") or {}).items():
            try:
                frequencies[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return cls(
            entries, frequencies,
            default_semantic=config.fallback_semantic,
            max_distance=config.prior_max_distance,
        )

    def classify(self, rgb: list[int]) -> tuple[int, float]:
        if not self.entries:
            return self.default_semantic, 0.0
        key = tuple(int(value) for value in rgb[:3])
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if self._matrix is None:
            self._matrix = np.asarray(
                [entry.rgb for entry in self.entries], dtype=np.float64,
            )
        color = np.asarray(key, dtype=np.float64)
        distances = np.linalg.norm(self._matrix - color, axis=1)
        index = int(np.argmin(distances))
        best = self.entries[index]
        best_distance = float(distances[index])
        if best_distance > self.max_distance:
            result = (self.default_semantic, 0.0)
            self._cache[key] = result
            return result
        # 距离越近、样本越多，置信度越高。
        distance_factor = max(0.0, 1.0 - best_distance / max(self.max_distance, 1e-6))
        confidence = float(max(0.0, min(1.0, best.confidence * (0.5 + 0.5 * distance_factor))))
        result = (best.semantic_id, confidence)
        self._cache[key] = result
        return result


def primitive_features(primitive: Primitive) -> list[float]:
    """分类器特征；训练与推理必须使用同一函数。"""
    xs = [point[0] for point in primitive.points]
    ys = [point[1] for point in primitive.points]
    width = max(max(xs) - min(xs), 1e-6)
    height = max(max(ys) - min(ys), 1e-6)
    aspect = max(width, height) / max(min(width, height), 1e-6)
    r, g, b = primitive.rgb
    # 不使用 layer_id：SVG 的 <g> 顺序与推理时按像素数排序的颜色层编号
    # 不是同一套编号，作为特征会造成训练/推理分布不一致。
    return [
        float(r), float(g), float(b),
        float(primitive.command),
        float(primitive.width),
        float(primitive.length),
        float(width), float(height), float(aspect),
        float(width * height),
    ]


class PrimitiveClassifier:
    """可选的 sklearn 分类器；参数量远低于 5B。"""

    def __init__(self, model: Any | None = None):
        self.model = model

    @classmethod
    def load(cls, directory: str | Path) -> "PrimitiveClassifier":
        path = Path(directory) / "primitive_classifier.joblib"
        if not path.exists():
            return cls()
        try:
            import joblib
            return cls(joblib.load(path))
        except Exception:  # noqa: BLE001 - 权重损坏时回退颜色先验
            return cls()

    @property
    def available(self) -> bool:
        return self.model is not None

    def classify(self, primitive: Primitive) -> tuple[int, float] | None:
        results = self.classify_many([primitive])
        return results[0] if results is not None else None

    def classify_many(
        self, primitives: list[Primitive],
    ) -> list[tuple[int, float] | None] | None:
        """批量预测，避免逐图元调用 sklearn 的固定开销。"""
        if self.model is None or not primitives:
            return None
        try:
            features = np.asarray(
                [primitive_features(primitive) for primitive in primitives],
                dtype=np.float64,
            )
            probabilities = self.model.predict_proba(features)
            classes = np.asarray(self.model.classes_)
            winners = probabilities.argmax(axis=1)
            results: list[tuple[int, float] | None] = []
            for row, index in enumerate(winners):
                semantic = int(classes[index])
                confidence = float(probabilities[row, index])
                if not 0 <= semantic <= 35:
                    results.append(None)
                else:
                    results.append((semantic, confidence))
            return results
        except Exception:  # noqa: BLE001 - 单图失败回退颜色先验
            return None


def apply_semantics(
    primitives: list[Primitive],
    prior: ColorPrior,
    classifier: PrimitiveClassifier,
    *,
    min_confidence: float = 0.0,
) -> None:
    predicted = classifier.classify_many(primitives)
    for index, primitive in enumerate(primitives):
        result = predicted[index] if predicted is not None else None
        if result is None:
            result = prior.classify(primitive.rgb)
        semantic, confidence = result
        if not 0 <= semantic <= 35:
            semantic, confidence = prior.default_semantic, 0.0
        # 分类器置信度不足时回退忽略类；颜色先验的 confidence 未经校准，
        # 不参与该阈值，避免把有效的先验映射一并压掉。
        if predicted is not None and semantic != prior.default_semantic:
            if confidence < min_confidence:
                semantic = prior.default_semantic
        primitive.semantic_id = int(semantic)
        primitive.confidence = float(max(primitive.confidence * 0.5, confidence))
