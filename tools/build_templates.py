#!/usr/bin/env python3
"""从训练集 SVG 构建符号模板库（GT 实例 → 归一化形状描述子 + 图元）。

模板拟合是文献里高 PQ 的关键（TempFit 思路）：实例 = 模板 + 刚体变换，
保证实例完整性。这里用经典实现：形状描述子做检索、质心+尺度+90° 旋转
做初始对齐，推理时再用距离变换做平移精修。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadvec.config import IGNORE_CLASS, STUFF_CLASS_MAX, THING_CLASS_MAX  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402

GRID = 16
ROTATIONS = 4


def _instance_key(primitive):
    semantic = primitive.semantic_id
    if semantic == IGNORE_CLASS:
        return None
    if semantic <= THING_CLASS_MAX:
        if primitive.instance_id < 0:
            return None
        return (semantic, primitive.instance_id)
    if semantic <= STUFF_CLASS_MAX:
        return (semantic, -1)
    return None


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


def _descriptor(points: np.ndarray, rotation: int = 0) -> np.ndarray:
    normalized, _, _ = _normalize(points, rotation)
    xs = np.clip(((normalized[:, 0] + 0.5) * (GRID - 1)).round().astype(int), 0, GRID - 1)
    ys = np.clip(((normalized[:, 1] + 0.5) * (GRID - 1)).round().astype(int), 0, GRID - 1)
    grid = np.zeros((GRID, GRID), dtype=np.float32)
    np.add.at(grid, (ys, xs), 1.0)
    return (grid / max(grid.sum(), 1.0)).ravel()


def _descriptors(points: np.ndarray) -> np.ndarray:
    return np.stack([_descriptor(points, rotation) for rotation in range(ROTATIONS)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--out", type=Path, default=Path("weights/symbol_templates.pkl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cluster-distance", type=float, default=0.02,
                        help="描述子 L2 距离小于该值的实例合并为同一模板")
    args = parser.parse_args()
    files = sorted(args.svg_dir.glob("*.svg"))
    if args.limit:
        files = files[: args.limit]
    print(f"解析 {len(files)} 个 SVG", file=sys.stderr)
    templates = []
    seen: dict[tuple, int] = {}

    def coarse_key(class_id: int, descriptor: np.ndarray) -> tuple:
        grid = descriptor.reshape(GRID, GRID)
        pooled = grid.reshape(GRID // 2, 2, GRID // 2, 2).mean(axis=(1, 3))
        return (class_id,) + tuple(np.round(pooled * 25).astype(int).ravel().tolist())

    for index, path in enumerate(files, 1):
        try:
            primitives = list(iter_primitives(path))
        except Exception:  # noqa: BLE001
            continue
        groups: dict[tuple[int, int], list] = defaultdict(list)
        for primitive in primitives:
            key = _instance_key(primitive)
            if key is not None:
                groups[key].append(primitive)
        for (class_id, _), items in groups.items():
            points = np.asarray([point for primitive in items for point in primitive.points],
                                dtype=np.float64)
            if len(points) < 3:
                continue
            descriptors = _descriptors(points)
            key = coarse_key(class_id, descriptors[0])
            if key in seen:
                templates[seen[key]]["count"] += 1
                continue
            seen[key] = len(templates)
            templates.append({
                "class": int(class_id),
                "descriptors": descriptors.astype(np.float16),
                "count": 1,
                "primitives": [
                    {
                        "points": [[float(x), float(y)] for x, y in primitive.points],
                        "rgb": [int(v) for v in primitive.rgb],
                        "width": float(primitive.width),
                        "command": int(primitive.command),
                    }
                    for primitive in items[:48]
                ],
            })
        if index % 400 == 0:
            print(f"  {index}/{len(files)} 模板 {len(templates)}", file=sys.stderr)

    payload = {
        "schema_version": 1,
        "grid": GRID,
        "rotations": ROTATIONS,
        "templates": templates,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    import zlib
    args.out.write_bytes(zlib.compress(pickle.dumps(payload, protocol=4), 9))
    print(f"写入 {args.out}: {len(templates)} 个模板, "
          f"{args.out.stat().st_size / 1e6:.1f}MB", file=sys.stderr)
    meta = {
        "templates": len(templates),
        "grid": GRID,
        "rotations": ROTATIONS,
        "svg_files": len(files),
        "cluster_distance": args.cluster_distance,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
