#!/usr/bin/env python3
"""构建 YOLO 实例分割数据集：GT 实例的凸包多边形 + 类别。

凸包作为“完整实例候选区域”：推理时用预测掩码选取区域内的图像笔画，
再矢量化——解决模板路线的候选不完整问题。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cadvec.config import IGNORE_CLASS, STUFF_CLASS_MAX, THING_CLASS_MAX  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402


def _key(primitive):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--svg-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--out", type=Path, default=Path("../_work/yolo_seg_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.06)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--hull-pad", type=float, default=0.35,
                        help="凸包外扩（画布单位），让掩码包含描边宽度")
    args = parser.parse_args()
    pngs = sorted(args.png_dir.glob("*.png"))
    if args.limit:
        pngs = pngs[: args.limit]
    split = max(1, int(len(pngs) * (1.0 - args.val_fraction)))
    for subset in ("train", "val"):
        (args.out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / subset).mkdir(parents=True, exist_ok=True)
    boxes = 0
    for index, png in enumerate(pngs):
        subset = "train" if index < split else "val"
        svg = args.svg_dir / f"{png.stem}.svg"
        if not svg.exists():
            continue
        try:
            primitives = list(iter_primitives(svg))
        except Exception:  # noqa: BLE001
            continue
        groups = defaultdict(list)
        for primitive in primitives:
            key = _key(primitive)
            if key is not None:
                groups[key].append(primitive)
        lines = []
        for (class_id, _), items in groups.items():
            points = np.asarray(
                [[x, y] for primitive in items for x, y in primitive.points],
                dtype=np.float32,
            )
            if len(points) < 3:
                continue
            hull = cv2.convexHull(points.reshape(-1, 1, 2)).reshape(-1, 2)
            if len(hull) < 3:
                continue
            # 外扩一点，覆盖 1px 描边与抗锯齿。
            center = hull.mean(axis=0)
            vector = hull - center
            length = np.linalg.norm(vector, axis=1, keepdims=True)
            padded = center + vector * (1.0 + args.hull_pad / np.maximum(length, 1e-3))
            normalized = np.clip(padded / 100.0, 0.0, 1.0)
            coordinates = " ".join(
                f"{value:.6f}" for value in normalized.reshape(-1).tolist()
            )
            lines.append(f"{class_id} {coordinates}")
            boxes += 1
        if not lines:
            continue
        (args.out / "labels" / subset / f"{png.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
        shutil.copy(png, args.out / "images" / subset / png.name)
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(pngs)} 实例 {boxes}", file=sys.stderr)
    names = "\n".join(f"  {class_id}: c{class_id}" for class_id in range(35))
    (args.out / "cad_seg.yaml").write_text(
        f"path: {(args.out.resolve()).as_posix()}\n"
        f"train: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(f"完成：{boxes} 个实例多边形 → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
