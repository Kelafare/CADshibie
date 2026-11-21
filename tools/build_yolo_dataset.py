#!/usr/bin/env python3
"""把训练集 SVG 转成 YOLO 检测数据集（GT 实例包围盒 + 类别）。"""
from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    parser.add_argument("--out", type=Path, default=Path("../_work/yolo_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.06)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    pngs = sorted(args.png_dir.glob("*.png"))
    if args.limit:
        pngs = pngs[: args.limit]
    split = max(1, int(len(pngs) * (1.0 - args.val_fraction)))
    for subset in ("train", "val"):
        (args.out / "images" / subset).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / subset).mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
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
            xs = [point[0] for primitive in items for point in primitive.points]
            ys = [point[1] for primitive in items for point in primitive.points]
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            width, height = x1 - x0, y1 - y0
            if width < 0.15 or height < 0.15:
                continue
            cx, cy = (x0 + x1) / 200.0, (y0 + y1) / 200.0
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width / 100:.6f} {height / 100:.6f}")
            counts[class_id] += 1
        if not lines:
            continue
        (args.out / "labels" / subset / f"{png.stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )
        shutil.copy(png, args.out / "images" / subset / png.name)
        if (index + 1) % 500 == 0:
            print(f"  {index + 1}/{len(pngs)}", file=sys.stderr)
    yaml = args.out / "cad.yaml"
    names = "\n".join(f"  {class_id}: c{class_id}" for class_id in range(35))
    yaml.write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(f"完成：{sum(counts.values())} 个框，{len(counts)} 类；YAML {yaml}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
