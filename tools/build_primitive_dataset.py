#!/usr/bin/env python3
"""构建图元图数据集：恢复笔画 ↔ GT 图元匹配，生成节点/边标签。

每个样本保存为 npz：
    features  N×14    edge_index 2×E   edge_features E×4
    labels    N        keys N          （keys: Thing=class*1000+inst, Stuff=90000+class, 忽略=-1）
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cadvec.config import IGNORE_CLASS, STUFF_CLASS_MAX, THING_CLASS_MAX  # noqa: E402
from cadvec.dataset import component_scene  # noqa: E402
from cadvec.layers import load_palette  # noqa: E402
from cadvec.primitive_graph import build_graph  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402
from cadvec.config import InferenceConfig  # noqa: E402

RASTER = 300
MAX_NODES = 1200
_STATE: dict = {}


def _init(palette_path: str, out_dir: str) -> None:
    _STATE["palette"] = load_palette(palette_path)
    _STATE["config"] = InferenceConfig()
    _STATE["out"] = Path(out_dir)


def _mask(points, size=RASTER) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    pixels = np.round(np.asarray(points, dtype=np.float64) * (size / 100.0)).astype(np.int32)
    cv2.polylines(mask, [pixels.reshape(-1, 1, 2)], False, 1, thickness=1)
    return mask.astype(bool)


def _effective(primitive) -> tuple[int, int]:
    semantic = primitive.semantic_id
    if semantic == IGNORE_CLASS:
        return IGNORE_CLASS, -1
    if semantic <= THING_CLASS_MAX and primitive.instance_id < 0:
        return IGNORE_CLASS, -1
    if semantic <= THING_CLASS_MAX:
        return semantic, primitive.instance_id
    if semantic <= STUFF_CLASS_MAX:
        return semantic, -2  # Stuff 按类别合并
    return IGNORE_CLASS, -1


def _match(stroke_mask: np.ndarray, gt_items) -> tuple[int, int] | None:
    area = int(stroke_mask.sum())
    if area < 3:
        return None
    rows, columns = np.nonzero(stroke_mask)
    box = (rows.min(), rows.max() + 1, columns.min(), columns.max() + 1)
    best_key, best_fraction = None, 0.0
    for mask, class_id, instance_id in gt_items:
        if not mask[box[0]:box[1], box[2]:box[3]].any():
            continue
        intersection = int(np.count_nonzero(
            stroke_mask[box[0]:box[1], box[2]:box[3]]
            & mask[box[0]:box[1], box[2]:box[3]]
        ))
        if not intersection:
            continue
        fraction = intersection / area
        if fraction > best_fraction:
            best_fraction = fraction
            best_key = (class_id, instance_id)
    return best_key if best_fraction >= 0.5 else None


def _process(job: tuple[str, str]) -> str:
    png_path, svg_path = job
    name = Path(png_path).stem
    output = _STATE["out"] / f"{name}.npz"
    if output.exists():
        return name
    scene = component_scene(png_path, _STATE["palette"], _STATE["config"])
    strokes = scene.strokes
    if len(strokes) > MAX_NODES:
        keep = sorted(range(len(strokes)), key=lambda i: -strokes[i].length)[:MAX_NODES]
        keep = sorted(keep)
        strokes = [strokes[i] for i in keep]
    features, edge_index, edge_features, _ = build_graph(strokes)
    gt_items = []
    for primitive in iter_primitives(svg_path):
        class_id, instance_id = _effective(primitive)
        gt_items.append((_mask(primitive.points), class_id, instance_id))
    labels = np.full(len(strokes), IGNORE_CLASS, dtype=np.int64)
    keys = np.full(len(strokes), -1, dtype=np.int64)
    for index, stroke in enumerate(strokes):
        matched = _match(_mask(stroke.points), gt_items)
        if matched is None:
            continue
        class_id, instance_id = matched
        labels[index] = class_id
        if class_id == IGNORE_CLASS:
            keys[index] = -1
        elif instance_id == -2:
            keys[index] = 90000 + class_id
        else:
            keys[index] = class_id * 1000 + instance_id
    np.savez_compressed(
        output, features=features, edge_index=edge_index,
        edge_features=edge_features, labels=labels, keys=keys,
    )
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--svg-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--palette", type=Path, default=Path("weights/stroke_palette.json"))
    parser.add_argument("--out", type=Path, default=Path("../_work/primitive_graphs"))
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pngs = sorted(args.png_dir.glob("*.png"))[: args.limit]
    jobs = [(str(png), str(args.svg_dir / f"{png.stem}.svg")) for png in pngs
            if (args.svg_dir / f"{png.stem}.svg").exists()]
    print(f"处理 {len(jobs)} 张", file=sys.stderr)
    with Pool(args.workers, initializer=_init,
              initargs=(str(args.palette), str(args.out))) as pool:
        for index, _ in enumerate(pool.imap_unordered(_process, jobs, chunksize=1), 1):
            if index % 100 == 0:
                print(f"  {index}/{len(jobs)}", file=sys.stderr)
    print("完成", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
