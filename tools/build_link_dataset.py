#!/usr/bin/env python3
"""构建笔画成对链接数据集：同一 GT 实例的邻近笔画对为正样本。

对每张训练图跑组件化几何，得到全部笔画；把笔画栅格化后与各 GT 实例掩码
求重叠，重叠 ≥ 50% 视为属于该实例。近邻笔画对按标签生成正负样本。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from cadvec.config import InferenceConfig  # noqa: E402
from cadvec.dataset import component_scene, gt_segment_masks  # noqa: E402
from cadvec.grouping import candidate_pairs, pair_features  # noqa: E402
from cadvec.layers import load_palette  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402

RASTER = 400
_STATE: dict = {}


def _init(palette_path: str) -> None:
    _STATE["palette"] = load_palette(palette_path)
    _STATE["config"] = InferenceConfig()


def _stroke_masks(strokes) -> list[np.ndarray]:
    masks = []
    for stroke in strokes:
        mask = np.zeros((RASTER, RASTER), dtype=np.uint8)
        points = np.round(np.asarray(stroke.points) * (RASTER / 100.0)).astype(np.int32)
        cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, thickness=1)
        masks.append(mask.astype(bool))
    return masks


def _assign(strokes, masks, gt_masks) -> list[tuple[int, int] | None]:
    assignment: list[tuple[int, int] | None] = []
    gt_items = list(gt_masks.items())
    for stroke, mask in zip(strokes, masks):
        area = int(mask.sum())
        if area < 3:
            assignment.append(None)
            continue
        rows, columns = np.nonzero(mask)
        box = (rows.min(), rows.max() + 1, columns.min(), columns.max() + 1)
        best_key, best_fraction = None, 0.0
        for key, gt in gt_items:
            if not gt[box[0]:box[1], box[2]:box[3]].any():
                continue
            intersection = int(np.count_nonzero(mask[box[0]:box[1], box[2]:box[3]]
                                               & gt[box[0]:box[1], box[2]:box[3]]))
            if not intersection:
                continue
            fraction = intersection / area
            if fraction > best_fraction:
                best_fraction = fraction
                best_key = key
        assignment.append(best_key if best_fraction >= 0.5 else None)
    return assignment


def _process(job: tuple[str, str]) -> tuple[np.ndarray, np.ndarray]:
    png_path, svg_path = job
    scene = component_scene(png_path, _STATE["palette"], _STATE["config"])
    if not scene.strokes:
        return np.zeros((0, 12)), np.zeros((0,))
    gt_masks = gt_segment_masks(list(iter_primitives(svg_path)), 100, 100)
    masks = _stroke_masks(scene.strokes)
    assignment = _assign(scene.strokes, masks, gt_masks)
    component_of: dict[int, int] = {}
    for hypothesis_id, hypothesis in enumerate(scene.hypotheses):
        for index in hypothesis.indices:
            component_of[index] = hypothesis_id
    component_sizes = defaultdict(int)
    for index, component in component_of.items():
        component_sizes[component] += 1
    pairs = candidate_pairs(scene.strokes)
    positives, negatives = [], []
    for first, second in pairs:
        a, b = assignment[first], assignment[second]
        feature = pair_features(
            scene.strokes[first], scene.strokes[second],
            same_component=component_of.get(first) == component_of.get(second),
            component_size=component_sizes.get(component_of.get(first), 1),
        )
        if a is not None and b is not None and a == b:
            positives.append(feature)
        else:
            negatives.append(feature)
    rng = np.random.default_rng(0)
    if len(negatives) > 2 * len(positives):
        indices = rng.choice(len(negatives), size=2 * len(positives), replace=False)
        negatives = [negatives[index] for index in indices]
    features = np.asarray(positives + negatives, dtype=np.float32)
    labels = np.concatenate([
        np.ones(len(positives), dtype=np.float32),
        np.zeros(len(negatives), dtype=np.float32),
    ])
    return features, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--svg-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--palette", type=Path, default=Path("weights/stroke_palette.json"))
    parser.add_argument("--out", type=Path, default=Path("../_work/link_dataset.npz"))
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pngs = sorted(args.png_dir.glob("*.png"))[: args.limit]
    jobs = [(str(png), str(args.svg_dir / f"{png.stem}.svg")) for png in pngs
            if (args.svg_dir / f"{png.stem}.svg").exists()]
    print(f"处理 {len(jobs)} 张（workers={args.workers}）", file=sys.stderr)
    features_all, labels_all = [], []
    with Pool(args.workers, initializer=_init, initargs=(str(args.palette),)) as pool:
        for index, (features, labels) in enumerate(
                pool.imap_unordered(_process, jobs, chunksize=1), start=1):
            if features.size:
                features_all.append(features)
                labels_all.append(labels)
            if index % 40 == 0:
                total = sum(len(item) for item in labels_all)
                print(f"  {index}/{len(jobs)} 样本 {total}", file=sys.stderr)
    if not features_all:
        print("没有样本", file=sys.stderr)
        return 1
    features = np.concatenate(features_all)
    labels = np.concatenate(labels_all)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, features=features, labels=labels)
    print(f"写入 {args.out}: {features.shape}, 正样本 {int(labels.sum())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
