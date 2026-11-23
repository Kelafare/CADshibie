#!/usr/bin/env python3
"""E1：真实 GT 实例的几何往返诊断。

对每个 GT Thing 实例，分别在原尺度概率和 256/192 标签概率下调用当前
``vectorize_mask``，再按评分器一致的 1000px PIL 栅格比较 IoU。用于回答：
损失发生在 ROI 标签缩放、骨架/切段，还是四点表示本身。

只读 GT，不参与正式推理。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402
from cadvec.roi_ink import PAD, vectorize_mask  # noqa: E402

RASTER = 1000


def crop_box(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return None
    box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    side = max(box[2] - box[0], box[3] - box[1])
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    half = side * (0.5 + PAD)
    left = int(round(max(cx - half, 0)))
    top = int(round(max(cy - half, 0)))
    right = int(round(min(cx + half, RASTER)))
    bottom = int(round(min(cy + half, RASTER)))
    if right - left < 8 or bottom - top < 8:
        return None
    return left, top, right, bottom


def rasterize(segments) -> np.ndarray:
    image = Image.new("1", (RASTER, RASTER), 0)
    draw = ImageDraw.Draw(image)
    for points, _length in segments:
        draw.line([(float(x) * RASTER / 100.0, float(y) * RASTER / 100.0)
                   for x, y in points], fill=1, width=1, joint="curve")
    return np.asarray(image, dtype=bool)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    return intersection / int(np.count_nonzero(first | second))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out", default="_work/geom_roundtrip.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--crop-sizes", default="192,256")
    parser.add_argument("--chunks", default="4,6,8,12,16")
    parser.add_argument("--adaptive-eps", default="",
                        help="逗号分隔的四点误差阈值；非空时额外扫描自适应切分")
    args = parser.parse_args()

    chunks = [int(value) for value in args.chunks.split(",") if value.strip()]
    adaptive_eps = [float(value) for value in args.adaptive_eps.split(",")
                    if value.strip()]
    crop_sizes = [int(value) for value in args.crop_sizes.split(",") if value.strip()]
    variants = [("native", 0)] + [(f"crop{size}", size) for size in crop_sizes]
    stats = defaultdict(lambda: {"ious": [], "segments": [], "hits": 0, "n": 0,
                                 "per_class": defaultdict(list)})
    paths = sorted(Path(args.gt_dir).glob("*.json"))[: args.limit]
    for index, path in enumerate(paths):
        gt = load_json(path)
        size = (max(float(gt.get("width", 100)), 1e-6),
                max(float(gt.get("height", 100)), 1e-6))
        masks, _ = _segments(gt, size, RASTER)
        per_class = defaultdict(int)
        for (class_id, instance), mask in masks.items():
            if class_id >= 30 or class_id == 35 or instance < 0:
                continue
            if per_class[class_id] >= args.per_class:
                continue
            per_class[class_id] += 1
            box = crop_box(mask)
            if box is None:
                continue
            left, top, right, bottom = box
            truth = mask[top:bottom, left:right]
            for variant, crop_size in variants:
                if variant == "native":
                    probability = truth.astype(np.float32)
                else:
                    low = cv2.resize(
                        truth.astype(np.uint8) * 255, (crop_size, crop_size),
                        interpolation=cv2.INTER_AREA)
                    low = cv2.dilate(low, np.ones((3, 3), np.uint8), iterations=1)
                    probability = cv2.resize(
                        low, (right - left, bottom - top),
                        interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                for chunk in chunks:
                    segments = vectorize_mask(
                        probability, left, top, right - left, bottom - top,
                        10.0, 10.0, 0.5, chunk=chunk)
                    value = iou(mask, rasterize(segments))
                    entry = stats[(variant, chunk)]
                    entry["ious"].append(value)
                    entry["segments"].append(len(segments))
                    entry["hits"] += int(value > 0.5)
                    entry["n"] += 1
                    entry["per_class"][class_id].append(value)
                for eps in adaptive_eps:
                    segments = vectorize_mask(
                        probability, left, top, right - left, bottom - top,
                        10.0, 10.0, 0.5, adaptive=True, eps=eps)
                    value = iou(mask, rasterize(segments))
                    key = (variant, f"a{eps:g}")
                    entry = stats[key]
                    entry["ious"].append(value)
                    entry["segments"].append(len(segments))
                    entry["hits"] += int(value > 0.5)
                    entry["n"] += 1
                    entry["per_class"][class_id].append(value)
        if index % 20 == 0:
            print(index, flush=True)

    payload = {"images": len(paths), "variants": {}}
    print(f"{'variant':>8} {'chunk':>5} {'n':>5} {'meanIoU':>8} "
          f"{'hit>0.5':>8} {'meanSeg':>8}")
    for (variant, chunk), entry in sorted(
            stats.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))):
        values = np.asarray(entry["ious"])
        record = {
            "n": entry["n"],
            "mean_iou": float(values.mean()),
            "hit_rate": entry["hits"] / max(entry["n"], 1),
            "mean_segments": float(np.mean(entry["segments"])),
            "per_class": {
                str(class_id): {
                    "n": len(items),
                    "mean_iou": float(np.mean(items)),
                    "hit_rate": float(np.mean(np.asarray(items) > 0.5)),
                }
                for class_id, items in sorted(entry["per_class"].items())
            },
        }
        payload["variants"][f"{variant}_chunk{chunk}"] = record
        print(f"{variant:>8} {chunk:>5} {entry['n']:>5} "
              f"{record['mean_iou']:>8.4f} {record['hit_rate']:>8.3f} "
              f"{record['mean_segments']:>8.2f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
