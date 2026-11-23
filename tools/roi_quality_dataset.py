#!/usr/bin/env python3
"""提取 ROI 细化实例的质量特征与 TP/FP 标签（训练学习式质量门控）。

对每张图用 YOLO-seg 检测框调用 ROI 模型，得到实例几何/概率特征；
再把实例折线按评分器口径栅格化，与同名 SVG 的 GT 实例做匈牙利匹配，
命中 IoU>0.5 记为 TP。输出 JSONL，每行一个预测实例。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cadvec.layers import load_rgba  # noqa: E402
from cadvec.roi_ink import RoiInkRefiner  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402

RASTER = 500


def render_gt(primitives, classes):
    masks = {}
    scale = RASTER / 100.0
    for primitive in primitives:
        semantic = int(primitive.semantic_id)
        instance = int(primitive.instance_id)
        if semantic not in classes or instance < 0:
            continue
        key = (semantic, instance)
        mask = masks.get(key)
        if mask is None:
            mask = np.zeros((RASTER, RASTER), dtype=np.uint8)
            masks[key] = mask
        points = np.round(
            np.asarray(primitive.points, dtype=np.float64) * scale).astype(np.int32)
        if int(primitive.command) in (0, 1):
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, 1)
        else:
            low = points.min(axis=0)
            high = points.max(axis=0)
            center = tuple(int(v) for v in (low + high) // 2)
            axes = tuple(max(int(v), 1) for v in (high - low) // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, 1)
    return {key: mask.astype(bool) for key, mask in masks.items()}


def render_instance(segments):
    scale = RASTER / 100.0
    mask = np.zeros((RASTER, RASTER), dtype=np.uint8)
    for segment in segments:
        points = np.round(
            np.asarray(segment["points"], dtype=np.float64) * scale).astype(np.int32)
        cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, 1)
    return mask.astype(bool)


def iou(first, second):
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    return intersection / int(np.count_nonzero(first | second))


def box_iou(first, second):
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = (first[2] - first[0]) * (first[3] - first[1])
    area_b = (second[2] - second[0]) * (second[3] - second[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--svg-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights-192",
                        default=str(ROOT / "weights" / "roi_ink.pt"))
    parser.add_argument("--weights-256",
                        default=str(ROOT / "weights" / "roi_ink_256.pt"))
    parser.add_argument("--classes-192", default="2")
    parser.add_argument("--classes-256",
                        default="8,12,14,15,18,19,20,21,22,25,27,28")
    parser.add_argument("--yolo", default=str(ROOT / "weights" / "yolo_seg.pt"))
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    classes_192 = frozenset(int(v) for v in args.classes_192.split(",") if v.strip())
    classes_256 = frozenset(int(v) for v in args.classes_256.split(",") if v.strip())
    classes = classes_192 | classes_256

    import torch
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    refiner_192 = RoiInkRefiner.load(args.weights_192, device=args.device)
    refiner_256 = RoiInkRefiner.load(args.weights_256, device=args.device)

    pngs = sorted(f for f in os.listdir(args.png_dir) if f.endswith(".png"))
    if args.limit:
        pngs = pngs[: args.limit]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    written = 0
    gt_counts = {}
    with out_path.open("w", encoding="utf-8") as handle:
        for index, name in enumerate(pngs):
            stem = os.path.splitext(name)[0]
            svg = os.path.join(args.svg_dir, stem + ".svg")
            if not os.path.exists(svg):
                continue
            png = os.path.join(args.png_dir, name)
            try:
                primitives = list(iter_primitives(svg))
            except Exception:
                continue
            gt = render_gt(primitives, classes)
            gt_count = defaultdict(set)
            for (semantic, instance) in gt:
                gt_count[semantic].add(instance)
            gt_counts[stem] = {str(k): len(v) for k, v in gt_count.items()}
            rgb, alpha = load_rgba(png)
            image = np.clip(
                rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                0, 255).astype(np.uint8)
            height, width = image.shape[:2]
            result = model.predict(source=png, imgsz=args.imgsz, conf=args.conf,
                                   iou=0.5, verbose=False, device=args.device)[0]
            boxes = []
            if result.boxes is not None and len(result.boxes):
                for class_id, confidence, box in zip(
                        result.boxes.cls.cpu().numpy().astype(int),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.xyxy.cpu().numpy()):
                    if int(class_id) in classes:
                        boxes.append((int(class_id), float(confidence),
                                      tuple(float(v) for v in box)))
            analyzed = []
            for classes_set, refiner in ((classes_192, refiner_192),
                                         (classes_256, refiner_256)):
                subset = [item for item in boxes if item[0] in classes_set]
                if subset:
                    analyzed.extend(refiner.analyze(
                        image, subset, width, height, 100, 100))
            # 与推理一致：按类别做框 IoU NMS + 每类 15 个上限
            kept = []
            counts = defaultdict(int)
            kept_boxes = defaultdict(list)
            for item in sorted(analyzed, key=lambda entry: -entry["confidence"]):
                class_id = int(item["class_id"])
                if counts[class_id] >= 15:
                    continue
                if any(box_iou(item["box"], other) > 0.5
                       for other in kept_boxes[class_id]):
                    continue
                kept_boxes[class_id].append(item["box"])
                counts[class_id] += 1
                kept.append(item)
            # 按类别做匈牙利匹配
            by_class = defaultdict(list)
            for item in kept:
                by_class[int(item["class_id"])].append(item)
            for class_id, items in by_class.items():
                gt_items = {key: mask for key, mask in gt.items() if key[0] == class_id}
                gt_keys = list(gt_items)
                if gt_keys:
                    scores = np.zeros((len(gt_keys), len(items)))
                    masks = []
                    for j, item in enumerate(items):
                        mask = render_instance(item["segments"])
                        masks.append(mask)
                        for i, key in enumerate(gt_keys):
                            scores[i, j] = iou(gt_items[key], mask)
                    assignment_rows, assignment_cols = linear_sum_assignment(-scores)
                    matched = set()
                    match_iou = {}
                    for r, c in zip(assignment_rows, assignment_cols):
                        if scores[r, c] > 0.5:
                            matched.add(int(c))
                            match_iou[int(c)] = float(scores[r, c])
                else:
                    matched, match_iou = set(), {}
                for j, item in enumerate(items):
                    row = {
                        "name": stem,
                        "class": class_id,
                        "gt_count": len(gt_keys),
                        "tp": j in matched,
                        "iou": match_iou.get(j, 0.0),
                        "features": item["features"],
                    }
                    handle.write(json.dumps(row) + "\n")
                    written += 1
            if index % 200 == 0:
                print(index, written, round(time.perf_counter() - started, 1),
                      flush=True)
    Path(str(out_path) + ".gt.json").write_text(
        json.dumps(gt_counts, ensure_ascii=False), encoding="utf-8")
    print("done", written, round(time.perf_counter() - started, 1))


if __name__ == "__main__":
    main()
