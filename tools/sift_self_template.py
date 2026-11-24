#!/usr/bin/env python3
"""同图自模板 SIFT：用高置信 YOLO 框作种子，在图内找同类重复符号。

避免跨图模板的外观/尺度差异；用于评估 SIFT 能否补充锚点。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402
from cadvec.layers import load_rgba  # noqa: E402

RASTER = 1000


def bbox_iou(first, second):
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / max(area, 1e-9)


def prep(gray):
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--yolo", default="weights/yolo_seg.pt")
    parser.add_argument("--classes", default="3,8,18,22,28")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--ratio", type=float, default=0.85)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dump", default="",
                        help="可选：输出每图 YOLO 检测与 SIFT 候选 JSONL")
    args = parser.parse_args()
    from ultralytics import YOLO

    classes = {int(v) for v in args.classes.split(",") if v.strip()}
    model = YOLO(args.yolo)
    sift = cv2.SIFT_create(nfeatures=1500, contrastThreshold=0.005,
                           edgeThreshold=20)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    stats = defaultdict(lambda: {"gt": 0, "cand": 0, "tp": 0})
    dump_handle = open(args.dump, "w", encoding="utf-8") if args.dump else None
    paths = sorted(Path(args.gt_dir).glob("*.json"))[: args.limit]
    for index, gt_path in enumerate(paths):
        png = Path(args.png_dir) / f"{gt_path.stem}.png"
        if not png.exists():
            continue
        gt = load_json(gt_path)
        masks, _ = _segments(gt, (100.0, 100.0), RASTER)
        gt_boxes = defaultdict(list)
        for (class_id, instance), mask in masks.items():
            if class_id in classes and class_id < 30 and instance >= 0 and mask.any():
                ys, xs = np.nonzero(mask)
                gt_boxes[class_id].append((float(xs.min()), float(ys.min()),
                                           float(xs.max()), float(ys.max())))
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        if image.shape[:2] != (RASTER, RASTER):
            image = cv2.resize(image, (RASTER, RASTER),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_points, image_desc = sift.detectAndCompute(prep(gray), None)
        if image_desc is None:
            continue
        image_points = np.asarray([kp.pt for kp in image_points], np.float32)
        result = model.predict(source=str(png), imgsz=640, conf=args.conf,
                               iou=0.5, verbose=False, device=args.device)[0]
        detections = defaultdict(list)
        if result.boxes is not None:
            for class_id, confidence, box in zip(
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.xyxy.cpu().numpy()):
                if int(class_id) in classes:
                    detections[int(class_id)].append(
                        (float(confidence), tuple(float(v) for v in box)))
        record = {
            "stem": gt_path.stem,
            "detections": [
                {"class": class_id, "conf": confidence, "box": list(box)}
                for class_id, items in detections.items()
                for confidence, box in items
            ],
            "candidates": [],
        }
        for class_id, items in detections.items():
            items.sort(reverse=True)
            candidates = []
            for _confidence, seed in items[:2]:
                x0, y0, x1, y1 = seed
                pad = 0.3 * max(x1 - x0, y1 - y0)
                left = max(int(x0 - pad), 0)
                top = max(int(y0 - pad), 0)
                right = min(int(x1 + pad), RASTER)
                bottom = min(int(y1 + pad), RASTER)
                crop_gray = prep(gray[top:bottom, left:right])
                seed_points, seed_desc = sift.detectAndCompute(crop_gray, None)
                if seed_desc is None or len(seed_points) < 5:
                    continue
                knn = matcher.knnMatch(seed_desc, image_desc, k=2)
                good = [pair[0] for pair in knn
                        if len(pair) == 2 and pair[0].distance <
                        args.ratio * pair[1].distance]
                if len(good) < args.min_matches:
                    continue
                src = np.asarray([seed_points[m.queryIdx].pt for m in good],
                                 np.float32)
                dst = image_points[[m.trainIdx for m in good]]
                matrix, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if matrix is None or inliers is None or int(inliers.sum()) < 6:
                    continue
                scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
                if not 0.7 <= scale <= 1.4:
                    continue
                local = (x0 - left, y0 - top, x1 - left, y1 - top)
                corners = np.asarray([[local[0], local[1], 1],
                                      [local[2], local[1], 1],
                                      [local[2], local[3], 1],
                                      [local[0], local[3], 1]],
                                     np.float32).T
                mapped = matrix @ corners
                xs = mapped[0]
                ys = mapped[1]
                box = (float(max(xs.min(), 0)), float(max(ys.min(), 0)),
                       float(min(xs.max(), RASTER)),
                       float(min(ys.max(), RASTER)))
                candidates.append((box, int(inliers.sum())))
            candidates.sort(key=lambda item: -item[1])
            kept = []
            for box, score in candidates:
                if all(bbox_iou(box, other) < 0.3 for other, _ in kept):
                    kept.append((box, score))
            record["candidates"].extend(
                {"class": class_id, "box": list(box), "score": score}
                for box, score in kept)
            stats[class_id]["gt"] += len(gt_boxes.get(class_id, []))
            stats[class_id]["cand"] += len(kept)
            if gt_boxes.get(class_id) and kept:
                scores = np.zeros((len(gt_boxes[class_id]), len(kept)))
                for row, gt_box in enumerate(gt_boxes[class_id]):
                    for col, (box, _score) in enumerate(kept):
                        scores[row, col] = bbox_iou(gt_box, box)
                rows, cols = linear_sum_assignment(-scores)
                stats[class_id]["tp"] += int((scores[rows, cols] > 0.5).sum())
        if dump_handle is not None:
            dump_handle.write(json.dumps(record) + "\n")
        if index % 20 == 0:
            print(index, flush=True)
    print(f"{'cls':>3} {'gt':>5} {'cand':>5} {'tp':>4} {'recall':>7} {'precision':>9}")
    for class_id in sorted(classes):
        entry = stats[class_id]
        print(f"{class_id:>3} {entry['gt']:>5} {entry['cand']:>5} "
              f"{entry['tp']:>4} {entry['tp']/max(entry['gt'],1):>7.3f} "
              f"{entry['tp']/max(entry['cand'],1):>9.3f}")
    if dump_handle is not None:
        dump_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
