#!/usr/bin/env python3
"""SIFT 符号模板匹配可行性探测。

从 train_final GT 实例构建模板，在 test 图上做 SIFT + RANSAC 仿射匹配，
统计同类候选的框召回/精度。只用于离线诊断。
"""
from __future__ import annotations

import argparse
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


def load_image(path):
    rgb, alpha = load_rgba(str(path))
    image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                    0, 255).astype(np.uint8)
    if image.shape[:2] != (RASTER, RASTER):
        image = cv2.resize(image, (RASTER, RASTER),
                           interpolation=cv2.INTER_AREA)
    return image


def build_templates(gt_dir, png_dir, classes, per_class, sift):
    templates = {class_id: [] for class_id in classes}
    for gt_path in sorted(Path(gt_dir).glob("*.json")):
        png = Path(png_dir) / f"{gt_path.stem}.png"
        if not png.exists():
            continue
        gt = load_json(gt_path)
        masks, _ = _segments(gt, (100.0, 100.0), RASTER)
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        if image.shape[:2] != (RASTER, RASTER):
            image = cv2.resize(image, (RASTER, RASTER),
                               interpolation=cv2.INTER_AREA)
        for (class_id, instance), mask in masks.items():
            if class_id not in templates or len(templates[class_id]) >= per_class:
                continue
            if class_id >= 30 or instance < 0 or not mask.any():
                continue
            ys, xs = np.nonzero(mask)
            side = max(int(xs.max() - xs.min()), int(ys.max() - ys.min()), 8)
            pad = int(side * 0.30)
            left = max(int(xs.min()) - pad, 0)
            top = max(int(ys.min()) - pad, 0)
            right = min(int(xs.max()) + pad, RASTER)
            bottom = min(int(ys.max()) + pad, RASTER)
            crop = image[top:bottom, left:right].copy()
            local = mask[top:bottom, left:right]
            crop[~local] = 255
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < 5:
                continue
            templates[class_id].append({
                "descriptors": descriptors,
                "points": np.asarray([kp.pt for kp in keypoints], np.float32),
                "size": (crop.shape[1], crop.shape[0]),
                "stem": gt_path.stem,
            })
        if all(len(items) >= per_class for items in templates.values()):
            break
    return templates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-gt", required=True)
    parser.add_argument("--train-png", required=True)
    parser.add_argument("--test-gt", required=True)
    parser.add_argument("--test-png", required=True)
    parser.add_argument("--classes", default="3,8,18,22,28")
    parser.add_argument("--per-class", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--ratio", type=float, default=0.75)
    parser.add_argument("--out", default="_work/sift_probe.json")
    args = parser.parse_args()
    classes = {int(value) for value in args.classes.split(",") if value.strip()}
    sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.01,
                           edgeThreshold=10)
    templates = build_templates(args.train_gt, args.train_png, classes,
                                args.per_class, sift)
    print("templates", {c: len(v) for c, v in templates.items()}, flush=True)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    stats = defaultdict(lambda: {"gt": 0, "cand": 0, "tp": 0, "stems": 0})
    paths = sorted(Path(args.test_gt).glob("*.json"))[: args.limit]
    for index, gt_path in enumerate(paths):
        png = Path(args.test_png) / f"{gt_path.stem}.png"
        if not png.exists():
            continue
        gt = load_json(gt_path)
        gt_masks, _ = _segments(gt, (100.0, 100.0), RASTER)
        gt_boxes = defaultdict(list)
        for (class_id, instance), mask in gt_masks.items():
            if class_id in classes and class_id < 30 and instance >= 0 and mask.any():
                ys, xs = np.nonzero(mask)
                gt_boxes[class_id].append(
                    (float(xs.min()), float(ys.min()),
                     float(xs.max()), float(ys.max())))
        image = load_image(png)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is None:
            continue
        image_points = np.asarray([kp.pt for kp in keypoints], np.float32)
        for class_id, items in templates.items():
            candidates = []
            for template in items:
                matches = matcher.knnMatch(template["descriptors"],
                                           descriptors, k=2)
                good = [pair[0] for pair in matches
                        if len(pair) == 2 and pair[0].distance <
                        args.ratio * pair[1].distance]
                if len(good) < args.min_matches:
                    continue
                src = template["points"][[m.queryIdx for m in good]]
                dst = image_points[[m.trainIdx for m in good]]
                matrix, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                if matrix is None or inliers is None or int(inliers.sum()) < 5:
                    continue
                scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
                if not 0.5 <= scale <= 2.0:
                    continue
                width, height = template["size"]
                corners = np.asarray([[0, 0, 1], [width, 0, 1],
                                      [width, height, 1], [0, height, 1]],
                                     np.float32).T
                mapped = matrix @ corners
                xs = mapped[0]
                ys = mapped[1]
                candidates.append((tuple(float(v) for v in (
                    max(xs.min(), 0), max(ys.min(), 0),
                    min(xs.max(), RASTER), min(ys.max(), RASTER))),
                    int(inliers.sum())))
            # 类内 NMS
            candidates.sort(key=lambda item: -item[1])
            kept = []
            for box, score in candidates:
                if all(bbox_iou(box, other) < 0.3 for other, _ in kept):
                    kept.append((box, score))
            stats[class_id]["stems"] += 1 if gt_boxes.get(class_id) else 0
            stats[class_id]["gt"] += len(gt_boxes.get(class_id, []))
            stats[class_id]["cand"] += len(kept)
            if gt_boxes.get(class_id) and kept:
                scores = np.zeros((len(gt_boxes[class_id]), len(kept)))
                for row, gt_box in enumerate(gt_boxes[class_id]):
                    for col, (box, _score) in enumerate(kept):
                        scores[row, col] = bbox_iou(gt_box, box)
                rows, cols = linear_sum_assignment(-scores)
                stats[class_id]["tp"] += int((scores[rows, cols] > 0.5).sum())
        if index % 20 == 0:
            print(index, flush=True)
    print(f"{'cls':>3} {'gt':>5} {'cand':>5} {'tp':>4} {'recall':>7} {'precision':>9}")
    for class_id in sorted(classes):
        entry = stats[class_id]
        recall = entry["tp"] / max(entry["gt"], 1)
        precision = entry["tp"] / max(entry["cand"], 1)
        print(f"{class_id:>3} {entry['gt']:>5} {entry['cand']:>5} "
              f"{entry['tp']:>4} {recall:>7.3f} {precision:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
