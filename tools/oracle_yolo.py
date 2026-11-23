#!/usr/bin/env python3
"""E7 区域 oracle：比较 YOLO 区域、GT 区域与现有 ROI 墨迹头的上限。

对每张图：
- YOLO 类内框与 GT 实例框做匈牙利匹配（框 IoU>0.5），统计 proposal 覆盖；
- YOLO 掩码与 GT 实例掩码的 IoU（>0.5 记为可匹配）；
- 对已有 ROI 权重的类别，分别用 GT 框和 YOLO 框调用细化头，比较墨迹 IoU。

只用于离线诊断；推理不读取 GT。
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
from cadvec.roi_ink import RoiInkRefiner  # noqa: E402

RASTER = 500
THING = 30


def bbox_iou(first, second):
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / max(area, 1e-9)


def mask_iou(first, second):
    inter = int(np.count_nonzero(first & second))
    if not inter:
        return 0.0
    return inter / int(np.count_nonzero(first | second))


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def raster_segments(segments):
    """与 cadvec.evaluate 相同的 PIL 1px 栅格化。"""
    from PIL import Image, ImageDraw
    image = Image.new("1", (RASTER, RASTER), 0)
    draw = ImageDraw.Draw(image)
    for segment in segments:
        points = [(float(x) * RASTER / 100.0, float(y) * RASTER / 100.0)
                  for x, y in segment["points"]]
        draw.line(points, fill=1, width=1, joint="curve")
    return np.asarray(image, dtype=bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--yolo", default="weights/yolo_seg.pt")
    parser.add_argument("--refiner-192", default="weights/roi_ink_c2.pt")
    parser.add_argument("--classes-192", default="2")
    parser.add_argument("--refiner-256", default="weights/roi_ink_256.pt")
    parser.add_argument("--classes-256",
                        default="8,12,14,15,18,19,20,21,22,25,27,28")
    parser.add_argument("--refiner-unified", default="")
    parser.add_argument("--classes-unified", default="")
    parser.add_argument("--out", default="_work/oracle_yolo.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf", type=float, default=0.2)
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    classes_192 = frozenset(int(v) for v in args.classes_192.split(",") if v.strip())
    classes_256 = frozenset(int(v) for v in args.classes_256.split(",") if v.strip())
    classes_unified = frozenset(
        int(v) for v in args.classes_unified.split(",") if v.strip())
    refiner_192 = RoiInkRefiner.load(args.refiner_192, device=args.device)
    refiner_256 = RoiInkRefiner.load(args.refiner_256, device=args.device)
    refiner_unified = (RoiInkRefiner.load(args.refiner_unified, device=args.device)
                       if args.refiner_unified and classes_unified else None)

    stats = defaultdict(lambda: {
        "gt": 0, "det": 0, "box_tp": 0, "mask_tp": 0, "mask_iou_sum": 0.0,
        "gt_box_refine_hit": 0, "gt_box_refine_iou": 0.0,
        "yolo_box_refine_hit": 0, "yolo_box_refine_iou": 0.0,
    })
    pngs = sorted(Path(args.png_dir).glob("*.png"))[: args.limit]
    for index, png in enumerate(pngs):
        gt_path = Path(args.gt_dir) / f"{png.stem}.json"
        if not gt_path.exists():
            continue
        gt = load_json(gt_path)
        size = (max(float(gt.get("width", 100)), 1e-6),
                max(float(gt.get("height", 100)), 1e-6))
        gt_masks, _ = _segments(gt, size, RASTER)
        gt_items = {key: mask for key, mask in gt_masks.items()
                    if key[0] < THING and key[1] >= 0}
        gt_boxes = {key: bbox(mask) for key, mask in gt_items.items()}
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        height, width = image.shape[:2]
        result = model.predict(source=str(png), imgsz=640, conf=args.conf,
                               iou=0.5, verbose=False, device=args.device)[0]
        dets = []
        if result.masks is not None and result.boxes is not None and len(result.masks):
            for polygon, class_id, confidence in zip(
                    result.masks.xy,
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy()):
                if polygon is None or len(polygon) < 3:
                    continue
                mask = np.zeros((RASTER, RASTER), np.uint8)
                points = np.asarray(polygon, dtype=np.float64)
                points[:, 0] *= RASTER / max(width, 1)
                points[:, 1] *= RASTER / max(height, 1)
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
                dets.append((int(class_id), float(confidence), mask.astype(bool)))
        det_boxes = {index: bbox(mask) for index, (_c, _conf, mask) in enumerate(dets)}

        by_class_gt = defaultdict(list)
        by_class_det = defaultdict(list)
        for key in gt_items:
            by_class_gt[key[0]].append(key)
        for det_index, (class_id, _conf, _mask) in enumerate(dets):
            by_class_det[class_id].append(det_index)
        matched_gt = set()
        matched_det = set()
        for class_id in sorted(set(by_class_gt) | set(by_class_det)):
            gts = by_class_gt.get(class_id, [])
            det_indices = by_class_det.get(class_id, [])
            stats[class_id]["gt"] += len(gts)
            stats[class_id]["det"] += len(det_indices)
            if not gts or not det_indices:
                continue
            scores = np.zeros((len(gts), len(det_indices)), dtype=np.float32)
            for row, gkey in enumerate(gts):
                for col, det_index in enumerate(det_indices):
                    if bbox_iou(gt_boxes[gkey], det_boxes[det_index]) < 0.05:
                        continue
                    scores[row, col] = mask_iou(gt_items[gkey], dets[det_index][2])
            rows, cols = linear_sum_assignment(-scores)
            for row, col in zip(rows, cols):
                if scores[row, col] <= 0.5:
                    continue
                stats[class_id]["mask_tp"] += 1
                stats[class_id]["mask_iou_sum"] += float(scores[row, col])
                matched_gt.add(gts[row])
                matched_det.add(det_indices[col])
        # 框匹配（与掩码独立；框 IoU>0.5）
        for class_id in sorted(set(by_class_gt) | set(by_class_det)):
            gts = by_class_gt.get(class_id, [])
            det_indices = by_class_det.get(class_id, [])
            if not gts or not det_indices:
                continue
            scores = np.zeros((len(gts), len(det_indices)), dtype=np.float32)
            for row, gkey in enumerate(gts):
                for col, det_index in enumerate(det_indices):
                    scores[row, col] = bbox_iou(gt_boxes[gkey], det_boxes[det_index])
            rows, cols = linear_sum_assignment(-scores)
            for row, col in zip(rows, cols):
                if scores[row, col] > 0.5:
                    stats[class_id]["box_tp"] += 1

        # GT 框 → 现有细化头；YOLO 框 → 现有细化头
        plans = [(classes_192, refiner_192), (classes_256, refiner_256)]
        if refiner_unified is not None:
            plans.append((classes_unified, refiner_unified))
        for classes, refiner in plans:
            if refiner is None:
                continue
            gt_boxes_px, gt_keys = [], []
            for key, mask in gt_items.items():
                if key[0] not in classes:
                    continue
                box = gt_boxes[key]
                if box is None:
                    continue
                pad = 2.0
                gt_boxes_px.append((
                    max(box[0] * width / RASTER - pad, 0.0),
                    max(box[1] * height / RASTER - pad, 0.0),
                    min(box[2] * width / RASTER + pad, width),
                    min(box[3] * height / RASTER + pad, height),
                ))
                gt_keys.append(key)
            if gt_boxes_px:
                analyzed = refiner.analyze(
                    image, [(key[0], 1.0, box)
                            for key, box in zip(gt_keys, gt_boxes_px)],
                    width, height, size[0], size[1])
                for item in analyzed:
                    key = (int(item["class_id"]), -1)
                    # 以实例框中心匹配最近 GT（同类且框 IoU 最大）
                    best = (0.0, None)
                    for gt_key, gt_box in zip(gt_keys, gt_boxes_px):
                        if gt_key[0] != item["class_id"]:
                            continue
                        value = bbox_iou(item.get("det_box", item["box"]), gt_box)
                        if value > best[0]:
                            best = (value, gt_key)
                    if best[1] is None:
                        continue
                    value = mask_iou(raster_segments(item["segments"]),
                                     gt_items[best[1]])
                    stats[item["class_id"]]["gt_box_refine_iou"] += value
                    if value > 0.5:
                        stats[item["class_id"]]["gt_box_refine_hit"] += 1
            yolo_boxes = []
            for class_id, confidence, mask in dets:
                if class_id not in classes:
                    continue
                box = bbox(mask)
                if box is None:
                    continue
                yolo_boxes.append((class_id, confidence, (
                    box[0] * width / RASTER, box[1] * height / RASTER,
                    box[2] * width / RASTER, box[3] * height / RASTER)))
            if yolo_boxes:
                for item in refiner.analyze(image, yolo_boxes, width, height,
                                            size[0], size[1]):
                    best = (0.0, None)
                    for gt_key, gt_box in zip(gt_keys, gt_boxes_px):
                        if gt_key[0] != item["class_id"]:
                            continue
                        value = bbox_iou(item.get("det_box", item["box"]), gt_box)
                        if value > best[0]:
                            best = (value, gt_key)
                    if best[1] is None:
                        continue
                    value = mask_iou(raster_segments(item["segments"]),
                                     gt_items[best[1]])
                    stats[item["class_id"]]["yolo_box_refine_iou"] += value
                    if value > 0.5:
                        stats[item["class_id"]]["yolo_box_refine_hit"] += 1
        if index % 20 == 0:
            print(index, flush=True)

    payload = {"images": len(pngs), "classes": {
        str(class_id): entry for class_id, entry in sorted(stats.items())}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'cls':>3} {'gt':>5} {'det':>5} {'boxTP':>5} {'maskTP':>6} "
          f"{'gtRef':>5} {'gtRefIoU':>8} {'yoloRef':>7} {'yoloIoU':>7}")
    for class_id, entry in payload["classes"].items():
        print(f"{class_id:>3} {entry['gt']:>5} {entry['det']:>5} "
              f"{entry['box_tp']:>5} {entry['mask_tp']:>6} "
              f"{entry['gt_box_refine_hit']:>5} {entry['gt_box_refine_iou']:>8.1f} "
              f"{entry['yolo_box_refine_hit']:>7} {entry['yolo_box_refine_iou']:>7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
