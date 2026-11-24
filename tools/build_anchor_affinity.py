#!/usr/bin/env python3
"""构建 atom-group × YOLO anchor 的亲和力训练对。

训练组使用 GT 实例（SVG/JSON 图元），锚点使用 YOLO 同类检测框；
标签为组框与锚点框 IoU>0.5。推理时同一特征函数作用在预测实例组上。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import load_json  # noqa: E402

THING = 30


def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_iou(first, second):
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / max(area, 1e-9)


def inside_ratio(points, box, samples=8):
    x0, y0, x1, y1 = box
    hits = total = 0
    for index in range(len(points) - 1):
        ax, ay = points[index]
        bx, by = points[index + 1]
        for step in range(samples):
            ratio = step / samples
            x = ax + (bx - ax) * ratio
            y = ay + (by - ay) * ratio
            total += 1
            if x0 <= x <= x1 and y0 <= y <= y1:
                hits += 1
    return hits / max(total, 1)


def features(group, anchor):
    points = group["points"]
    box = group["box"]
    anchor_box = anchor["box"]
    ax0, ay0, ax1, ay1 = anchor_box
    aw = max(ax1 - ax0, 1e-6)
    ah = max(ay1 - ay0, 1e-6)
    gx = (box[0] + box[2]) / 2.0
    gy = (box[1] + box[3]) / 2.0
    cx = (ax0 + ax1) / 2.0
    cy = (ay0 + ay1) / 2.0
    gw = max(box[2] - box[0], 1e-6)
    gh = max(box[3] - box[1], 1e-6)
    dx = (gx - cx) / aw
    dy = (gy - cy) / ah
    ratios = [inside_ratio(atom, anchor_box) for atom in group["atoms"]]
    return {
        "inside_mean": float(np.mean(ratios)) if ratios else 0.0,
        "inside_max": float(np.max(ratios)) if ratios else 0.0,
        "dx": dx, "dy": dy,
        "dist": float(np.hypot(dx, dy)),
        "w_ratio": gw / aw, "h_ratio": gh / ah,
        "area_ratio": (gw * gh) / (aw * ah),
        "length_ratio": group["length"] / max(np.hypot(aw, ah), 1e-6),
        "n_atoms": float(len(group["atoms"])),
        "anchor_conf": float(anchor["conf"]),
        "bbox_iou": bbox_iou(box, anchor_box),
        "group_class": float(group["class"]),
        "anchor_class": float(anchor["class"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--yolo", default="weights/yolo_seg.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--per-image", type=int, default=0)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(args.gt_dir).glob("*.json"))[: args.limit]
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for index, gt_path in enumerate(paths):
            png = Path(args.png_dir) / f"{gt_path.stem}.png"
            if not png.exists():
                continue
            gt = load_json(gt_path)
            width, height = Image.open(png).size
            groups = defaultdict(list)
            for args8, semantic, instance in zip(
                    gt["args"], gt["semanticIds"], gt["instanceIds"]):
                semantic = int(semantic)
                instance = int(instance)
                if semantic >= THING or instance < 0:
                    continue
                points = [(float(args8[k]) / 100.0 * width,
                           float(args8[k + 1]) / 100.0 * height)
                          for k in range(0, 8, 2)]
                groups[(semantic, instance)].append(points)
            result = model.predict(source=str(png), imgsz=640, conf=args.conf,
                                   iou=0.5, verbose=False, device=args.device)[0]
            anchors = []
            if result.boxes is not None:
                for class_id, confidence, box in zip(
                        result.boxes.cls.cpu().numpy().astype(int),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.xyxy.cpu().numpy()):
                    anchors.append({
                        "class": int(class_id),
                        "conf": float(confidence),
                        "box": tuple(float(v) for v in box),
                    })
            rows = 0
            for (semantic, instance), atoms in groups.items():
                points = [point for atom in atoms for point in atom]
                bbox_value = bbox(points)
                length = sum(
                    np.linalg.norm(np.asarray(atom[index + 1]) -
                                   np.asarray(atom[index])).sum()
                    for atom in atoms for index in range(3))
                group = {
                    "class": semantic, "instance": instance,
                    "points": points, "atoms": atoms,
                    "box": bbox_value, "length": float(length),
                }
                for anchor_index, anchor in enumerate(anchors):
                    if anchor["class"] != semantic:
                        continue
                    row = features(group, anchor)
                    row["label"] = int(bbox_iou(bbox_value, anchor["box"]) > 0.5)
                    row["stem"] = gt_path.stem
                    row["class"] = semantic
                    row["anchor_idx"] = anchor_index
                    handle.write(json.dumps(row) + "\n")
                    rows += 1
                    written += 1
                    if args.per_image and rows >= args.per_image:
                        break
            if index % 100 == 0:
                print(index, written, flush=True)
    print("done", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
