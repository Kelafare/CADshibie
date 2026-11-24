#!/usr/bin/env python3
"""构建原子级 atom-anchor 亲和力数据。

原子=GT 图元（推理时换成预测图元），锚点=同类 YOLO 框。
正样本=锚点与原子所属 GT 实例框 IoU>0.5。
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


def features(atom, anchor):
    points = atom["points"]
    box = atom["box"]
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
    direction = np.asarray(points[-1]) - np.asarray(points[0])
    norm = float(np.linalg.norm(direction))
    angle = abs(direction[0]) / max(norm, 1e-6) if norm else 0.0
    return {
        "inside": inside_ratio(points, anchor_box),
        "dx": (gx - cx) / aw, "dy": (gy - cy) / ah,
        "dist": float(np.hypot((gx - cx) / aw, (gy - cy) / ah)),
        "w_ratio": gw / aw, "h_ratio": gh / ah,
        "area_ratio": (gw * gh) / (aw * ah),
        "length_ratio": atom["length"] / max(np.hypot(aw, ah), 1e-6),
        "axis_align": angle,
        "bbox_iou": bbox_iou(box, anchor_box),
        "anchor_conf": float(anchor["conf"]),
        "atom_length": atom["length"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--yolo", default="weights/yolo_seg.pt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--max-anchors", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    paths = sorted(Path(args.gt_dir).glob("*.json"))[: args.limit]
    with out.open("w", encoding="utf-8") as handle:
        for index, gt_path in enumerate(paths):
            png = Path(args.png_dir) / f"{gt_path.stem}.png"
            if not png.exists():
                continue
            gt = load_json(gt_path)
            width, height = Image.open(png).size
            groups = defaultdict(list)
            atoms = []
            for args8, semantic, instance in zip(
                    gt["args"], gt["semanticIds"], gt["instanceIds"]):
                semantic = int(semantic)
                instance = int(instance)
                if semantic >= THING or instance < 0:
                    continue
                points = [(float(args8[k]) / 100.0 * width,
                           float(args8[k + 1]) / 100.0 * height)
                          for k in range(0, 8, 2)]
                length = float(sum(np.linalg.norm(
                    np.asarray(points[k + 1]) - np.asarray(points[k]))
                    for k in range(3)))
                atoms.append({
                    "class": semantic, "instance": instance,
                    "points": points, "box": bbox(points), "length": length,
                })
                groups[(semantic, instance)].append(points)
            result = model.predict(source=str(png), imgsz=640, conf=args.conf,
                                   iou=0.5, verbose=False, device=args.device)[0]
            anchors = defaultdict(list)
            if result.boxes is not None:
                for class_id, confidence, box in zip(
                        result.boxes.cls.cpu().numpy().astype(int),
                        result.boxes.conf.cpu().numpy(),
                        result.boxes.xyxy.cpu().numpy()):
                    anchors[int(class_id)].append({
                        "class": int(class_id), "conf": float(confidence),
                        "box": tuple(float(v) for v in box),
                    })
            for class_id in anchors:
                anchors[class_id].sort(key=lambda item: -item["conf"])
                anchors[class_id] = anchors[class_id][: args.max_anchors]
            group_boxes = {
                key: bbox([point for atom in points for point in atom])
                for key, points in groups.items()
            }
            for atom_index, atom in enumerate(atoms):
                for anchor_index, anchor in enumerate(anchors.get(atom["class"], [])):
                    if anchor_index >= args.max_anchors:
                        continue
                    row = features(atom, anchor)
                    gt_box = group_boxes[(atom["class"], atom["instance"])]
                    row["label"] = int(bbox_iou(gt_box, anchor["box"]) > 0.5)
                    row["stem"] = gt_path.stem
                    row["class"] = atom["class"]
                    row["atom_idx"] = atom_index
                    row["anchor_idx"] = anchor_index
                    handle.write(json.dumps(row) + "\n")
                    written += 1
            if index % 100 == 0:
                print(index, written, flush=True)
    print("done", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
