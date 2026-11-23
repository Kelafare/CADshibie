#!/usr/bin/env python3
"""挖掘 YOLO 的真实难负例：与任何 GT 实例框都不重叠的检测框。

这些框通常落在文字、尺寸线、表格线或空白处；条件化类别后目标墨迹为全 0，
用于训练 ROI 墨迹头抑制整框误检。只使用 train_final 的合法标注。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.layers import load_rgba  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402

SIZE = 256
PAD = 0.30


def box_iou(first, second):
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / max(area, 1e-9)


def crop(image, box):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = side * (0.5 + PAD)
    left = int(round(max(cx - half, 0)))
    top = int(round(max(cy - half, 0)))
    right = int(round(min(cx + half, width)))
    bottom = int(round(min(cy + half, height)))
    if right - left < 8 or bottom - top < 8:
        return None
    return cv2.resize(image[top:bottom, left:right], (SIZE, SIZE),
                      interpolation=cv2.INTER_AREA)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--svg-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--split-role", default="train")
    parser.add_argument("--yolo", default="weights/yolo_seg.pt")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-image", type=int, default=2)
    parser.add_argument("--class-cap", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    pngs = sorted(Path(args.png_dir).glob("*.png"))
    if args.split_file:
        split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
        keep = set(split.get("train_final", {}).get(args.split_role, []))
        if not keep:
            raise SystemExit(f"split role {args.split_role!r} 为空")
        pngs = [path for path in pngs if path.stem in keep]
    if args.limit:
        pngs = pngs[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    counts = defaultdict(int)
    bucket_images, bucket_masks, bucket_labels = [], [], []
    part = 0
    written = 0

    def flush():
        nonlocal part, bucket_images, bucket_masks, bucket_labels
        if not bucket_images:
            return
        path = out / f"roi_neg_p{part:02d}.npz"
        np.savez_compressed(path, images=np.stack(bucket_images),
                            masks=np.stack(bucket_masks),
                            labels=np.asarray(bucket_labels, dtype=np.int64))
        print("wrote", path, len(bucket_images), flush=True)
        part += 1
        bucket_images, bucket_masks, bucket_labels = [], [], []

    for index, png in enumerate(pngs):
        svg = Path(args.svg_dir) / f"{png.stem}.svg"
        if not svg.exists():
            continue
        try:
            primitives = list(iter_primitives(svg))
        except Exception:  # noqa: BLE001
            continue
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        height, width = image.shape[:2]
        gt_boxes = defaultdict(list)
        for primitive in primitives:
            sem = int(primitive.semantic_id)
            if sem >= 30 or int(primitive.instance_id) < 0:
                continue
            xs = [point[0] for point in primitive.points]
            ys = [point[1] for point in primitive.points]
            gt_boxes[sem].append((
                min(xs) * width / 100.0, min(ys) * height / 100.0,
                max(xs) * width / 100.0, max(ys) * height / 100.0))
        result = model.predict(source=str(png), imgsz=640, conf=args.conf,
                               iou=0.5, verbose=False, device=args.device)[0]
        picked = 0
        if result.boxes is not None and len(result.boxes):
            for class_id, confidence, coordinates in zip(
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.xyxy.cpu().numpy()):
                if picked >= args.per_image or counts[int(class_id)] >= args.class_cap:
                    continue
                box = tuple(float(v) for v in coordinates)
                if any(box_iou(box, other) >= 0.1
                       for boxes in gt_boxes.values() for other in boxes):
                    continue
                patch = crop(image, box)
                if patch is None:
                    continue
                bucket_images.append(patch)
                bucket_masks.append(np.zeros((SIZE, SIZE), dtype=np.uint8))
                bucket_labels.append(int(class_id))
                counts[int(class_id)] += 1
                picked += 1
                written += 1
        if len(bucket_images) >= 1000:
            flush()
        if index % 500 == 0:
            print(index, written, flush=True)
    flush()
    (out / "counts.json").write_text(json.dumps(dict(counts), indent=1),
                                     encoding="utf-8")
    print("done", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
