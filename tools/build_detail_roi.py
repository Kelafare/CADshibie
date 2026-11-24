#!/usr/bin/env python3
"""构建原分辨率细节头数据：RGB + 512 粗概率 → GT native 墨迹。

只使用 train_final 的 GT 框与 GT 墨迹；粗概率来自 512 ROI 模型。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402
from cadvec.layers import load_rgba  # noqa: E402
from cadvec.roi_ink import RoiInkRefiner  # noqa: E402

SIZE = 384
RASTER = 1000


def main() -> int:
    global SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    SIZE = int(args.size)
    refiner = RoiInkRefiner.load(args.weights, device=args.device)
    if refiner is None:
        raise SystemExit("无法加载 512 ROI 权重")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images, coarse, masks = [], [], []
    stems, labels, lefts, tops, cws, chs, vws, vhs = [], [], [], [], [], [], [], []
    paths = sorted(Path(args.gt_dir).glob("*.json"))[: args.limit]
    for index, gt_path in enumerate(paths):
        png = Path(args.png_dir) / f"{gt_path.stem}.png"
        if not png.exists():
            continue
        gt = load_json(gt_path)
        rendered, _ = _segments(gt, (100, 100), RASTER)
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        if image.shape[:2] != (RASTER, RASTER):
            image = cv2.resize(image, (RASTER, RASTER),
                               interpolation=cv2.INTER_AREA)
        per_class = defaultdict(int)
        for (class_id, instance), truth in rendered.items():
            if class_id >= 30 or instance < 0 or not truth.any():
                continue
            if per_class[class_id] >= args.per_class:
                continue
            per_class[class_id] += 1
            ys, xs = np.nonzero(truth)
            box = (float(xs.min()), float(ys.min()),
                   float(xs.max()), float(ys.max()))
            analyzed = refiner.analyze(
                image, [(class_id, 1.0, box)], RASTER, RASTER, 100, 100)
            if not analyzed:
                continue
            item = analyzed[0]
            left, top, right, bottom = item["box"]
            probability = item["probability"]
            crop_w, crop_h = right - left, bottom - top
            rgb_crop = image[top:bottom, left:right]
            target = truth[top:bottom, left:right].astype(np.uint8) * 255
            valid_w, valid_h = crop_w, crop_h
            if crop_w > SIZE or crop_h > SIZE:
                ratio = min(SIZE / crop_w, SIZE / crop_h)
                valid_w = max(1, int(round(crop_w * ratio)))
                valid_h = max(1, int(round(crop_h * ratio)))
                rgb_crop = cv2.resize(rgb_crop, (valid_w, valid_h),
                                      interpolation=cv2.INTER_AREA)
                target = cv2.resize(target, (valid_w, valid_h),
                                    interpolation=cv2.INTER_NEAREST)
                probability = cv2.resize(probability, (valid_w, valid_h),
                                         interpolation=cv2.INTER_LINEAR)
            rgb_canvas = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
            rgb_canvas[:valid_h, :valid_w] = rgb_crop
            coarse_canvas = np.zeros((SIZE, SIZE), dtype=np.uint8)
            coarse_canvas[:valid_h, :valid_w] = np.clip(
                probability * 255.0, 0, 255).astype(np.uint8)
            mask_canvas = np.zeros((SIZE, SIZE), dtype=np.uint8)
            mask_canvas[:valid_h, :valid_w] = target
            images.append(rgb_canvas)
            coarse.append(coarse_canvas)
            masks.append(mask_canvas)
            stems.append(gt_path.stem)
            labels.append(class_id)
            lefts.append(left)
            tops.append(top)
            cws.append(crop_w)
            chs.append(crop_h)
            vws.append(valid_w)
            vhs.append(valid_h)
        if index % 20 == 0:
            print(index, len(images), flush=True)
    np.savez_compressed(
        out / "detail_roi.npz",
        images=np.stack(images), coarse=np.stack(coarse),
        masks=np.stack(masks), stems=np.asarray(stems),
        labels=np.asarray(labels), lefts=np.asarray(lefts),
        tops=np.asarray(tops), cws=np.asarray(cws), chs=np.asarray(chs),
        vws=np.asarray(vws), vhs=np.asarray(vhs))
    print("done", len(images), "->", out / "detail_roi.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
