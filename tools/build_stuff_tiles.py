#!/usr/bin/env python3
"""构建全图 Stuff 语义分割的 512 重叠分块数据。

GT 来自 JSON；30–34 每类按 instance=-1 合并成一个掩码。分块在原图
1000px 上直接裁切，不缩放，保留 1px 细线。只用于离线训练/诊断。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402
from cadvec.layers import load_rgba  # noqa: E402

CLASSES = tuple(range(30, 35))
RASTER = 1000


def starts(size: int, tile: int, stride: int) -> list[int]:
    values = list(range(0, max(size - tile, 0) + 1, stride))
    if not values or values[-1] != size - tile:
        values.append(size - tile)
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--stride", type=int, default=384)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--shard", type=int, default=150)
    args = parser.parse_args()

    paths = sorted(Path(args.gt_dir).glob("*.json"))
    selected = []
    for path in paths:
        data = load_json(path)
        if any(30 <= int(value) <= 34 for value in data["semanticIds"]):
            selected.append(path)
            if len(selected) >= args.limit:
                break
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"images with Stuff: {len(selected)}", flush=True)

    images, masks, stems, xs, ys = [], [], [], [], []
    part = 0
    written = 0

    def flush():
        nonlocal part, images, masks, stems, xs, ys
        if not images:
            return
        path = out / f"stuff_tiles_p{part:02d}.npz"
        np.savez_compressed(
            path, images=np.stack(images), masks=np.stack(masks),
            stems=np.asarray(stems), xs=np.asarray(xs), ys=np.asarray(ys))
        print("wrote", path, len(images), flush=True)
        part += 1
        images, masks, stems, xs, ys = [], [], [], [], []

    for index, gt_path in enumerate(selected):
        stem = gt_path.stem
        png = Path(args.png_dir) / f"{stem}.png"
        if not png.exists():
            continue
        gt = load_json(gt_path)
        rendered, _ = _segments(gt, (100, 100), RASTER)
        class_masks = np.stack([
            rendered.get((class_id, -1), np.zeros((RASTER, RASTER), bool))
            for class_id in CLASSES
        ]).astype(np.uint8)
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        if image.shape[:2] != (RASTER, RASTER):
            # 少数图纸为 999/1000 混排；统一到评分栅格，保证与 GT 对齐。
            image = cv2.resize(image, (RASTER, RASTER),
                               interpolation=cv2.INTER_AREA)
        for top in starts(RASTER, args.tile, args.stride):
            for left in starts(RASTER, args.tile, args.stride):
                crop = image[top:top + args.tile, left:left + args.tile]
                target = class_masks[:, top:top + args.tile,
                                     left:left + args.tile] * 255
                images.append(crop)
                masks.append(target)
                stems.append(stem)
                xs.append(left)
                ys.append(top)
                written += 1
                if len(images) >= args.shard:
                    flush()
        if index % 20 == 0:
            print(index, written, flush=True)
    flush()
    print("done", written)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
