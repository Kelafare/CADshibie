"""构建 ROI 实例墨迹数据集：GT 实例框裁剪 + 真值笔画栅格化目标。

输出按类别的 npz：images (N,192,192,3) uint8，masks (N,192,192) uint8。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cadvec.layers import load_rgba  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

SIZE = 192
PAD = 0.30
CLASSES = tuple(range(2, 30))


def _init_worker(size: int, classes) -> None:
    global SIZE, CLASSES
    SIZE = int(size)
    CLASSES = tuple(classes)


def render_instances(primitives, raster=1000):
    """与评测器一致的 PIL 栅格化（1px 线、joint=curve、椭圆按四点外接框）。"""
    scale = raster / 100.0
    masks = {}
    for primitive in primitives:
        semantic = int(primitive.semantic_id)
        if semantic not in CLASSES or semantic == 35:
            continue
        instance = int(primitive.instance_id)
        if semantic < 30 and instance < 0:
            continue
        key = (semantic, instance)
        image = masks.get(key)
        if image is None:
            image = Image.new("1", (raster, raster), 0)
            masks[key] = image
        draw = ImageDraw.Draw(image)
        points = [(float(x) * scale, float(y) * scale) for x, y in primitive.points]
        if int(primitive.command) in (0, 1):
            draw.line(points, fill=1, width=1, joint="curve")
        else:
            xs, ys = zip(*points)
            draw.ellipse((min(xs), min(ys), max(xs), max(ys)), outline=1, width=1)
    return {key: np.asarray(image, dtype=np.uint8) for key, image in masks.items()}


def crop_instance(image, mask, box):
    height, width = image.shape[:2]
    x0, y0, x1, y1 = box
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = side * (0.5 + PAD)
    left = int(round(max(cx - half, 0)))
    top = int(round(max(cy - half, 0)))
    right = int(round(min(cx + half, width)))
    bottom = int(round(min(cy + half, height)))
    if right - left < 4 or bottom - top < 4:
        return None
    crop = image[top:bottom, left:right]
    target = mask[top:bottom, left:right]
    # gt_segment_masks 返回 0/1；先放大到 0/255，再缩放，避免细线被 AREA 取整成 0。
    target = (target.astype(np.uint8) * 255)
    crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    target = cv2.resize(target, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return crop, target


def sample_background(image, rng, attempts=24):
    """采样几乎空白的裁剪作为负样本（目标全 0）。"""
    height, width = image.shape[:2]
    for _ in range(attempts):
        side = rng.uniform(0.04, 0.18) * min(height, width)
        side = max(int(round(side)), 16)
        if side >= height or side >= width:
            continue
        top = rng.randint(0, height - side)
        left = rng.randint(0, width - side)
        crop = image[top:top + side, left:left + side]
        if float((crop.min(axis=2) < 245).mean()) > 0.01:
            continue
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        return crop, np.zeros((SIZE, SIZE), dtype=np.uint8)
    return None


def process(args):
    png, svg, negatives = args
    stem = os.path.splitext(os.path.basename(png))[0]
    try:
        primitives = list(iter_primitives(svg))
        masks = render_instances(primitives)
    except Exception:
        return stem, []
    rgb, alpha = load_rgba(png)
    image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                    0, 255).astype(np.uint8)
    out = []
    for (semantic, instance), mask in masks.items():
        if semantic not in CLASSES:
            continue
        ys, xs = np.nonzero(mask)
        if len(xs) < 8:
            continue
        box = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
        pair = crop_instance(image, mask, box)
        if pair is None:
            continue
        crop, target = pair
        target = cv2.dilate(target, np.ones((3, 3), np.uint8), iterations=1)
        out.append((semantic, crop, target))
    if negatives:
        rng = random.Random(stem)
        valid = [class_id for class_id in CLASSES if class_id < 30]
        for _ in range(negatives):
            pair = sample_background(image, rng)
            if pair is None:
                break
            class_id = rng.choice(valid) if valid else 0
            out.append((class_id, pair[0], pair[1]))
    return stem, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--svg-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--size", type=int, default=192)
    parser.add_argument("--classes", default="",
                        help="逗号分隔的类别；空串表示 2..29 全部")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--split-file", default=None,
                        help="tools/build_splits.py 生成的 splits.json")
    parser.add_argument("--split-role", default="train",
                        help="只处理 train_final 中该角色的图片")
    parser.add_argument("--negatives", type=int, default=0,
                        help="每张训练图附加的空白负样本数（val 不加）")
    args = parser.parse_args()
    if args.classes.strip():
        globals()["CLASSES"] = tuple(
            int(v) for v in args.classes.split(",") if v.strip())
    os.makedirs(args.out, exist_ok=True)
    pngs = sorted(f for f in os.listdir(args.png_dir) if f.endswith(".png"))
    keep = None
    if args.split_file:
        split = json.loads(open(args.split_file, encoding="utf-8").read())
        keep = set(split.get("train_final", {}).get(args.split_role, []))
        if not keep:
            raise SystemExit(f"split role {args.split_role!r} 为空")
        pngs = [name for name in pngs if os.path.splitext(name)[0] in keep]
        print(f"split {args.split_role}: {len(pngs)} 张", flush=True)
    if args.limit:
        pngs = pngs[: args.limit]
    negatives = args.negatives if args.split_role == "train" else 0
    jobs = [(os.path.join(args.png_dir, p), os.path.join(args.svg_dir, p[:-4] + ".svg"),
             negatives)
            for p in pngs
            if os.path.exists(os.path.join(args.svg_dir, p[:-4] + ".svg"))]
    shard = 1500
    buckets = {c: {"images": [], "masks": [], "part": 0} for c in CLASSES}

    def flush(semantic, force=False):
        bucket = buckets[semantic]
        if not bucket["images"] or (len(bucket["images"]) < shard and not force):
            return
        path = os.path.join(
            args.out, f"roi_ink_c{semantic}_p{bucket['part']:02d}.npz")
        np.savez_compressed(
            path,
            images=np.stack(bucket["images"]),
            masks=np.stack(bucket["masks"]),
        )
        print(semantic, len(bucket["images"]), path, flush=True)
        bucket["images"] = []
        bucket["masks"] = []
        bucket["part"] += 1

    started = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(args.size, CLASSES)) as pool:
        for index, (_stem, items) in enumerate(pool.imap_unordered(process, jobs, chunksize=8)):
            for semantic, crop, target in items:
                buckets[semantic]["images"].append(crop)
                buckets[semantic]["masks"].append(target)
            for semantic in {item[0] for item in items}:
                flush(semantic)
            if index % 200 == 0:
                print(index, round(time.perf_counter() - started, 1), flush=True)
    for semantic in CLASSES:
        flush(semantic, force=True)
    print("done", round(time.perf_counter() - started, 1))


if __name__ == "__main__":
    main()
