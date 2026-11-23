#!/usr/bin/env python3
"""拼接 Stuff 分块预测，比较原始掩码与矢量输出的严格 IoU。"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402
from cadvec.roi_ink import vectorize_mask  # noqa: E402
from cadvec.stuff_unet import StuffUNet  # noqa: E402

CLASSES = tuple(range(30, 35))
RASTER = 1000


def rasterize(segments) -> np.ndarray:
    image = Image.new("1", (RASTER, RASTER), 0)
    draw = ImageDraw.Draw(image)
    for points, _length in segments:
        draw.line([(float(x) * RASTER / 100.0, float(y) * RASTER / 100.0)
                   for x, y in points], fill=1, width=1, joint="curve")
    return np.asarray(image, dtype=bool)


def iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    return intersection / int(np.count_nonzero(first | second))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="_work/stuff_eval.json")
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--vector-images", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = StuffUNet(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    tiles = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(args.shards, "stuff_tiles_p*.npz"))):
        data = np.load(path)
        for image, stem, x, y in zip(data["images"], data["stems"],
                                     data["xs"], data["ys"]):
            tiles[str(stem)].append((int(x), int(y), image))
    unique = sorted(tiles)
    val_stems = set(unique[-args.val_images:])
    vector_stems = set(unique[-args.vector_images:]) | set(unique[:10])
    stats = defaultdict(lambda: {"raw": [], "vec": [], "hit": 0,
                                 "vec_n": 0})
    for index, stem in enumerate(unique):
        accum = np.zeros((5, RASTER, RASTER), dtype=np.float32)
        counts = np.zeros((RASTER, RASTER), dtype=np.float32)
        batch, meta = [], []
        for x, y, image in tiles[stem]:
            batch.append(image)
            meta.append((x, y))
            if len(batch) == args.batch_size:
                tensor = torch.from_numpy(np.stack(batch)).to(device)
                tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
                with torch.no_grad():
                    probability = torch.sigmoid(model(tensor)).cpu().numpy()
                for (left, top), value in zip(meta, probability):
                    accum[:, top:top + value.shape[1],
                          left:left + value.shape[2]] += value
                    counts[top:top + value.shape[1],
                           left:left + value.shape[2]] += 1
                batch, meta = [], []
        if batch:
            tensor = torch.from_numpy(np.stack(batch)).to(device)
            tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
            with torch.no_grad():
                probability = torch.sigmoid(model(tensor)).cpu().numpy()
            for (left, top), value in zip(meta, probability):
                accum[:, top:top + value.shape[1],
                      left:left + value.shape[2]] += value
                counts[top:top + value.shape[1],
                       left:left + value.shape[2]] += 1
        accum /= np.maximum(counts, 1.0)[None]
        gt_path = Path(args.gt_dir) / f"{stem}.json"
        if not gt_path.exists():
            continue
        gt = load_json(gt_path)
        rendered, _ = _segments(gt, (100, 100), RASTER)
        split = "val" if stem in val_stems else "train"
        for class_index, class_id in enumerate(CLASSES):
            truth = rendered.get((class_id, -1))
            if truth is None or not truth.any():
                continue
            raw = iou(accum[class_index] > 0.5, truth)
            stats[class_id]["raw"].append(raw)
            if stem in vector_stems:
                best = 0.0
                for threshold in (0.3, 0.4, 0.5, 0.6):
                    segments = vectorize_mask(
                        accum[class_index], 0, 0, RASTER, RASTER,
                        10.0, 10.0, threshold)
                    best = max(best, iou(rasterize(segments), truth))
                stats[class_id]["vec"].append(best)
                stats[class_id]["vec_n"] += 1
                stats[class_id]["hit"] += int(best > 0.5)
        if index % 10 == 0:
            print(index, flush=True)

    payload = {"images": len(unique), "classes": {}}
    print(f"{'cls':>3} {'nRaw':>5} {'rawIoU':>7} {'nVec':>5} {'vecIoU':>7} {'hit':>5}")
    for class_id in CLASSES:
        entry = stats[class_id]
        if not entry["raw"]:
            continue
        record = {
            "n_raw": len(entry["raw"]),
            "raw_iou": float(np.mean(entry["raw"])),
            "n_vec": entry["vec_n"],
            "vec_iou": float(np.mean(entry["vec"])) if entry["vec"] else 0.0,
            "vector_hit_rate": entry["hit"] / max(entry["vec_n"], 1),
        }
        payload["classes"][str(class_id)] = record
        print(f"{class_id:>3} {record['n_raw']:>5} {record['raw_iou']:>7.3f} "
              f"{record['n_vec']:>5} {record['vec_iou']:>7.3f} "
              f"{record['vector_hit_rate']:>5.2f}")
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
