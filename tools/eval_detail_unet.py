#!/usr/bin/env python3
"""比较粗概率与细节头在原分辨率下的矢量严格 IoU。"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.detail_unet import DetailUNet  # noqa: E402
from cadvec.roi_ink import vectorize_mask  # noqa: E402

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
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out", default="_work/detail_eval.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = DetailUNet(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    data = np.load(args.data)
    stems = np.asarray([str(v) for v in data["stems"]])
    unique = sorted(set(stems))
    val_stems = set(unique[-args.val_images:])
    stats = defaultdict(lambda: {"coarse": [], "detail": [], "hits": 0, "n": 0})

    def evaluate(index):
        left = int(data["lefts"][index]); top = int(data["tops"][index])
        crop_w = int(data["cws"][index]); crop_h = int(data["chs"][index])
        valid_w = int(data["vws"][index]); valid_h = int(data["vhs"][index])
        mask_canvas = data["masks"][index]
        truth_full = np.zeros((RASTER, RASTER), bool)
        target = mask_canvas[:valid_h, :valid_w]
        if (valid_w, valid_h) != (crop_w, crop_h):
            target = cv2.resize(target, (crop_w, crop_h),
                                interpolation=cv2.INTER_NEAREST)
        truth_full[top:top + crop_h, left:left + crop_w] = target > 0
        class_id = int(data["labels"][index])

        coarse = data["coarse"][index].astype(np.float32) / 255.0
        coarse = coarse[:valid_h, :valid_w]
        if (valid_w, valid_h) != (crop_w, crop_h):
            coarse = cv2.resize(coarse, (crop_w, crop_h),
                                interpolation=cv2.INTER_LINEAR)
        x = np.concatenate([data["images"][index],
                            data["coarse"][index][..., None]], axis=2)
        tensor = torch.from_numpy(x).to(device).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            probability = torch.sigmoid(model(tensor[None]))[0, 0].cpu().numpy()
        probability = probability[:valid_h, :valid_w]
        if (valid_w, valid_h) != (crop_w, crop_h):
            probability = cv2.resize(probability, (crop_w, crop_h),
                                     interpolation=cv2.INTER_LINEAR)
        coarse_mask = rasterize(vectorize_mask(
            coarse, left, top, crop_w, crop_h, 10.0, 10.0, 0.5))
        detail_mask = rasterize(vectorize_mask(
            probability, left, top, crop_w, crop_h, 10.0, 10.0, 0.5))
        entry = stats[class_id]
        entry["coarse"].append(iou(coarse_mask, truth_full))
        entry["detail"].append(iou(detail_mask, truth_full))
        entry["n"] += 1
        entry["hits"] += int(iou(detail_mask, truth_full) > 0.5)

    count = 0
    for index in range(len(stems)):
        if stems[index] in val_stems:
            evaluate(index)
            count += 1
            if count >= args.limit:
                break
    print(f"{'cls':>3} {'n':>5} {'coarse':>7} {'detail':>7} {'hit':>5}")
    all_coarse, all_detail = [], []
    for class_id, entry in sorted(stats.items()):
        all_coarse.extend(entry["coarse"])
        all_detail.extend(entry["detail"])
        print(f"{class_id:>3} {entry['n']:>5} "
              f"{np.mean(entry['coarse']):>7.3f} "
              f"{np.mean(entry['detail']):>7.3f} "
              f"{entry['hits']/max(entry['n'],1):>5.2f}")
    if all_coarse:
        print(f"all {len(all_coarse):>5} {np.mean(all_coarse):>7.3f} "
              f"{np.mean(all_detail):>7.3f} "
              f"{np.mean(np.asarray(all_detail) > 0.5):>5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
