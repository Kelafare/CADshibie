#!/usr/bin/env python3
"""逐类评估 ROI 墨迹模型在独立验证集上的 Dice / mask IoU。"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cadvec.roi_ink import RoiInkUNet  # noqa: E402


def load_class(shard_dir: str, class_id: int, cap: int):
    images, masks = [], []
    for path in sorted(glob.glob(os.path.join(shard_dir, f"roi_ink_c{class_id}_p*.npz"))):
        data = np.load(path)
        images.append(data["images"])
        masks.append(data["masks"])
    if not images:
        return None, None
    return (np.concatenate(images)[:cap],
            np.concatenate(masks)[:cap].astype(np.float32) / 255.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--classes", default=",".join(str(i) for i in range(30)))
    parser.add_argument("--cap", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = RoiInkUNet(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    onehot = torch.eye(36, device=device)
    results = {}
    print(f"{'cls':>3} {'n':>5} {'dice':>7} {'iou':>7} {'thr':>5}")
    for class_id in [int(v) for v in args.classes.split(",") if v.strip()]:
        images, masks = load_class(args.shards, class_id, args.cap)
        if images is None:
            continue
        probability = []
        with torch.no_grad():
            for start in range(0, len(images), args.batch_size):
                x = torch.from_numpy(images[start:start + args.batch_size])
                x = x.to(device).permute(0, 3, 1, 2).float() / 255.0
                c = onehot[torch.full((len(x),), class_id, device=device)]
                logits = model(x, c)
                probability.append(torch.sigmoid(logits.float()).cpu())
        probability = torch.cat(probability)
        truth = torch.from_numpy(masks).unsqueeze(1)
        best = (0.0, 0.5, 0.0)
        for threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            prediction = (probability > threshold).float()
            intersection = float((prediction * truth).sum())
            pred_sum = float(prediction.sum())
            gt_sum = float(truth.sum())
            dice = 2 * intersection / max(pred_sum + gt_sum, 1e-6)
            iou = intersection / max(pred_sum + gt_sum - intersection, 1e-6)
            if dice > best[0]:
                best = (dice, threshold, iou)
        dice, threshold, iou = best
        results[str(class_id)] = {"n": len(images), "dice": dice,
                                  "iou": iou, "threshold": threshold}
        print(f"{class_id:>3} {len(images):>5} {dice:>7.4f} {iou:>7.4f} "
              f"{threshold:>5.2f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"weights": args.weights, "classes": results}, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
