#!/usr/bin/env python3
"""训练全图 Stuff 语义 U-Net（30–34），按图纸划分训练/留出。"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cadvec.stuff_unet import StuffUNet, multiclass_dice  # noqa: E402


def load(shard_dir: str):
    images, masks, stems = [], [], []
    for path in sorted(glob.glob(os.path.join(shard_dir, "stuff_tiles_p*.npz"))):
        data = np.load(path)
        images.append(data["images"])
        masks.append(data["masks"])
        stems.append(data["stems"])
    return (np.concatenate(images), np.concatenate(masks),
            np.concatenate(stems))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", required=True)
    parser.add_argument("--out", default="_work/stuff_unet.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", default="all",
                        help="all 或逗号分隔的 GPU 编号，例如 0,1")
    args = parser.parse_args()
    random = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)

    images, masks, stems = load(args.shards)
    unique = sorted({str(stem) for stem in stems})
    val_stems = set(unique[-args.val_images:])
    is_val = np.asarray([str(stem) in val_stems for stem in stems])
    train_x = images[~is_val].astype(np.float32) / 255.0
    train_y = masks[~is_val].astype(np.float32) / 255.0
    val_x = images[is_val].astype(np.float32) / 255.0
    val_y = masks[is_val].astype(np.float32) / 255.0
    print("tiles train", len(train_x), "val", len(val_x),
          "images", len(unique) - len(val_stems), "/", len(val_stems),
          flush=True)

    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        if args.gpus.strip().lower() == "all":
            device_ids = list(range(available))
        else:
            device_ids = [int(value) for value in args.gpus.split(",")
                          if value.strip()]
            device_ids = [value for value in device_ids if value < available]
        device = torch.device(f"cuda:{device_ids[0]}")
    else:
        device_ids = []
    model = StuffUNet(classes=5, base=32).to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
        print("DataParallel GPUs", device_ids, flush=True)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    steps = math.ceil(len(train_x) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps,
        pct_start=0.15)
    # Stuff 前景占比约 0.003%–0.5%，pos_weight 过小会退化成近似全零。
    pos_weight = torch.tensor(50.0, device=device)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        started = time.perf_counter()
        order = random.permutation(len(train_x))
        running = 0.0
        for step in range(steps):
            batch = order[step * args.batch_size:(step + 1) * args.batch_size]
            if len(batch) < 2:
                continue
            x = torch.from_numpy(train_x[batch]).to(device).permute(0, 3, 1, 2)
            y = torch.from_numpy(train_y[batch]).to(device)
            if random.rand() < 0.5:
                x = torch.flip(x, dims=[3])
                y = torch.flip(y, dims=[3])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(
                logits, y, pos_weight=pos_weight) + multiclass_dice(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss)
        model.eval()
        inter = np.zeros(5, dtype=np.float64)
        union = np.zeros(5, dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(val_x), 16):
                x = torch.from_numpy(val_x[start:start + 16]).to(device)
                x = x.permute(0, 3, 1, 2)
                y = torch.from_numpy(val_y[start:start + 16]).to(device)
                probability = torch.sigmoid(model(x))
                for class_id in range(5):
                    pred = probability[:, class_id] > 0.5
                    truth = y[:, class_id] > 0.5
                    inter[class_id] += float((pred & truth).sum())
                    union[class_id] += float((pred | truth).sum())
        per_class = (inter / np.maximum(union, 1.0)).tolist()
        mean_iou = float(np.mean(per_class))
        elapsed = time.perf_counter() - started
        print(f"epoch {epoch + 1}: loss {running / max(steps,1):.4f} "
              f"val mIoU {mean_iou:.4f} per-class "
              + " ".join(f"{value:.3f}" for value in per_class)
              + f" ({elapsed:.0f}s)", flush=True)
        if mean_iou > best:
            best = mean_iou
            state = (model.module.state_dict()
                     if isinstance(model, nn.DataParallel)
                     else model.state_dict())
            torch.save({"state_dict": state,
                        "model_kwargs": {"classes": 5, "base": 32},
                        "classes": list(range(30, 35)),
                        "miou": mean_iou}, args.out)
            print("saved", args.out, flush=True)
    print("best mIoU", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
