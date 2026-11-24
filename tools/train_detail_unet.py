#!/usr/bin/env python3
"""训练原分辨率 ROI 细节头（RGB + 512 粗概率 → native 墨迹）。"""
from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cadvec.detail_unet import DetailUNet  # noqa: E402
from cadvec.roi_ink import dice_loss  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="_work/detail_unet.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data)
    images, coarse, masks = data["images"], data["coarse"], data["masks"]
    stems = np.asarray([str(v) for v in data["stems"]])
    unique = sorted(set(stems))
    val_stems = set(unique[-args.val_images:])
    is_val = np.asarray([stem in val_stems for stem in stems])
    x = np.concatenate([images, coarse[..., None]], axis=3).astype(np.float32) / 255.0
    y = masks.astype(np.float32)[..., None] / 255.0
    train_x, train_y = x[~is_val], y[~is_val]
    val_x, val_y = x[is_val], y[is_val]
    print("device", device, "train", len(train_x), "val", len(val_x), flush=True)

    model = DetailUNet(base=24).to(device)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.01)
    steps = math.ceil(len(train_x) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps,
        pct_start=0.15)
    pos_weight = torch.tensor(10.0, device=device)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        started = time.perf_counter()
        order = rng.permutation(len(train_x))
        running = 0.0
        for step in range(steps):
            batch = order[step * args.batch_size:(step + 1) * args.batch_size]
            if len(batch) < 2:
                continue
            bx = torch.from_numpy(train_x[batch]).to(device).permute(0, 3, 1, 2)
            by = torch.from_numpy(train_y[batch]).to(device).permute(0, 3, 1, 2)
            if rng.rand() < 0.5:
                bx = torch.flip(bx, dims=[3])
                by = torch.flip(by, dims=[3])
            optimizer.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.binary_cross_entropy_with_logits(
                logits, by, pos_weight=pos_weight) + dice_loss(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss)
        model.eval()
        inter = union = 0.0
        with torch.no_grad():
            for start in range(0, len(val_x), 16):
                bx = torch.from_numpy(val_x[start:start + 16]).to(device)
                bx = bx.permute(0, 3, 1, 2)
                by = torch.from_numpy(val_y[start:start + 16]).to(device)
                by = by.permute(0, 3, 1, 2)
                pred = torch.sigmoid(model(bx)) > 0.5
                inter += float((pred & (by > 0.5)).sum())
                union += float((pred | (by > 0.5)).sum())
        iou = inter / max(union, 1.0)
        print(f"epoch {epoch + 1}: loss {running / max(steps,1):.4f} "
              f"val IoU {iou:.4f} ({time.perf_counter() - started:.0f}s)",
              flush=True)
        if iou > best:
            best = iou
            torch.save({"state_dict": model.state_dict(),
                        "model_kwargs": {"base": 24},
                        "size": int(images.shape[1]),
                        "iou": iou}, args.out)
            print("saved", args.out, flush=True)
    print("best IoU", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
