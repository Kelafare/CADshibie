"""在 GPU 上训练 ROI 实例墨迹 U-Net。"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from cadvec.roi_ink import RoiInkUNet, dice_loss  # noqa: E402

SIZE = 192


def load_class(shard_dir: str, class_id: int, cap: int):
    images, masks = [], []
    for path in sorted(glob.glob(os.path.join(shard_dir, f"roi_ink_c{class_id}_p*.npz"))):
        data = np.load(path)
        images.append(data["images"])
        masks.append(data["masks"])
    if not images:
        return None, None
    images = np.concatenate(images, axis=0)[:cap]
    masks = np.concatenate(masks, axis=0)[:cap]
    return images, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", default="_work/roi_ink")
    parser.add_argument("--val-shards", default=None,
                        help="独立验证集 shard 目录；为空则按 90/10 随机切分")
    parser.add_argument("--val-cap", type=int, default=400,
                        help="每类验证样本上限")
    parser.add_argument("--neg-shards", default=None,
                        help="tools/mine_negatives.py 输出的负样本目录")
    parser.add_argument("--neg-cap", type=int, default=3000)
    parser.add_argument("--init", default=None,
                        help="从已有权重继续训练（fine-tune）")
    parser.add_argument("--out", default="_work/roi_ink.pt")
    parser.add_argument("--classes", default=",".join(str(i) for i in range(2, 30)))
    parser.add_argument("--cap", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=192)
    args = parser.parse_args()
    global SIZE
    SIZE = int(args.size)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "",
          flush=True)

    images, masks, labels = [], [], []
    for class_id in [int(v) for v in args.classes.split(",")]:
        x, y = load_class(args.shards, class_id, args.cap)
        if x is None:
            print("missing class", class_id, flush=True)
            continue
        images.append(x)
        masks.append(y)
        labels.append(np.full(len(x), class_id, dtype=np.int64))
        print("class", class_id, len(x), flush=True)
    images = np.concatenate(images)
    masks = np.concatenate(masks).astype(np.float32) / 255.0
    labels = np.concatenate(labels)
    order = np.arange(len(images))
    np.random.shuffle(order)
    images, masks, labels = images[order], masks[order], labels[order]
    if args.val_shards:
        val_images, val_masks, val_labels = [], [], []
        for class_id in [int(v) for v in args.classes.split(",")]:
            x, y = load_class(args.val_shards, class_id, args.val_cap)
            if x is None:
                continue
            val_images.append(x)
            val_masks.append(y)
            val_labels.append(np.full(len(x), class_id, dtype=np.int64))
        if val_images:
            val_x = np.concatenate(val_images)
            val_m = np.concatenate(val_masks).astype(np.float32) / 255.0
            val_c = np.concatenate(val_labels)
            train_x, train_m, train_c = images, masks, labels
        else:
            print("val shards 为空，回退到 90/10 切分", flush=True)
            split = int(len(images) * 0.9)
            train_x, train_m, train_c = images[:split], masks[:split], labels[:split]
            val_x, val_m, val_c = images[split:], masks[split:], labels[split:]
    else:
        split = int(len(images) * 0.9)
        train_x, train_m, train_c = images[:split], masks[:split], labels[:split]
        val_x, val_m, val_c = images[split:], masks[split:], labels[split:]
    print("train", len(train_x), "val", len(val_x), flush=True)

    if args.neg_shards:
        negative_x, negative_m, negative_c = [], [], []
        for path in sorted(glob.glob(os.path.join(args.neg_shards, "roi_neg_p*.npz"))):
            data = np.load(path)
            negative_x.append(data["images"])
            negative_m.append(data["masks"])
            negative_c.append(data["labels"])
        if negative_x:
            nx = np.concatenate(negative_x)
            nm = np.concatenate(negative_m).astype(np.float32) / 255.0
            nc = np.concatenate(negative_c)
            order = np.random.permutation(len(nx))[: args.neg_cap]
            train_x = np.concatenate([train_x, nx[order]])
            train_m = np.concatenate([train_m, nm[order]])
            train_c = np.concatenate([train_c, nc[order]])
            print("added negatives", len(order), flush=True)

    model = RoiInkUNet(base=32).to(device)
    if args.init:
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        print("initialized from", args.init, flush=True)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = math.ceil(len(train_x) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps,
        pct_start=0.15)
    pos_weight = torch.tensor(4.0, device=device)
    onehot_table = torch.eye(36, device=device)
    best = 0.0
    for epoch in range(args.epochs):
        model.train()
        started = time.perf_counter()
        permutation = np.random.permutation(len(train_x))
        running = 0.0
        for step in range(steps):
            batch = permutation[step * args.batch_size:(step + 1) * args.batch_size]
            if len(batch) < 2:
                continue
            x = torch.from_numpy(train_x[batch]).to(device).permute(0, 3, 1, 2).float() / 255.0
            y = torch.from_numpy(train_m[batch]).to(device).unsqueeze(1)
            c = onehot_table[torch.from_numpy(train_c[batch]).to(device)]
            if random.random() < 0.5:
                x = torch.flip(x, dims=[3])
                y = torch.flip(y, dims=[3])
            # 模拟检测框比 GT 框更紧/更偏的情形：随机放大并平移裁剪窗口。
            if random.random() < 0.5:
                for index in range(x.shape[0]):
                    if random.random() >= 0.5:
                        continue
                    factor = random.uniform(0.78, 1.0)
                    side = int(round(SIZE * factor))
                    top = random.randint(0, SIZE - side)
                    left = random.randint(0, SIZE - side)
                    xi = x[index:index + 1, :, top:top + side, left:left + side]
                    yi = y[index:index + 1, :, top:top + side, left:left + side]
                    x[index:index + 1] = F.interpolate(
                        xi, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
                    y[index:index + 1] = F.interpolate(
                        yi, size=(SIZE, SIZE), mode="nearest")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(x, c)
                loss = F.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=pos_weight) + dice_loss(logits.float(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss)
        # validation：概率图累积后扫描阈值，避免 pos_weight 造成的阈值偏移
        model.eval()
        probability, truth = [], []
        with torch.no_grad():
            for start in range(0, len(val_x), 128):
                x = torch.from_numpy(val_x[start:start + 128]).to(device)
                x = x.permute(0, 3, 1, 2).float() / 255.0
                y = torch.from_numpy(val_m[start:start + 128]).to(device).unsqueeze(1)
                c = onehot_table[torch.from_numpy(val_c[start:start + 128]).to(device)]
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == "cuda"):
                    logits = model(x, c)
                probability.append(torch.sigmoid(logits.float()).cpu())
                truth.append(y.cpu())
        probability = torch.cat(probability)
        truth = torch.cat(truth)
        gt_sum = float(truth.sum())
        best_local = (0.0, 0.5, 0.0)
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
            prediction = (probability > threshold).float()
            intersection = float((prediction * truth).sum())
            pred_sum = float(prediction.sum())
            dice_value = 2 * intersection / max(pred_sum + gt_sum, 1e-6)
            if dice_value > best_local[0]:
                best_local = (dice_value, threshold,
                              intersection / max(pred_sum + gt_sum - intersection, 1e-6))
        dice, threshold, iou = best_local
        elapsed = time.perf_counter() - started
        print(f"epoch {epoch + 1}: loss {running / max(steps,1):.4f} "
              f"val dice {dice:.4f} iou {iou:.4f} @t={threshold} ({elapsed:.0f}s)",
              flush=True)
        if dice > best:
            best = dice
            torch.save({"state_dict": model.state_dict(),
                        "model_kwargs": {"classes": 36, "base": 32, "class_dim": 64},
                        "crop_size": int(args.size),
                        "dice": dice, "iou": iou, "threshold": threshold,
                        "epoch": epoch + 1}, args.out)
            print("saved", args.out, flush=True)
    print("best dice", best)


if __name__ == "__main__":
    main()
