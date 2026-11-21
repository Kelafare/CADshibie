#!/usr/bin/env python3
"""在 GPU 上训练 ViT 对象分类器。

样本由 ``tools/build_candidates.py`` 生成：真值实例正样本 + 几何流水线
假设样本（按覆盖率打标签，低覆盖为忽略类 35）。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadvec.crops import CROP_SIZE, prepare_image, render_crop  # noqa: E402
from cadvec.vit import (  # noqa: E402
    CLASS_COUNT,
    StrokeViT,
    class_weights,
    parameter_count,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, IterableDataset
except ImportError:  # pragma: no cover
    print("需要 torch：pip install torch --index-url https://download.pytorch.org/whl/cu124",
          file=sys.stderr)
    raise


class StreamDataset(IterableDataset):
    """按图像流式产出样本：每张图只解码一次，随后连续渲染它的全部裁剪。

    随机打乱 ImageNet 式的逐样本读取会让 1000×1000 PNG 反复解码，成为训练
    瓶颈；流式读取既省内存也让 GPU 保持满载。
    """

    def __init__(self, samples, image_dir, training: bool = False):
        self.groups: dict[str, list] = {}
        for sample in samples:
            self.groups.setdefault(sample["name"], []).append(sample)
        self.image_dir = Path(image_dir)
        self.training = training
        self.epoch = 0

    def __iter__(self):
        names = list(self.groups)
        if self.training:
            random.shuffle(names)
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            names = names[worker.id::worker.num_workers]
        for name in names:
            image, scale_x, scale_y = prepare_image(self.image_dir / f"{name}.png")
            items = list(self.groups[name])
            if self.training:
                random.shuffle(items)
            for sample in items:
                flip = 0
                if self.training and random.random() < 0.5:
                    flip = 1 + random.randrange(2)
                tensor = render_crop(
                    image, tuple(sample["bbox"]), sample["strokes"],
                    scale_x, scale_y, size=CROP_SIZE, flip=flip,
                )
                yield (torch.from_numpy(tensor), int(sample["label"]),
                       float(sample.get("iou", 0.0)))


def _load_samples(directory: Path, image_dir: Path, limit_per_image: int = 40):
    paths = sorted(directory.glob("*.json"))
    samples = []
    for path in paths:
        if not (image_dir / f"{path.stem}.png").exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload.get("samples", [])
        gt = [item for item in items if item.get("kind") == "gt"]
        hyp = [item for item in items if item.get("kind") != "gt"]
        # 忽略类过多，做下采样，控制正负比。
        negatives = [item for item in hyp if item["label"] == 35]
        positives = [item for item in hyp if item["label"] != 35]
        negatives.sort(key=lambda item: -item["length"])
        keep_negatives = negatives[: max(limit_per_image - len(positives), 8)]
        chosen = gt + positives + keep_negatives
        for item in chosen:
            item["name"] = path.stem
        samples.extend(chosen)
    return samples


def evaluate(model, loader, device, ignore_index: int = 35):
    model.eval()
    correct = total = 0
    confusion = Counter()
    per_class_correct = Counter()
    per_class_total = Counter()
    with torch.no_grad():
        for images, labels, _iou in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(images)
            logits = output[0] if isinstance(output, tuple) else output
            predictions = logits.argmax(dim=1)
            correct += int((predictions == labels).sum())
            total += labels.numel()
            for target, prediction in zip(labels.tolist(), predictions.tolist()):
                per_class_total[target] += 1
                if target == prediction:
                    per_class_correct[target] += 1
                else:
                    confusion[(target, prediction)] += 1
    accuracy = correct / max(total, 1)
    recalls = {
        key: per_class_correct[key] / per_class_total[key]
        for key in per_class_total if key != ignore_index
    }
    macro = float(np.mean(list(recalls.values()))) if recalls else 0.0
    return accuracy, macro, per_class_total, per_class_correct, confusion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("../_work/candidates"))
    parser.add_argument("--image-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--out", type=Path, default=Path("../_work/vit_strokes.pt"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-image", type=int, default=36)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", type=Path, default=None,
                        help="从已有权重继续训练（可选）")
    parser.add_argument("--objectness", action="store_true",
                        help="增加 IoU/objectness 回归头（第二输出）")
    parser.add_argument("--iou-weight", type=float, default=0.5,
                        help="IoU 回归损失权重")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        print("GPU:", torch.cuda.get_device_name(0),
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB",
              file=sys.stderr)
    else:
        print("未检测到 CUDA，使用 CPU 训练（明显更慢）", file=sys.stderr)

    samples = _load_samples(args.candidates, args.image_dir, args.limit_per_image)
    if not samples:
        print("没有训练样本", file=sys.stderr)
        return 1
    names = sorted({sample["name"] for sample in samples})
    random.Random(args.seed).shuffle(names)
    split = max(1, int(len(names) * 0.88))
    train_names = set(names[:split])
    train_samples = [sample for sample in samples if sample["name"] in train_names]
    valid_samples = [sample for sample in samples if sample["name"] not in train_names]
    print(f"图像 {len(names)}（训练 {len(train_names)} / 验证 {len(names) - split}），"
          f"样本 {len(train_samples)}+{len(valid_samples)}", file=sys.stderr)
    labels = np.array([sample["label"] for sample in train_samples])
    weights = torch.from_numpy(class_weights(labels, power=0.5)).to(device)

    model = StrokeViT(dim=args.dim, depth=args.depth, heads=args.heads,
                      classes=CLASS_COUNT, image=CROP_SIZE,
                      objectness=args.objectness)
    if args.init is not None and args.init.exists():
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"从 {args.init} 继续训练（epoch {checkpoint.get('epoch')}）；"
              f"缺失键 {len(missing)}，多余键 {len(unexpected)}", file=sys.stderr)
    model = model.to(device)
    print(f"参数 {parameter_count(model):,}", file=sys.stderr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = math.ceil(len(train_samples) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, pct_start=0.15,
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)

    train_loader = DataLoader(
        StreamDataset(train_samples, args.image_dir, training=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None,
    )
    valid_loader = DataLoader(
        StreamDataset(valid_samples, args.image_dir, training=False),
        batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.workers // 2),
        pin_memory=device.type == "cuda",
    )

    best = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        for step, (images, target, iou_target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            iou_target = iou_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(images)
                if isinstance(output, tuple):
                    logits, iou_logit = output
                    loss = criterion(logits, target) + args.iou_weight * torch.nn.functional.binary_cross_entropy_with_logits(
                        iou_logit, iou_target.clamp(0.0, 1.0)
                    )
                else:
                    logits = output
                    loss = criterion(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler.last_epoch < scheduler.total_steps:
                scheduler.step()
            running += float(loss.detach()) * target.numel()
            seen += target.numel()
            if (step + 1) % 50 == 0:
                rate = (time.perf_counter() - started) / (step + 1)
                print(f"  epoch {epoch + 1} step {step + 1}/{steps_per_epoch} "
                      f"loss {running / max(seen, 1):.4f} {rate:.3f}s/step",
                      file=sys.stderr, flush=True)
        accuracy, macro, totals, correct, _ = evaluate(model, valid_loader, device)
        elapsed = time.perf_counter() - started
        entry = {
            "epoch": epoch + 1, "loss": running / max(seen, 1),
            "accuracy": accuracy, "macro_recall": macro, "seconds": round(elapsed, 1),
        }
        history.append(entry)
        print(f"epoch {epoch + 1}: loss {entry['loss']:.4f} acc {accuracy:.4f} "
              f"macro-recall {macro:.4f} ({elapsed:.0f}s)", file=sys.stderr)
        score = macro * 0.6 + accuracy * 0.4
        if score > best:
            best = score
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "model_kwargs": {
                    "dim": args.dim, "depth": args.depth, "heads": args.heads,
                    "classes": CLASS_COUNT, "image": CROP_SIZE,
                    "in_channels": 4, "patch": 16, "objectness": args.objectness,
                },
                "crop_size": CROP_SIZE,
                "epoch": epoch + 1,
                "metrics": entry,
                "history": history,
                "classes": list(range(CLASS_COUNT)),
            }, args.out)
    print(f"最佳权重已保存到 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
