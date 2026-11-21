#!/usr/bin/env python3
"""训练 dense segmentation 的 U-Net（512×512 裁剪、正样本偏置采样）。"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from cadvec.segmodel import BACKGROUND_INDEX, NUM_CLASSES, UNet, parameter_count  # noqa: E402

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, IterableDataset
except ImportError:  # pragma: no cover
    print("需要 torch", file=sys.stderr)
    raise

NUM_CLASSES = NUM_CLASSES  # 0-35 + 背景 36


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    composite = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


class SegDataset(IterableDataset):
    def __init__(self, names: list[str], image_dir: Path, label_dir: Path,
                 crop: int = 512, training: bool = True, positive_bias: float = 0.7):
        self.names = names
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.crop = crop
        self.training = training
        self.positive_bias = positive_bias

    def __iter__(self):
        names = list(self.names)
        if self.training:
            random.shuffle(names)
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            names = names[worker.id::worker.num_workers]
        for name in names:
            image = load_image(self.image_dir / f"{name}.png")
            label = np.asarray(Image.open(self.label_dir / f"{name}.png"), dtype=np.uint8)
            height = min(image.shape[0], label.shape[0])
            width = min(image.shape[1], label.shape[1])
            image = image[:height, :width]
            label = label[:height, :width]
            crop = self.crop
            if random.random() < self.positive_bias:
                rows, columns = np.nonzero(label != 255)
                if len(rows):
                    index = random.randrange(len(rows))
                    center_y, center_x = int(rows[index]), int(columns[index])
                    y0 = min(max(center_y - crop // 2, 0), height - crop)
                    x0 = min(max(center_x - crop // 2, 0), width - crop)
                else:
                    y0 = random.randint(0, height - crop)
                    x0 = random.randint(0, width - crop)
            else:
                y0 = random.randint(0, height - crop) if height > crop else 0
                x0 = random.randint(0, width - crop) if width > crop else 0
            image_crop = image[y0:y0 + crop, x0:x0 + crop]
            label_crop = label[y0:y0 + crop, x0:x0 + crop].astype(np.int64)
            label_crop[label_crop == 255] = BACKGROUND_INDEX
            if self.training:
                if random.random() < 0.5:
                    image_crop = image_crop[:, ::-1]
                    label_crop = label_crop[:, ::-1]
                if random.random() < 0.5:
                    image_crop = image_crop[::-1]
                    label_crop = label_crop[::-1]
            tensor = np.ascontiguousarray(
                (image_crop.astype(np.float32) / 255.0 - 0.5).transpose(2, 0, 1)
            )
            yield tensor, np.ascontiguousarray(label_crop)


def collate(batch):
    images = torch.tensor(np.stack([item[0] for item in batch]), dtype=torch.float32)
    labels = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.int64)
    return images, labels


def class_weights(label_dir: Path, names: list[str], sample: int = 200) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for name in names[:sample]:
        label = np.asarray(Image.open(label_dir / f"{name}.png"), dtype=np.uint8)
        label = label.copy()
        label[label == 255] = BACKGROUND_INDEX
        counts += np.bincount(label.ravel(), minlength=NUM_CLASSES)
    frequencies = counts / max(counts.sum(), 1.0)
    # 稀有类权重封顶，避免个别 1-2 像素的类把均值拉爆、把常见前景压成 0。
    weights = np.power(1.0 / np.maximum(frequencies, 1e-5), 0.5)
    weights = np.clip(weights, 0.5, 40.0)
    # 绝对值而非按均值归一：背景像素占 98%，必须显式压低它的单像素权重，
    # 否则模型只要全预测背景就能把损失压到接近 0。
    weights[BACKGROUND_INDEX] = 0.03
    weights[35] = 0.5
    return torch.from_numpy(weights.astype(np.float32))


def dice_loss(logits, labels) -> torch.Tensor:
    """前景类（0–35）的 soft Dice，直接优化细线的重叠。"""
    probabilities = torch.softmax(logits.float(), dim=1)
    valid = labels != BACKGROUND_INDEX
    total = 0.0
    count = 0
    for class_id in range(NUM_CLASSES - 1):
        target = (labels == class_id) & valid
        if not target.any():
            continue
        prediction = probabilities[:, class_id] * valid
        intersection = (prediction * target).sum()
        denominator = prediction.sum() + target.sum()
        total = total + (2.0 * intersection + 1.0) / (denominator + 1.0)
        count += 1
    if count == 0:
        return logits.sum() * 0.0
    return 1.0 - total / count


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    intersections = np.zeros(NUM_CLASSES, dtype=np.float64)
    unions = np.zeros(NUM_CLASSES, dtype=np.float64)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(images)
            predictions = logits.float().argmax(dim=1)
            stroke = labels != BACKGROUND_INDEX
            correct += int(((predictions == labels) & stroke).sum())
            total += int(stroke.sum())
            for class_id in range(NUM_CLASSES):
                predicted = predictions == class_id
                actual = labels == class_id
                intersections[class_id] += float((predicted & actual).sum())
                unions[class_id] += float((predicted | actual).sum())
    accuracy = correct / max(total, 1)
    present = unions > 0
    iou = intersections[present] / np.maximum(unions[present], 1.0)
    classes = np.arange(NUM_CLASSES)[present]
    foreground = iou[(classes != BACKGROUND_INDEX) & (classes != 35)]
    foreground_miou = float(foreground.mean()) if len(foreground) else 0.0
    return accuracy, float(iou.mean()) if len(iou) else 0.0, foreground_miou


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--label-dir", type=Path, default=Path("../_work/seg_labels"))
    parser.add_argument("--out", type=Path, default=Path("weights/seg_unet.pt"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop", type=int, default=512)
    parser.add_argument("--base", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = sorted(path.stem for path in args.label_dir.glob("*.png"))
    if args.limit:
        names = names[: args.limit]
    random.Random(args.seed).shuffle(names)
    split = max(1, int(len(names) * 0.95))
    train_names, valid_names = names[:split], names[split:]
    print(f"图像 {len(names)}（训练 {len(train_names)} / 验证 {len(valid_names)}）",
          file=sys.stderr)

    weights = class_weights(args.label_dir, train_names).to(device)
    model = UNet(base=args.base, depth=args.depth).to(device)
    print(f"参数 {parameter_count(model):,}", file=sys.stderr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(train_names) // args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(
        SegDataset(train_names, args.image_dir, args.label_dir, args.crop, True),
        batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True,
        persistent_workers=args.workers > 0, collate_fn=collate,
    )
    valid_loader = DataLoader(
        SegDataset(valid_names, args.image_dir, args.label_dir, args.crop, False),
        batch_size=max(1, args.batch_size), num_workers=max(1, args.workers // 2),
        pin_memory=device.type == "cuda", collate_fn=collate,
    )

    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        started = time.perf_counter()
        running = 0.0
        seen = 0
        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits.float(), labels) + args.dice_weight * dice_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler.last_epoch < scheduler.total_steps:
                scheduler.step()
            running += float(loss.detach()) * labels.numel()
            seen += labels.numel()
            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch + 1} step {step + 1} loss {running / max(seen, 1):.4f}",
                      file=sys.stderr, flush=True)
        accuracy, miou, foreground_miou = evaluate(model, valid_loader, device)
        elapsed = time.perf_counter() - started
        print(f"epoch {epoch + 1}: loss {running / max(seen, 1):.4f} "
              f"stroke-acc {accuracy:.4f} mIoU {miou:.4f} fg-mIoU {foreground_miou:.4f} "
              f"({elapsed:.0f}s)", file=sys.stderr)
        score = foreground_miou * 0.6 + miou * 0.2 + accuracy * 0.2
        if score > best:
            best = score
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "model_kwargs": {"in_channels": 3, "classes": NUM_CLASSES,
                                 "base": args.base, "depth": args.depth},
                "metrics": {"stroke_accuracy": accuracy, "miou": miou,
                            "foreground_miou": foreground_miou, "epoch": epoch + 1},
            }, args.out)
    print(f"最佳权重 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
