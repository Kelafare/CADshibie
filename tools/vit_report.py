#!/usr/bin/env python3
"""ViT 验证集逐类报告：precision / recall / 主要混淆。"""
from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadvec.crops import CROP_SIZE, prepare_image, render_crop  # noqa: E402
from cadvec.vit import StrokeViT  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    print("需要 torch", file=sys.stderr)
    raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("../_work/candidates"))
    parser.add_argument("--image-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--weights", type=Path, default=Path("../_work/vit_strokes.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-per-image", type=int, default=36)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_vit import _load_samples  # noqa: E402

    samples = _load_samples(args.candidates, args.image_dir, args.limit_per_image)
    names = sorted({sample["name"] for sample in samples})
    random.Random(args.seed).shuffle(names)
    split = max(1, int(len(names) * 0.88))
    valid_names = set(names[split:])
    valid = [sample for sample in samples if sample["name"] in valid_names]
    print(f"验证样本 {len(valid)}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = StrokeViT(**checkpoint.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    groups: dict[str, list] = defaultdict(list)
    for sample in valid:
        groups[sample["name"]].append(sample)
    true_positive = Counter()
    predicted_total = Counter()
    actual_total = Counter()
    confusion = Counter()
    with torch.no_grad():
        for name, items in groups.items():
            image, sx, sy = prepare_image(args.image_dir / f"{name}.png")
            crops = [
                render_crop(image, tuple(item["bbox"]), item["strokes"], sx, sy, size=CROP_SIZE)
                for item in items
            ]
            block = torch.from_numpy(np.stack(crops)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                logits = model(block)
            predictions = logits.float().argmax(dim=1).cpu().numpy()
            for item, prediction in zip(items, predictions):
                label = int(item["label"])
                prediction = int(prediction)
                actual_total[label] += 1
                predicted_total[prediction] += 1
                if prediction == label:
                    true_positive[label] += 1
                else:
                    confusion[(label, prediction)] += 1
    print("class  precision  recall  n_true  n_pred")
    for class_id in sorted(actual_total):
        precision = true_positive[class_id] / max(predicted_total[class_id], 1)
        recall = true_positive[class_id] / max(actual_total[class_id], 1)
        print(f"{class_id:5d}  {precision:9.3f}  {recall:6.3f}  {actual_total[class_id]:6d}  {predicted_total[class_id]:6d}")
    print("\ntop confusions (true -> pred : n)")
    for (label, prediction), count in confusion.most_common(15):
        print(f"  {label:2d} -> {prediction:2d} : {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
