#!/usr/bin/env python3
"""逐类阈值校准：用验证集假设的 (置信度, 是否真匹配) 曲线选阈值与抑制类。

样本来自 `tools/build_candidates.py` 的验证集划分（默认与训练相同的
88/12 图像切分，种子 0）。"正确" 的定义与评分口径一致：模型 argmax 类别
等于样本类别，且样本与 GT 实例的 IoU > 0.5。

输出 `weights/class_policy.json`：
    {"thresholds": {"<class>": t, ...}, "suppress": [...], "calibration": {...}}
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadvec.crops import CROP_SIZE, prepare_image, render_crop  # noqa: E402
from cadvec.pipeline_v2 import apply_color_prior, load_color_prior  # noqa: E402
from cadvec.vit import CLASS_COUNT, StrokeViT  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover
    print("需要 torch", file=sys.stderr)
    raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=Path("../_work/candidates_comp"))
    parser.add_argument("--image-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--weights", type=Path, default=Path("weights/vit_strokes.pt"))
    parser.add_argument("--color-prior", type=Path, default=Path("weights/color_class_prior.json"))
    parser.add_argument("--color-prior-weight", type=float, default=0.6)
    parser.add_argument("--out", type=Path, default=Path("weights/class_policy.json"))
    parser.add_argument("--min-precision", type=float, default=0.15,
                        help="选阈值时的精度下限；达不到则抑制该类")
    parser.add_argument("--min-support", type=int, default=8,
                        help="验证集里 argmax 为该类的样本少于该值则不校准")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    samples = []
    for path in sorted(args.candidates.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("samples", []):
            item["name"] = payload["name"]
            samples.append(item)
    names = sorted({item["name"] for item in samples})
    random.Random(args.seed).shuffle(names)
    split = max(1, int(len(names) * 0.88))
    valid_names = set(names[split:])
    samples = [item for item in samples if item["name"] in valid_names]
    if args.limit:
        samples = samples[: args.limit]
    print(f"校准样本 {len(samples)}（{len(valid_names)} 图）", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = StrokeViT(**checkpoint.get("model_kwargs", {})).to(device).eval()
    model.load_state_dict(checkpoint["state_dict"])

    prior_table = load_color_prior(args.color_prior)

    class Sample:
        pass

    rows: list[dict] = []
    groups: dict[str, list] = defaultdict(list)
    for item in samples:
        groups[item["name"]].append(item)
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
                output = model(block)
            logits = output[0] if isinstance(output, tuple) else output
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            if args.color_prior_weight > 0:
                # 用与推理相同的融合：构造最小假设对象。
                hypotheses = []
                for item in items:
                    hypothesis = Sample()
                    hypothesis.indices = list(range(len(item["strokes"])))
                    hypotheses.append(hypothesis)
                strokes = [
                    type("S", (), {"rgb": stroke["rgb"]})()
                    for item in items for stroke in item["strokes"]
                ]
                # 每个假设的 rgb 取第一根笔画，索引需与 strokes 对齐。
                offset = 0
                for hypothesis, item in zip(hypotheses, items):
                    hypothesis.indices = [offset]
                    offset += len(item["strokes"])
                probabilities = apply_color_prior(
                    probabilities, hypotheses, strokes, prior_table,
                    args.color_prior_weight,
                )
            predictions = probabilities.argmax(axis=1)
            confidences = probabilities.max(axis=1)
            for item, prediction, confidence in zip(items, predictions, confidences):
                rows.append({
                    "prediction": int(prediction),
                    "confidence": float(confidence),
                    "label": int(item["label"]),
                    "iou": float(item.get("iou", 0.0)),
                })

    by_class: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        correct = row["prediction"] == row["label"] and row["iou"] > 0.5
        by_class[row["prediction"]].append((row["confidence"], correct))

    thresholds: dict[str, float] = {}
    suppressed: list[int] = []
    report: dict[str, dict] = {}
    for class_id in range(CLASS_COUNT):
        if class_id == 35:
            continue
        entries = by_class.get(class_id, [])
        if len(entries) < args.min_support:
            if entries:
                suppressed.append(class_id)
                report[str(class_id)] = {"support": len(entries), "action": "suppress_low_support"}
            continue
        values = np.array([entry[0] for entry in entries])
        correct = np.array([entry[1] for entry in entries], dtype=bool)
        best = None
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.95, 19):
            kept = values >= threshold
            tp = int(np.count_nonzero(kept & correct))
            fp = int(np.count_nonzero(kept & ~correct))
            fn = int(np.count_nonzero(~kept & correct))
            if tp == 0:
                continue
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            if precision < args.min_precision:
                continue
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best = (float(threshold), precision, recall, tp, fp, fn)
        if best is None:
            suppressed.append(class_id)
            report[str(class_id)] = {
                "support": len(entries), "action": "suppress_low_precision",
                "max_precision_below": args.min_precision,
            }
            continue
        threshold, precision, recall, tp, fp, fn = best
        thresholds[str(class_id)] = threshold
        report[str(class_id)] = {
            "support": len(entries), "threshold": threshold,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    payload = {
        "schema_version": 1,
        "source": str(args.candidates),
        "samples": len(rows),
        "target_precision_floor": args.min_precision,
        "thresholds": thresholds,
        "suppress": suppressed,
        "report": report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"写入 {args.out}: {len(thresholds)} 类阈值, 抑制 {len(suppressed)} 类", file=sys.stderr)
    for class_id in sorted(report, key=int):
        print(class_id, report[class_id], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
