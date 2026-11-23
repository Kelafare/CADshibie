#!/usr/bin/env python3
"""在校准集 ROI 行上重新选择逐类质量门限，只改 threshold，不改模型。"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def probability(spec: dict, features: dict) -> np.ndarray:
    names = spec["features"]
    values = np.asarray([[float(features.get(name, 0.0)) for name in names]])
    mean = np.asarray(spec["mean"])
    scale = np.asarray(spec["scale"])
    coef = np.asarray(spec["coef"])
    z = (values - mean) / np.maximum(scale, 1e-9) @ coef + float(spec["intercept"])
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target-pq", type=float, default=0.18)
    args = parser.parse_args()

    payload = json.loads(Path(args.model).read_text(encoding="utf-8"))
    rows = load_rows(args.rows)
    gt_counts = json.loads(Path(args.rows + ".gt.json").read_text(encoding="utf-8"))
    gt_total = defaultdict(int)
    for counts in gt_counts.values():
        for class_id, value in counts.items():
            gt_total[int(class_id)] += int(value)
    by_class = defaultdict(list)
    for row in rows:
        by_class[int(row["class"])].append(row)
    classes = payload["classes"]
    for class_id, items in sorted(by_class.items()):
        spec = classes.get(str(class_id))
        if not spec:
            continue
        scores = np.asarray([
            float(probability({"features": payload["features"], **spec},
                              row["features"])[0]) for row in items])
        truth = np.asarray([bool(row["tp"]) for row in items])
        ious = np.asarray([float(row["iou"]) for row in items])
        best = None
        for threshold in np.arange(0.05, 0.96, 0.01):
            keep = scores >= threshold
            tp = int((keep & truth).sum())
            fp = int((keep & ~truth).sum())
            fn = max(gt_total.get(class_id, 0) - tp, 0)
            s = float(ious[keep & truth].sum())
            value = s - args.target_pq * (tp + 0.5 * (fp + fn))
            if best is None or value > best[0]:
                best = (value, float(threshold), tp, fp, fn, s)
        value, threshold, tp, fp, fn, s = best
        old = float(spec.get("threshold", 0.5))
        spec["threshold"] = threshold
        denominator = tp + 0.5 * (fp + fn)
        print(f"class {class_id}: {old:.2f} -> {threshold:.2f}  "
              f"TP {tp} FP {fp} FN {fn} PQ {100*s/max(denominator,1):.2f}%  "
              f"precision {tp/max(tp+fp,1):.1%}")
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
