#!/usr/bin/env python3
"""训练 atom-group × YOLO anchor 亲和力（逐类逻辑回归/共享模型）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "inside_mean", "inside_max", "dx", "dy", "dist", "w_ratio",
    "h_ratio", "area_ratio", "length_ratio", "n_atoms", "anchor_conf",
    "bbox_iou",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", default="weights/anchor_affinity.json")
    parser.add_argument("--calib-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = []
    with open(args.rows, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    names = sorted({row["stem"] for row in rows})
    rng = np.random.RandomState(args.seed)
    rng.shuffle(names)
    split = max(1, int(len(names) * (1.0 - args.calib_fraction)))
    fit_names = set(names[:split])
    calib_names = set(names[split:])
    fit = [row for row in rows if row["stem"] in fit_names]
    calib = [row for row in rows if row["stem"] in calib_names]
    x = np.asarray([[row[name] for name in FEATURES] for row in fit],
                   dtype=np.float64)
    y = np.asarray([row["label"] for row in fit], dtype=np.int64)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    clf.fit(x_scaled, y)
    cx = np.asarray([[row[name] for name in FEATURES] for row in calib],
                    dtype=np.float64)
    cy = np.asarray([row["label"] for row in calib], dtype=np.int64)
    prob = clf.predict_proba(scaler.transform(cx))[:, 1]
    best = None
    for threshold in np.arange(0.05, 0.96, 0.01):
        pred = prob >= threshold
        tp = int((pred & (cy == 1)).sum())
        fp = int((pred & (cy == 0)).sum())
        fn = int((~pred & (cy == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        score = f1
        if best is None or score > best[0]:
            best = (score, float(threshold), precision, recall, tp, fp, fn)
    _, threshold, precision, recall, tp, fp, fn = best
    print(f"fit {len(fit)} calib {len(calib)}; threshold {threshold:.2f} "
          f"precision {precision:.3f} recall {recall:.3f} "
          f"TP {tp} FP {fp} FN {fn}")
    payload = {
        "schema_version": 1,
        "features": list(FEATURES),
        "mean": [float(v) for v in scaler.mean_],
        "scale": [float(v) for v in scaler.scale_],
        "coef": [float(v) for v in clf.coef_[0]],
        "intercept": float(clf.intercept_[0]),
        "threshold": threshold,
    }
    Path(args.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
