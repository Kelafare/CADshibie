#!/usr/bin/env python3
"""训练 ROI 细化实例的学习式质量门控（逐类逻辑回归）。

输入 tools/roi_quality_dataset.py 的 JSONL 与 .gt.json；按图像切分拟合/校准，
逐类拟合标准化逻辑回归，并在校准集上按“对总 PQ 的边际贡献”选阈值。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "conf", "prob_mean", "prob_max", "prob_p90", "mask_fraction",
    "n_seg", "total_len", "mean_len", "max_len", "w", "h",
    "aspect", "density", "log_box_area", "ink_box_ratio",
)


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def vector(row: dict) -> list[float]:
    return [float(row["features"].get(name, 0.0)) for name in FEATURES]


def metrics(tp: int, fp: int, fn: int, s: float) -> float:
    d = tp + 0.5 * (fp + fn)
    return s / d if d else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", default="weights/roi_quality.json")
    parser.add_argument("--calib-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-pq", type=float, default=0.16,
                        help="选阈值时用于衡量边际贡献的目标 PQ")
    parser.add_argument("--min-rows", type=int, default=80)
    args = parser.parse_args()

    rows = load_rows(args.rows)
    gt_counts = json.loads(Path(args.rows + ".gt.json").read_text(encoding="utf-8"))
    names = sorted({row["name"] for row in rows})
    random.Random(args.seed).shuffle(names)
    split = max(1, int(len(names) * (1.0 - args.calib_fraction)))
    fit_names = set(names[:split])
    calib_names = set(names[split:])
    fit_rows = [row for row in rows if row["name"] in fit_names]
    calib_rows = [row for row in rows if row["name"] in calib_names]
    print(f"rows {len(rows)} fit {len(fit_rows)} calib {len(calib_rows)} "
          f"images {len(fit_names)}/{len(calib_names)}", flush=True)

    by_class_fit = defaultdict(list)
    by_class_calib = defaultdict(list)
    for row in fit_rows:
        by_class_fit[int(row["class"])].append(row)
    for row in calib_rows:
        by_class_calib[int(row["class"])].append(row)

    # 校准集每个类别的 GT 总数（含没有预测的图）
    calib_gt = defaultdict(int)
    for name in calib_names:
        for class_id, count in (gt_counts.get(name) or {}).items():
            calib_gt[int(class_id)] += int(count)

    models = {}
    for class_id in sorted(by_class_fit):
        fit = by_class_fit[class_id]
        labels = np.array([1 if row["tp"] else 0 for row in fit])
        if len(fit) < args.min_rows or labels.sum() < 10 or (1 - labels).sum() < 10:
            print(f"class {class_id}: skip rows={len(fit)} tp={int(labels.sum())}")
            continue
        x = np.asarray([vector(row) for row in fit], dtype=np.float64)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        clf = LogisticRegression(C=1.0, class_weight="balanced",
                                 max_iter=3000, solver="lbfgs")
        clf.fit(x_scaled, labels)

        calib = by_class_calib.get(class_id, [])
        best = None
        for threshold in np.arange(0.05, 0.96, 0.01):
            tp = fp = 0
            s = 0.0
            if calib:
                cx = np.asarray([vector(row) for row in calib], dtype=np.float64)
                prob = clf.predict_proba(scaler.transform(cx))[:, 1]
                keep = prob >= threshold
                for row, kept in zip(calib, keep):
                    if not kept:
                        continue
                    if row["tp"]:
                        tp += 1
                        s += float(row["iou"])
                    else:
                        fp += 1
            fn = max(calib_gt.get(class_id, 0) - tp, 0)
            value = s - args.target_pq * (tp + 0.5 * (fp + fn))
            if best is None or value > best[0]:
                best = (value, float(threshold), tp, fp, fn, s)
        _, threshold, tp, fp, fn, s = best
        print(f"class {class_id}: rows={len(fit)} tp={int(labels.sum())} "
              f"-> threshold {threshold:.2f} calib TP={tp} FP={fp} FN={fn} "
              f"PQ={metrics(tp, fp, fn, s)*100:.2f}%", flush=True)
        models[str(class_id)] = {
            "mean": [float(v) for v in scaler.mean_],
            "scale": [float(v) for v in scaler.scale_],
            "coef": [float(v) for v in clf.coef_[0]],
            "intercept": float(clf.intercept_[0]),
            "threshold": threshold,
        }

    # 汇总校准集 ROI 类别指标
    total = {"tp": 0, "fp": 0, "fn": 0, "s": 0.0}
    for class_id, model in models.items():
        cid = int(class_id)
        calib = by_class_calib.get(cid, [])
        tp = fp = 0
        s = 0.0
        if calib:
            cx = np.asarray([vector(row) for row in calib], dtype=np.float64)
            mean = np.asarray(model["mean"])
            scale = np.asarray(model["scale"])
            coef = np.asarray(model["coef"])
            z = (cx - mean) / scale @ coef + model["intercept"]
            prob = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            keep = prob >= model["threshold"]
            for row, kept in zip(calib, keep):
                if not kept:
                    continue
                if row["tp"]:
                    tp += 1
                    s += float(row["iou"])
                else:
                    fp += 1
        fn = max(calib_gt.get(cid, 0) - tp, 0)
        total["tp"] += tp
        total["fp"] += fp
        total["fn"] += fn
        total["s"] += s
    print(f"calib ROI total TP={total['tp']} FP={total['fp']} FN={total['fn']} "
          f"PQ={metrics(total['tp'], total['fp'], total['fn'], total['s'])*100:.2f}%")

    payload = {
        "schema_version": 1,
        "source": "train_final ROI predictions, logistic regression per class",
        "features": list(FEATURES),
        "classes": models,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
