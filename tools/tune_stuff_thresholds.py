#!/usr/bin/env python3
"""在 calib 上联合选择 Stuff 每类的概率阈值与最小面积。

Stuff 每图每类只输出一段；空图上的低阈值噪声会造成 FP，因此只看正样本
IoU 不够。这里对全部 calib 图统计 TP/FP/FN，并用边际 PQ 目标选阈值。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402
from cadvec.stuff_branch import CLASSES, RASTER, StuffBranch  # noqa: E402


def iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    return intersection / int(np.count_nonzero(first | second))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="weights/stuff_thresholds.json")
    parser.add_argument("--report", default="_work/stuff_threshold_report.json")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--target-pq", type=float, default=0.22)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    branch = StuffBranch.load(args.weights, None, args.device)
    if branch is None:
        raise SystemExit("无法加载 Stuff 权重")
    thresholds = np.round(np.arange(0.30, 0.91, 0.05), 2)
    areas = [0, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    records = {class_id: [] for class_id in CLASSES}
    paths = sorted(Path(args.png_dir).glob("*.png"))[: args.limit]
    for index, png in enumerate(paths):
        gt_path = Path(args.gt_dir) / f"{png.stem}.json"
        if not gt_path.exists():
            continue
        gt = load_json(gt_path)
        rendered, _ = _segments(gt, (100, 100), RASTER)
        probability = branch.probability(png)
        for class_index, class_id in enumerate(CLASSES):
            truth = rendered.get((class_id, -1))
            present = truth is not None and bool(truth.any())
            values = {}
            for threshold in thresholds:
                mask = probability[class_index] > threshold
                area = int(mask.sum())
                values[float(threshold)] = {
                    "area": area,
                    "iou": iou(mask, truth) if present else 0.0,
                }
            records[class_id].append({"gt": present, "values": values})
        if index % 25 == 0:
            print(index, flush=True)

    output = {}
    report = {}
    for class_id in CLASSES:
        rows = records[class_id]
        if not rows:
            continue
        best = None
        for threshold in thresholds:
            for min_area in areas:
                tp = fp = fn = 0
                total_iou = 0.0
                for row in rows:
                    value = row["values"][float(threshold)]
                    pred = value["area"] >= min_area
                    matched = pred and value["iou"] > 0.5
                    if matched:
                        tp += 1
                        total_iou += value["iou"]
                    elif pred:
                        fp += 1
                    if row["gt"] and not matched:
                        fn += 1
                denominator = tp + 0.5 * (fp + fn)
                score = total_iou - args.target_pq * denominator
                if best is None or score > best[0]:
                    best = (score, float(threshold), int(min_area),
                            tp, fp, fn, total_iou)
        score, threshold, min_area, tp, fp, fn, total_iou = best
        denominator = tp + 0.5 * (fp + fn)
        output[str(class_id)] = {"threshold": threshold, "min_area": min_area}
        report[str(class_id)] = {
            "threshold": threshold, "min_area": min_area,
            "TP": tp, "FP": fp, "FN": fn,
            "PQ": total_iou / denominator if denominator else 0.0,
            "precision": tp / max(tp + fp, 1),
        }
        print(f"class {class_id}: thr={threshold:.2f} area>={min_area} "
              f"TP={tp} FP={fp} FN={fn} "
              f"PQ={100*report[str(class_id)]['PQ']:.2f}%")
    Path(args.out).write_text(json.dumps(output, indent=1), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
    print("wrote", args.out, args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
