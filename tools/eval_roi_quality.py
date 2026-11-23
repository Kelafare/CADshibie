#!/usr/bin/env python3
"""在 test_final 的 ROI 质量行上评估学习式门控（calib/holdout 分列）。"""
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


def probability(model: dict, row: dict) -> float:
    features = row["features"]
    z = float(model["intercept"])
    for name, coef, mean, scale in zip(model["features"], model["coef"],
                                       model["mean"], model["scale"]):
        z += float(coef) * (float(features.get(name, 0.0)) - float(mean)) / max(
            float(scale), 1e-9)
    return 1.0 / (1.0 + np.exp(-max(min(z, 50.0), -50.0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--model", default="weights/roi_quality.json")
    parser.add_argument("--calib-dir", default=None,
                        help="calib GT JSON 目录（用于区分 calib/holdout）")
    parser.add_argument("--calib-list", default=None,
                        help="calib 文件名列表（每行一个 stem）")
    args = parser.parse_args()

    payload = json.loads(Path(args.model).read_text(encoding="utf-8"))
    names = payload["features"]
    models = {int(k): v for k, v in payload["classes"].items()}
    if args.calib_list:
        calib_names = {line.strip() for line in Path(args.calib_list).read_text().splitlines() if line.strip()}
    else:
        calib_names = {Path(f).stem for f in Path(args.calib_dir).glob("*.json")}
    rows = load_rows(args.rows)
    gt_counts = json.loads(Path(args.rows + ".gt.json").read_text(encoding="utf-8"))

    splits = ("calib", "hold", "all")
    stats = {split: {"base": defaultdict(lambda: [0, 0, 0, 0.0]),
                     "gate": defaultdict(lambda: [0, 0, 0, 0.0])}
             for split in splits}
    gt_total = {split: defaultdict(int) for split in splits}
    for name, counts in gt_counts.items():
        split = "calib" if name in calib_names else "hold"
        for class_id, count in counts.items():
            if int(class_id) in models:
                for key in (split, "all"):
                    gt_total[key][int(class_id)] += int(count)

    for row in rows:
        class_id = int(row["class"])
        model = models.get(class_id)
        if model is None:
            continue
        split = "calib" if row["name"] in calib_names else "hold"
        keep = probability({"features": names, **model}, row) >= float(model["threshold"])
        for key in (split, "all"):
            for bucket, selected in (("base", True), ("gate", keep)):
                if not selected:
                    continue
                entry = stats[key][bucket][class_id]
                if row["tp"]:
                    entry[0] += 1
                    entry[3] += float(row["iou"])
                else:
                    entry[1] += 1

    for split in splits:
        print(f"== {split} ==")
        for bucket in ("base", "gate"):
            total = [0, 0, 0, 0.0]
            for class_id, entry in stats[split][bucket].items():
                tp, fp, _fn, s = entry
                fn = max(gt_total[split][class_id] - tp, 0)
                d = tp + 0.5 * (fp + fn)
                print(f"  {bucket} class {class_id:2d}: TP {tp:4d} FP {fp:4d} "
                      f"FN {fn:4d} PQ {100*s/max(d,1):5.2f}%")
                total[0] += tp; total[1] += fp; total[2] += fn; total[3] += s
            d = total[0] + 0.5 * (total[1] + total[2])
            print(f"  {bucket} ROI total: TP {total[0]} FP {total[1]} "
                  f"FN {total[2]} PQ {100*total[3]/max(d,1):.3f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
