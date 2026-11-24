#!/usr/bin/env python3
"""E5 候选池 oracle：冻结多个分支预测，用 GT 做最优选择的上界。

不训练、不改变预测。对每类把各分支候选并集与 GT 做匈牙利匹配，
统计 oracle TP/FN、最优 ΣIoU 与各分支的唯一命中。FP 视为可被 oracle 丢弃。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402

THING = 30


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_iou(first, second):
    if first is None or second is None:
        return 0.0
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / max(area, 1e-9)


def mask_iou(first, second):
    inter = int(np.count_nonzero(first & second))
    if not inter:
        return 0.0
    return inter / int(np.count_nonzero(first | second))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--raster", type=int, default=1000)
    parser.add_argument("--out", default="_work/e5_oracle.json")
    parser.add_argument("--pred-dirs", nargs="+", required=True,
                        help="名称=目录，例如 v2=/path")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    sources = []
    for value in args.pred_dirs:
        name, directory = value.split("=", 1)
        sources.append((name, Path(directory)))

    stats = defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0, "iou": 0.0,
                                 "unique": Counter(), "best": Counter()})
    paths = sorted(Path(args.gt_dir).glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    for index, gt_path in enumerate(paths):
        gt = load_json(gt_path)
        gt_masks, _ = _segments(gt, (100.0, 100.0), args.raster)
        gt_items = [(key, mask) for key, mask in gt_masks.items()
                    if key[0] < THING and key[1] >= 0]
        if not gt_items:
            continue
        gt_boxes = [bbox(mask) for _key, mask in gt_items]
        candidates = []
        per_source = {}
        for name, directory in sources:
            path = directory / gt_path.name
            if not path.exists():
                continue
            data = load_json(path)
            masks, _ = _segments(data, (100.0, 100.0), args.raster)
            entries = []
            for key, mask in masks.items():
                if key[0] >= THING or key[1] < 0:
                    continue
                entries.append((key[0], key[1], mask, bbox(mask), name))
            per_source[name] = entries
            candidates.extend(entries)
        by_class_gt = defaultdict(list)
        by_class_cand = defaultdict(list)
        for index_gt, (key, _mask) in enumerate(gt_items):
            by_class_gt[key[0]].append(index_gt)
        for index_cand, entry in enumerate(candidates):
            by_class_cand[entry[0]].append(index_cand)
        for class_id in sorted(set(by_class_gt) | set(by_class_cand)):
            gts = by_class_gt.get(class_id, [])
            cands = by_class_cand.get(class_id, [])
            entry = stats[class_id]
            entry["gt"] += len(gts)
            if not gts or not cands:
                entry["fn"] += len(gts)
                continue
            scores = np.zeros((len(gts), len(cands)), dtype=np.float32)
            for row, gi in enumerate(gts):
                for col, ci in enumerate(cands):
                    if bbox_iou(gt_boxes[gi], candidates[ci][3]) < 0.02:
                        continue
                    scores[row, col] = mask_iou(gt_items[gi][1],
                                                candidates[ci][2])
            rows, cols = linear_sum_assignment(-scores)
            matched = 0
            for row, col in zip(rows, cols):
                if scores[row, col] <= 0.5:
                    continue
                matched += 1
                entry["iou"] += float(scores[row, col])
                entry["tp"] += 1
                # 哪些来源能独立命中该 GT
                gi = gts[row]
                hit_sources = []
                for name, entries in per_source.items():
                    best = 0.0
                    for _c, _i, mask, box, _n in entries:
                        if _c != class_id:
                            continue
                        if bbox_iou(gt_boxes[gi], box) < 0.02:
                            continue
                        best = max(best, mask_iou(gt_items[gi][1], mask))
                    if best > 0.5:
                        hit_sources.append(name)
                if len(hit_sources) == 1:
                    entry["unique"][hit_sources[0]] += 1
                entry["best"]["+".join(hit_sources)] += 1
            entry["fn"] += len(gts) - matched
        if index % 30 == 0:
            print(index, flush=True)

    payload = {"sources": [name for name, _ in sources], "classes": {}}
    print(f"{'cls':>3} {'gt':>5} {'tp':>4} {'fn':>4} {'oraclePQ':>9} {'unique'}")
    total_tp = total_fn = 0
    total_iou = 0.0
    for class_id, entry in sorted(stats.items()):
        denominator = entry["tp"] + 0.5 * entry["fn"]
        pq = entry["iou"] / denominator if denominator else 0.0
        total_tp += entry["tp"]; total_fn += entry["fn"]
        total_iou += entry["iou"]
        record = {
            "gt": entry["gt"], "tp": entry["tp"], "fn": entry["fn"],
            "oracle_pq": pq, "unique": dict(entry["unique"]),
            "best_sources": dict(entry["best"]),
        }
        payload["classes"][str(class_id)] = record
        print(f"{class_id:>3} {entry['gt']:>5} {entry['tp']:>4} "
              f"{entry['fn']:>4} {100*pq:>8.2f}% "
              f"{dict(entry['unique'])}")
    denominator = total_tp + 0.5 * total_fn
    payload["total"] = {
        "tp": total_tp, "fn": total_fn,
        "oracle_pq": total_iou / denominator if denominator else 0.0,
    }
    print(f"TOTAL oracle TP {total_tp} FN {total_fn} "
          f"PQ {100*payload['total']['oracle_pq']:.2f}%")
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
