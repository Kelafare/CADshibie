#!/usr/bin/env python3
"""E4 原子归属 oracle：把各分支图元原子直接分配给 GT 实例，测表示上界。

每个原子按其采样点落入 GT mask 的比例赋给 GT；再按 GT 实例分组导出，
忽略原预测的实例编号。FP 数视为 0（每个组都对应一个 GT），用于判断
几何/原子表示是否足够。
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


def mask_iou(first, second):
    inter = int(np.count_nonzero(first & second))
    if not inter:
        return 0.0
    return inter / int(np.count_nonzero(first | second))


def bbox(mask):
    ys, xs = np.nonzero(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def bbox_iou(first, second):
    x0 = max(first[0], second[0]); y0 = max(first[1], second[1])
    x1 = min(first[2], second[2]); y1 = min(first[3], second[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = ((first[2] - first[0]) * (first[3] - first[1]) +
            (second[2] - second[0]) * (second[3] - second[1]) - inter)
    return inter / max(area, 1e-9)


def sample_points(args, raster):
    points = [(float(args[i]), float(args[i + 1])) for i in range(0, 8, 2)]
    out = []
    for index in range(3):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        for step in range(8):
            ratio = step / 8.0
            out.append((int((x0 + (x1 - x0) * ratio) * raster / 100.0),
                        int((y0 + (y1 - y0) * ratio) * raster / 100.0)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--raster", type=int, default=1000)
    parser.add_argument("--pred-dirs", nargs="+", required=True)
    parser.add_argument("--out", default="_work/e4_atoms.json")
    args = parser.parse_args()
    sources = [(name, Path(directory))
               for name, directory in (value.split("=", 1)
                                       for value in args.pred_dirs)]
    stats = defaultdict(lambda: {"gt": 0, "tp": 0, "fn": 0, "iou": 0.0,
                                 "covered": 0})
    paths = sorted(Path(args.gt_dir).glob("*.json"))
    for index, gt_path in enumerate(paths):
        gt = load_json(gt_path)
        gt_masks, _ = _segments(gt, (100.0, 100.0), args.raster)
        gt_items = {key: mask for key, mask in gt_masks.items()
                    if key[0] < THING and key[1] >= 0}
        if not gt_items:
            continue
        gt_boxes = {key: bbox(mask) for key, mask in gt_items.items()}
        groups = defaultdict(list)
        assigned_atoms = Counter()
        for name, directory in sources:
            path = directory / gt_path.name
            if not path.exists():
                continue
            data = load_json(path)
            for command, semantic, instance, args8 in zip(
                    data["commands"], data["semanticIds"],
                    data["instanceIds"], data["args"]):
                if int(semantic) >= THING or int(instance) < 0:
                    continue
                points = sample_points(args8, args.raster)
                best = (0.0, None)
                for key, mask in gt_items.items():
                    hits = sum(1 for x, y in points
                               if 0 <= y < args.raster and 0 <= x < args.raster
                               and mask[y, x])
                    ratio = hits / max(len(points), 1)
                    if ratio > best[0]:
                        best = (ratio, key)
                if best[1] is None or best[0] < 0.30:
                    continue
                key = best[1]
                groups[key].append((int(command), args8))
                assigned_atoms[key] += 1
        for key, _mask in gt_items.items():
            entry = stats[key[0]]
            entry["gt"] += 1
            if assigned_atoms.get(key, 0) > 0:
                entry["covered"] += 1
        # 每个 GT 组按原样导出，计算严格 IoU
        pred_data = {
            "width": 100, "height": 100,
            "commands": [], "args": [], "semanticIds": [],
            "instanceIds": [], "lengths": [], "rgb": [], "layerIds": [],
            "widths": [],
        }
        keys = []
        for key, items in groups.items():
            keys.append(key)
            for command, args8 in items:
                pred_data["commands"].append(command)
                pred_data["args"].append(list(args8))
                pred_data["semanticIds"].append(key[0])
                pred_data["instanceIds"].append(key[1])
                pred_data["lengths"].append(0.0)
                pred_data["rgb"].append([0, 0, 0])
                pred_data["layerIds"].append(0)
                pred_data["widths"].append(1.0)
        pred_masks, _ = _segments(pred_data, (100.0, 100.0), args.raster)
        by_class_gt = defaultdict(list)
        by_class_pred = defaultdict(list)
        for key in gt_items:
            by_class_gt[key[0]].append(key)
        for key in pred_masks:
            by_class_pred[key[0]].append(key)
        for class_id in sorted(by_class_gt):
            gts = by_class_gt[class_id]
            preds = by_class_pred.get(class_id, [])
            entry = stats[class_id]
            if not preds:
                entry["fn"] += len(gts)
                continue
            scores = np.zeros((len(gts), len(preds)), dtype=np.float32)
            for row, gkey in enumerate(gts):
                for col, pkey in enumerate(preds):
                    if bbox_iou(gt_boxes[gkey], bbox(pred_masks[pkey])) < 0.02:
                        continue
                    scores[row, col] = mask_iou(gt_items[gkey],
                                                pred_masks[pkey])
            rows, cols = linear_sum_assignment(-scores)
            matched = 0
            for row, col in zip(rows, cols):
                if scores[row, col] <= 0.5:
                    continue
                matched += 1
                entry["tp"] += 1
                entry["iou"] += float(scores[row, col])
            entry["fn"] += len(gts) - matched
        if index % 30 == 0:
            print(index, flush=True)
    payload = {"classes": {}}
    total_tp = total_fn = 0
    total_iou = 0.0
    print(f"{'cls':>3} {'gt':>5} {'cover':>5} {'tp':>4} {'fn':>4} {'oraclePQ':>9}")
    for class_id, entry in sorted(stats.items()):
        denominator = entry["tp"] + 0.5 * entry["fn"]
        pq = entry["iou"] / denominator if denominator else 0.0
        total_tp += entry["tp"]; total_fn += entry["fn"]
        total_iou += entry["iou"]
        payload["classes"][str(class_id)] = {
            "gt": entry["gt"], "covered": entry["covered"],
            "tp": entry["tp"], "fn": entry["fn"], "oracle_pq": pq,
        }
        print(f"{class_id:>3} {entry['gt']:>5} {entry['covered']:>5} "
              f"{entry['tp']:>4} {entry['fn']:>4} {100*pq:>8.2f}%")
    denominator = total_tp + 0.5 * total_fn
    payload["total"] = {"tp": total_tp, "fn": total_fn,
                        "oracle_pq": total_iou / denominator
                        if denominator else 0.0}
    print(f"TOTAL oracle TP {total_tp} FN {total_fn} "
          f"PQ {100*payload['total']['oracle_pq']:.2f}%")
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
