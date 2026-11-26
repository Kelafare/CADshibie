#!/usr/bin/env python3
"""用同图 SIFT 自模板候选给现有实例打分，评估能否过滤 FP。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
    return (float(xs.min()), float(ys.min()),
            float(xs.max()), float(ys.max()))


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
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--sift-jsonl", required=True)
    parser.add_argument("--classes", default="3,8,18,22,28")
    parser.add_argument("--out", default="_work/sift_rerank.json")
    args = parser.parse_args()
    classes = {int(v) for v in args.classes.split(",") if v.strip()}
    sift = {}
    with open(args.sift_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                sift[row["stem"]] = row
    records = defaultdict(list)
    gt_total = defaultdict(int)
    base = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "s": 0.0})
    for gt_path in sorted(Path(args.gt_dir).glob("*.json")):
        pred_path = Path(args.pred_dir) / gt_path.name
        row = sift.get(gt_path.stem)
        if not pred_path.exists() or row is None:
            continue
        gt = load_json(gt_path)
        pred = load_json(pred_path)
        gt_masks, _ = _segments(gt, (100.0, 100.0), 1000)
        pred_masks, _ = _segments(pred, (100.0, 100.0), 1000)
        gt_items = {key: mask for key, mask in gt_masks.items()
                    if key[0] < THING and key[1] >= 0}
        pred_items = {key: mask for key, mask in pred_masks.items()
                      if key[0] < THING and key[1] >= 0}
        candidates = defaultdict(list)
        for item in row["candidates"]:
            class_id = int(item["class"])
            if class_id in classes:
                candidates[class_id].append(
                    (tuple(float(v) for v in item["box"]),
                     float(item.get("score", 0))))
        for class_id in classes:
            gts = [key for key in gt_items if key[0] == class_id]
            preds = [key for key in pred_items if key[0] == class_id]
            gt_total[class_id] += len(gts)
            scores = np.zeros((len(gts), len(preds)), dtype=np.float32)
            for r, gkey in enumerate(gts):
                for c, pkey in enumerate(preds):
                    if bbox_iou(bbox(gt_items[gkey]), bbox(pred_items[pkey])) < 0.02:
                        continue
                    scores[r, c] = mask_iou(gt_items[gkey], pred_items[pkey])
            rows, cols = linear_sum_assignment(-scores)
            matched = set()
            for r, c in zip(rows, cols):
                if scores[r, c] > 0.5:
                    matched.add(c)
                    base[class_id]["tp"] += 1
                    base[class_id]["s"] += float(scores[r, c])
            for c, pkey in enumerate(preds):
                pred_box = bbox(pred_items[pkey])
                sift_score = 0.0
                for cand_box, score in candidates.get(class_id, []):
                    if bbox_iou(pred_box, cand_box) > 0.3:
                        sift_score = max(sift_score, score)
                records[class_id].append({
                    "score": sift_score,
                    "tp": c in matched,
                    "iou": float(scores[:, c].max()) if len(gts) else 0.0,
                })
            base[class_id]["fp"] += len(preds) - len(matched)
            base[class_id]["fn"] += len(gts) - len(matched)

    # 先汇总全局 base
    tp0 = sum(entry["tp"] for entry in base.values())
    fp0 = sum(entry["fp"] for entry in base.values())
    fn0 = sum(entry["fn"] for entry in base.values())
    s0 = sum(entry["s"] for entry in base.values())
    d0 = tp0 + 0.5 * (fp0 + fn0)
    print(f"base PQ {100*s0/max(d0,1):.3f}% TP {tp0} FP {fp0} FN {fn0}")
    thresholds = {}
    removed = {"tp": 0, "fp": 0, "s": 0.0}
    for class_id in sorted(classes):
        items = records[class_id]
        if not items:
            continue
        values = np.asarray([item["score"] for item in items])
        labels = np.asarray([item["tp"] for item in items])
        tp_scores = values[labels]
        fp_scores = values[~labels]
        print(f"class {class_id}: n={len(items)} TP={len(tp_scores)} "
              f"FP={len(fp_scores)} TP score mean={tp_scores.mean():.1f} "
              f"median={np.median(tp_scores) if len(tp_scores) else 0:.1f} "
              f"FP mean={fp_scores.mean():.1f} "
              f"median={np.median(fp_scores) if len(fp_scores) else 0:.1f}")
        best = None
        for threshold in sorted(set(values.tolist())):
            keep = values >= threshold
            rtp = int((labels & ~keep).sum())
            rfp = int((~labels & ~keep).sum())
            rs = float(sum(item["iou"] for item, kept in zip(items, keep)
                           if item["tp"] and not kept))
            tp = tp0 - removed["tp"] - rtp
            fp = fp0 - removed["fp"] - rfp
            fn = fn0 + removed["tp"] + rtp
            s = s0 - removed["s"] - rs
            d = tp + 0.5 * (fp + fn)
            pq = s / max(d, 1)
            if best is None or pq > best[0]:
                best = (pq, float(threshold), rtp, rfp, rs)
        pq, threshold, rtp, rfp, rs = best
        if pq * d0 > s0:
            thresholds[class_id] = threshold
            removed["tp"] += rtp
            removed["fp"] += rfp
            removed["s"] += rs
            print(f"  keep score>={threshold:.1f} removes TP {rtp} FP {rfp} "
                  f"-> PQ {100*pq:.3f}%")
    tp = tp0 - removed["tp"]
    fp = fp0 - removed["fp"]
    fn = fn0 + removed["tp"]
    s = s0 - removed["s"]
    d = tp + 0.5 * (fp + fn)
    print(f"filtered PQ {100*s/max(d,1):.3f}% TP {tp} FP {fp} FN {fn}")
    payload = {
        "base_pq": s0 / max(d0, 1),
        "filtered_pq": s / max(d, 1),
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "removed": {"tp": removed["tp"], "fp": removed["fp"]},
    }
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
