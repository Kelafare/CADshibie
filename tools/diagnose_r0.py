#!/usr/bin/env python3
"""R0 损失归因：在固定预测/GT 上分解 TP/FP/FN 的来源。

输出：
- 逐类 TP/FP/FN、SQ/RQ/PQ、GT 实例数、尺寸分桶；
- 忽略类别的实例定位上界（best any-class IoU>0.5）；
- 匹配对的类别一致性；
- FP/FN 的可复核原因标签（无重叠、近阈值、错类、碎裂、重复、多余笔画）。

只读预测与 GT JSON，不参与正式推理；用于决定后续主攻方向。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402

THING = 30
IGNORE = 35
RASTER = 500
MATCH = 0.5


def _entry():
    return {
        "gt": 0, "tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0,
        "oracle_hit": 0, "oracle_class_ok": 0, "oracle_iou_sum": 0.0,
        "fn_reasons": Counter(), "fp_reasons": Counter(),
        "size_buckets": Counter(), "size_hits": Counter(),
    }


def _bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(area, 1e-9)


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero(a & b))
    if not inter:
        return 0.0
    return inter / int(np.count_nonzero(a | b))


def _best(gt_mask, box, preds, pred_boxes, exclude=None):
    best = (0.0, None)
    for index, (_key, other) in enumerate(preds):
        if exclude is not None and index == exclude:
            continue
        if _bbox_iou(box, pred_boxes[index]) < 0.10:
            continue
        value = _mask_iou(gt_mask, other)
        if value > best[0]:
            best = (value, index)
    return best


def _tag_gt(gt_mask, gt_class, gt_box, gt_index, preds, pred_boxes):
    values = []
    for index, (_key, mask) in enumerate(preds):
        if _bbox_iou(gt_box, pred_boxes[index]) < 0.05:
            continue
        value = _mask_iou(gt_mask, mask)
        if value > 0.0:
            values.append((value, index))
    values.sort(reverse=True)
    if not values:
        return "no_candidate", 0.0
    best_iou, best_index = values[0]
    best_class = preds[best_index][0][0]
    if best_iou > MATCH and best_class == gt_class:
        return "duplicate_or_merge", best_iou
    if best_iou > MATCH:
        return "wrong_class", best_iou
    if best_iou >= 0.30:
        return "near_miss", best_iou
    if len(values) >= 2 and values[0][0] + values[1][0] > 0.45:
        return "fragmented", best_iou
    return "misaligned", best_iou


def _tag_pred(pred_mask, pred_class, pred_box, pred_index, gts, gt_boxes):
    values = []
    for index, ((gt_class, _gt_instance), mask) in enumerate(gts):
        if _bbox_iou(pred_box, gt_boxes[index]) < 0.05:
            continue
        value = _mask_iou(pred_mask, mask)
        if value > 0.0:
            values.append((value, index, gt_class))
    values.sort(reverse=True)
    if not values:
        return "no_gt_overlap", 0.0
    best_iou, _best_index, best_class = values[0]
    if best_iou > MATCH and best_class == pred_class:
        return "duplicate", best_iou
    if best_iou > MATCH:
        return "wrong_class", best_iou
    if best_iou >= 0.30:
        return "near_miss", best_iou
    return "extra_or_misaligned", best_iou


def process(args):
    pred_path, gt_path, raster = args
    try:
        pred = load_json(pred_path)
        gt = load_json(gt_path)
    except Exception as error:  # noqa: BLE001
        return {"error": f"{os.path.basename(pred_path)}: {error}"}
    size = (max(float(gt.get("width", 100)), 1e-6),
            max(float(gt.get("height", 100)), 1e-6))
    gt_masks, _ = _segments(gt, size, raster)
    pred_masks, _ = _segments(pred, size, raster)
    gts = [(key, mask) for key, mask in gt_masks.items()
           if key[0] < THING and key[1] >= 0]
    preds = [(key, mask) for key, mask in pred_masks.items()
             if key[0] < THING and key[1] >= 0 and key[0] != IGNORE]
    gt_boxes = [_bbox(mask) for _key, mask in gts]
    pred_boxes = [_bbox(mask) for _key, mask in preds]

    by_class_gt = defaultdict(list)
    by_class_pred = defaultdict(list)
    for index, (key, _mask) in enumerate(gts):
        by_class_gt[key[0]].append(index)
    for index, (key, _mask) in enumerate(preds):
        by_class_pred[key[0]].append(index)

    out = {
        "classes": defaultdict(_entry),
        "fn_examples": [],
        "fp_examples": [],
    }
    matched_gt = set()
    matched_pred = set()
    for class_id in sorted(set(by_class_gt) | set(by_class_pred)):
        gt_indices = by_class_gt.get(class_id, [])
        pred_indices = by_class_pred.get(class_id, [])
        entry = out["classes"][class_id]
        entry["gt"] = len(gt_indices)
        scores = np.zeros((len(gt_indices), len(pred_indices)), dtype=np.float32)
        for row, gi in enumerate(gt_indices):
            for col, pi in enumerate(pred_indices):
                if _bbox_iou(gt_boxes[gi], pred_boxes[pi]) < 0.02:
                    continue
                scores[row, col] = _mask_iou(gts[gi][1], preds[pi][1])
        if len(gt_indices) and len(pred_indices):
            rows, cols = linear_sum_assignment(-scores)
        else:
            rows, cols = [], []
        for row, col in zip(rows, cols):
            gi = gt_indices[row]
            pi = pred_indices[col]
            if scores[row, col] > MATCH:
                entry["tp"] += 1
                entry["iou_sum"] += float(scores[row, col])
                matched_gt.add(gi)
                matched_pred.add(pi)
        entry["fn"] = len(gt_indices) - entry["tp"]
        entry["fp"] = len(pred_indices) - entry["tp"]

    # 忽略类别的实例定位上界 + 类别一致性
    for gi, (key, mask) in enumerate(gts):
        entry = out["classes"][key[0]]
        best_iou, best_index = _best(mask, gt_boxes[gi], preds, pred_boxes)
        entry["oracle_iou_sum"] += best_iou
        if best_iou > MATCH:
            entry["oracle_hit"] += 1
            if preds[best_index][0][0] == key[0]:
                entry["oracle_class_ok"] += 1
        # 尺寸分桶（GT 墨迹像素数）
        pixels = int(np.count_nonzero(mask))
        bucket = "tiny" if pixels < 40 else ("small" if pixels < 120 else
                                             ("medium" if pixels < 400 else "large"))
        entry["size_buckets"][bucket] += 1
        if gi in matched_gt:
            entry["size_hits"][bucket] += 1
        elif gi < 100000:
            reason, value = _tag_gt(mask, key[0], gt_boxes[gi], gi, preds, pred_boxes)
            entry["fn_reasons"][reason] += 1
            if len(out["fn_examples"]) < 40:
                out["fn_examples"].append({
                    "image": os.path.basename(gt_path), "class": key[0],
                    "instance": key[1], "reason": reason,
                    "best_iou": round(float(value), 3), "pixels": pixels})

    for pi, (key, mask) in enumerate(preds):
        if pi in matched_pred:
            continue
        entry = out["classes"][key[0]]
        reason, value = _tag_pred(mask, key[0], pred_boxes[pi], pi, gts, gt_boxes)
        entry["fp_reasons"][reason] += 1
        if len(out["fp_examples"]) < 40:
            out["fp_examples"].append({
                "image": os.path.basename(pred_path), "class": key[0],
                "instance": key[1], "reason": reason,
                "best_iou": round(float(value), 3)})
    return out


def merge(total, part):
    if part is None:
        return
    if "error" in part:
        total["errors"].append(part["error"])
        return
    total["images"] += 1
    for class_id, entry in part["classes"].items():
        target = total["classes"][class_id]
        for key in ("gt", "tp", "fp", "fn", "oracle_hit", "oracle_class_ok"):
            target[key] += entry[key]
        for key in ("iou_sum", "oracle_iou_sum"):
            target[key] += entry[key]
        for key in ("fn_reasons", "fp_reasons", "size_buckets", "size_hits"):
            target[key].update(entry[key])
    total["fn_examples"].extend(part["fn_examples"])
    total["fp_examples"].extend(part["fp_examples"])


def finalize(total):
    rows = []
    all_reasons_fn = Counter()
    all_reasons_fp = Counter()
    for class_id, entry in sorted(total["classes"].items()):
        tp, fp, fn = entry["tp"], entry["fp"], entry["fn"]
        denominator = tp + 0.5 * (fp + fn)
        entry["PQ"] = entry["iou_sum"] / denominator if denominator else 0.0
        entry["SQ"] = entry["iou_sum"] / tp if tp else 0.0
        entry["RQ"] = tp / denominator if denominator else 0.0
        entry["oracle_recall"] = (entry["oracle_hit"] / entry["gt"]
                                  if entry["gt"] else 0.0)
        entry["oracle_class_purity"] = (entry["oracle_class_ok"] /
                                        entry["oracle_hit"]
                                        if entry["oracle_hit"] else 0.0)
        all_reasons_fn.update(entry["fn_reasons"])
        all_reasons_fp.update(entry["fp_reasons"])
        rows.append((class_id, entry))
    total["classes"] = {str(k): {
        **v, "fn_reasons": dict(v["fn_reasons"]),
        "fp_reasons": dict(v["fp_reasons"]),
        "size_buckets": dict(v["size_buckets"]),
        "size_hits": dict(v["size_hits"]),
    } for k, v in rows}
    total["fn_reasons"] = dict(all_reasons_fn)
    total["fp_reasons"] = dict(all_reasons_fp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out", default="_work/r0_report.json")
    parser.add_argument("--raster", type=int, default=RASTER)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    preds = sorted(Path(args.pred_dir).glob("*.json"))
    if args.limit:
        preds = preds[: args.limit]
    jobs = []
    for pred_path in preds:
        gt_path = Path(args.gt_dir) / pred_path.name
        if gt_path.exists():
            jobs.append((str(pred_path), str(gt_path), args.raster))
    total = {"images": 0, "errors": [], "classes": defaultdict(_entry),
             "fn_examples": [], "fp_examples": []}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(process, jobs, chunksize=4):
            merge(total, part)
    finalize(total)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(total, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"images {total['images']} errors {len(total['errors'])}")
    tp = sum(v["tp"] for v in total["classes"].values())
    fp = sum(v["fp"] for v in total["classes"].values())
    fn = sum(v["fn"] for v in total["classes"].values())
    s = sum(v["iou_sum"] for v in total["classes"].values())
    d = tp + 0.5 * (fp + fn)
    print(f"TOTAL TP {tp} FP {fp} FN {fn} PQ {100*s/max(d,1):.3f}% "
          f"SQ {100*s/max(tp,1):.3f}% RQ {100*tp/max(d,1):.3f}%")
    print("FN reasons", dict(total["fn_reasons"]))
    print("FP reasons", dict(total["fp_reasons"]))
    print(f"{'cls':>3} {'gt':>5} {'tp':>4} {'fp':>4} {'fn':>4} "
          f"{'PQ':>6} {'oracle':>6} {'purity':>6}")
    for class_id, entry in sorted(total["classes"].items(),
                                  key=lambda item: -item[1]["fn"]):
        print(f"{class_id:>3} {entry['gt']:>5} {entry['tp']:>4} "
              f"{entry['fp']:>4} {entry['fn']:>4} "
              f"{100*entry['PQ']:>5.2f}% {100*entry['oracle_recall']:>5.1f}% "
              f"{100*entry['oracle_class_purity']:>5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
