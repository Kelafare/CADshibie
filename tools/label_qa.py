#!/usr/bin/env python3
"""标签质量诊断：同一 GT 实例在不同评分栅格下的稳定性。

对每个 Thing 实例在 1000px 栅格上的掩码，降采样到 300/400px，与直接在
300/400px 栅格化的掩码比较 IoU；小实例（1000px 下 <8 像素）单独统计。
结果用于训练集过滤与按类别解释不稳定的 PQ 上界。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.evaluate import _segments, load_json  # noqa: E402

THING = 30


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    return intersection / int(np.count_nonzero(first | second))


def process(path: str):
    try:
        gt = load_json(path)
    except Exception:  # noqa: BLE001
        return {}
    size = (max(float(gt.get("width", 100)), 1e-6),
            max(float(gt.get("height", 100)), 1e-6))
    masks = {}
    for raster in (300, 400, 1000):
        masks[raster], _ = _segments(gt, size, raster)
    out = defaultdict(lambda: {"n": 0, "tiny": 0, "iou300": [], "iou400": []})
    for key, mask in masks[1000].items():
        if key[0] >= THING or key[1] < 0:
            continue
        pixels = int(np.count_nonzero(mask))
        entry = out[key[0]]
        entry["n"] += 1
        if pixels < 8:
            entry["tiny"] += 1
        for raster in (300, 400):
            direct = masks[raster].get(key)
            if direct is None:
                continue
            downsampled = cv2.resize(
                mask.astype(np.uint8), (raster, raster),
                interpolation=cv2.INTER_NEAREST).astype(bool)
            entry[f"iou{raster}"].append(_iou(direct, downsampled))
    return {str(class_id): {
        "n": entry["n"], "tiny": entry["tiny"],
        "iou300": entry["iou300"], "iou400": entry["iou400"],
    } for class_id, entry in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--out", default="weights/label_qa.json")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    paths = sorted(Path(args.gt_dir).glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    aggregates = defaultdict(lambda: {"n": 0, "tiny": 0,
                                      "iou300": [], "iou400": []})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(process, [str(path) for path in paths], chunksize=8):
            for class_id, entry in part.items():
                target = aggregates[class_id]
                target["n"] += entry["n"]
                target["tiny"] += entry["tiny"]
                target["iou300"].extend(entry["iou300"])
                target["iou400"].extend(entry["iou400"])

    payload = {"schema_version": 1, "source": "GT JSON raster stability",
               "classes": {}}
    for class_id, entry in sorted(aggregates.items(), key=lambda item: int(item[0])):
        record = {"n": entry["n"], "tiny": entry["tiny"]}
        for raster in (300, 400):
            values = np.asarray(entry[f"iou{raster}"], dtype=np.float64)
            record[f"mean_iou{raster}"] = float(values.mean()) if len(values) else 0.0
            record[f"stable{raster}"] = float((values > 0.5).mean()) if len(values) else 0.0
        payload["classes"][class_id] = record
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'cls':>3} {'n':>6} {'tiny':>5} {'mIoU300':>8} {'mIoU400':>8} "
          f"{'stable400':>9}")
    for class_id, record in payload["classes"].items():
        print(f"{class_id:>3} {record['n']:>6} {record['tiny']:>5} "
              f"{record['mean_iou300']:>8.3f} {record['mean_iou400']:>8.3f} "
              f"{record['stable400']:>8.1%}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
