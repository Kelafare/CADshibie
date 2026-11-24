#!/usr/bin/env python3
"""把 SIFT 自模板候选接入 512 ROI，测试新增锚点能否转成 PQ。

只添加与 YOLO 检测/现有实例不重复的候选；每个候选用 512 ROI 细化，
按类内 NMS/质量门控决定是否输出。
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
from cadvec.layers import load_rgba  # noqa: E402
from cadvec.roi_ink import RoiInkRefiner  # noqa: E402
from cadvec.schema import assemble_output  # noqa: E402


class _P:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")


def box_iou(first, second):
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
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--sift-jsonl", required=True)
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--quality", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dedup", type=float, default=0.4)
    parser.add_argument("--limits", default="3,8,18,22,28")
    args = parser.parse_args()
    limit_classes = {int(v) for v in args.limits.split(",") if v.strip()}
    refiner = RoiInkRefiner.load(args.weights, device="cuda:0",
                                 quality_path=args.quality)
    rows = {}
    with open(args.sift_jsonl, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["stem"]] = row
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for index, (stem, row) in enumerate(rows.items()):
        png = Path(args.png_dir) / f"{stem}.png"
        base_path = Path(args.base_dir) / f"{stem}.json"
        if not png.exists() or not base_path.exists():
            continue
        data = json.loads(base_path.read_text(encoding="utf-8"))
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        height, width = image.shape[:2]
        base_masks, _ = _segments(data, (float(data.get("width", 100)),
                                         float(data.get("height", 100))), 1000)
        detections = [item for item in row["detections"]
                      if int(item["class"]) in limit_classes]
        candidates = [item for item in row["candidates"]
                      if int(item["class"]) in limit_classes]
        candidates.sort(key=lambda item: -float(item.get("score", 0)))
        added = []
        for candidate in candidates:
            class_id = int(candidate["class"])
            box = tuple(float(v) for v in candidate["box"])
            if any(int(d["class"]) == class_id and
                   box_iou(box, tuple(d["box"])) > args.dedup
                   for d in detections):
                continue
            # 候选框内的实例掩码与现有同类实例去重
            duplicate = False
            for (pred_class, _instance), pred_mask in base_masks.items():
                if pred_class != class_id:
                    continue
                local = np.zeros((1000, 1000), bool)
                x0, y0, x1, y1 = [int(v) for v in box]
                local[max(y0, 0):min(y1, 1000), max(x0, 0):min(x1, 1000)] = True
                if mask_iou(pred_mask, local) > args.dedup:
                    duplicate = True
                    break
            if duplicate:
                continue
            refined = refiner.refine(image, [(class_id, float(candidate["score"]),
                                              box)], width, height, 100, 100)
            if refined:
                added.extend(refined)
        primitives = []
        for i, (semantic, instance, command, args8, length, color, layer,
                line_width) in enumerate(zip(
                data["semanticIds"], data["instanceIds"], data["commands"],
                data["args"], data["lengths"], data["rgb"], data["layerIds"],
                data["widths"])):
            item = _P()
            item.command = int(command)
            item.points = [(float(args8[k]), float(args8[k + 1]))
                           for k in range(0, 8, 2)]
            item.length = float(length)
            item.semantic_id = int(semantic)
            item.instance_id = int(instance)
            item.rgb = [int(v) for v in color]
            item.layer_id = int(layer)
            item.width = float(line_width)
            primitives.append(item)
        next_instance = max([p.instance_id for p in primitives] + [0]) + 1
        for class_id, _confidence, segments in added:
            for segment in segments:
                item = _P()
                item.command = int(segment["command"])
                item.points = [(float(p[0]), float(p[1]))
                               for p in segment["points"]]
                item.length = float(segment["length"])
                item.semantic_id = int(class_id)
                item.instance_id = next_instance
                item.rgb = [0, 0, 0]
                item.layer_id = 0
                item.width = 1.0
                primitives.append(item)
            next_instance += 1
        output = assemble_output(primitives, width=int(data.get("width", 100)),
                                 height=int(data.get("height", 100)))
        (out / f"{stem}.json").write_text(
            json.dumps(output, ensure_ascii=False), encoding="utf-8")
        if index % 25 == 0:
            print(index, len(added), flush=True)
    print("done", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
