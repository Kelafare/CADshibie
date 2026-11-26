#!/usr/bin/env python3
"""B-lite 原型：用 YOLO 框作锚点，对现有预测的原子做重新归属。

原子取自现有 hybrid JSON；每个锚点按“点落入框内比例 + 类别一致”竞争原子。
未分配给锚点的原子保留原实例。只改归属，不改几何。
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cadvec.layers import load_rgba  # noqa: E402
from cadvec.schema import assemble_output, validate_output  # noqa: E402


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")


def _inside_ratio(args, box, width, height, samples=8):
    x0, y0, x1, y1 = box
    points = [(args[i], args[i + 1]) for i in range(0, 8, 2)]
    hits = total = 0
    for index in range(3):
        ax, ay = points[index]
        bx, by = points[index + 1]
        for step in range(samples):
            ratio = step / samples
            x = (ax + (bx - ax) * ratio) / 100.0 * width
            y = (ay + (by - ay) * ratio) / 100.0 * height
            total += 1
            if x0 <= x <= x1 and y0 <= y <= y1:
                hits += 1
    return hits / max(total, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--yolo", default="weights/yolo_seg.pt")
    parser.add_argument("--classes", default="3,8,18,22,28",
                        help="只对这些类别做锚点重归属")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--inside", type=float, default=0.6)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.yolo)
    classes = {int(v) for v in args.classes.split(",") if v.strip()}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pngs = sorted(Path(args.png_dir).glob("*.png"))
    for index, png in enumerate(pngs):
        pred_path = Path(args.pred_dir) / f"{png.stem}.json"
        if not pred_path.exists():
            continue
        import json
        data = json.loads(pred_path.read_text(encoding="utf-8"))
        rgb, alpha = load_rgba(str(png))
        image = np.clip(rgb * alpha[..., None] + 255.0 * (1.0 - alpha[..., None]),
                        0, 255).astype(np.uint8)
        height, width = image.shape[:2]
        result = model.predict(source=str(png), imgsz=640, conf=args.conf,
                               iou=0.5, verbose=False, device=args.device)[0]
        anchors = []
        if result.boxes is not None and len(result.boxes):
            for class_id, confidence, box in zip(
                    result.boxes.cls.cpu().numpy().astype(int),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.xyxy.cpu().numpy()):
                if int(class_id) in classes:
                    anchors.append((int(class_id), float(confidence),
                                    tuple(float(v) for v in box)))
        # 按原实例整体竞争锚点：避免只把部分原子分给锚点而产生重复实例。
        groups = defaultdict(list)
        for item_index, (semantic, instance) in enumerate(zip(
                data["semanticIds"], data["instanceIds"])):
            groups[(int(semantic), int(instance))].append(item_index)
        atom_anchor = [None] * len(data["semanticIds"])
        for (semantic, _instance), indices in groups.items():
            if semantic not in classes:
                continue
            best = (0.0, None)
            for anchor_index, (class_id, confidence, box) in enumerate(anchors):
                if class_id != semantic:
                    continue
                ratios = [_inside_ratio(data["args"][index], box, width, height)
                          for index in indices]
                ratio = float(np.mean(ratios))
                score = ratio + 0.1 * confidence
                if ratio >= args.inside and score > best[0]:
                    best = (score, anchor_index)
            for index in indices:
                atom_anchor[index] = best[1]

        primitives = []
        for item_index, (semantic, instance, command, args8, length, rgb,
                         layer, line_width) in enumerate(zip(
                data["semanticIds"], data["instanceIds"], data["commands"],
                data["args"], data["lengths"], data["rgb"], data["layerIds"],
                data["widths"])):
            item = _Primitive()
            item.command = int(command)
            item.points = [(float(args8[k]), float(args8[k + 1]))
                           for k in range(0, 8, 2)]
            item.length = float(length)
            item.semantic_id = int(semantic)
            anchor_index = atom_anchor[item_index]
            if anchor_index is not None:
                class_id, _confidence, _box = anchors[anchor_index]
                item.semantic_id = int(class_id)
                item.instance_id = 1_000_000 + anchor_index
            else:
                item.instance_id = int(instance)
            item.rgb = [int(v) for v in rgb]
            item.layer_id = int(layer)
            item.width = float(line_width)
            primitives.append(item)
        output = assemble_output(primitives, width=int(data.get("width", 100)),
                                 height=int(data.get("height", 100)))
        validate_output(output)
        (out / f"{png.stem}.json").write_text(
            json.dumps(output, ensure_ascii=False), encoding="utf-8")
        if index % 30 == 0:
            print(index, len(anchors), flush=True)
    print("done", len(pngs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
