#!/usr/bin/env python3
"""类别级集成：主引擎保留除目标类以外的图元；目标类从多个引擎取并集，
按实例掩码 IoU 去重（保留先出现的引擎实例）。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from cadvec.schema import assemble_output, validate_output, write_output  # noqa: E402

RASTER = 400


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, data, index, instance_offset=0):
        self.command = int(data["commands"][index])
        points = data["args"][index]
        self.points = [(float(points[k]), float(points[k + 1])) for k in range(0, 8, 2)]
        self.length = float(data["lengths"][index])
        self.semantic_id = int(data["semanticIds"][index])
        self.instance_id = int(data["instanceIds"][index])
        if self.instance_id >= 0:
            self.instance_id += instance_offset
        self.rgb = [int(v) for v in data["rgb"][index]]
        self.layer_id = int(data["layerIds"][index])
        self.width = float(data["widths"][index])


def _instance_masks(data, class_id):
    """返回 {instance_id: (mask, [indices])}。"""
    groups = defaultdict(list)
    for index in range(len(data["commands"])):
        if int(data["semanticIds"][index]) == class_id:
            groups[int(data["instanceIds"][index])].append(index)
    result = {}
    for instance, indices in groups.items():
        mask = np.zeros((RASTER, RASTER), np.uint8)
        for index in indices:
            points = np.round(np.asarray(data["args"][index]) * (RASTER / 100.0)).astype(np.int32)
            cv2.polylines(mask, [points.reshape(-1, 1, 2)], False, 1, thickness=1)
        result[instance] = (mask.astype(bool), indices)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondaries", type=Path, nargs="+", required=True)
    parser.add_argument("--class-id", type=int, default=8)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in args.primary.glob("*.json"))
    for name in names:
        primary = json.loads((args.primary / name).read_text(encoding="utf-8"))
        kept = []  # (mask, primitives)
        for instance, (mask, indices) in _instance_masks(primary, args.class_id).items():
            kept.append((mask, [_Primitive(primary, index) for index in indices]))
        for offset, secondary_dir in enumerate(args.secondaries, 1):
            path = secondary_dir / name
            if not path.exists():
                continue
            secondary = json.loads(path.read_text(encoding="utf-8"))
            for instance, (mask, indices) in _instance_masks(secondary, args.class_id).items():
                duplicate = False
                for existing_mask, _ in kept:
                    intersection = int(np.count_nonzero(mask & existing_mask))
                    if not intersection:
                        continue
                    union = int(np.count_nonzero(mask | existing_mask))
                    if intersection / max(union, 1) > args.iou_threshold:
                        duplicate = True
                        break
                if not duplicate:
                    kept.append((mask, [
                        _Primitive(secondary, index, instance_offset=offset * 1_000_000)
                        for index in indices
                    ]))
        primitives = [
            _Primitive(primary, index)
            for index in range(len(primary["commands"]))
            if int(primary["semanticIds"][index]) != args.class_id
        ]
        for _, items in kept:
            primitives.extend(items)
        data = assemble_output(primitives, width=int(primary.get("width", 100)),
                               height=int(primary.get("height", 100)))
        validate_output(data)
        write_output(data, args.out / name)
    print(f"集成 {len(names)} 张 → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
