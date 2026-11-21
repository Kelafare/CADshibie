#!/usr/bin/env python3
"""Oracle 实验 O2/O3：已知 GT bbox（不提供位姿/图元/模板 ID），模板拟合。

用途：在训练 YOLO 之前验证“模板系统能否从不精确的位置恢复实例”。
- O2：使用 GT 类别作为模板检索条件（class_filter=GT 类别）；
- O3：不使用类别（class_filter=None），同时检验类别识别。

输出官方 12 字段 JSON，并给出实例级召回、模板命中 IoU、PQ。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

from cadvec.config import InferenceConfig, STUFF_CLASS_MAX, THING_CLASS_MAX  # noqa: E402
from cadvec.dataset import component_scene  # noqa: E402
from cadvec.layers import load_palette  # noqa: E402
from cadvec.schema import assemble_output, validate_output, write_output  # noqa: E402
from cadvec.templating import TemplateLibrary  # noqa: E402


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, payload, instance_id):
        self.command = int(payload["command"])
        self.points = payload["points"]
        self.length = float(payload["length"])
        self.semantic_id = int(payload["semantic"])
        self.instance_id = int(instance_id)
        self.rgb = [int(v) for v in payload["rgb"]]
        self.layer_id = int(payload["layer"])
        self.width = float(payload["width"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-json", type=Path, default=Path("../eval_run/gt_json"))
    parser.add_argument("--png-dir", type=Path, default=Path("../datasets/test_final"))
    parser.add_argument("--templates", type=Path, default=Path("../_work/symbol_templates.pkl"))
    parser.add_argument("--palette", type=Path, default=Path("weights/stroke_palette.json"))
    parser.add_argument("--out", type=Path, default=Path("../_work/oracle_bbox"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--class-mode", choices=("gt", "free"), default="gt")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-score", type=float, default=4.0)
    args = parser.parse_args()

    library = TemplateLibrary.load(args.templates)
    if library is None:
        print("缺少模板库", file=sys.stderr)
        return 1
    palette = load_palette(args.palette)
    config = InferenceConfig()
    files = sorted(args.gt_json.glob("*.json"))[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    total_instances = 0
    emitted = 0
    for index, gt_path in enumerate(files, 1):
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        png = args.png_dir / f"{gt_path.stem}.png"
        if not png.exists():
            continue
        scene = component_scene(png, palette, config)
        if not scene.layers:
            continue
        stroke_mask = np.zeros_like(scene.layers[0].mask, dtype=bool)
        for layer in scene.layers:
            stroke_mask |= layer.mask
        distance_transform = ndimage.distance_transform_edt(~stroke_mask)
        height, width = stroke_mask.shape
        canvas_to_pixel = width / 100.0

        # GT 实例框（只取 boxes 字段，不读实例图元与位姿）。
        instances = []
        for row in gt.get("boxes", []):
            x0, y0, x1, y1, class_id = row
            class_id = int(class_id)
            if class_id > THING_CLASS_MAX:
                continue
            instances.append((float(x0), float(y0), float(x1), float(y1), class_id))
        total_instances += len(instances)

        primitives = []
        instance_id = 0
        for (x0, y0, x1, y1, class_id) in instances:
            center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            scale = max(x1 - x0, y1 - y0)
            variants = []
            for mode in ("intersect", "inside", "midpoint"):
                points = []
                local_mask = np.zeros_like(stroke_mask, dtype=np.uint8)
                for stroke in scene.strokes:
                    sx0, sy0, sx1, sy1 = stroke.bbox()
                    area = max((sx1 - sx0) * (sy1 - sy0), 1e-6)
                    ix = max(0.0, min(sx1, x1) - max(sx0, x0))
                    iy = max(0.0, min(sy1, y1) - max(sy0, y0))
                    fraction = ix * iy / area
                    cx, cy = (sx0 + sx1) / 2.0, (sy0 + sy1) / 2.0
                    if mode == "intersect":
                        use = fraction > 0.0
                    elif mode == "inside":
                        use = fraction >= 0.35
                    else:
                        use = x0 - 0.5 <= cx <= x1 + 0.5 and y0 - 0.5 <= cy <= y1 + 0.5
                    if not use:
                        continue
                    points.extend(stroke.points)
                    pixels = np.round(np.asarray(stroke.points) * canvas_to_pixel).astype(np.int32)
                    cv2.polylines(local_mask, [pixels.reshape(-1, 1, 2)], False, 1,
                                  thickness=2)
                if len(points) >= 8:
                    variants.append((np.asarray(points, dtype=np.float64),
                                     local_mask.astype(bool)))
            best_fit = None
            for points, local_mask in variants:
                fit = library.fit(
                    points, distance_transform, canvas_to_pixel,
                    top_k=args.top_k,
                    class_filter=class_id if args.class_mode == "gt" else None,
                    center=center, scale_override=scale,
                    stroke_mask=local_mask,
                )
                if fit is not None and fit.class_id <= THING_CLASS_MAX:
                    if best_fit is None or fit.score < best_fit.score:
                        best_fit = fit
            if best_fit is None or best_fit.score > args.max_score:
                continue
            for payload in library.instantiate(best_fit, np.zeros((0, 2))):
                primitives.append(_Primitive(payload, instance_id))
            instance_id += 1
            emitted += 1
        data = assemble_output(primitives, width=int(gt.get("width", 100)),
                               height=int(gt.get("height", 100)))
        validate_output(data)
        write_output(data, args.out / gt_path.name)
        if index % 20 == 0:
            print(f"  {index}/{len(files)} 实例 {emitted}/{total_instances}",
                  file=sys.stderr)
    print(f"完成：{emitted}/{total_instances} 个 GT 实例尝试输出 → {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
