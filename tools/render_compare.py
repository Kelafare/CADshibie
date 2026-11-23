#!/usr/bin/env python3
"""把原图、预测 JSON、GT JSON 渲染成 2×2 对比图。

用法：
  python tools/render_compare.py --png a.png --pred a.json --gt gt/a.json --out a.png
颜色按语义类别分配；Overlay 面板红色=预测，蓝色=GT，灰色=原图墨迹。
"""
from __future__ import annotations

import argparse
import colorsys
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

IGNORE = 35


def class_color(class_id: int):
    if class_id == IGNORE:
        return (190, 190, 190)
    if 30 <= class_id <= 34:
        return (120, 140, 160)
    hue = (class_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.78, 0.82)
    return int(r * 255), int(g * 255), int(b * 255)


def load_image(path: str) -> Image.Image:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, image).convert("RGB")


def draw_data(draw: ImageDraw.ImageDraw, data: dict, width: int, height: int,
              thickness: int = 2, label: bool = False,
              color_override=None):
    canvas_w = max(float(data.get("width", 100)), 1e-6)
    canvas_h = max(float(data.get("height", 100)), 1e-6)
    sx, sy = width / canvas_w, height / canvas_h
    groups = defaultdict(list)
    for index, (semantic, instance, command, args) in enumerate(zip(
            data["semanticIds"], data["instanceIds"],
            data["commands"], data["args"])):
        points = [(float(args[i]) * sx, float(args[i + 1]) * sy)
                  for i in range(0, 8, 2)]
        color = (class_color(int(semantic)) if color_override is None
                 else color_override)
        if command in (0, 1):
            draw.line(points, fill=color, width=thickness, joint="curve")
        else:
            xs, ys = zip(*points)
            draw.ellipse((min(xs), min(ys), max(xs), max(ys)),
                         outline=color, width=thickness)
        groups[(int(semantic), int(instance))].append(points)
    if label:
        font = ImageFont.load_default()
        for (semantic, instance), polylines in groups.items():
            if semantic == IGNORE:
                continue
            xs = [x for points in polylines for x, _ in points]
            ys = [y for points in polylines for _, y in points]
            if not xs:
                continue
            text = f"c{semantic}" + (f"#{instance}" if instance >= 0 else "")
            draw.text((min(xs), min(ys) - 10), text, fill=(20, 20, 20), font=font)


def panel(image: Image.Image | None, data: dict | None, size: int, title: str,
          thickness: int = 2, label: bool = True) -> Image.Image:
    tile = Image.new("RGB", (size, size), (255, 255, 255))
    if image is not None:
        tile = image.resize((size, size), Image.Resampling.LANCZOS)
    if data is not None:
        draw = ImageDraw.Draw(tile)
        draw_data(draw, data, size, size, thickness=thickness, label=label)
    canvas = Image.new("RGB", (size, size + 28), (255, 255, 255))
    canvas.paste(tile, (0, 28))
    ImageDraw.Draw(canvas).text((8, 8), title, fill=(0, 0, 0),
                                font=ImageFont.load_default())
    return canvas


def overlay(image: Image.Image, pred: dict, gt: dict | None, size: int) -> Image.Image:
    tile = image.resize((size, size), Image.Resampling.LANCZOS)
    tile = Image.blend(tile, Image.new("RGB", tile.size, (255, 255, 255)), 0.65)
    if gt is not None:
        gt_mask = Image.new("L", (size, size), 0)
        draw_data(ImageDraw.Draw(gt_mask), gt, size, size, thickness=2,
                  label=False, color_override=255)
        tile.paste(Image.new("RGB", (size, size), (30, 90, 220)), mask=gt_mask)
    pred_mask = Image.new("L", (size, size), 0)
    draw_data(ImageDraw.Draw(pred_mask), pred, size, size, thickness=2,
              label=False, color_override=255)
    tile.paste(Image.new("RGB", (size, size), (220, 40, 40)), mask=pred_mask)
    canvas = Image.new("RGB", (size, size + 28), (255, 255, 255))
    canvas.paste(tile, (0, 28))
    ImageDraw.Draw(canvas).text((8, 8), "Overlay: red=pred, blue=GT",
                                fill=(0, 0, 0), font=ImageFont.load_default())
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=900)
    args = parser.parse_args()
    image = load_image(args.png)
    pred = json.loads(Path(args.pred).read_text(encoding="utf-8"))
    gt = (json.loads(Path(args.gt).read_text(encoding="utf-8"))
          if args.gt else None)
    panels = [
        panel(image, None, args.size, "Input"),
        panel(None, pred, args.size, "Prediction"),
        panel(None, gt, args.size, "Ground truth") if gt else
        panel(None, pred, args.size, "Prediction (no GT)"),
        overlay(image, pred, gt, args.size),
    ]
    gap = 12
    width = args.size * 2 + gap
    height = (args.size + 28) * 2 + gap
    sheet = Image.new("RGB", (width, height), (235, 235, 235))
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (args.size + gap, 0))
    sheet.paste(panels[2], (0, args.size + 28 + gap))
    sheet.paste(panels[3], (args.size + gap, args.size + 28 + gap))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
