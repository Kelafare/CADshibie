#!/usr/bin/env python3
"""生成合成 CAD 平面图（PNG + SVG + GT JSON），用于离线冒烟测试。

注意：合成数据只用于验证代码链路，不代表真实赛题分布。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.svg import svg_to_output  # noqa: E402

# 固定的颜色 -> 语义映射；真实数据请用 tools/build_prior.py 从训练集统计。
PALETTE = [
    ([220, 30, 30], 0),
    ([30, 90, 220], 1),
    ([30, 150, 60], 2),
    ([180, 60, 190], 3),
    ([230, 140, 20], 4),
    ([20, 160, 170], 5),
]


@dataclass
class SceneShape:
    kind: str
    params: dict[str, float]
    color: list[int]
    semantic: int
    instance: int

    def svg_element(self) -> str:
        r, g, b = self.color
        stroke = f"#{r:02x}{g:02x}{b:02x}"
        common = (
            f'stroke="{stroke}" fill="none" stroke-width="0.8" '
            f'semantic-id="{self.semantic + 1}" instance-id="{self.instance}"'
        )
        if self.kind == "circle":
            return (
                f'<circle cx="{self.params["cx"]:.3f}" cy="{self.params["cy"]:.3f}" '
                f'r="{self.params["r"]:.3f}" {common}/>'
            )
        if self.kind == "ellipse":
            return (
                f'<ellipse cx="{self.params["cx"]:.3f}" cy="{self.params["cy"]:.3f}" '
                f'rx="{self.params["rx"]:.3f}" ry="{self.params["ry"]:.3f}" {common}/>'
            )
        if self.kind == "line":
            return (
                f'<path d="M {self.params["x1"]:.3f} {self.params["y1"]:.3f} '
                f'L {self.params["x2"]:.3f} {self.params["y2"]:.3f}" {common}/>'
            )
        if self.kind == "arc":
            return (
                f'<path d="M {self.params["x1"]:.3f} {self.params["y1"]:.3f} '
                f'A {self.params["r"]:.3f} {self.params["r"]:.3f} 0 0 1 '
                f'{self.params["x2"]:.3f} {self.params["y2"]:.3f}" {common}/>'
            )
        raise ValueError(self.kind)

    def draw(self, draw: ImageDraw.ImageDraw, scale: float) -> None:
        color = tuple(self.color)
        width = max(1, int(round(0.8 * scale)))
        if self.kind == "circle":
            cx, cy, radius = self.params["cx"] * scale, self.params["cy"] * scale, self.params["r"] * scale
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                         outline=color, width=width)
        elif self.kind == "ellipse":
            cx, cy = self.params["cx"] * scale, self.params["cy"] * scale
            rx, ry = self.params["rx"] * scale, self.params["ry"] * scale
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=color, width=width)
        elif self.kind == "line":
            draw.line((
                self.params["x1"] * scale, self.params["y1"] * scale,
                self.params["x2"] * scale, self.params["y2"] * scale,
            ), fill=color, width=width)
        elif self.kind == "arc":
            # 与 SVG 的 A 指令一致：PIL 的角度从 3 点钟方向起、屏幕坐标顺时针增加。
            cx, cy = self.params["cx"], self.params["cy"]
            radius = self.params["r"]
            start_angle = self.params["start_angle"]
            end_angle = self.params["end_angle"]
            draw.arc(
                ((cx - radius) * scale, (cy - radius) * scale,
                 (cx + radius) * scale, (cy + radius) * scale),
                start_angle, end_angle, fill=color, width=width,
            )


def make_scene(rng: random.Random, shape_count: int) -> list[SceneShape]:
    shapes: list[SceneShape] = []
    used_instances = 0
    for _ in range(shape_count):
        color, semantic = rng.choice(PALETTE)
        kind = rng.choice(["circle", "ellipse", "line", "arc", "rect"])
        cx = rng.uniform(18, 82)
        cy = rng.uniform(18, 82)
        if kind == "circle":
            params = {"cx": cx, "cy": cy, "r": rng.uniform(5, 12)}
            shapes.append(SceneShape("circle", params, color, semantic, used_instances))
        elif kind == "ellipse":
            rx, ry = rng.uniform(7, 16), rng.uniform(4, 9)
            if rx < ry:
                rx, ry = ry, rx
            params = {"cx": cx, "cy": cy, "rx": rx, "ry": ry}
            shapes.append(SceneShape("ellipse", params, color, semantic, used_instances))
        elif kind == "line":
            angle = rng.uniform(0, math.pi * 2)
            length = rng.uniform(12, 25)
            x2 = min(98.0, max(2.0, cx + length * math.cos(angle)))
            y2 = min(98.0, max(2.0, cy + length * math.sin(angle)))
            params = {"x1": cx, "y1": cy, "x2": x2, "y2": y2}
            shapes.append(SceneShape("line", params, color, semantic, used_instances))
        elif kind == "arc":
            radius = rng.uniform(7, 14)
            start_angle = rng.uniform(140, 220)
            end_angle = start_angle + rng.uniform(80, 150)
            params = {
                "cx": cx, "cy": cy, "r": radius,
                "start_angle": start_angle, "end_angle": end_angle,
                "x1": cx + radius * math.cos(math.radians(start_angle)),
                "y1": cy + radius * math.sin(math.radians(start_angle)),
                "x2": cx + radius * math.cos(math.radians(end_angle)),
                "y2": cy + radius * math.sin(math.radians(end_angle)),
            }
            shapes.append(SceneShape("arc", params, color, semantic, used_instances))
        else:
            # 矩形用 4 条独立线段表示，共享同一个实例号。
            width, height = rng.uniform(10, 24), rng.uniform(8, 20)
            x1, y1, x2, y2 = cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            for index in range(4):
                (ax, ay), (bx, by) = corners[index], corners[(index + 1) % 4]
                shapes.append(SceneShape(
                    "line", {"x1": ax, "y1": ay, "x2": bx, "y2": by},
                    color, semantic, used_instances,
                ))
        used_instances += 1
    return shapes


def build_svg(shapes: list[SceneShape], width: int, height: int) -> str:
    body = "\n  ".join(shape.svg_element() for shape in shapes)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">\n'
        f"  {body}\n</svg>\n"
    )


def build_png(shapes: list[SceneShape], canvas: int, path: Path) -> None:
    image = Image.new("RGB", (canvas, canvas), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    scale = canvas / 100.0
    for shape in shapes:
        shape.draw(draw, scale)
    image.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("synthetic"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--shapes", type=int, default=5)
    parser.add_argument("--canvas", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    for index in range(args.count):
        stem = f"syn_{index:04d}"
        shapes = make_scene(rng, args.shapes)
        build_png(shapes, args.canvas, args.out / f"{stem}.png")
        svg_path = args.out / f"{stem}.svg"
        svg_path.write_text(build_svg(shapes, 100, 100), encoding="utf-8")
        gt = svg_to_output(svg_path)
        (args.out / f"{stem}.json").write_text(
            json.dumps(gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
    print(f"已生成 {args.count} 组合成数据到 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
