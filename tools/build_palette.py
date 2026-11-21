#!/usr/bin/env python3
"""从训练集 SVG 提取精确描边色，生成颜色调色板（v2 颜色恢复用）。"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COLOR_RE = re.compile(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)")


def collect(svg_dir: Path, max_files: int = 0) -> tuple[list[list[int]], dict]:
    counts: Counter = Counter()
    user_files = 0
    files = sorted(svg_dir.rglob("*.svg"))
    if max_files:
        files = files[:max_files]
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        user_files += 1
        for element in root.iter():
            if element.tag.split("}")[-1] not in {"path", "circle", "ellipse"}:
                continue
            match = COLOR_RE.match((element.get("stroke") or "").strip())
            if not match:
                continue
            color = tuple(int(float(value)) for value in match.groups())
            if 0 <= color[0] <= 255 and color[1] < 256 and color[2] < 256:
                counts[color] += 1
    colors = [list(color) for color, _ in counts.most_common()]
    return colors, {"files": user_files, "colors": len(colors), "strokes": sum(counts.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("weights/stroke_palette.json"))
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()
    colors, stats = collect(args.svg_dir, args.max_files)
    if not colors:
        print("没有找到颜色", file=sys.stderr)
        return 1
    payload = {"schema_version": 2, "stats": stats, "colors": colors}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"写入 {args.out}: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
