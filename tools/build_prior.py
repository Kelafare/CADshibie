#!/usr/bin/env python3
"""从训练集 SVG 标签统计「描边颜色 -> 语义类别」先验。

只使用训练/本地测试数据的 SVG 真值；dev_final 只有 PNG，不参与统计。
生成的 ``weights/color_prior.json`` 会在推理时按颜色最近邻分类。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.config import IGNORE_CLASS, THING_CLASS_MAX  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402


def build_prior(svg_dir: Path) -> dict:
    color_counts: dict[tuple[int, int, int], Counter] = defaultdict(Counter)
    class_counts: Counter = Counter()
    files = sorted(svg_dir.rglob("*.svg"))
    if not files:
        raise SystemExit(f"{svg_dir} 中没有 SVG")
    for path in files:
        try:
            primitives = list(iter_primitives(path))
        except Exception as error:  # noqa: BLE001 - 单个 SVG 损坏时跳过并提示
            print(f"[跳过] {path}: {error}", file=sys.stderr)
            continue
        for primitive in primitives:
            color = tuple(int(channel) for channel in primitive.rgb)
            semantic = primitive.semantic_id
            # 与评估口径对齐：Thing（0-29）若没有实例号，评测时会整条跳过，
            # 不应作为颜色先验的正类样本，否则黑色墙线会被学成 class 0。
            if semantic <= THING_CLASS_MAX and primitive.instance_id < 0:
                semantic = IGNORE_CLASS
            color_counts[color][semantic] += 1
            class_counts[semantic] += 1

    total = sum(class_counts.values())
    colors = []
    for color, counter in sorted(
        color_counts.items(), key=lambda item: sum(item[1].values()), reverse=True,
    ):
        # 主导类别只在“已标注”类别 0-34 中选：类别 35（未标注/背景）在图元
        # 数量上通常占绝对多数，若直接取众数会让几乎所有颜色都落到 35，
        # 导致预测只剩忽略类、PQ 恒为 0。
        labeled = {key: value for key, value in counter.items() if 0 <= key <= 34}
        if labeled:
            semantic, count = max(labeled.items(), key=lambda item: item[1])
            labeled_total = sum(labeled.values())
            confidence = round(count / max(labeled_total, 1), 6)
        else:
            semantic, count, confidence = 35, sum(counter.values()), 1.0
        colors.append({
            "rgb": list(color),
            "semantic": int(semantic),
            "count": int(sum(counter.values())),
            "confidence": confidence,
            "counts": {str(key): int(value) for key, value in counter.items()},
        })
    return {
        "schema_version": 1,
        "source": str(svg_dir),
        "svg_files": len(files),
        "primitive_count": int(total),
        "class_frequencies": {
            str(key): round(value / max(total, 1), 8)
            for key, value in sorted(class_counts.items())
        },
        "colors": colors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, required=True, help="训练/本地测试 SVG 目录")
    parser.add_argument(
        "--out", type=Path, default=Path("weights/color_prior.json"),
        help="输出先验 JSON",
    )
    args = parser.parse_args()
    payload = build_prior(args.svg_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(
        f"已写入 {args.out}: {payload['primitive_count']} 个图元，"
        f"{len(payload['colors'])} 种颜色，{payload['svg_files']} 个 SVG"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
