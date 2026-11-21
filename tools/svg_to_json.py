#!/usr/bin/env python3
"""把带标注的 SVG 转成赛题 12 字段 JSON（仅用于训练集/本地测试集）。

dev_final 没有 SVG，正式推理只使用 infer.py 读取 PNG。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.schema import validate_output, write_output  # noqa: E402
from cadvec.svg import svg_to_output  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, required=True, help="SVG 文件或目录")
    parser.add_argument("--out", type=Path, required=True, help="输出 JSON 目录")
    parser.add_argument("--limit", type=int, default=0, help="只转换前 N 个")
    parser.add_argument("--skip-validation", action="store_true", help="不校验输出")
    args = parser.parse_args()

    if args.svg_dir.is_file():
        files = [args.svg_dir]
    elif args.svg_dir.is_dir():
        files = sorted(args.svg_dir.rglob("*.svg"))
    else:
        raise SystemExit(f"SVG 路径不存在: {args.svg_dir}")
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit("没有可转换的 SVG")

    failures = 0
    for path in files:
        try:
            data = svg_to_output(path)
            output = args.out / f"{path.stem}.json"
            if args.skip_validation:
                # GT 允许出现 Thing + instance-id=-1（参考评估会跳过这些图元），
                # 此时不能走 write_output 的提交格式校验。
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(
                    json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                validate_output(data)
                write_output(data, output)
        except Exception as error:  # noqa: BLE001 - 批量转换汇总失败
            failures += 1
            print(f"[失败] {path}: {type(error).__name__}: {error}", file=sys.stderr)
    print(f"转换 {len(files) - failures}/{len(files)} 个 SVG -> {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
