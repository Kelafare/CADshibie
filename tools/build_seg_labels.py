#!/usr/bin/env python3
"""生成 dense segmentation 的像素级标签 PNG（1000×1000，255=背景）。"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from cadvec.segmentation import rasterize_label_map  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402

_STATE: dict = {}


def _init(out_dir: str, size: int) -> None:
    _STATE["out"] = Path(out_dir)
    _STATE["size"] = size


def _process(job: tuple[str, str]) -> str:
    svg_path, name = job
    output = _STATE["out"] / f"{name}.png"
    if output.exists():
        return name
    primitives = list(iter_primitives(svg_path))
    label = rasterize_label_map(primitives, 100, 100, _STATE["size"])
    Image.fromarray(label, mode="L").save(output)
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--out", type=Path, default=Path("../_work/seg_labels"))
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    files = sorted(args.svg_dir.glob("*.svg"))
    if args.limit:
        files = files[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"生成 {len(files)} 张标签", file=sys.stderr)
    with Pool(args.workers, initializer=_init,
              initargs=(str(args.out), args.size)) as pool:
        for index, _ in enumerate(pool.imap_unordered(
                _process, [(str(path), path.stem) for path in files], chunksize=4), 1):
            if index % 200 == 0:
                print(f"  {index}/{len(files)}", file=sys.stderr)
    print(f"完成：{args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
