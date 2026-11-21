#!/usr/bin/env python3
"""构建对象假设训练集：PNG 几何流水线 + SVG 真值匹配 → 每图 JSON。"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadvec.config import InferenceConfig  # noqa: E402
from cadvec.dataset import (  # noqa: E402
    component_scene,
    gt_samples,
    gt_segment_masks,
    label_hypotheses,
)
from cadvec.layers import load_palette  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402

_STATE: dict = {}


def _init(palette_path: str, config_kwargs: dict) -> None:
    _STATE["palette"] = load_palette(palette_path)
    _STATE["config"] = InferenceConfig(**config_kwargs)


def _process(job: tuple[str, str, str]) -> dict:
    png_path, svg_path, out_dir = job
    name = Path(png_path).stem
    output = Path(out_dir) / f"{name}.json"
    if output.exists():
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
            return {"name": name, "skipped": True, "count": len(payload["samples"])}
        except Exception:  # noqa: BLE001
            pass
    started = time.perf_counter()
    scene = component_scene(png_path, _STATE["palette"], _STATE["config"])
    primitives = list(iter_primitives(svg_path))
    masks = gt_segment_masks(primitives, 100, 100)
    samples = gt_samples(primitives) + label_hypotheses(scene, masks)
    # 假设按总长度排序，只保留较大的 120 个，控制样本量。
    hypotheses_samples = [item for item in samples if item.get("kind") == "hyp"]
    hypotheses_samples.sort(key=lambda item: -item["length"])
    samples = [item for item in samples if item.get("kind") == "gt"] + hypotheses_samples[:120]
    payload = {
        "name": name,
        "width": 100,
        "height": 100,
        "hypotheses": len(scene.hypotheses),
        "strokes": len(scene.strokes),
        "seconds": round(time.perf_counter() - started, 3),
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return {"name": name, "count": len(samples), "seconds": payload["seconds"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png-dir", type=Path, required=True)
    parser.add_argument("--svg-dir", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=Path("weights/stroke_palette.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pngs = sorted(args.png_dir.glob("*.png"))
    if args.limit:
        pngs = pngs[: args.limit]
    jobs = [
        (str(png), str(args.svg_dir / f"{png.stem}.svg"), str(args.out))
        for png in pngs if (args.svg_dir / f"{png.stem}.svg").exists()
    ]
    print(f"处理 {len(jobs)} 张（workers={args.workers}）", file=sys.stderr)
    started = time.perf_counter()
    counts: Counter = Counter()
    total = 0
    if args.workers <= 1:
        _init(str(args.palette), {})
        for index, job in enumerate(jobs, start=1):
            result = _process(job)
            total += result["count"]
            if index % 20 == 0:
                print(f"  {index}/{len(jobs)} 样本 {total}", file=sys.stderr)
    else:
        with Pool(args.workers, initializer=_init,
                  initargs=(str(args.palette), {})) as pool:
            for index, result in enumerate(pool.imap_unordered(_process, jobs, chunksize=1), start=1):
                total += result["count"]
                if index % 50 == 0:
                    print(f"  {index}/{len(jobs)} 样本 {total}", file=sys.stderr)
    print(f"完成：{len(jobs)} 图，{total} 个假设样本，"
          f"{time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
