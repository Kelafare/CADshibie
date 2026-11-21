#!/usr/bin/env python3
"""为图元图预计算节点图像特征（ResNet18 layer3，256 维）。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadvec.image_features import load_rgb, node_image_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, default=Path("../_work/primitive_graphs"))
    parser.add_argument("--image-dir", type=Path, default=Path("../datasets/train_final"))
    parser.add_argument("--out", type=Path, default=Path("../_work/node_image_features"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    graphs = sorted(args.graphs.glob("*.npz"))
    if args.limit:
        graphs = graphs[: args.limit]
    graphs = graphs[args.shard_index::max(1, args.shard_count)]
    print(f"处理 {len(graphs)} 张", file=sys.stderr)
    for index, graph_path in enumerate(graphs, 1):
        output = args.out / f"{graph_path.stem}.npy"
        if output.exists():
            continue
        image_path = args.image_dir / f"{graph_path.stem}.png"
        if not image_path.exists():
            continue
        payload = np.load(graph_path)
        centers = payload["features"][:, :2].astype(np.float64) * 100.0
        image = load_rgb(image_path)
        features = node_image_features(image, centers, device=args.device)
        np.save(output, features)
        if index % 200 == 0:
            print(f"  {index}/{len(graphs)}", file=sys.stderr)
    print("完成", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
