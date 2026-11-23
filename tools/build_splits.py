#!/usr/bin/env python3
"""按感知哈希分组的数据划分，避免同源/近似重复图纸跨划分泄漏。

对每张图计算 64 位 dHash；Hamming 距离 <= threshold 的图并为一组；整组
分配到 train/val（train_final）或 calib/hold/final（test_final）。final
只用于最终报告，不参与任何模型、阈值或类别决策。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image


def dhash(path: Path, size: int = 8) -> int:
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image = Image.alpha_composite(background, image).convert("L")
    image = image.resize((size + 1, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.int16)
    bits = (array[:, 1:] > array[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def group_hashes(hashes: dict[str, int], threshold: int) -> dict[str, int]:
    names = sorted(hashes)
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(first, second):
        root_a, root_b = find(first), find(second)
        if root_a != root_b:
            parent[root_b] = root_a

    for index, first in enumerate(names):
        for second in names[index + 1:]:
            if bin(hashes[first] ^ hashes[second]).count("1") <= threshold:
                union(first, second)
    return {name: find(name) for name in names}


def assign(groups: dict[str, int], ratios: list[tuple[str, float]], seed: int):
    by_group: dict[int, list[str]] = {}
    for name, group in groups.items():
        by_group.setdefault(group, []).append(name)
    members = sorted(by_group.values(), key=len, reverse=True)
    rng = random.Random(seed)
    rng.shuffle(members)
    total = sum(len(item) for item in members)
    targets = [(role, ratio * total) for role, ratio in ratios]
    counts = {role: 0 for role, _ in ratios}
    out = {role: [] for role, _ in ratios}
    for group in members:
        role = min(
            (item for item in targets),
            key=lambda item: (counts[item[0]] + len(group)) / max(item[1], 1e-9),
        )[0]
        counts[role] += len(group)
        out[role].extend(group)
    for role in out:
        out[role].sort()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--out", default="weights/splits.json")
    parser.add_argument("--threshold", type=int, default=4,
                        help="dHash Hamming 距离阈值（<= 视为同组）")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    def scan(directory: str) -> dict[str, int]:
        paths = sorted(Path(directory).glob("*.png"))
        return {path.stem: dhash(path) for path in paths}

    train_hashes = scan(args.train_dir)
    test_hashes = scan(args.test_dir)
    train_groups = group_hashes(train_hashes, args.threshold)
    test_groups = group_hashes(test_hashes, args.threshold)
    train_split = assign(train_groups, [("train", 0.85), ("val", 0.15)], args.seed)
    test_split = assign(test_groups, [("calib", 0.30), ("hold", 0.30),
                                      ("final", 0.40)], args.seed)
    payload = {
        "schema_version": 1,
        "method": "64-bit dHash, connected components at Hamming<=threshold",
        "threshold": args.threshold,
        "seed": args.seed,
        "train_final": train_split,
        "test_final": test_split,
        "groups": {
            "train_final": {name: str(group) for name, group in train_groups.items()},
            "test_final": {name: str(group) for name, group in test_groups.items()},
        },
        "counts": {
            "train_final": {role: len(names) for role, names in train_split.items()},
            "test_final": {role: len(names) for role, names in test_split.items()},
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print("train_final", payload["counts"]["train_final"],
          "groups", len(set(train_groups.values())))
    print("test_final", payload["counts"]["test_final"],
          "groups", len(set(test_groups.values())))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
