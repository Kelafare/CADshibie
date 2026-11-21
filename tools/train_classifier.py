#!/usr/bin/env python3
"""可选：用训练集 SVG 训练小型图元语义分类器。

默认推理只依赖颜色先验（``weights/color_prior.json``），分类器用于在颜色
重复或颜色缺失时补充几何特征。模型是 sklearn 树模型，参数量远低于 5B。

生成 ``weights/primitive_classifier.joblib`` 后，``PrimitiveClassifier.load``
会自动启用；请同时更新 ``weights/manifest.json`` 中的模型描述（参数量按
树模型节点数计，可在 manifest 中声明）。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from cadvec.prior import primitive_features  # noqa: E402
from cadvec.svg import iter_primitives  # noqa: E402


def collect(svg_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features = []
    labels = []
    sources = []
    files = sorted(svg_dir.rglob("*.svg"))
    if not files:
        raise SystemExit(f"{svg_dir} 中没有 SVG")
    for path in files:
        try:
            primitives = list(iter_primitives(path))
        except Exception as error:  # noqa: BLE001 - 跳过损坏文件
            print(f"[跳过] {path}: {error}", file=sys.stderr)
            continue
        for primitive in primitives:
            features.append(primitive_features(primitive))
            labels.append(primitive.semantic_id)
            sources.append(str(path))
    if not features:
        raise SystemExit("没有可用的训练图元")
    return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64), sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("weights/primitive_classifier.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--update-manifest", action="store_true",
        help="训练后自动把树节点数写回 weights/manifest.json 并重算总参数量",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("weights/manifest.json"),
        help="与 --update-manifest 配合使用的清单路径",
    )
    args = parser.parse_args()

    features, labels, sources = collect(args.svg_dir)
    unique_sources = sorted(set(sources))
    if len(unique_sources) >= 2:
        train_sources, valid_sources = train_test_split(
            unique_sources, test_size=args.test_size, random_state=args.seed,
        )
        train_mask = np.array([source in set(train_sources) for source in sources])
        valid_mask = ~train_mask
    else:
        train_mask = np.ones(len(labels), dtype=bool)
        valid_mask = train_mask

    model = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=1,
    )
    model.fit(features[train_mask], labels[train_mask])
    if valid_mask.any():
        predictions = model.predict(features[valid_mask])
        accuracy = accuracy_score(labels[valid_mask], predictions)
        print(f"验证集准确率: {accuracy:.4f}（{int(valid_mask.sum())} 个图元）")
        print(classification_report(
            labels[valid_mask], predictions, zero_division=0, digits=3,
        ))
    print("训练类别分布:", dict(sorted(Counter(labels.tolist()).items())))

    import joblib
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    node_count = int(sum(tree.tree_.node_count for tree in model.estimators_))
    metadata = {
        "schema_version": 1,
        "feature_count": int(features.shape[1]),
        "classes": [int(value) for value in model.classes_],
        "trees": args.trees,
        "tree_node_count": node_count,
        "train_primitives": int(train_mask.sum()),
        "train_svg_files": len(unique_sources),
        "seed": args.seed,
    }
    metadata_path = args.out.with_suffix(".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"已写入 {args.out} 与 {metadata_path}；树节点数 {node_count}")
    if args.update_manifest:
        _update_manifest(args.manifest, args.out.name, node_count)
        print(f"已更新 {args.manifest}：primitive-classifier 参数量 = {node_count}")
    else:
        print(
            "请同步更新 weights/manifest.json 的模型条目与参数量说明，"
            "或加 --update-manifest 自动写入。"
        )
    return 0


def _update_manifest(path: Path, filename: str, node_count: int) -> None:
    """把分类器节点数写回清单；命名必须与 PrimitiveClassifier.load 一致。"""
    if not path.exists():
        raise SystemExit(f"清单不存在: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload.get("models", []):
        if isinstance(entry, dict) and Path(str(entry.get("path", ""))).name == filename:
            entry["parameters"] = int(node_count)
            entry["parameter_source"] = "sklearn-tree-nodes"
            break
    else:
        payload.setdefault("models", []).append({
            "name": "primitive-classifier",
            "kind": "sklearn-classifier",
            "path": filename,
            "parameters": int(node_count),
            "parameter_source": "sklearn-tree-nodes",
            "optional": True,
            "description": "可选的小型 sklearn 分类器；参数量按树节点数计。",
        })
    payload["total_parameters"] = int(sum(
        int(entry.get("parameters", 0))
        for entry in payload.get("models", [])
        if isinstance(entry, dict)
    ))
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
