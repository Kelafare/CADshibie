#!/usr/bin/env python3
"""训练笔画成对链接模型（同一 GT 实例 = 正样本）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from cadvec.grouping import FEATURE_NAMES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("../_work/link_dataset.npz"))
    parser.add_argument("--out", type=Path, default=Path("weights/group_model.joblib"))
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    payload = np.load(args.dataset)
    features, labels = payload["features"], payload["labels"]
    train_x, valid_x, train_y, valid_y = train_test_split(
        features, labels, test_size=0.15, random_state=0, stratify=labels,
    )
    model = HistGradientBoostingClassifier(
        max_iter=args.iterations, learning_rate=args.learning_rate,
        max_leaf_nodes=63, min_samples_leaf=30, l2_regularization=0.1,
        random_state=0,
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(valid_x)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        valid_y, predictions, average="binary", zero_division=0,
    )
    auc = roc_auc_score(valid_y, probabilities)
    ap = average_precision_score(valid_y, probabilities)
    metrics = {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "average_precision": round(float(ap), 4),
        "train_pairs": int(len(train_y)),
        "valid_pairs": int(len(valid_y)),
        "positive_rate": round(float(labels.mean()), 4),
        "threshold": args.threshold,
        "feature_names": list(FEATURE_NAMES),
    }
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    import joblib
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "link_threshold": args.threshold,
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics,
    }, args.out)
    print(f"写入 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
