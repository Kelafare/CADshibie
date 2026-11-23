#!/usr/bin/env python3
"""比较两份 cadvec.evaluate JSON 的逐类边际 PQ，用于类别/分支取舍。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    per_class = data.get("per_class") or {}
    return data, {int(key): value for key, value in per_class.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()
    base, base_classes = load(args.base)
    new, new_classes = load(args.new)

    def pq(entry):
        tp, fp, fn = entry["TP"], entry["FP"], entry["FN"]
        denominator = tp + 0.5 * (fp + fn)
        return entry.get("sum_iou", 0.0) / denominator if denominator else 0.0

    denominator_old = base["TP"] + 0.5 * (base["FP"] + base["FN"])
    denominator_new = new["TP"] + 0.5 * (new["FP"] + new["FN"])
    base_pq = (base.get("micro") or base).get("PQ", 0.0)
    new_pq = (new.get("micro") or new).get("PQ", 0.0)
    print(f"base PQ {100*base_pq:.3f}%  new PQ {100*new_pq:.3f}%")
    rows = []
    for class_id in sorted(set(base_classes) | set(new_classes)):
        old = base_classes.get(class_id, {"TP": 0, "FP": 0, "FN": 0,
                                          "sum_iou": 0.0})
        cur = new_classes.get(class_id, {"TP": 0, "FP": 0, "FN": 0,
                                         "sum_iou": 0.0})
        marginal = (cur.get("sum_iou", 0.0) / max(denominator_new, 1.0) -
                    old.get("sum_iou", 0.0) / max(denominator_old, 1.0))
        rows.append((marginal, class_id, old, cur))
    print(f"{'cls':>3} {'dTP':>5} {'dFP':>5} {'dFN':>5} {'dS':>8} "
          f"{'PQold':>7} {'PQnew':>7}")
    for marginal, class_id, old, cur in sorted(rows, reverse=True)[: args.top]:
        print(f"{class_id:>3} {cur['TP']-old['TP']:>5} {cur['FP']-old['FP']:>5} "
              f"{cur['FN']-old['FN']:>5} "
              f"{cur.get('sum_iou',0)-old.get('sum_iou',0):>8.1f} "
              f"{100*pq(old):>6.2f}% {100*pq(cur):>6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
