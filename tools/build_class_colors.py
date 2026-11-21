#!/usr/bin/env python3
"""统计每个类别的代表描边颜色（用于检测框内的笔画筛选）。"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.config import IGNORE_CLASS, STUFF_CLASS_MAX, THING_CLASS_MAX
from cadvec.svg import iter_primitives

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--svg-dir",type=Path,default=Path("../datasets/train_final"))
    ap.add_argument("--out",type=Path,default=Path("weights/class_colors.json"))
    ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    files=sorted(a.svg_dir.glob("*.svg"))
    if a.limit: files=files[:a.limit]
    counts=defaultdict(Counter)
    for f in files:
        try: prims=list(iter_primitives(f))
        except Exception: continue
        for p in prims:
            s=p.semantic_id
            if s==IGNORE_CLASS: continue
            if s<=THING_CLASS_MAX and p.instance_id<0: continue
            if s<=STUFF_CLASS_MAX:
                counts[s][tuple(p.rgb)]+=1
    payload={str(c):[[list(rgb),int(n)] for rgb,n in ctr.most_common(12)] for c,ctr in counts.items()}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(payload,ensure_ascii=False,indent=1),encoding="utf-8")
    print("classes",len(payload),"->",a.out)
main()
