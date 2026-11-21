#!/usr/bin/env python3
"""按类别抑制：删除指定语义类别的图元并重算字段。"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cadvec.schema import assemble_output, validate_output, write_output

class P:
    __slots__=("command","points","length","semantic_id","instance_id","rgb","layer_id","width")
    def __init__(self,d,i):
        self.command=int(d["commands"][i]); p=d["args"][i]
        self.points=[(float(p[k]),float(p[k+1])) for k in range(0,8,2)]
        self.length=float(d["lengths"][i]); self.semantic_id=int(d["semanticIds"][i])
        self.instance_id=int(d["instanceIds"][i]); self.rgb=[int(v) for v in d["rgb"][i]]
        self.layer_id=int(d["layerIds"][i]); self.width=float(d["widths"][i])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pred",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--classes",required=True)
    a=ap.parse_args(); drop={int(v) for v in a.classes.split(",") if v.strip()}
    a.out.mkdir(parents=True,exist_ok=True); n=0
    for path in sorted(a.pred.glob("*.json")):
        d=json.loads(path.read_text(encoding="utf-8"))
        prims=[P(d,i) for i in range(len(d["commands"])) if int(d["semanticIds"][i]) not in drop]
        data=assemble_output(prims,width=int(d.get("width",100)),height=int(d.get("height",100)))
        validate_output(data); write_output(data,a.out/path.name); n+=1
    print("suppressed",n)

main()
