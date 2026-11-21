#!/usr/bin/env python3
"""用 YOLO 检测框过滤像素管线的指定类别预测（保留与检测框重叠的实例）。"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("YOLO_OFFLINE", "1")
import numpy as np
from cadvec.schema import assemble_output, validate_output, write_output

class _Prim:
    __slots__=("command","points","length","semantic_id","instance_id","rgb","layer_id","width")
    def __init__(self,d,i,off=0):
        self.command=int(d["commands"][i]); p=d["args"][i]
        self.points=[(float(p[k]),float(p[k+1])) for k in range(0,8,2)]
        self.length=float(d["lengths"][i]); self.semantic_id=int(d["semanticIds"][i])
        self.instance_id=int(d["instanceIds"][i]); off = off if self.instance_id>=0 else 0
        self.instance_id += off
        self.rgb=[int(v) for v in d["rgb"][i]]; self.layer_id=int(d["layerIds"][i]); self.width=float(d["widths"][i])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pred",type=Path,required=True); ap.add_argument("--png-dir",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True); ap.add_argument("--classes",default="3")
    ap.add_argument("--weights",type=Path,default=Path("weights/yolo_det.pt"))
    ap.add_argument("--conf",type=float,default=0.25); ap.add_argument("--margin",type=float,default=1.0)
    a=ap.parse_args()
    classes={int(v) for v in a.classes.split(",")}
    from ultralytics import YOLO
    model=YOLO(str(a.weights))
    a.out.mkdir(parents=True,exist_ok=True)
    kept_inst=0; dropped=0
    for path in sorted(a.pred.glob("*.json")):
        d=json.loads(path.read_text(encoding="utf-8"))
        png=a.png_dir/f"{path.stem}.png"
        r=model.predict(source=str(png),imgsz=640,conf=a.conf,iou=0.5,verbose=False,device=0)[0]
        boxes=[]
        if r.boxes is not None and len(r.boxes):
            for b,c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                if c in classes:
                    boxes.append((b[0]/10-a.margin,b[1]/10-a.margin,b[2]/10+a.margin,b[3]/10+a.margin))
        prims=[]
        for i in range(len(d["commands"])):
            s=int(d["semanticIds"][i])
            if s in classes and int(d["instanceIds"][i])>=0:
                pts=d["args"][i]; cx=(pts[0]+pts[2]+pts[4]+pts[6])/4; cy=(pts[1]+pts[3]+pts[5]+pts[7])/4
                ok=any(b[0]<=cx<=b[2] and b[1]<=cy<=b[3] for b in boxes)
                if not ok:
                    dropped+=1; continue
            prims.append(_Prim(d,i))
        data=assemble_output(prims,width=int(d.get("width",100)),height=int(d.get("height",100)))
        validate_output(data); write_output(data,a.out/path.name)
    print("dropped prims",dropped)

main()
