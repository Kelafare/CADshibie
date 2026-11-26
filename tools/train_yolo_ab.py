#!/usr/bin/env python3
"""YOLOv8n-seg A/B：从零训练 vs ImageNet/COCO 预训练初始化。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True,
                        help="yolov8n-seg.yaml（从零）或预训练 .pt")
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0,1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="/home/user/cadcomp/_work/yolo_runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    devices = [int(value) for value in args.device.split(",") if value.strip()]
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=devices,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        deterministic=True,
        cache="ram",
        plots=False,
        exist_ok=True,
        pretrained=args.model.endswith(".pt"),
    )
    print("results", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
