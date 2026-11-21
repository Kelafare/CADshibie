#!/usr/bin/env python3
"""按类别合并两个引擎的预测：某些类别取 secondary，其余取 primary。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadvec.schema import assemble_output, validate_output, write_output  # noqa: E402


class _Primitive:
    __slots__ = ("command", "points", "length", "semantic_id", "instance_id",
                 "rgb", "layer_id", "width")

    def __init__(self, payload):
        self.command = int(payload["commands"])
        points = payload["args"]
        self.points = [
            (float(points[index]), float(points[index + 1]))
            for index in range(0, 8, 2)
        ]
        self.length = float(payload["lengths"])
        self.semantic_id = int(payload["semanticIds"])
        self.instance_id = int(payload["instanceIds"])
        self.rgb = [int(value) for value in payload["rgb"]]
        self.layer_id = int(payload["layerIds"])
        self.width = float(payload["widths"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--classes", required=True,
                        help="从 secondary 取哪些类别，逗号分隔，例如 8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    classes = {int(value) for value in args.classes.split(",") if value.strip()}

    args.out.mkdir(parents=True, exist_ok=True)
    names = sorted(path.name for path in args.primary.glob("*.json"))
    merged = 0
    for name in names:
        secondary_path = args.secondary / name
        if not classes or not secondary_path.exists():
            payload = json.loads((args.primary / name).read_text(encoding="utf-8"))
            write_output(payload, args.out / name)
            continue
        primary = json.loads((args.primary / name).read_text(encoding="utf-8"))
        secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
        primitives = []
        mapping = {}
        next_instance = [0]

        def renumber(prefix, instance):
            if instance < 0:
                return -1
            key = (prefix, instance)
            if key not in mapping:
                mapping[key] = next_instance[0]
                next_instance[0] += 1
            return mapping[key]

        for index in range(len(primary["commands"])):
            row = {key: primary[key][index] for key in (
                "commands", "args", "lengths", "semanticIds", "instanceIds",
                "rgb", "layerIds", "widths",
            )}
            if int(row["semanticIds"]) not in classes:
                primitive = _Primitive(row)
                primitive.instance_id = renumber("p", primitive.instance_id)
                primitives.append(primitive)
        for index in range(len(secondary["commands"])):
            row = {key: secondary[key][index] for key in (
                "commands", "args", "lengths", "semanticIds", "instanceIds",
                "rgb", "layerIds", "widths",
            )}
            if int(row["semanticIds"]) in classes:
                primitive = _Primitive(row)
                primitive.instance_id = renumber("s", primitive.instance_id)
                primitives.append(primitive)
        data = assemble_output(primitives, width=primary.get("width", 100),
                               height=primary.get("height", 100))
        validate_output(data)
        write_output(data, args.out / name)
        merged += 1
    print(f"合并 {merged} 张 → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
