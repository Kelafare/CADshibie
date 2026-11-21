"""预测 JSON 的构造、校验与写出。

字段定义严格按赛事 README 第 4 节。所有校验在写文件前执行，保证提交物
不会出现 NaN、无穷大、长度不一致或实例号越界。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import (
    IGNORE_CLASS,
    OUTPUT_FIELDS,
    STUFF_CLASS_MAX,
    STUFF_CLASS_MIN,
    THING_CLASS_MAX,
    THING_CLASS_MIN,
    VALID_COMMANDS,
    VALID_SEMANTIC_IDS,
)


class SubmissionValidationError(ValueError):
    """预测 JSON 不符合赛题格式。"""


def is_thing(semantic_id: int) -> bool:
    return THING_CLASS_MIN <= semantic_id <= THING_CLASS_MAX


def is_stuff(semantic_id: int) -> bool:
    return STUFF_CLASS_MIN <= semantic_id <= STUFF_CLASS_MAX


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _as_int(value: Any, field: str, index: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        where = f"{field}[{index}]" if index is not None else field
        raise SubmissionValidationError(f"{where} 必须是整数，实际为 {value!r}")
    return value


def validate_output(data: dict[str, Any], *, check_instance_consistency: bool = True) -> None:
    """按 README 4.4 校验一个预测 JSON。"""
    if not isinstance(data, dict):
        raise SubmissionValidationError("预测 JSON 根节点必须是对象")

    missing = [field for field in OUTPUT_FIELDS if field not in data]
    if missing:
        raise SubmissionValidationError(f"缺少字段: {', '.join(missing)}")

    commands = data["commands"]
    args = data["args"]
    lengths = data["lengths"]
    semantic_ids = data["semanticIds"]
    instance_ids = data["instanceIds"]
    colors = data["rgb"]
    layer_ids = data["layerIds"]
    widths = data["widths"]

    if not all(isinstance(value, list) for value in (
        commands, args, lengths, semantic_ids, instance_ids, colors, layer_ids, widths,
    )):
        raise SubmissionValidationError("图元字段必须是数组")

    n = len(commands)
    for field, value in (
        ("args", args), ("lengths", lengths), ("semanticIds", semantic_ids),
        ("instanceIds", instance_ids), ("rgb", colors), ("layerIds", layer_ids),
        ("widths", widths),
    ):
        if len(value) != n:
            raise SubmissionValidationError(
                f"len({field})={len(value)} 与 len(commands)={n} 不一致"
            )

    if not isinstance(data["width"], int) or data["width"] <= 0:
        raise SubmissionValidationError("width 必须是正整数")
    if not isinstance(data["height"], int) or data["height"] <= 0:
        raise SubmissionValidationError("height 必须是正整数")

    instance_semantics: dict[int, set[int]] = defaultdict(set)
    for index in range(n):
        command = _as_int(commands[index], "commands", index)
        if command not in VALID_COMMANDS:
            raise SubmissionValidationError(f"commands[{index}]={command} 不在 0-3 内")

        points = args[index]
        if not isinstance(points, (list, tuple)) or len(points) != 8:
            raise SubmissionValidationError(f"args[{index}] 必须恰好 8 个数")
        if not all(_finite(value) for value in points):
            raise SubmissionValidationError(f"args[{index}] 含非有限数值")

        if not _finite(lengths[index]) or float(lengths[index]) < 0:
            raise SubmissionValidationError(f"lengths[{index}] 必须是非负有限数")

        semantic = _as_int(semantic_ids[index], "semanticIds", index)
        if semantic not in VALID_SEMANTIC_IDS:
            raise SubmissionValidationError(f"semanticIds[{index}]={semantic} 不在 0-35 内")

        instance = _as_int(instance_ids[index], "instanceIds", index)
        if instance < -1:
            raise SubmissionValidationError(f"instanceIds[{index}]={instance} 不能小于 -1")
        if is_thing(semantic) and instance < 0:
            raise SubmissionValidationError(
                f"semanticIds[{index}]={semantic} 是 Thing，instanceIds 必须非负"
            )
        if (is_stuff(semantic) or semantic == IGNORE_CLASS) and instance != -1:
            raise SubmissionValidationError(
                f"semanticIds[{index}]={semantic} 是 Stuff/忽略类，instanceIds 必须为 -1"
            )
        if instance >= 0:
            instance_semantics[instance].add(semantic)

        color = colors[index]
        if not isinstance(color, (list, tuple)) or len(color) != 3:
            raise SubmissionValidationError(f"rgb[{index}] 必须恰好 3 个整数")
        for channel in color:
            channel_value = _as_int(channel, f"rgb[{index}]")
            if not 0 <= channel_value <= 255:
                raise SubmissionValidationError(f"rgb[{index}] 通道必须在 0-255 内")

        _as_int(layer_ids[index], "layerIds", index)
        if not _finite(widths[index]) or float(widths[index]) < 0:
            raise SubmissionValidationError(f"widths[{index}] 必须是非负有限数")

    if check_instance_consistency:
        for instance, semantics in instance_semantics.items():
            if len(semantics) != 1:
                raise SubmissionValidationError(
                    f"实例 {instance} 同时包含多个类别 {sorted(semantics)}；"
                    "同一对象的图元类别必须一致"
                )

    obj_cts = data["obj_cts"]
    boxes = data["boxes"]
    if not isinstance(obj_cts, list) or not isinstance(boxes, list):
        raise SubmissionValidationError("obj_cts 和 boxes 必须是数组")
    if len(obj_cts) != len(boxes):
        raise SubmissionValidationError("obj_cts 与 boxes 行数必须相同")

    expected_instances = {
        instance: next(iter(semantics))
        for instance, semantics in instance_semantics.items()
    }
    seen_instances: set[int] = set()
    for row_index, (center, box) in enumerate(zip(obj_cts, boxes)):
        if not isinstance(center, (list, tuple)) or len(center) != 4:
            raise SubmissionValidationError(f"obj_cts[{row_index}] 必须恰好 4 个数")
        if not all(_finite(value) for value in center):
            raise SubmissionValidationError(f"obj_cts[{row_index}] 含非有限数值")
        center_instance = _as_int(center[3], f"obj_cts[{row_index}]")
        if center_instance in seen_instances:
            raise SubmissionValidationError(f"obj_cts 中实例 {center_instance} 重复")
        seen_instances.add(center_instance)
        if center_instance not in expected_instances:
            raise SubmissionValidationError(
                f"obj_cts 实例 {center_instance} 未出现在图元 instanceIds 中"
            )

        if not isinstance(box, (list, tuple)) or len(box) != 5:
            raise SubmissionValidationError(f"boxes[{row_index}] 必须恰好 5 个数")
        if not all(_finite(value) for value in box):
            raise SubmissionValidationError(f"boxes[{row_index}] 含非有限数值")
        # boxes 形如 [x_min, y_min, x_max, y_max, semantic_id]；与 obj_cts 按行对齐。
        box_semantic = _as_int(box[4], f"boxes[{row_index}]")
        if box_semantic != expected_instances[center_instance]:
            raise SubmissionValidationError(
                f"boxes[{row_index}] 类别与图元类别不一致"
            )
        if not (float(box[0]) <= float(box[2]) and float(box[1]) <= float(box[3])):
            raise SubmissionValidationError(f"boxes[{row_index}] 的 min/max 顺序错误")

    if seen_instances != set(expected_instances):
        missing_instances = sorted(set(expected_instances) - seen_instances)
        raise SubmissionValidationError(f"obj_cts/boxes 缺少实例: {missing_instances}")


def empty_output(width: int, height: int) -> dict[str, Any]:
    """一张图没有任何可输出图元时的合法 JSON。"""
    return {
        "commands": [],
        "args": [],
        "lengths": [],
        "semanticIds": [],
        "instanceIds": [],
        "width": int(width),
        "height": int(height),
        "obj_cts": [],
        "boxes": [],
        "rgb": [],
        "layerIds": [],
        "widths": [],
    }


def assemble_output(
    primitives: Iterable[Any],
    *,
    width: int,
    height: int,
) -> dict[str, Any]:
    """把图元对象装配成 12 字段 JSON。

    ``primitive`` 需要提供 command/points(4x2)/length/semantic_id/instance_id/
    rgb/layer_id/width。
    """
    primitives = list(primitives)
    data: dict[str, Any] = {
        "commands": [],
        "args": [],
        "lengths": [],
        "semanticIds": [],
        "instanceIds": [],
        "width": int(width),
        "height": int(height),
        "obj_cts": [],
        "boxes": [],
        "rgb": [],
        "layerIds": [],
        "widths": [],
    }

    instance_points: dict[int, list[list[float]]] = defaultdict(list)
    instance_semantic: dict[int, int] = {}
    for primitive in primitives:
        points = [[round(float(x), 6), round(float(y), 6)] for x, y in primitive.points]
        data["commands"].append(int(primitive.command))
        data["args"].append([value for point in points for value in point])
        data["lengths"].append(round(float(primitive.length), 6))
        data["semanticIds"].append(int(primitive.semantic_id))
        data["instanceIds"].append(int(primitive.instance_id))
        data["rgb"].append([int(channel) for channel in primitive.rgb])
        data["layerIds"].append(int(primitive.layer_id))
        data["widths"].append(round(float(primitive.width), 6))
        if primitive.instance_id >= 0:
            instance_points[primitive.instance_id].extend(points)
            instance_semantic[primitive.instance_id] = int(primitive.semantic_id)

    for instance_id in sorted(instance_points):
        points = instance_points[instance_id]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        data["obj_cts"].append([
            round((x_min + x_max) / 2.0, 6),
            round((y_min + y_max) / 2.0, 6),
            0,
            int(instance_id),
        ])
        data["boxes"].append([
            round(x_min, 6), round(y_min, 6), round(x_max, 6), round(y_max, 6),
            int(instance_semantic[instance_id]),
        ])

    return data


def write_output(data: dict[str, Any], path: str | Path) -> Path:
    """先校验再原子写出 UTF-8 JSON。"""
    validate_output(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    temporary.replace(path)
    return path
