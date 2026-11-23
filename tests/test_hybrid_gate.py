from __future__ import annotations

from cadvec.hybrid import merge_final


def _output(segments, semantic=8):
    """segments: [(instance, (x0, y0), (x1, y1))]"""
    data = {
        "commands": [], "args": [], "lengths": [], "semanticIds": [],
        "instanceIds": [], "rgb": [], "layerIds": [], "widths": [],
        "width": 100, "height": 100, "obj_cts": [], "boxes": [],
    }
    for instance, start, end in segments:
        data["commands"].append(0)
        data["args"].append([start[0], start[1], start[0], start[1],
                             end[0], end[1], end[0], end[1]])
        data["lengths"].append(((end[0] - start[0]) ** 2
                                + (end[1] - start[1]) ** 2) ** 0.5)
        data["semanticIds"].append(semantic)
        data["instanceIds"].append(instance)
        data["rgb"].append([0, 0, 0])
        data["layerIds"].append(0)
        data["widths"].append(1.0)
    return data


def _class8_instances(data):
    return {instance for instance, semantic in
            zip(data["instanceIds"], data["semanticIds"]) if semantic == 8}


def test_support_gate_drops_unsupported_instances():
    primary = _output([(0, (10.0, 10.0), (20.0, 10.0)),
                       (1, (80.0, 80.0), (90.0, 80.0))])
    support = {
        "size": [100, 100],
        "polygons": {8: [[[8.0, 8.0], [24.0, 8.0], [24.0, 13.0], [8.0, 13.0]]]},
    }
    merged = merge_final(
        primary, None, None, gat_classes=frozenset(),
        gate_classes={8}, gate_threshold=0.1, gate_metric="cover",
        support=support,
    )
    assert len(_class8_instances(merged)) == 1


def test_support_gate_keeps_everything_without_support_source():
    primary = _output([(0, (10.0, 10.0), (20.0, 10.0)),
                       (1, (80.0, 80.0), (90.0, 80.0))])
    merged = merge_final(
        primary, None, None, gat_classes=frozenset(),
        gate_classes={8}, support=None,
    )
    assert len(_class8_instances(merged)) == 2


def test_support_gate_drops_all_when_class_absent_from_detections():
    primary = _output([(0, (10.0, 10.0), (20.0, 10.0))])
    merged = merge_final(
        primary, None, None, gat_classes=frozenset(),
        gate_classes={8}, support={"size": [100, 100], "polygons": {}},
    )
    assert not merged["commands"]
