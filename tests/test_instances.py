from __future__ import annotations

from cadvec.config import COMMAND_CIRCLE, COMMAND_LINE, InferenceConfig
from cadvec.instances import assign_instances
from cadvec.vectorize import Primitive


def _line(points, semantic=0):
    return Primitive(
        command=COMMAND_LINE,
        points=points,
        length=1.0,
        rgb=[0, 0, 0],
        layer_id=1,
        width=0.5,
        semantic_id=semantic,
    )


def _rect(x0, y0, x1, y1, semantic=0):
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    primitives = []
    for index in range(4):
        first, second = corners[index], corners[(index + 1) % 4]
        primitives.append(_line([
            first,
            (first[0] + (second[0] - first[0]) / 3, first[1] + (second[1] - first[1]) / 3),
            (first[0] + 2 * (second[0] - first[0]) / 3, first[1] + 2 * (second[1] - first[1]) / 3),
            second,
        ], semantic))
    return primitives


def test_crossing_rectangles_get_separate_instances():
    primitives = _rect(0, 0, 10, 10) + _rect(5, -5, 15, 5)
    assign_instances(primitives, InferenceConfig())
    first = {item.instance_id for item in primitives[:4]}
    second = {item.instance_id for item in primitives[4:]}
    assert len(first) == 1 and len(second) == 1
    assert first != second
    assert min(first | second) >= 0


def test_chain_shares_instance():
    first = _line([(0, 0), (3, 0), (7, 0), (10, 0)])
    second = _line([(10, 0), (13, 0), (17, 0), (20, 0)])
    assign_instances([first, second], InferenceConfig())
    assert first.instance_id == second.instance_id >= 0


def test_circle_is_own_instance():
    circle = Primitive(
        command=COMMAND_CIRCLE,
        points=[(30, 20), (40, 30), (30, 40), (20, 30)],
        length=62.8,
        rgb=[0, 0, 0],
        layer_id=1,
        width=0.5,
        semantic_id=0,
    )
    line = _line([(0, 0), (3, 0), (7, 0), (10, 0)])
    assign_instances([circle, line], InferenceConfig())
    assert circle.instance_id != line.instance_id
    assert circle.instance_id >= 0 and line.instance_id >= 0


def test_stuff_and_ignore_stay_minus_one():
    stuff = _line([(0, 0), (1, 0), (2, 0), (3, 0)], semantic=31)
    ignored = _line([(0, 0), (1, 0), (2, 0), (3, 0)], semantic=35)
    assign_instances([stuff, ignored], InferenceConfig())
    assert stuff.instance_id == -1
    assert ignored.instance_id == -1
