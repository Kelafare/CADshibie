from __future__ import annotations

import math

from cadvec.config import InferenceConfig
from cadvec.vectorize import _Shape, _hugs_curve, _merge_lines


def test_merge_lines_bridges_fragments_in_any_order():
    # 一条水平线被 Hough 拆成三段，其中桥接小段长度最小；旧的按长度贪心
    # 一次分配会把它留到最后，导致左右两段无法合并。
    segments = [
        ((243.0, 387.0), (204.0, 387.0)),
        ((282.0, 388.0), (254.0, 388.0)),
        ((253.0, 388.0), (244.0, 388.0)),
    ]
    merged = _merge_lines(segments, InferenceConfig())
    assert len(merged) == 1
    asserted = merged[0]
    assert math.dist(*asserted) > 70.0


def test_merge_lines_keeps_parallel_lines_apart():
    segments = [
        ((0.0, 0.0), (100.0, 0.0)),
        ((0.0, 20.0), (100.0, 20.0)),
    ]
    merged = _merge_lines(segments, InferenceConfig())
    assert len(merged) == 2


def test_hugs_curve_detects_rim_residual():
    circle = _Shape("circle", 50.0, 50.0, 20.0, 20.0, 0.0, 2 * math.pi, 0.0, 2 * math.pi, 0.0)
    rim = ((29.6, 52.0), (30.4, 48.0))
    assert _hugs_curve(rim[0], rim[1], [circle], tolerance=4.0)


def test_hugs_curve_keeps_chord():
    circle = _Shape("circle", 50.0, 50.0, 20.0, 20.0, 0.0, 2 * math.pi, 0.0, 2 * math.pi, 0.0)
    chord = ((31.0, 50.0), (69.0, 50.0))
    assert not _hugs_curve(chord[0], chord[1], [circle], tolerance=4.0)
