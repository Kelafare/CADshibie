from __future__ import annotations

import numpy as np

from cadvec.pipeline_segdet import _local_attribution
from cadvec.strokes import Stroke


def _stroke(points) -> Stroke:
    return Stroke(command=0, points=list(points), length=1.0,
                  p0=points[0], p1=points[-1])


def test_local_attribution_splits_at_mask_boundary():
    stroke = _stroke([(10.0, 10.0), (90.0, 10.0), (90.0, 10.0), (90.0, 10.0)])
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, :50] = True
    assigned = _local_attribution(
        [(0.9, 2, mask)], [stroke], 1.0, 1.0, 100, 100, min_length=0.5,
    )
    assert len(assigned) == 1
    segments = assigned[0][2]
    assert len(segments) == 1
    xs = [point[0] for point in segments[0]["points"]]
    assert min(xs) >= 9.5
    assert max(xs) <= 50.0
    assert segments[0]["length"] < 45.0


def test_local_attribution_first_detection_owns_overlap():
    stroke = _stroke([(10.0, 10.0), (90.0, 10.0), (90.0, 10.0), (90.0, 10.0)])
    left = np.zeros((100, 100), dtype=bool)
    left[:, :60] = True
    right = np.zeros((100, 100), dtype=bool)
    right[:, 40:] = True
    assigned = _local_attribution(
        [(0.9, 2, left), (0.8, 3, right)], [stroke], 1.0, 1.0, 100, 100,
        min_length=0.5,
    )
    classes = [item[1] for item in assigned]
    assert classes == [2, 3]
    first = assigned[0][2][0]["points"]
    second = assigned[1][2][0]["points"]
    assert max(point[0] for point in first) <= 60.0
    # 第二个检测只拿到未被占用的部分，起点在左检测终点之后。
    assert min(point[0] for point in second) >= 59.0


def test_local_attribution_drops_short_fragments():
    stroke = _stroke([(10.0, 10.0), (90.0, 10.0), (90.0, 10.0), (90.0, 10.0)])
    mask = np.zeros((100, 100), dtype=bool)
    mask[:, 20:21] = True  # 1 像素宽的窄带，片段短于 min_length
    assigned = _local_attribution(
        [(0.9, 2, mask)], [stroke], 1.0, 1.0, 100, 100, min_length=2.0,
    )
    assert assigned == []
