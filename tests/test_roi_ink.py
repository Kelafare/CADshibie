from __future__ import annotations

import numpy as np

from cadvec.roi_ink import RoiInkRefiner, vectorize_mask


def test_vectorize_mask_recovers_canvas_coordinates():
    mask = np.zeros((192, 192), dtype=np.float32)
    mask[90:94, 20:170] = 1.0
    segments = vectorize_mask(mask, 0, 0, 192, 192, 10.0, 10.0)
    assert segments
    xs = [point[0] for points, _length in segments for point in points]
    ys = [point[1] for points, _length in segments for point in points]
    assert 1.5 <= min(xs) <= 3.0
    assert 16.5 <= max(xs) <= 17.5
    assert 8.5 <= np.mean(ys) <= 9.5


def test_vectorize_mask_drops_small_noise():
    mask = np.zeros((192, 192), dtype=np.float32)
    mask[10:12, 10:12] = 1.0
    assert vectorize_mask(mask, 0, 0, 192, 192, 10.0, 10.0) == []


def test_refiner_missing_weights_returns_none(tmp_path):
    assert RoiInkRefiner.load(tmp_path / "missing.pt") is None
