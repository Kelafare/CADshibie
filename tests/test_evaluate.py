from __future__ import annotations

from cadvec.config import COMMAND_CIRCLE
from cadvec.evaluate import evaluate
from cadvec.schema import assemble_output
from cadvec.vectorize import Primitive


def _circle(cx, cy, radius, semantic=0, instance=0):
    return Primitive(
        command=COMMAND_CIRCLE,
        points=[
            (cx - radius, cy), (cx, cy - radius),
            (cx + radius, cy), (cx, cy + radius),
        ],
        length=2 * 3.141592653589793 * radius,
        rgb=[0, 0, 0],
        layer_id=1,
        width=0.5,
        semantic_id=semantic,
        instance_id=instance,
    )


def test_perfect_prediction_scores_one():
    data = assemble_output([_circle(50, 50, 20)], width=100, height=100)
    result = evaluate(data, data, (100, 100), raster_size=400)
    assert result["TP"] == 1
    assert result["FP"] == 0 and result["FN"] == 0
    assert abs(result["micro"]["PQ"] - 1.0) < 1e-9


def test_shifted_prediction_scores_zero():
    gt = assemble_output([_circle(30, 50, 10)], width=100, height=100)
    pred = assemble_output([_circle(70, 50, 10)], width=100, height=100)
    result = evaluate(gt, pred, (100, 100), raster_size=400)
    assert result["TP"] == 0
    assert result["FP"] == 1 and result["FN"] == 1
    assert result["micro"]["PQ"] == 0.0


def test_ignore_class_is_skipped():
    gt = assemble_output([
        _circle(50, 50, 20),
        _circle(20, 20, 5, semantic=35, instance=-1),
    ], width=100, height=100)
    pred = assemble_output([_circle(50, 50, 20)], width=100, height=100)
    result = evaluate(gt, pred, (100, 100), raster_size=400)
    assert result["skipped_gt"] == 1
    assert result["TP"] == 1 and result["FN"] == 0


def test_thing_without_instance_is_skipped():
    gt = assemble_output([
        _circle(50, 50, 20, instance=-1),
    ], width=100, height=100)
    pred = assemble_output([_circle(50, 50, 20)], width=100, height=100)
    result = evaluate(gt, pred, (100, 100), raster_size=400)
    assert result["skipped_gt"] == 1
    assert result["TP"] == 0 and result["FN"] == 0
