from __future__ import annotations

import copy

import pytest

from cadvec.schema import (
    SubmissionValidationError,
    assemble_output,
    empty_output,
    validate_output,
)
from cadvec.vectorize import Primitive


def _primitive(**overrides):
    values = dict(
        command=0,
        points=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],
        length=4.24,
        rgb=[10, 20, 30],
        layer_id=1,
        width=0.5,
        semantic_id=3,
        instance_id=7,
    )
    values.update(overrides)
    return Primitive(**values)


def test_empty_output_is_valid():
    data = empty_output(100, 100)
    validate_output(data)
    assert data["commands"] == []
    assert data["width"] == 100


def test_assemble_and_validate():
    first = _primitive(instance_id=0)
    second = _primitive(
        points=[(4.0, 4.0), (5.0, 5.0), (6.0, 6.0), (7.0, 7.0)],
        instance_id=0,
    )
    data = assemble_output([first, second], width=100, height=100)
    validate_output(data)
    assert len(data["commands"]) == 2
    assert data["instanceIds"] == [0, 0]
    assert len(data["obj_cts"]) == 1
    assert data["boxes"][0][4] == 3


def test_stuff_must_use_minus_one():
    data = assemble_output(
        [_primitive(semantic_id=31, instance_id=-1)], width=100, height=100,
    )
    validate_output(data)
    bad = copy.deepcopy(data)
    bad["instanceIds"][0] = 3
    with pytest.raises(SubmissionValidationError):
        validate_output(bad)


def test_thing_requires_instance():
    data = assemble_output(
        [_primitive(semantic_id=0, instance_id=-1)], width=100, height=100,
    )
    with pytest.raises(SubmissionValidationError):
        validate_output(data)


def test_length_mismatch_rejected():
    data = empty_output(100, 100)
    data["commands"] = [0]
    with pytest.raises(SubmissionValidationError):
        validate_output(data)


def test_rgb_range_rejected():
    data = assemble_output([_primitive()], width=100, height=100)
    data["rgb"][0] = [0, 0, 256]
    with pytest.raises(SubmissionValidationError):
        validate_output(data)


def test_args_must_have_eight_finite_numbers():
    data = assemble_output([_primitive()], width=100, height=100)
    data["args"][0] = [0.0] * 7
    with pytest.raises(SubmissionValidationError):
        validate_output(data)
    data = assemble_output([_primitive()], width=100, height=100)
    data["args"][0][0] = float("nan")
    with pytest.raises(SubmissionValidationError):
        validate_output(data)


def test_instance_cannot_have_multiple_semantics():
    data = assemble_output(
        [
            _primitive(semantic_id=2, instance_id=1),
            _primitive(semantic_id=3, instance_id=1),
        ],
        width=100, height=100,
    )
    with pytest.raises(SubmissionValidationError):
        validate_output(data)


def test_obj_cts_must_match_instances():
    data = assemble_output([_primitive()], width=100, height=100)
    data["obj_cts"] = []
    data["boxes"] = []
    with pytest.raises(SubmissionValidationError):
        validate_output(data)


def test_boxes_semantic_mismatch_rejected():
    data = assemble_output([_primitive()], width=100, height=100)
    data["boxes"][0][4] = 4
    with pytest.raises(SubmissionValidationError):
        validate_output(data)
