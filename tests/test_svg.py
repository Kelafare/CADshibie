from __future__ import annotations

import math

from cadvec.config import COMMAND_ARC, COMMAND_CIRCLE, COMMAND_LINE
from cadvec.svg import parse_color, parse_path, parse_viewbox, svg_to_output
from cadvec.schema import validate_output

import xml.etree.ElementTree as ET


def test_parse_color():
    assert parse_color("#f00") == [255, 0, 0]
    assert parse_color("#102030") == [16, 32, 48]
    assert parse_color("rgb(1, 2, 3)") == [1, 2, 3]
    assert parse_color("none") is None


def test_parse_line_path():
    subpaths, command = parse_path("M 0 0 L 10 0 L 10 10")
    assert command == COMMAND_LINE
    assert subpaths[0][0] == (0.0, 0.0)
    assert subpaths[0][-1] == (10.0, 10.0)


def test_parse_arc_path():
    subpaths, command = parse_path("M 0 0 A 5 5 0 0 1 10 0")
    assert command == COMMAND_ARC
    assert len(subpaths[0]) >= 3
    assert math.dist(subpaths[0][0], (0.0, 0.0)) < 1e-9
    assert math.dist(subpaths[0][-1], (10.0, 0.0)) < 1e-6


def test_parse_relative_and_bezier():
    subpaths, command = parse_path("M 0 0 l 10 0 c 1 1 2 2 3 3")
    assert command == COMMAND_LINE
    assert math.dist(subpaths[0][-1], (13.0, 3.0)) < 1e-6


def test_svg_to_output(tmp_path):
    svg = tmp_path / "sample.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 100">'
        '<circle cx="50" cy="50" r="10" stroke="#ff0000" fill="none" '
        'stroke-width="0.5" semantic-id="5" instance-id="2"/>'
        '<path d="M 10 10 L 90 10" stroke="#0000ff" fill="none" '
        'stroke-width="0.5" semantic-id="3" instance-id="1"/>'
        "</svg>",
        encoding="utf-8",
    )
    data = svg_to_output(svg)
    validate_output(data)
    assert data["width"] == 200 and data["height"] == 100
    assert data["commands"] == [COMMAND_CIRCLE, COMMAND_LINE]
    assert data["semanticIds"] == [4, 2]
    assert data["instanceIds"] == [2, 1]
    assert data["rgb"] == [[255, 0, 0], [0, 0, 255]]


def test_parse_viewbox():
    root = ET.fromstring(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100.4 50.6"/>'
    )
    assert parse_viewbox(root) == (100, 51)
