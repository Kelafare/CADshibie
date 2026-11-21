from __future__ import annotations

from PIL import Image, ImageDraw

from cadvec.config import COMMAND_CIRCLE, InferenceConfig
from cadvec.pipeline import run_image
from cadvec.prior import ColorEntry, ColorPrior, PrimitiveClassifier
from cadvec.schema import validate_output


def _prior() -> ColorPrior:
    return ColorPrior([
        ColorEntry([220, 30, 30], 0, 1.0, 10, {0: 10}),
        ColorEntry([30, 90, 220], 1, 1.0, 10, {1: 10}),
    ])


def test_pipeline_on_synthetic_png(tmp_path):
    image = Image.new("RGB", (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((120, 120, 360, 360), outline=(220, 30, 30), width=5)
    draw.line((60, 420, 440, 420), fill=(30, 90, 220), width=3)
    path = tmp_path / "plan.png"
    image.save(path)

    result = run_image(
        path, InferenceConfig(), _prior(), PrimitiveClassifier(),
        canvas_width=100, canvas_height=100,
    )
    validate_output(result.data)
    assert result.data["commands"], "至少应提取出一个图元"
    assert COMMAND_CIRCLE in result.data["commands"]
    for semantic, instance in zip(
        result.data["semanticIds"], result.data["instanceIds"],
    ):
        if 0 <= semantic <= 29:
            assert instance >= 0
    assert result.data["width"] == 100
    assert result.data["height"] == 100


def test_pipeline_blank_image(tmp_path):
    image = Image.new("RGB", (64, 64), (255, 255, 255))
    path = tmp_path / "blank.png"
    image.save(path)
    result = run_image(
        path, InferenceConfig(), _prior(), PrimitiveClassifier(),
        canvas_width=100, canvas_height=100,
    )
    validate_output(result.data)
    assert result.data["commands"] == []
    assert result.data["obj_cts"] == []
