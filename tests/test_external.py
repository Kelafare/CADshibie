from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from cadvec.config import InferenceConfig
from cadvec.external import (
    ExternalApiConfig,
    ExternalRefiner,
    _extract_content,
    _parse_labels,
    load_refiner,
)
from cadvec.pipeline import run_image
from cadvec.prior import ColorEntry, ColorPrior, PrimitiveClassifier
from cadvec.schema import validate_output
from cadvec.vectorize import Primitive

ROOT = Path(__file__).resolve().parents[1]


def _primitive(index: int, rgb: list[int] | None = None) -> Primitive:
    return Primitive(
        command=0,
        points=[(float(index), 1.0), (float(index) + 2, 1.0),
                (float(index) + 4, 1.0), (float(index) + 6, 1.0)],
        length=6.0,
        rgb=rgb or [220, 30, 30],
        layer_id=1,
        width=0.4,
    )


def test_load_refiner_disabled_by_default():
    assert load_refiner(ExternalApiConfig()) is None
    refiner = load_refiner(ExternalApiConfig(enabled=True))
    assert refiner is not None


def test_enabled_refiner_handles_transport_failure(tmp_path, capsys, monkeypatch):
    image = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(image)
    primitive = _primitive(0)
    primitive.semantic_id = 12
    refiner = ExternalRefiner(ExternalApiConfig(enabled=True))

    def fail(payload):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(refiner, "_request", fail)
    assert refiner.refine(image, [primitive]) == 0
    assert primitive.semantic_id == 12
    assert "外部接口调用失败" in capsys.readouterr().err


def test_chat_completions_url_normalizes_v1_suffix():
    cases = {
        "http://host:11434": "http://host:11434/v1/chat/completions",
        "http://host:11434/": "http://host:11434/v1/chat/completions",
        "http://host:11434/v1": "http://host:11434/v1/chat/completions",
        "http://host:11434/v1/": "http://host:11434/v1/chat/completions",
        "http://host:11434/v1/chat/completions": (
            "http://host:11434/v1/chat/completions"
        ),
    }
    for base_url, expected in cases.items():
        refiner = ExternalRefiner(ExternalApiConfig(enabled=True, base_url=base_url))
        assert refiner.chat_completions_url == expected


def test_build_payload_uses_openai_chat_completions_format(tmp_path):
    image = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(image)
    refiner = ExternalRefiner(ExternalApiConfig(
        enabled=True, base_url="http://host:11434/v1", model="qwen2.5vl:3b",
    ))
    payload = refiner._build_payload(image, [_primitive(0)])
    assert payload["model"] == "qwen2.5vl:3b"
    assert payload["stream"] is False
    message = payload["messages"][0]
    assert message["role"] == "user"
    parts = message["content"]
    assert parts[0]["type"] == "text"
    assert "图元列表" in parts[0]["text"]
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_request_posts_to_chat_completions(tmp_path, monkeypatch):
    image = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(image)
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return (
                b'{"choices":[{"message":{"content":'
                b'"{\\"labels\\": {\\"0\\": 7}}"}}]}'
            )

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["json"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    primitive = _primitive(0)
    refiner = ExternalRefiner(ExternalApiConfig(
        enabled=True, base_url="http://host:11434/v1",
        model="qwen2.5vl:3b", timeout=12.5,
    ))
    assert refiner.refine(image, [primitive]) == 1
    assert primitive.semantic_id == 7
    assert captured["url"] == "http://host:11434/v1/chat/completions"
    assert captured["timeout"] == 12.5
    assert captured["json"]["model"] == "qwen2.5vl:3b"
    assert captured["json"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )


def test_extract_content_accepts_openai_and_ollama_shapes():
    assert _extract_content({
        "choices": [{"message": {"content": '{"labels": {}}'}}],
    }) == '{"labels": {}}'
    assert _extract_content({
        "choices": [{"message": {"content": [
            {"type": "text", "text": "part-a"},
            {"type": "text", "text": "part-b"},
        ]}}],
    }) == "part-apart-b"
    assert _extract_content({"message": {"content": "native"}}) == "native"
    assert _extract_content({"response": "generate"}) == "generate"
    assert _extract_content({"choices": []}) == ""


def test_parse_labels_accepts_fences_and_rejects_invalid():
    labels = _parse_labels('```json\n{"labels": {"0": 3, "2": 31}}\n```')
    assert labels == {0: 3, 2: 31}
    assert _parse_labels("not json") == {}
    assert _parse_labels('{"labels": {"x": 3, "1": 99, "2": true}}') == {}


def test_refiner_applies_valid_labels(tmp_path, monkeypatch):
    image = tmp_path / "blank.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(image)
    primitives = [_primitive(0), _primitive(1)]
    refiner = ExternalRefiner(ExternalApiConfig(enabled=True))
    monkeypatch.setattr(
        refiner, "_request",
        lambda payload: {"message": {"content": '{"labels": {"0": 7, "1": 99}}'}},
    )
    assert refiner.refine(image, primitives) == 1
    assert primitives[0].semantic_id == 7
    assert primitives[1].semantic_id == 35  # 越界值被忽略，保持默认忽略类


def test_pipeline_with_refiner_reassigns_instances(tmp_path):
    image = Image.new("RGB", (512, 512), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.line((60, 200, 200, 200), fill=(220, 30, 30), width=4)
    draw.line((300, 200, 440, 200), fill=(220, 30, 30), width=4)
    path = tmp_path / "two_lines.png"
    image.save(path)

    class DummyRefiner:
        def __init__(self) -> None:
            self.calls = 0

        def refine(self, image_path, primitives) -> int:
            self.calls += 1
            for primitive in primitives:
                primitive.semantic_id = 5
            return len(primitives)

    prior = ColorPrior([ColorEntry([220, 30, 30], 0, 1.0, 10, {0: 10})])
    refiner = DummyRefiner()
    result = run_image(
        path, InferenceConfig(), prior, PrimitiveClassifier(),
        canvas_width=100, canvas_height=100, refiner=refiner,
    )
    validate_output(result.data)
    assert refiner.calls == 1
    assert result.data["semanticIds"]
    assert all(semantic == 5 for semantic in result.data["semanticIds"])
    assert all(instance >= 0 for instance in result.data["instanceIds"])
    assert result.stats["external_refined"] == len(result.data["commands"])


def test_cli_rejects_external_api_without_network_flag():
    completed = subprocess.run(
        [sys.executable, "infer.py", "--external-api", "--check-only"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 2
    assert "--allow-network" in completed.stderr
