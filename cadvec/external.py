"""可选的外部多模态模型接口（默认关闭，正式评测禁止使用）。

赛题要求评测容器完全离线、模型参数总量 ≤5B、评测期间不得访问外部模型
API。因此本模块遵守三条约定：

1. 默认路径从不创建 ``ExternalRefiner``，纯几何 + 颜色先验即可完成推理；
2. 只有同时指定 ``infer.py --external-api --allow-network`` 才会启用；
3. 网络传输只在 ``_request`` 函数内部延迟导入 ``urllib.request``，模块
   顶层不导入任何联网模块，默认推理路径不会加载网络代码。

推荐的合规用法是**开发期**在一台安装 Ollama 的机器上运行小尺寸 Qwen，
对少量图纸做语义复核或生成弱标签，再把结论固化为
``weights/color_prior.json`` 或 ``weights/primitive_classifier.joblib``；
正式评测仍使用这些离线权重，不调用外部服务。

本接口使用 OpenAI 兼容的 Chat Completions 协议（Ollama 从 0.1.24 起在
``/v1`` 下提供该协议），基础地址通常以 ``/v1`` 结尾::

    ollama pull qwen2.5vl:3b
    OLLAMA_HOST=0.0.0.0 ollama serve

    python infer.py --input /data/dev_final --output results \
        --external-api --allow-network \
        --external-url http://10.11.64.137:11434/v1 \
        --external-model qwen2.5vl:3b

注意：如果计划在正式评测时调用模型，必须把 Ollama 与模型权重一起打包进
离线容器、申报参数量（Qwen2.5-VL-3B 约 3.75B 参数，仍低于 5B 上限），
并确认平台允许容器内的回环服务；否则会被判定为违规访问外部 API。
"""
from __future__ import annotations

import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .vectorize import COMMAND_ARC, COMMAND_CIRCLE, COMMAND_ELLIPSE, COMMAND_LINE, Primitive

_COMMAND_NAMES = {
    COMMAND_LINE: "line",
    COMMAND_ARC: "arc",
    COMMAND_CIRCLE: "circle",
    COMMAND_ELLIPSE: "ellipse",
}

_IMAGE_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

_PROMPT = (
    "你是 CAD 平面图语义标注助手。图中已提取出若干矢量图元，每个图元给出"
    "序号、类型、颜色、长度、中心与包围盒。请根据整张图判断每个图元属于哪个"
    "语义类别，只输出 JSON，不要解释：\n"
    '{"labels": {"<序号>": <0-34 的整数>}}\n'
    "类别 0-29 为可数对象，30-34 为不可数区域，无法判断时用 35。\n"
    "图元列表：\n"
)


@dataclass(frozen=True)
class ExternalApiConfig:
    """外部模型连接参数（仅开发期使用）。"""

    enabled: bool = False
    #: OpenAI 兼容服务的基础地址，通常以 ``/v1`` 结尾。
    base_url: str = "http://10.11.64.137:11434/v1"
    model: str = "qwen2.5vl:3b"
    timeout: float = 60.0
    #: 单张图最多送给模型多少个图元，避免提示词过长。
    max_primitives: int = 40
    #: 可选 Bearer Token；Ollama 本地服务通常不需要。
    api_key: str = ""


class ExternalRefiner:
    """用外部多模态模型复核图元语义；失败时保持原有结果。"""

    def __init__(self, config: ExternalApiConfig):
        self.config = config
        self._warned = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def chat_completions_url(self) -> str:
        """规范化 Chat Completions 端点地址。

        接受 ``http://host:11434``、``http://host:11434/v1``、带尾部斜杠
        或完整 ``/chat/completions`` 的写法；缺 ``/v1`` 时自动补上。
        """
        base = self.config.base_url.strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def refine(self, image_path: str | Path, primitives: list[Primitive]) -> int:
        """改写图元语义，返回实际采纳的条数；任何异常都不会中断推理。"""
        if not self.config.enabled or not primitives:
            return 0
        selected = sorted(
            primitives,
            key=lambda item: (-item.length, item.points[0][0], item.points[0][1]),
        )[: max(1, self.config.max_primitives)]
        try:
            response = self._request(self._build_payload(image_path, selected))
        except Exception as error:  # noqa: BLE001 - 外部服务异常不影响本地推理
            if not self._warned:
                print(f"[外部接口调用失败] {type(error).__name__}: {error}", file=sys.stderr)
                self._warned = True
            return 0

        labels = _parse_labels(_extract_content(response))
        applied = 0
        for index, primitive in enumerate(selected):
            semantic = labels.get(index)
            if semantic is None or not 0 <= semantic <= 35:
                continue
            primitive.semantic_id = int(semantic)
            primitive.confidence = max(primitive.confidence, 0.5)
            applied += 1
        return applied

    def _build_payload(
        self, image_path: str | Path, primitives: Iterable[Primitive],
    ) -> dict[str, Any]:
        """构造 OpenAI Chat Completions 多模态请求体。"""
        path = Path(image_path)
        image = base64.b64encode(path.read_bytes()).decode("ascii")
        mime = _IMAGE_MIME.get(path.suffix.lower(), "image/png")
        lines = []
        for index, primitive in enumerate(primitives):
            xs = [point[0] for point in primitive.points]
            ys = [point[1] for point in primitive.points]
            lines.append(
                f"{index}: {_COMMAND_NAMES.get(primitive.command, 'unknown')}, "
                f"rgb={primitive.rgb}, length={primitive.length:.2f}, "
                f"center=({(min(xs) + max(xs)) / 2:.2f},{(min(ys) + max(ys)) / 2:.2f}), "
                f"bbox=({min(xs):.2f},{min(ys):.2f},{max(xs):.2f},{max(ys):.2f})"
            )
        return {
            "model": self.config.model,
            "stream": False,
            "temperature": 0,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT + "\n".join(lines)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image}"},
                    },
                ],
            }],
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """向 OpenAI 兼容端点发送请求。

        ``urllib.request`` 只在函数内部延迟导入：模块顶层保持无联网依赖，
        未显式启用外部接口时，执行任何推理路径都不会加载网络代码。
        """
        import urllib.error
        import urllib.request

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        url = self.chat_completions_url
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout,
            ) as response:
                body = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"HTTP {error.code} {error.reason} @ {url}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"无法连接 {url}: {error.reason}") from error

        try:
            data = json.loads(body)
        except ValueError as error:
            raise RuntimeError(f"响应不是合法 JSON: {body[:200]}") from error
        if not isinstance(data, dict):
            raise RuntimeError(f"响应结构异常: {body[:200]}")
        return data


def load_refiner(config: ExternalApiConfig) -> ExternalRefiner | None:
    """按配置返回外部复核器；默认关闭时返回 None，推理路径零网络依赖。"""
    return ExternalRefiner(config) if config.enabled else None


def _extract_content(response: Any) -> str:
    """提取模型文本，兼容 OpenAI Chat Completions 与 Ollama 原生返回结构。"""
    if not isinstance(response, dict):
        return ""
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and isinstance(part.get("text"), str)
                    ]
                    if parts:
                        return "".join(parts)
    message = response.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    if isinstance(response.get("response"), str):
        return response["response"]
    return ""


def _parse_labels(text: str) -> dict[int, int]:
    """从模型输出中提取 {"序号": 类别}；解析失败时返回空字典。"""
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(cleaned[start:end + 1])
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw = payload.get("labels", payload)
    if not isinstance(raw, dict):
        return {}
    labels: dict[int, int] = {}
    for key, value in raw.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, bool):
            continue
        try:
            semantic = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= semantic <= 35:
            labels[index] = semantic
    return labels
