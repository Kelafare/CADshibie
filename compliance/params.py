"""模型清单与参数量审计。

赛题要求提交模型清单，并保证视觉编码器、检测、OCR 等全部模型合计不超过
5B 参数。本方案是传统几何算法 + 颜色查找表，没有神经网络权重；如果后续
加入分类器或检测器，必须同步更新 ``weights/manifest.json`` 并由本模块审计。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cadvec.config import MAX_MODEL_PARAMETERS

MANIFEST_NAME = "manifest.json"


class ModelComplianceError(RuntimeError):
    """模型清单缺失、不一致或超出参数上限。"""


@dataclass
class ModelEntry:
    name: str
    kind: str
    path: str
    parameters: int
    parameter_source: str
    exists: bool
    size_bytes: int = 0


@dataclass
class AuditReport:
    models: list[ModelEntry] = field(default_factory=list)
    total_parameters: int = 0
    limit: int = MAX_MODEL_PARAMETERS
    manifest_path: str = ""

    @property
    def ok(self) -> bool:
        return self.total_parameters <= self.limit

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "total_parameters": self.total_parameters,
            "limit": self.limit,
            "manifest": self.manifest_path,
            "models": [
                {
                    "name": item.name,
                    "kind": item.kind,
                    "path": item.path,
                    "parameters": item.parameters,
                    "parameter_source": item.parameter_source,
                    "exists": item.exists,
                    "size_bytes": item.size_bytes,
                }
                for item in self.models
            ],
        }


def _count_torch_parameters(path: Path) -> int:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # 旧版 torch 没有 weights_only
        payload = torch.load(path, map_location="cpu")
    except Exception:  # noqa: BLE001 - ultralytics 等自定义模型类
        # 本地自建/可信权重（评测容器内只加载提交包内文件）。
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.nn.Module):
        return int(sum(parameter.numel() for parameter in payload.parameters()))
    total = 0
    stack = [payload]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            total += int(item.numel())
        elif isinstance(item, torch.nn.Module):
            total += int(sum(parameter.numel() for parameter in item.parameters()))
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
    return total


def _count_onnx_parameters(path: Path) -> int:
    try:
        import onnx
    except ImportError as error:
        raise ModelComplianceError(
            f"ONNX 模型 {path.name} 需要安装 onnx 才能审计参数量；"
            "请在 manifest 中声明 parameters 并移除此文件的自动审计"
        ) from error
    model = onnx.load(str(path), load_external_data=False)
    total = 0
    for initializer in model.graph.initializer:
        count = 1
        for dimension in initializer.dims:
            count *= int(dimension)
        total += count
    return total


def _count_joblib_parameters(path: Path) -> int:
    """按 sklearn 树模型节点数计参数；无法识别的模型返回 0。"""
    try:
        import joblib
    except ImportError as error:
        raise ModelComplianceError(
            f"审计 {path.name} 需要安装 joblib（已在 requirements.txt 中固定）"
        ) from error
    try:
        model = joblib.load(path)
    except Exception as error:  # noqa: BLE001 - 损坏/不兼容的权重
        raise ModelComplianceError(f"无法读取模型 {path.name}: {error}") from error
    total = 0
    estimators = getattr(model, "estimators_", None)
    if estimators is not None:
        for estimator in np.asarray(estimators).ravel():
            tree = getattr(estimator, "tree_", None)
            if tree is not None:
                total += int(tree.node_count)
    if total == 0:
        tree = getattr(model, "tree_", None)
        if tree is not None:
            total = int(tree.node_count)
    return total


def _count_file(path: Path, entry: dict[str, Any]) -> tuple[int, str]:
    declared = int(entry.get("parameters", 0))
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth", ".ckpt", ".bin"}:
        return _count_torch_parameters(path), "torch-state-dict"
    if suffix == ".onnx":
        return _count_onnx_parameters(path), "onnx-initializers"
    if suffix == ".joblib":
        counted = _count_joblib_parameters(path)
        if counted > 0:
            return counted, "sklearn-tree-nodes"
    # json / 无法识别结构的 joblib / npy 等：使用清单声明值。
    return declared, "manifest"


def audit_models(weights_dir: str | Path = "weights") -> AuditReport:
    directory = Path(weights_dir)
    manifest_path = directory / MANIFEST_NAME
    try:
        # 报告里写相对路径，避免把本机绝对路径带进提交物。
        display_path = str(manifest_path.resolve().relative_to(Path.cwd()))
    except ValueError:
        display_path = str(manifest_path)
    report = AuditReport(manifest_path=display_path)
    if not manifest_path.exists():
        raise ModelComplianceError(
            f"缺少模型清单 {manifest_path}；提交物必须声明所有模型及参数量"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ModelComplianceError(f"无法读取模型清单: {error}") from error

    if manifest.get("external_network_required") is not False:
        raise ModelComplianceError("模型清单必须声明 external_network_required=false")
    external_apis = manifest.get("external_apis") or []
    if external_apis:
        raise ModelComplianceError(f"评测期间禁止外部 API，清单仍声明了: {external_apis}")

    declared_limit = int(manifest.get("limit", MAX_MODEL_PARAMETERS))
    if declared_limit > MAX_MODEL_PARAMETERS:
        raise ModelComplianceError(
            f"清单 limit={declared_limit} 超过赛题上限 {MAX_MODEL_PARAMETERS}"
        )
    report.limit = declared_limit

    for entry in manifest.get("models", []):
        if not isinstance(entry, dict):
            raise ModelComplianceError("models 数组元素必须是对象")
        try:
            name = str(entry["name"])
            kind = str(entry.get("kind", "unknown"))
            relative = str(entry["path"])
        except KeyError as error:
            raise ModelComplianceError(f"模型条目缺少字段 {error}") from error
        path = (directory / relative).resolve()
        if directory.resolve() not in path.parents and path != directory.resolve():
            raise ModelComplianceError(f"模型路径越界: {relative}")
        exists = path.exists()
        if not exists and not entry.get("optional", False):
            raise ModelComplianceError(f"清单中的模型文件不存在: {relative}")
        if exists and path.is_file():
            parameters, source = _count_file(path, entry)
            size = path.stat().st_size
        else:
            parameters, source, size = int(entry.get("parameters", 0)), "manifest", 0
        report.models.append(ModelEntry(
            name=name, kind=kind, path=relative,
            parameters=parameters, parameter_source=source,
            exists=exists and path.is_file(), size_bytes=size,
        ))
        report.total_parameters += parameters

    declared_total = manifest.get("total_parameters")
    if declared_total is not None and int(declared_total) != report.total_parameters:
        raise ModelComplianceError(
            f"清单声明总参数量 {declared_total} 与实际审计 {report.total_parameters} 不一致"
        )
    if report.total_parameters > report.limit:
        raise ModelComplianceError(
            f"模型总参数量 {report.total_parameters} 超过上限 {report.limit}"
        )
    return report
