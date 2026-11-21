#!/usr/bin/env python3
"""训练图元级 GAT 全景检测模型（语义 + 邻接）。"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from cadvec.gat_model import PrimitiveGAT, parameter_count  # noqa: E402
from cadvec.primitive_graph import polar_edge_features  # noqa: E402

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    print("需要 torch", file=sys.stderr)
    raise

IGNORE_CLASS = 35


def _augment(features, edge_index, edge_features, labels, keys, extra):
    """几何抖动 + 翻转 + 少量节点丢弃，模拟 PNG 恢复误差。"""
    count = features.shape[0]
    if count == 0:
        return features, edge_index, edge_features, labels, keys, extra
    if torch.rand(1).item() < 0.5:
        features = features.clone()
        features[:, 0] = 1.0 - features[:, 0]
        features[:, 4] = -features[:, 4]
    if torch.rand(1).item() < 0.5:
        features = features.clone()
        features[:, 1] = 1.0 - features[:, 1]
        features[:, 4] = -features[:, 4]
    jitter = torch.randn(count, 2) * 0.004
    features = features.clone()
    features[:, 0:2] = features[:, 0:2] + jitter
    scale = torch.exp(torch.randn(count) * 0.1)
    features[:, 2] = features[:, 2] * scale
    features[:, 10] = features[:, 10] * scale
    features[:, 11] = features[:, 11] * scale
    color = torch.randn(count, 3) * 0.02
    features[:, 5:8] = (features[:, 5:8] + color).clamp(0.0, 1.0)
    # 节点丢弃（5%），同步过滤边并重映射。
    keep = torch.rand(count) >= 0.05
    if keep.sum() < 10:
        keep[:] = True
    if not bool(keep.all()):
        mapping = torch.full((count,), -1, dtype=torch.long)
        mapping[keep.nonzero().flatten()] = torch.arange(int(keep.sum()))
        edge_mask = keep[edge_index[0]] & keep[edge_index[1]]
        edge_index = mapping[edge_index[:, edge_mask]]
        edge_features = edge_features[edge_mask]
        features = features[keep]
        labels = labels[keep]
        keys = keys[keep]
        if extra is not None:
            extra = extra[keep]
    return features, edge_index, edge_features, labels, keys, extra


def load_sample(path: Path, node_features_dir: Path | None = None,
                augment: bool = False, node_feature_dim: int = 0):
    payload = np.load(path)
    features = torch.from_numpy(payload["features"]).float()
    edge_index = torch.from_numpy(payload["edge_index"]).long()
    edge_features = torch.from_numpy(payload["edge_features"]).float()
    labels = torch.from_numpy(payload["labels"]).long()
    keys = torch.from_numpy(payload["keys"]).long()
    extra = None
    if node_features_dir is not None:
        image_path = node_features_dir / f"{path.stem}.npy"
        if image_path.exists():
            extra = torch.from_numpy(np.load(image_path)).float()
        elif node_feature_dim > 0:
            extra = torch.zeros((features.shape[0], node_feature_dim))
    if augment:
        features, edge_index, edge_features, labels, keys, extra = _augment(
            features, edge_index, edge_features, labels, keys, extra,
        )
    edge_features = torch.from_numpy(polar_edge_features(
        features.numpy(), edge_index.numpy(), edge_features.numpy(),
    ))
    if extra is not None:
        features = torch.cat([features, extra], dim=1)
    count = features.shape[0]
    adjacency = torch.zeros((count, count), dtype=torch.bool)
    if edge_index.shape[1]:
        adjacency[edge_index[0], edge_index[1]] = True
        adjacency[edge_index[1], edge_index[0]] = True
    return features, edge_index, edge_features, adjacency, labels, keys


def class_weights(paths, sample: int = 300):
    counts = np.zeros(36, dtype=np.float64)
    for path in paths[:sample]:
        labels = np.load(path)["labels"]
        counts += np.bincount(labels, minlength=36)
    frequencies = counts / max(counts.sum(), 1.0)
    weights = np.power(1.0 / np.maximum(frequencies, 1e-4), 0.5)
    weights = np.clip(weights, 0.2, 20.0)
    weights[IGNORE_CLASS] = 0.5
    return torch.from_numpy(weights.astype(np.float32))




def run_epoch(model, paths, optimizer, criterion, device, training: bool,
              edge_pos_weight: float = 2.0, node_features_dir=None,
              node_feature_dim: int = 0):
    if training:
        model.train()
        random.shuffle(paths)
    else:
        model.eval()
    total_loss = 0.0
    node_correct = node_total = 0
    foreground_correct = foreground_total = 0
    tp = fp = fn = 0
    with torch.set_grad_enabled(training):
        for path in paths:
            features, edge_index, edge_features, adjacency, labels, keys = load_sample(
                path, node_features_dir, augment=training,
                node_feature_dim=node_feature_dim,
            )
            features = features.to(device)
            edge_index = edge_index.to(device)
            edge_features = edge_features.to(device)
            adjacency = adjacency.to(device)
            labels = labels.to(device)
            keys = keys.to(device)
            logits, edge_logits = model(features, adjacency, edge_index, edge_features)
            loss = criterion(logits, labels)
            if edge_index.shape[1]:
                edge_target = (
                    (keys[edge_index[0]] == keys[edge_index[1]])
                    & (keys[edge_index[0]] >= 0)
                ).float()
                pos_weight = torch.tensor(edge_pos_weight, device=device)
                loss = loss + nn.functional.binary_cross_entropy_with_logits(
                    edge_logits, edge_target, pos_weight=pos_weight,
                )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += float(loss)
            predictions = logits.argmax(dim=1)
            node_correct += int((predictions == labels).sum())
            node_total += labels.numel()
            foreground = labels != IGNORE_CLASS
            foreground_correct += int(((predictions == labels) & foreground).sum())
            foreground_total += int(foreground.sum())
            if edge_index.shape[1]:
                predicted = (torch.sigmoid(edge_logits) >= 0.5)
                target = edge_target > 0.5
                tp += int((predicted & target).sum())
                fp += int((predicted & ~target).sum())
                fn += int((~predicted & target).sum())
    accuracy = node_correct / max(node_total, 1)
    foreground_accuracy = foreground_correct / max(foreground_total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "loss": total_loss / max(len(paths), 1),
        "node_accuracy": accuracy,
        "foreground_accuracy": foreground_accuracy,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("../_work/primitive_graphs"))
    parser.add_argument("--out", type=Path, default=Path("weights/gat_primitive.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", type=Path, default=None,
                        help="从已有权重继续训练")
    parser.add_argument("--node-features", type=Path, default=None,
                        help="预计算的节点图像特征目录（可选）")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = sorted(args.data.glob("*.npz"))
    if args.limit:
        paths = paths[: args.limit]
    random.Random(args.seed).shuffle(paths)
    split = max(1, int(len(paths) * 0.9))
    train_paths, valid_paths = paths[:split], paths[split:]
    print(f"图 {len(paths)}（训练 {len(train_paths)} / 验证 {len(valid_paths)}）",
          file=sys.stderr)

    in_dim = 14
    node_feature_dim = 0
    if args.node_features is not None and args.node_features.exists():
        for candidate in train_paths:
            sample_path = args.node_features / f"{candidate.stem}.npy"
            if sample_path.exists():
                node_feature_dim = int(np.load(sample_path).shape[1])
                break
        in_dim = 14 + node_feature_dim
    print(f"节点输入维度 {in_dim}", file=sys.stderr)
    weights = class_weights(train_paths).to(device)
    model = PrimitiveGAT(in_dim=in_dim, dim=args.dim, layers=args.layers,
                         heads=args.heads)
    if args.init is not None and args.init.exists():
        checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(f"从 {args.init} 继续训练（缺失 {len(missing)} 多余 {len(unexpected)}）",
              file=sys.stderr)
    model = model.to(device)
    print(f"参数 {parameter_count(model):,}", file=sys.stderr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_paths),
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    best = -1.0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        metrics = run_epoch(model, train_paths, optimizer, criterion, device, True,
                            node_features_dir=args.node_features,
                            node_feature_dim=node_feature_dim)
        valid = run_epoch(model, valid_paths, None, criterion, device, False,
                          node_features_dir=args.node_features,
                          node_feature_dim=node_feature_dim)
        if scheduler.last_epoch < scheduler.total_steps:
            scheduler.step()
        elapsed = time.perf_counter() - started
        print(f"epoch {epoch + 1}: train loss {metrics['loss']:.3f} "
              f"node {metrics['node_accuracy']:.3f} fg {metrics['foreground_accuracy']:.3f} "
              f"edge-F1 {metrics['edge_f1']:.3f} | val fg {valid['foreground_accuracy']:.3f} "
              f"edge-F1 {valid['edge_f1']:.3f} ({elapsed:.0f}s)", file=sys.stderr)
        score = valid["edge_f1"] * 0.5 + valid["foreground_accuracy"] * 0.5
        if score > best:
            best = score
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "model_kwargs": {"in_dim": in_dim, "dim": args.dim,
                                 "layers": args.layers, "heads": args.heads},
                "uses_node_image_features": in_dim > 14,
                "metrics": valid, "epoch": epoch + 1,
            }, args.out)
    print(f"最佳权重 {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
