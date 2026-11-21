"""图元级全景检测：图注意力网络（GAT-lite）。

节点 = 笔画，边 = KNN/端点连接；两个头：
- 语义头：36 类（含忽略 35）；
- 邻接头：边两端是否属于同一实例（同一 GT 实例 / 同类别 Stuff）。

推理时同类且邻接概率高的节点用并查集聚成实例。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitive_graph import EDGE_FEATURES, NODE_FEATURES

CLASS_COUNT = 36


class GraphAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, edge_dim: int = EDGE_FEATURES):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_bias = nn.Linear(edge_dim, heads)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adjacency, edge_index, edge_features):
        count = x.shape[0]
        residual = x
        q = self.q(x).view(count, self.heads, self.head_dim)
        k = self.k(x).view(count, self.heads, self.head_dim)
        v = self.v(x).view(count, self.heads, self.head_dim)
        scores = torch.einsum("ihd,jhd->ijh", q, k) / math.sqrt(self.head_dim)
        if edge_features.numel():
            bias = self.edge_bias(edge_features)  # E×H
            full = torch.full((count, count, self.heads), -10.0, device=x.device,
                              dtype=bias.dtype)
            rows, columns = edge_index[0], edge_index[1]
            full[rows, columns] = bias
            full[columns, rows] = bias
            scores = scores + full
        scores = scores.masked_fill(~adjacency.unsqueeze(-1), float("-inf"))
        attention = torch.softmax(scores, dim=1)
        attention = torch.nan_to_num(attention, nan=0.0)
        out = torch.einsum("ijh,jhd->ihd", attention, v).reshape(count, -1)
        return self.norm(residual + F.gelu(self.out(out)))


class PrimitiveGAT(nn.Module):
    def __init__(self, in_dim: int = NODE_FEATURES, dim: int = 128,
                 layers: int = 4, heads: int = 4, classes: int = CLASS_COUNT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, dim), nn.GELU(), nn.LayerNorm(dim),
        )
        self.layers = nn.ModuleList([GraphAttention(dim, heads) for _ in range(layers)])
        self.cls_head = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, classes),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(dim * 2 + EDGE_FEATURES, dim), nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, features, adjacency, edge_index, edge_features):
        x = self.encoder(features)
        for layer in self.layers:
            x = layer(x, adjacency, edge_index, edge_features)
        logits = self.cls_head(x)
        if edge_index.shape[1]:
            rows, columns = edge_index[0], edge_index[1]
            pair = torch.cat([x[rows], x[columns], edge_features], dim=1)
            edge_logits = self.edge_head(pair).squeeze(-1)
        else:
            edge_logits = torch.zeros(0, device=x.device)
        return logits, edge_logits


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
