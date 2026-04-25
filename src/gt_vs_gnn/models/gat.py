"""GAT baseline model for ogbn-arxiv (Phase 3)."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Three-layer GAT with multi-head attention and dropout.

    Hidden layers use multi-head attention with concatenation where
    ``hidden_dim`` is the hidden size *per attention head*.

    This variant includes BatchNorm + residual connections to improve
    optimization stability on ogbn-arxiv.
    """

    def __init__(self, cfg: dict) -> None:
        """Initialise GAT from a configuration dictionary."""
        super().__init__()
        in_channels = int(cfg["in_channels"])
        hidden_dim = int(cfg.get("hidden_dim", 256))
        out_channels = int(cfg["num_classes"])
        num_layers = int(cfg.get("num_layers", 3))
        num_heads = int(cfg.get("num_heads", 8))
        out_heads = int(cfg.get("out_heads", 1))
        dropout = float(cfg.get("dropout", 0.5))

        if num_layers != 3:
            raise ValueError("Phase 3 GAT baseline expects num_layers == 3")
        if num_heads <= 0:
            raise ValueError("num_heads must be > 0")

        self.dropout: float = dropout
        total_hidden = hidden_dim * num_heads

        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.conv2 = GATConv(
            in_channels=total_hidden,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.conv3 = GATConv(
            in_channels=total_hidden,
            out_channels=out_channels,
            heads=out_heads,
            dropout=dropout,
            concat=False,
        )

        self.bn1 = nn.BatchNorm1d(total_hidden)
        self.bn2 = nn.BatchNorm1d(total_hidden)

        self.res1 = nn.Linear(in_channels, total_hidden, bias=False)
        self.res2 = nn.Identity()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Run a forward pass and return class logits."""
        x0 = x
        x = F.dropout(x0, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x + self.res1(x0))

        x_res = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x + self.res2(x_res))

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x
