"""GAT baseline model for ogbn-arxiv (Phase 3)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    """Three-layer GAT with multi-head attention and dropout.

    Hidden layers use multi-head attention with concatenation while preserving
    total hidden width ``hidden_dim`` by setting each head dimension to
    ``hidden_dim // num_heads``.
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
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.dropout: float = dropout
        head_dim = hidden_dim // num_heads

        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=head_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.conv2 = GATConv(
            in_channels=hidden_dim,
            out_channels=head_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.conv3 = GATConv(
            in_channels=hidden_dim,
            out_channels=out_channels,
            heads=out_heads,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Run a forward pass and return class logits."""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x
