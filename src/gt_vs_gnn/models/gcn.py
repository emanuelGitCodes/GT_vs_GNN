"""GCN baseline model for ogbn-arxiv (Phase 2)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    """Three-layer GCN with BatchNorm, ReLU, and dropout."""

    def __init__(self, cfg: dict) -> None:
        """Initialise GCN from a configuration dictionary."""
        super().__init__()
        in_channels = int(cfg["in_channels"])
        hidden_dim = int(cfg.get("hidden_dim", 256))
        out_channels = int(cfg["num_classes"])
        num_layers = int(cfg.get("num_layers", 3))
        dropout = float(cfg.get("dropout", 0.5))

        if num_layers < 2:
            raise ValueError("GCN requires num_layers >= 2")

        self.dropout: float = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_channels, hidden_dim, cached=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, cached=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, out_channels, cached=True))

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Run a forward pass and return class logits."""
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x
