"""GPS (Graph Transformer) model for ogbn-arxiv (Phase 4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, GPSConv


class GPS(nn.Module):
    """GraphGPS-style node classifier with Laplacian PE input projection."""

    def __init__(self, cfg: dict) -> None:
        """Initialise GPS from a configuration dictionary."""
        super().__init__()
        in_channels = int(cfg["in_channels"])
        num_classes = int(cfg["num_classes"])
        hidden_dim = int(cfg.get("hidden_dim", 256))
        num_layers = int(cfg.get("num_layers", 4))
        num_heads = int(cfg.get("num_heads", 8))
        dropout = float(cfg.get("dropout", 0.5))
        lap_pe_k = int(cfg.get("lap_pe_k", 16))

        if num_layers < 1:
            raise ValueError("GPS requires num_layers >= 1")
        if num_heads <= 0:
            raise ValueError("GPS requires num_heads > 0")

        self.dropout: float = dropout
        self.lap_pe_k: int = lap_pe_k

        self.input_proj = nn.Linear(in_channels + lap_pe_k, hidden_dim)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            # For ogbn-arxiv node classification, a simple GCN local MPNN tends
            # to be more stable than gated recurrent local propagation.
            local_mpnn = GCNConv(hidden_dim, hidden_dim)
            self.layers.append(
                GPSConv(
                    channels=hidden_dim,
                    conv=local_mpnn,
                    heads=num_heads,
                    dropout=dropout,
                    act="relu",
                    norm="layer_norm",
                )
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def _prepare_pe(self, lap_pe: Tensor) -> Tensor:
        """Return LapPE tensor with expected dimensionality."""
        if lap_pe.dim() == 1:
            lap_pe = lap_pe.unsqueeze(-1)

        if lap_pe.size(-1) != self.lap_pe_k:
            raise ValueError(
                f"Expected LapPE dim {self.lap_pe_k}, got {lap_pe.size(-1)}. "
                "Check lap_pe_k config and cached PE tensor."
            )
        return lap_pe

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        lap_pe: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Run a forward pass and return class logits."""
        lap_pe = self._prepare_pe(lap_pe)
        x = torch.cat([x, lap_pe], dim=-1)
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, edge_index, batch=batch)

        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
