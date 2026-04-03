"""Visualization utilities for training curves, per-class accuracy, and embeddings.

Phase 0: Stubs only — full implementations added in Phases 5 & 6.

Usage (once implemented)
------------------------
    from utils.viz import plot_training_curves, plot_per_class_accuracy
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Phase 2–4: Training diagnostics
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_accs: list[float],
    val_accs: list[float],
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot train/val accuracy curves over epochs.

    Implemented in Phase 2 once the training loop exists.
    """
    raise NotImplementedError("plot_training_curves — implemented in Phase 2")


# ---------------------------------------------------------------------------
# Phase 5: Comparative evaluation
# ---------------------------------------------------------------------------


def plot_per_class_accuracy(
    per_class_dict: dict[int, Optional[float]],
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart of per-class accuracy for a single model.

    Implemented in Phase 5.
    """
    raise NotImplementedError("plot_per_class_accuracy — implemented in Phase 5")


def plot_accuracy_delta(
    base_per_class: dict[int, Optional[float]],
    improved_per_class: dict[int, Optional[float]],
    base_name: str = "GCN",
    improved_name: str = "GPS",
    save_path: Optional[Path] = None,
) -> None:
    """Sorted delta bar chart: (improved - base) accuracy per class.

    Highlights where GPS gains/loses most relative to GCN.
    Implemented in Phase 5.
    """
    raise NotImplementedError("plot_accuracy_delta — implemented in Phase 5")


def plot_grouped_accuracy(
    per_class_results: dict[str, dict[int, Optional[float]]],
    save_path: Optional[Path] = None,
) -> None:
    """Grouped bar chart: all models × all 40 classes.

    Parameters
    ----------
    per_class_results:
        ``{"GCN": {0: 0.72, ...}, "GAT": {...}, "GPS": {...}}``

    Implemented in Phase 5.
    """
    raise NotImplementedError("plot_grouped_accuracy — implemented in Phase 5")


# ---------------------------------------------------------------------------
# Phase 6: Embedding visualization
# ---------------------------------------------------------------------------


def plot_tsne(
    embeddings,  # np.ndarray (N, D)
    labels,  # np.ndarray (N,)
    model_name: str,
    random_state: int = 42,
    perplexity: float = 30.0,
    save_path: Optional[Path] = None,
) -> None:
    """t-SNE projection of node embeddings, coloured by class.

    Fix ``random_state`` and ``perplexity`` across models so differences
    reflect the embeddings, not the projection hyperparameters.
    Implemented in Phase 6.
    """
    raise NotImplementedError("plot_tsne — implemented in Phase 6")


def plot_attention_heatmap(
    attention_matrix,  # np.ndarray (num_cs_hc_nodes, num_classes)
    save_path: Optional[Path] = None,
) -> None:
    """Heatmap of GPS attention distribution for cs.HC nodes across categories.

    Implemented in Phase 6.
    """
    raise NotImplementedError("plot_attention_heatmap — implemented in Phase 6")
