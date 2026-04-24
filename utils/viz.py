"""Visualization utilities for training curves, per-class accuracy, and embeddings.

Usage (once implemented)
------------------------
    from utils.viz import plot_training_curves, plot_per_class_accuracy
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Phase 2–4: Training diagnostics
# ---------------------------------------------------------------------------


def plot_training_curves(
    train_accs: list[float],
    val_accs: list[float],
    model_name: str,
    save_path: Optional[Path] = None,
) -> None:
    """Plot train/val accuracy curves over epochs."""
    if len(train_accs) == 0 or len(val_accs) == 0:
        raise ValueError("train_accs and val_accs must be non-empty")
    if len(train_accs) != len(val_accs):
        raise ValueError("train_accs and val_accs must have equal length")

    epochs = list(range(1, len(train_accs) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy", linewidth=2)
    plt.plot(epochs, val_accs, label="Validation Accuracy", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} Training Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[viz] Saved training curves → {save_path}")

    plt.close()


# ---------------------------------------------------------------------------
# Phase 5: Comparative evaluation
# ---------------------------------------------------------------------------


def plot_per_class_accuracy(
    per_class_dict: dict[int, Optional[float]],
    model_name: str,
    save_path: Optional[Path] = None,
    label_names: Optional[dict[int, str]] = None,
) -> None:
    """Bar chart of per-class accuracy for a single model."""
    items = sorted(per_class_dict.items())
    classes = [int(c) for c, _ in items]
    values = [np.nan if acc is None else float(acc) for _, acc in items]
    labels = _class_labels(classes, label_names)

    fig_width = max(10, len(classes) * 0.32)
    plt.figure(figsize=(fig_width, 5))
    plt.bar(labels, values, color="#4c78a8")
    plt.ylim(0, 1)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name.upper()} Per-Class Accuracy")
    plt.xticks(rotation=75, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_accuracy_delta(
    base_per_class: dict[int, Optional[float]],
    improved_per_class: dict[int, Optional[float]],
    base_name: str = "GCN",
    improved_name: str = "GPS",
    save_path: Optional[Path] = None,
    label_names: Optional[dict[int, str]] = None,
) -> None:
    """Sorted delta bar chart: (improved - base) accuracy per class.

    Highlights where GPS gains/loses most relative to GCN.
    """
    common = sorted(set(base_per_class).intersection(improved_per_class))
    deltas = []
    for cls in common:
        base_acc = base_per_class[cls]
        improved_acc = improved_per_class[cls]
        if base_acc is None or improved_acc is None:
            continue
        deltas.append((cls, float(improved_acc) - float(base_acc)))

    deltas.sort(key=lambda item: item[1])
    classes = [cls for cls, _ in deltas]
    values = [delta for _, delta in deltas]
    labels = _class_labels(classes, label_names)
    colors = ["#d95f02" if v < 0 else "#1b9e77" for v in values]

    fig_width = max(10, len(classes) * 0.32)
    plt.figure(figsize=(fig_width, 5))
    plt.bar(labels, values, color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Class")
    plt.ylabel(f"{improved_name.upper()} - {base_name.upper()} Accuracy")
    plt.title(f"Per-Class Accuracy Delta: {improved_name.upper()} vs {base_name.upper()}")
    plt.xticks(rotation=75, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_grouped_accuracy(
    per_class_results: dict[str, dict[int, Optional[float]]],
    save_path: Optional[Path] = None,
    label_names: Optional[dict[int, str]] = None,
) -> None:
    """Grouped bar chart: all models × all 40 classes.

    Parameters
    ----------
    per_class_results:
        ``{"GCN": {0: 0.72, ...}, "GAT": {...}, "GPS": {...}}``
    """
    if not per_class_results:
        raise ValueError("per_class_results must not be empty")

    model_names = list(per_class_results.keys())
    classes = sorted({int(c) for values in per_class_results.values() for c in values})
    x = np.arange(len(classes))
    width = min(0.8 / len(model_names), 0.25)

    fig_width = max(11, len(classes) * 0.36)
    plt.figure(figsize=(fig_width, 5.5))
    for idx, model_name in enumerate(model_names):
        values = [
            np.nan
            if per_class_results[model_name].get(cls) is None
            else float(per_class_results[model_name][cls])
            for cls in classes
        ]
        offset = (idx - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=model_name.upper())

    plt.ylim(0, 1)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Accuracy by Model")
    plt.xticks(x, _class_labels(classes, label_names), rotation=75, ha="right")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_cross_domain_delta(
    cross_domain_ratio: dict[int, float],
    accuracy_delta: dict[int, float],
    save_path: Optional[Path] = None,
    label_names: Optional[dict[int, str]] = None,
) -> None:
    """Scatter plot of cross-domain ratio vs model accuracy delta."""
    common = sorted(set(cross_domain_ratio).intersection(accuracy_delta))
    if not common:
        raise ValueError("No overlapping classes for cross-domain delta plot")

    x = np.array([float(cross_domain_ratio[cls]) for cls in common])
    y = np.array([float(accuracy_delta[cls]) for cls in common])

    corr = float(np.corrcoef(x, y)[0, 1]) if len(common) > 1 else float("nan")

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, color="#4c78a8", edgecolor="white", linewidth=0.6, s=55)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Cross-Domain Citation Ratio")
    plt.ylabel("Accuracy Delta")
    plt.title(f"Cross-Domain Ratio vs Accuracy Delta (r={corr:.2f})")
    plt.grid(alpha=0.3)

    if label_names is not None:
        for cls, x_val, y_val in zip(common, x, y):
            if abs(y_val) >= 0.07 or x_val >= 0.75:
                plt.annotate(
                    label_names.get(cls, str(cls)),
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

    plt.tight_layout()
    _save_or_show(save_path)


def _class_labels(classes: list[int], label_names: Optional[dict[int, str]]) -> list[str]:
    """Return compact x-axis labels for class IDs."""
    if label_names is None:
        return [str(cls) for cls in classes]
    return [label_names.get(cls, str(cls)) for cls in classes]


def _save_or_show(save_path: Optional[Path]) -> None:
    """Save the active Matplotlib figure when a path is provided."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"[viz] Saved -> {save_path}")
    plt.close()


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
