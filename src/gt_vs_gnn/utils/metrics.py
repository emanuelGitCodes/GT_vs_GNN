"""Metrics utilities: OGB evaluator wrapper and per-class accuracy helpers.

All functions accept CPU tensors. Move tensors off the device before calling.

Usage
-----
    from gt_vs_gnn.utils.metrics import get_evaluator, eval_acc, per_class_accuracy

    evaluator = get_evaluator()
    acc = eval_acc(evaluator, y_pred, y_true)
    per_cls = per_class_accuracy(y_pred, y_true)
"""

import json
from pathlib import Path
from typing import Optional

import torch
from ogb.nodeproppred import Evaluator


def get_evaluator(dataset_name: str = "ogbn-arxiv") -> Evaluator:
    """Return an OGB Evaluator for the given dataset."""
    return Evaluator(name=dataset_name)


def eval_acc(
    evaluator: Evaluator,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """Compute OGB accuracy.

    Parameters
    ----------
    evaluator:
        OGB Evaluator instance (from ``get_evaluator()``).
    y_pred:
        Predicted class indices, shape ``(N,)`` or ``(N, 1)``.
    y_true:
        Ground-truth class indices, shape ``(N,)`` or ``(N, 1)``.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    # OGB evaluator expects shape (N, 1)
    if y_pred.dim() == 1:
        y_pred = y_pred.unsqueeze(-1)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)

    return evaluator.eval({"y_pred": y_pred, "y_true": y_true})["acc"]


def per_class_accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int = 40,
) -> dict[int, Optional[float]]:
    """Compute per-class accuracy.

    Parameters
    ----------
    y_pred:
        Predicted class indices, shape ``(N,)``.
    y_true:
        Ground-truth class indices, shape ``(N,)``.
    num_classes:
        Total number of classes (40 for ogbn-arxiv).

    Returns
    -------
    dict[int, float | None]
        Mapping from class id → accuracy (or ``None`` if the class has no
        samples in the evaluated split).
    """
    # Flatten to 1-D if needed
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)

    results: dict[int, Optional[float]] = {}
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() == 0:
            results[c] = None
        else:
            results[c] = (y_pred[mask] == y_true[mask]).float().mean().item()
    return results


def save_metrics(metrics: dict, save_dir: Path, filename: str = "metrics.json") -> None:
    """Persist a metrics dictionary as JSON.

    Parameters
    ----------
    metrics:
        Arbitrary dict of scalar values (floats, ints, lists).
    save_dir:
        Directory to write into (created if it does not exist).
    filename:
        Output filename, default ``metrics.json``.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / filename
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[metrics] Saved → {out_path}")


def load_metrics(path: Path) -> dict:
    """Load a JSON metrics file."""
    with open(path) as f:
        return json.load(f)
