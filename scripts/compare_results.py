"""Generate local comparison artifacts from saved ogbn-arxiv results.

This script is intentionally analysis-only: it reads JSON outputs under
``results/<model>/`` and writes report-ready tables/plots under
``results/comparisons/``. It does not require Colab or GPU access.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.eda import load_dataset  # noqa: E402
from utils.viz import (  # noqa: E402
    plot_accuracy_delta,
    plot_cross_domain_delta,
    plot_grouped_accuracy,
    plot_per_class_accuracy,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Create local comparison tables and plots from saved metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory containing dataset_stats.json and results/<model>/ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <results-dir>/comparisons.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gcn", "gat", "gps"],
        help="Model result directories to include.",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="Skip loading OGB locally. Test support counts will be omitted.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write rows to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] Wrote {path}")


def load_available_models(
    results_dir: Path, requested_models: list[str]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, float | None]]]:
    """Load metrics and per-class accuracy for models with complete artifacts."""
    metrics_by_model: dict[str, dict[str, Any]] = {}
    per_class_by_model: dict[str, dict[int, float | None]] = {}

    for model in requested_models:
        model_dir = results_dir / model
        metrics_path = model_dir / "metrics.json"
        per_class_path = model_dir / "per_class_acc.json"
        if not metrics_path.exists() or not per_class_path.exists():
            print(f"[compare] Skipping {model}: missing metrics or per-class JSON")
            continue

        metrics_by_model[model] = load_json(metrics_path)
        raw_per_class = load_json(per_class_path)
        per_class_by_model[model] = {
            int(cls): (None if acc is None else float(acc))
            for cls, acc in raw_per_class.items()
        }

    if not metrics_by_model:
        raise FileNotFoundError(f"No model metrics found under {results_dir}")

    return metrics_by_model, per_class_by_model


def load_dataset_metadata(results_dir: Path) -> tuple[dict[int, str], dict[int, float], dict[int, int]]:
    """Load label names, cross-domain ratios, and total class counts."""
    stats = load_json(results_dir / "dataset_stats.json")
    label_names = {int(k): str(v) for k, v in stats.get("label_names", {}).items()}
    cross_domain = {
        int(k): float(v) for k, v in stats.get("cross_domain_citation_ratio", {}).items()
    }
    total_support = {
        int(k): int(v) for k, v in stats.get("class_distribution", {}).items()
    }
    return label_names, cross_domain, total_support


def compute_test_supports(data_root: Path) -> dict[int, int]:
    """Load local OGB data and compute test split support per class."""
    data, split_idx, dataset, _ = load_dataset(root=data_root)
    labels = data.y.view(-1).to(torch.long)
    test_labels = labels[split_idx["test"]]
    counts = torch.bincount(test_labels, minlength=int(dataset.num_classes))
    return {cls: int(counts[cls].item()) for cls in range(int(dataset.num_classes))}


def wilson_interval(acc: float | None, n: int | None, z: float = 1.96) -> tuple[float | None, float | None]:
    """Return a Wilson 95% confidence interval for a binomial accuracy."""
    if acc is None or n is None or n <= 0:
        return None, None

    p = float(acc)
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def macro_per_class_accuracy(per_class: dict[int, float | None]) -> float:
    """Average per-class accuracy, ignoring classes without support."""
    values = [float(v) for v in per_class.values() if v is not None]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def build_overall_rows(
    metrics_by_model: dict[str, dict[str, Any]],
    per_class_by_model: dict[str, dict[int, float | None]],
) -> list[dict[str, Any]]:
    """Build rows for overall_metrics.csv."""
    rows = []
    for model, metrics in metrics_by_model.items():
        rows.append(
            {
                "model": model.upper(),
                "best_val_acc": metrics.get("best_val_acc"),
                "test_acc": metrics.get("test_acc"),
                "macro_per_class_acc": macro_per_class_accuracy(per_class_by_model[model]),
                "best_epoch": metrics.get("best_epoch"),
                "epochs_ran": metrics.get("epochs_ran"),
                "final_train_acc": _last_or_none(metrics.get("train_acc")),
                "final_val_acc": _last_or_none(metrics.get("val_acc")),
                "final_train_loss": _last_or_none(metrics.get("train_loss")),
            }
        )
    return rows


def build_per_class_rows(
    per_class_by_model: dict[str, dict[int, float | None]],
    label_names: dict[int, str],
    cross_domain: dict[int, float],
    total_support: dict[int, int],
    test_support: dict[int, int] | None,
) -> list[dict[str, Any]]:
    """Build rows for per_class_accuracy.csv."""
    classes = sorted({cls for values in per_class_by_model.values() for cls in values})
    rows: list[dict[str, Any]] = []

    for cls in classes:
        row: dict[str, Any] = {
            "class_id": cls,
            "label": label_names.get(cls, str(cls)),
            "total_support": total_support.get(cls),
            "test_support": None if test_support is None else test_support.get(cls),
            "cross_domain_ratio": cross_domain.get(cls),
        }

        best_model = None
        best_acc = -1.0
        for model, per_class in per_class_by_model.items():
            acc = per_class.get(cls)
            support = None if test_support is None else test_support.get(cls)
            ci_low, ci_high = wilson_interval(acc, support)
            row[f"{model}_acc"] = acc
            row[f"{model}_ci_low"] = ci_low
            row[f"{model}_ci_high"] = ci_high
            if acc is not None and acc > best_acc:
                best_model = model.upper()
                best_acc = acc

        if "gps" in per_class_by_model and "gcn" in per_class_by_model:
            gps_acc = per_class_by_model["gps"].get(cls)
            gcn_acc = per_class_by_model["gcn"].get(cls)
            row["gps_minus_gcn"] = (
                None if gps_acc is None or gcn_acc is None else gps_acc - gcn_acc
            )

        row["best_model"] = best_model
        rows.append(row)
    return rows


def load_prediction_metrics(results_dir: Path, models: list[str]) -> list[dict[str, Any]]:
    """Compute F1/precision/recall if raw prediction exports are present."""
    try:
        from sklearn.metrics import precision_recall_fscore_support
    except ImportError:
        print("[compare] sklearn unavailable; skipping prediction metrics")
        return []

    rows: list[dict[str, Any]] = []
    for model in models:
        pred_path = results_dir / model / "test_predictions.npz"
        if not pred_path.exists():
            continue

        arr = np.load(pred_path)
        y_true = arr["y_true"].reshape(-1)
        y_pred = arr["y_pred"].reshape(-1)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(40)),
            zero_division=0,
        )
        for cls in range(40):
            rows.append(
                {
                    "model": model.upper(),
                    "class_id": cls,
                    "precision": float(precision[cls]),
                    "recall": float(recall[cls]),
                    "f1": float(f1[cls]),
                    "test_support": int(support[cls]),
                }
            )
    return rows


def write_summary_json(
    path: Path,
    metrics_by_model: dict[str, dict[str, Any]],
    per_class_by_model: dict[str, dict[int, float | None]],
    cross_domain: dict[int, float],
    label_names: dict[int, str],
) -> None:
    """Write a compact JSON summary for reports."""
    summary: dict[str, Any] = {
        "overall_test_acc": {
            model: metrics.get("test_acc") for model, metrics in metrics_by_model.items()
        },
        "macro_per_class_acc": {
            model: macro_per_class_accuracy(per_class)
            for model, per_class in per_class_by_model.items()
        },
    }

    if "gps" in per_class_by_model and "gcn" in per_class_by_model:
        deltas = {
            cls: float(per_class_by_model["gps"][cls] - per_class_by_model["gcn"][cls])
            for cls in sorted(set(per_class_by_model["gps"]).intersection(per_class_by_model["gcn"]))
            if per_class_by_model["gps"][cls] is not None
            and per_class_by_model["gcn"][cls] is not None
        }
        common = sorted(set(cross_domain).intersection(deltas))
        corr = None
        if len(common) > 1:
            corr = float(
                np.corrcoef(
                    [cross_domain[cls] for cls in common],
                    [deltas[cls] for cls in common],
                )[0, 1]
            )
        summary["gps_minus_gcn"] = {
            "cross_domain_correlation": corr,
            "largest_gains": _ranked_delta_rows(deltas, label_names, reverse=True),
            "largest_losses": _ranked_delta_rows(deltas, label_names, reverse=False),
        }

    if 6 in label_names:
        summary["cs_hc"] = {
            "class_id": 6,
            "label": label_names[6],
            "accuracy": {
                model: per_class.get(6) for model, per_class in per_class_by_model.items()
            },
            "cross_domain_ratio": cross_domain.get(6),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[compare] Wrote {path}")


def generate_plots(
    output_dir: Path,
    per_class_by_model: dict[str, dict[int, float | None]],
    cross_domain: dict[int, float],
    label_names: dict[int, str],
) -> None:
    """Generate comparison plots."""
    plot_grouped_accuracy(
        per_class_by_model,
        save_path=output_dir / "grouped_per_class_accuracy.png",
        label_names=label_names,
    )

    for model, per_class in per_class_by_model.items():
        plot_per_class_accuracy(
            per_class,
            model_name=model,
            save_path=output_dir / f"{model}_per_class_accuracy.png",
            label_names=label_names,
        )

    if "gcn" in per_class_by_model and "gps" in per_class_by_model:
        plot_accuracy_delta(
            base_per_class=per_class_by_model["gcn"],
            improved_per_class=per_class_by_model["gps"],
            base_name="GCN",
            improved_name="GPS",
            save_path=output_dir / "gps_minus_gcn_delta.png",
            label_names=label_names,
        )
        deltas = {
            cls: float(per_class_by_model["gps"][cls] - per_class_by_model["gcn"][cls])
            for cls in set(per_class_by_model["gps"]).intersection(per_class_by_model["gcn"])
            if per_class_by_model["gps"][cls] is not None
            and per_class_by_model["gcn"][cls] is not None
        }
        plot_cross_domain_delta(
            cross_domain_ratio=cross_domain,
            accuracy_delta=deltas,
            save_path=output_dir / "cross_domain_vs_gps_delta.png",
            label_names=label_names,
        )


def _last_or_none(values: Any) -> Any:
    """Return the last list item, or None."""
    if isinstance(values, list) and values:
        return values[-1]
    return None


def _ranked_delta_rows(
    deltas: dict[int, float],
    label_names: dict[int, str],
    reverse: bool,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return largest gain/loss rows."""
    ranked = sorted(deltas.items(), key=lambda item: item[1], reverse=reverse)[:limit]
    return [
        {"class_id": cls, "label": label_names.get(cls, str(cls)), "delta": delta}
        for cls, delta in ranked
    ]


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir / "comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_model, per_class_by_model = load_available_models(
        results_dir, args.models
    )
    label_names, cross_domain, total_support = load_dataset_metadata(results_dir)

    test_support = None
    if not args.skip_dataset:
        try:
            test_support = compute_test_supports(PROJECT_ROOT / "data")
            print("[compare] Loaded local OGB dataset for test support counts")
        except Exception as error:
            print(f"[compare] Could not compute test supports: {error}")

    overall_rows = build_overall_rows(metrics_by_model, per_class_by_model)
    write_csv(
        output_dir / "overall_metrics.csv",
        overall_rows,
        [
            "model",
            "best_val_acc",
            "test_acc",
            "macro_per_class_acc",
            "best_epoch",
            "epochs_ran",
            "final_train_acc",
            "final_val_acc",
            "final_train_loss",
        ],
    )

    per_class_rows = build_per_class_rows(
        per_class_by_model=per_class_by_model,
        label_names=label_names,
        cross_domain=cross_domain,
        total_support=total_support,
        test_support=test_support,
    )
    model_fields = []
    for model in metrics_by_model:
        model_fields.extend([f"{model}_acc", f"{model}_ci_low", f"{model}_ci_high"])
    write_csv(
        output_dir / "per_class_accuracy.csv",
        per_class_rows,
        [
            "class_id",
            "label",
            "total_support",
            "test_support",
            "cross_domain_ratio",
            *model_fields,
            "gps_minus_gcn",
            "best_model",
        ],
    )

    prediction_rows = load_prediction_metrics(results_dir, list(metrics_by_model))
    if prediction_rows:
        write_csv(
            output_dir / "prediction_metrics.csv",
            prediction_rows,
            ["model", "class_id", "precision", "recall", "f1", "test_support"],
        )
    else:
        print(
            "[compare] No test_predictions.npz files found; "
            "precision/F1 will be available after future train.py runs."
        )

    write_summary_json(
        output_dir / "summary.json",
        metrics_by_model=metrics_by_model,
        per_class_by_model=per_class_by_model,
        cross_domain=cross_domain,
        label_names=label_names,
    )
    generate_plots(
        output_dir=output_dir,
        per_class_by_model=per_class_by_model,
        cross_domain=cross_domain,
        label_names=label_names,
    )

    print(f"[compare] Complete. Artifacts are in {output_dir}")


if __name__ == "__main__":
    main()
