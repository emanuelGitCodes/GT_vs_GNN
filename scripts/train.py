"""train.py — Training entry point for GCN / GAT / GPS on ogbn-arxiv.

Phases 2–3 implement full-batch training pipelines for GCN and GAT baselines.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# Make sure the project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.device import empty_cache, get_device, sanity_check  # noqa: E402
from utils.eda import load_dataset  # noqa: E402
from utils.metrics import (  # noqa: E402
    eval_acc,
    get_evaluator,
    per_class_accuracy,
    save_metrics,
)
from utils.viz import plot_training_curves  # noqa: E402


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a GNN (GCN / GAT / GPS) on ogbn-arxiv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--model",
        type=str,
        choices=["gcn", "gat", "gps"],
        required=True,
        help="Which model architecture to train.",
    )

    # Optional config override
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a YAML config file. Defaults to configs/<model>.yaml.",
    )

    # Hyperparameter overrides (all optional — fall back to YAML values)
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--wd",
        type=float,
        default=None,
        dest="weight_decay",
        help="Weight decay (L2 regularisation).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience (epochs without val improvement).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "mps", "cuda", "cpu"],
        default=None,
        help="Device backend override. 'auto' uses MPS → CUDA → CPU.",
    )

    # Output
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Directory to save checkpoints and metrics. Defaults to results/<model>/.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(args: argparse.Namespace) -> dict:
    """Load YAML config, then apply any CLI overrides.

    Priority: CLI flags > YAML file > (nothing — required fields must be in YAML).
    """
    config_path = (
        Path(args.config)
        if args.config
        else PROJECT_ROOT / "configs" / f"{args.model}.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg: dict = yaml.safe_load(f)

    # Apply CLI overrides for any explicitly provided argument
    cli_overrides = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "seed": args.seed,
        "device": args.device,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            cfg[key] = val
            print(f"[config] CLI override: {key} = {val}")

    # Resolve results directory
    if args.results_dir:
        cfg["results_dir"] = Path(args.results_dir)
    else:
        cfg["results_dir"] = PROJECT_ROOT / "results" / args.model

    return cfg


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        # MPS does not expose a separate seed API; torch.manual_seed covers it.
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed] Random seed set to {seed}")


def _is_mps_scatter_error(error: Exception) -> bool:
    """Return True when the exception likely indicates an MPS sparse/scatter gap."""
    message = str(error).lower()
    markers = ["mps", "scatter", "sparse", "not implemented"]
    return any(marker in message for marker in markers)


def _mps_memory_info(device: torch.device) -> str:
    """Return allocated MPS memory string for epoch logs."""
    if device.type != "mps":
        return ""
    allocated_mb = torch.mps.current_allocated_memory() / 1e6
    return f" | mps_mem {allocated_mb:.1f}MB"


def train_epoch(
    model: torch.nn.Module,
    data: Data,
    split_idx: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
) -> tuple[float, float]:
    """Run one training epoch and return (loss, train_acc)."""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    train_idx = split_idx["train"]
    y_train = data.y[train_idx].view(-1)
    loss = F.cross_entropy(out[train_idx], y_train)
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        pred = out[train_idx].argmax(dim=-1)
        train_acc = float((pred == y_train).float().mean().item())

    return float(loss.item()), train_acc


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: Data,
    split_idx: dict[str, Tensor],
    evaluator: Any,
    split_name: str,
    device: torch.device,
) -> float:
    """Evaluate model accuracy on a split using OGB evaluator."""
    model.eval()
    idx = split_idx[split_name]

    out = model(data.x, data.edge_index)
    y_pred = out[idx].argmax(dim=-1).cpu()
    y_true = data.y[idx].view(-1).cpu()
    acc = float(eval_acc(evaluator, y_pred, y_true))

    # Required for MPS memory stability after evaluation loops.
    empty_cache(device)
    return acc


@torch.no_grad()
def evaluate_with_predictions(
    model: torch.nn.Module,
    data: Data,
    split_idx: dict[str, Tensor],
    evaluator: Any,
    device: torch.device,
) -> tuple[float, Tensor, Tensor]:
    """Evaluate test split and return (acc, y_pred, y_true)."""
    model.eval()
    test_idx = split_idx["test"]
    out = model(data.x, data.edge_index)

    y_pred = out[test_idx].argmax(dim=-1).cpu()
    y_true = data.y[test_idx].view(-1).cpu()
    test_acc = float(eval_acc(evaluator, y_pred, y_true))

    # Required for MPS memory stability after evaluation loops.
    empty_cache(device)
    return test_acc, y_pred, y_true


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()
    cfg = load_config(args)

    print(f"\n{'=' * 60}")
    print(f"  Model : {cfg['model'].upper()}")
    print(f"  Config: {cfg}")
    print(f"{'=' * 60}\n")

    # Reproducibility
    set_seed(cfg["seed"])

    # Device setup
    requested_device = get_device(str(cfg.get("device", "auto")))
    sanity_check(requested_device)
    device = requested_device

    # Ensure results directory exists
    results_dir: Path = cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] Results will be saved to: {results_dir}")

    # Load ogbn-arxiv dataset
    data, split_idx, dataset, _ = load_dataset(root=PROJECT_ROOT / "data")

    # ogbn-arxiv is directed (citation graph). Our full-batch GCN/GAT baselines
    # use the commonly adopted undirected variant for stable message passing.
    if cfg["model"] in {"gcn", "gat"}:
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    cfg.setdefault("in_channels", int(data.x.size(1)))
    cfg.setdefault("num_classes", int(dataset.num_classes))

    # Model instantiation (Phases 2–3)
    if cfg["model"] == "gcn":
        from models.gcn import GCN

        model = GCN(cfg).to(device)
    elif cfg["model"] == "gat":
        from models.gat import GAT

        model = GAT(cfg).to(device)
    else:
        raise NotImplementedError(
            f"Model '{cfg['model']}' is not implemented in train.py yet. "
            "Current support: --model gcn and --model gat."
        )

    data = data.to(device)
    split_idx = {k: v.to(device) for k, v in split_idx.items()}
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    evaluator = get_evaluator("ogbn-arxiv")

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    train_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    checkpoint_path = results_dir / "best_model.pt"

    for epoch in range(1, int(cfg["epochs"]) + 1):
        try:
            train_loss, train_acc = train_epoch(model, data, split_idx, optimizer)
            val_acc = evaluate(
                model=model,
                data=data,
                split_idx=split_idx,
                evaluator=evaluator,
                split_name="valid",
                device=device,
            )
        except (RuntimeError, NotImplementedError) as error:
            if device.type == "mps" and _is_mps_scatter_error(error):
                # MPS can fail on some PyG sparse/scatter kernels.
                # We fall back to CPU to keep training functional.
                print(
                    "[warn] MPS sparse/scatter op issue detected. "
                    "Falling back to CPU for training."
                )
                device = torch.device("cpu")
                model = model.to(device)
                data = data.to(device)
                split_idx = {k: v.to(device) for k, v in split_idx.items()}
                train_loss, train_acc = train_epoch(model, data, split_idx, optimizer)
                val_acc = evaluate(
                    model=model,
                    data=data,
                    split_idx=split_idx,
                    evaluator=evaluator,
                    split_name="valid",
                    device=device,
                )
            else:
                raise

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if epoch % 10 == 0:
            mem_info = _mps_memory_info(device)
            print(
                f"Epoch {epoch:04d} | "
                f"loss {train_loss:.4f} | "
                f"train {train_acc:.4f} | "
                f"val {val_acc:.4f}"
                f"{mem_info}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= int(cfg["patience"]):
                print(f"[early-stop] No val improvement for {cfg['patience']} epochs.")
                print(f"[early-stop] Stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_acc, y_pred_test, y_true_test = evaluate_with_predictions(
        model=model,
        data=data,
        split_idx=split_idx,
        evaluator=evaluator,
        device=device,
    )
    per_class = per_class_accuracy(
        y_pred=y_pred_test,
        y_true=y_true_test,
        num_classes=int(cfg["num_classes"]),
    )

    plot_training_curves(
        train_accs=train_accs,
        val_accs=val_accs,
        model_name=cfg["model"],
        save_path=results_dir / "training_curves.png",
    )

    save_metrics(
        metrics={
            "model": cfg["model"],
            "best_val_acc": best_val_acc,
            "best_epoch": best_epoch,
            "test_acc": test_acc,
            "epochs_ran": len(train_losses),
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_acc": val_accs,
        },
        save_dir=results_dir,
        filename="metrics.json",
    )
    save_metrics(
        metrics={str(k): v for k, v in per_class.items()},
        save_dir=results_dir,
        filename="per_class_acc.json",
    )

    phase_label = "phase-2" if cfg["model"] == "gcn" else "phase-3"
    print(f"\n[{phase_label}] Training complete.")
    print(
        f"[{phase_label}] Best validation accuracy: {best_val_acc:.4f} "
        f"(epoch {best_epoch})"
    )
    print(f"[{phase_label}] Test accuracy: {test_acc:.4f}")
    print(f"[{phase_label}] Checkpoint: {checkpoint_path}")
    print(f"[{phase_label}] Per-class metrics: {results_dir / 'per_class_acc.json'}")


if __name__ == "__main__":
    main()
