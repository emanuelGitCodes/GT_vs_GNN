"""train.py — Training entry point for GCN / GAT / GPS on ogbn-arxiv.

Phase 0: Skeleton with argparse + device detection.
         Model instantiation and training loop added in Phases 2–4.

Example usage
-------------
    # Use defaults from configs/gcn.yaml
    python scripts/train.py --model gcn

    # Override specific hyperparameters via CLI
    python scripts/train.py --model gps --epochs 200 --lr 0.0005 --seed 0

    # Point to a custom config file
    python scripts/train.py --model gat --config configs/gat.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make sure the project root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.device import get_device, sanity_check  # noqa: E402


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
    device = get_device()
    sanity_check(device)

    # Ensure results directory exists
    results_dir: Path = cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] Results will be saved to: {results_dir}")

    # ------------------------------------------------------------------
    # TODO (Phase 1): Load ogbn-arxiv dataset
    #   from ogb.nodeproppred import PygNodePropPredDataset
    #   dataset = PygNodePropPredDataset(name="ogbn-arxiv", root="data/")
    #   data = dataset[0]
    #   split_idx = dataset.get_idx_split()
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO (Phase 2): Instantiate and train GCN
    #   from models.gcn import GCN
    #   model = GCN(cfg).to(device)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO (Phase 3): Instantiate and train GAT
    #   from models.gat import GAT
    #   model = GAT(cfg).to(device)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO (Phase 4): Instantiate and train GPS
    #   from models.gps import GPS
    #   model = GPS(cfg).to(device)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # TODO (Phase 2–4): Training loop skeleton
    #
    #   optimizer = torch.optim.Adam(model.parameters(),
    #                                lr=cfg["lr"],
    #                                weight_decay=cfg["weight_decay"])
    #   evaluator = get_evaluator()
    #   writer = SummaryWriter(log_dir=results_dir / "tb_logs")
    #
    #   best_val_acc, patience_counter = 0.0, 0
    #   for epoch in range(1, cfg["epochs"] + 1):
    #       train_loss = train_epoch(model, data, optimizer, device)
    #       val_acc    = evaluate(model, data, split_idx["valid"], evaluator, device)
    #
    #       if epoch % 10 == 0:
    #           print(f"Epoch {epoch:04d} | loss {train_loss:.4f} | val {val_acc:.4f}")
    #
    #       writer.add_scalar("Loss/train", train_loss, epoch)
    #       writer.add_scalar("Acc/val",    val_acc,    epoch)
    #
    #       # Early stopping
    #       if val_acc > best_val_acc:
    #           best_val_acc = val_acc
    #           patience_counter = 0
    #           torch.save(model.state_dict(), results_dir / "best_model.pt")
    #       else:
    #           patience_counter += 1
    #           if patience_counter >= cfg["patience"]:
    #               print(f"Early stopping at epoch {epoch}")
    #               break
    #
    #   # Final test evaluation
    #   model.load_state_dict(torch.load(results_dir / "best_model.pt"))
    #   test_acc = evaluate(model, data, split_idx["test"], evaluator, device)
    #   print(f"Test accuracy: {test_acc:.4f}")
    #   save_metrics({"test_acc": test_acc, ...}, results_dir)
    # ------------------------------------------------------------------

    print("\n[train.py] Phase 0 scaffold complete. Training loop added in Phases 2–4.")


if __name__ == "__main__":
    main()
