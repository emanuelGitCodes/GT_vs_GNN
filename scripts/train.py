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
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.typing import WITH_PYG_LIB, WITH_TORCH_SPARSE
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


def build_split_masks(data: Data, split_idx: dict[str, Tensor]) -> None:
    """Attach boolean train/valid/test masks to *data* from split indices."""
    num_nodes = int(data.num_nodes)
    for split_name in ("train", "valid", "test"):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[split_idx[split_name].cpu()] = True
        setattr(data, f"{split_name}_mask", mask)


def lap_pe_cache_path(root: Path, lap_pe_k: int) -> Path:
    """Return cache path for Laplacian positional encodings."""
    return root / "processed" / f"ogbn_arxiv_lap_pe_k{lap_pe_k}.pt"


def add_or_load_laplacian_pe(data: Data, root: Path, lap_pe_k: int) -> Data:
    """Attach cached Laplacian PE (or compute it once and cache)."""
    cache_path = lap_pe_cache_path(root=root, lap_pe_k=lap_pe_k)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    lap_pe: Tensor | None = None
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu")
        if isinstance(cached, torch.Tensor):
            lap_pe = cached

    if (
        lap_pe is None
        or lap_pe.size(0) != int(data.num_nodes)
        or lap_pe.size(1) != lap_pe_k
    ):
        transform = AddLaplacianEigenvectorPE(
            k=lap_pe_k,
            attr_name="lap_pe",
            is_undirected=True,
        )
        pe_data = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
        pe_data = transform(pe_data)
        lap_pe = pe_data.lap_pe.to(torch.float32)
        torch.save(lap_pe, cache_path)
        print(f"[phase-4] Computed and cached LapPE → {cache_path}")
    else:
        lap_pe = lap_pe.to(torch.float32)
        print(f"[phase-4] Loaded cached LapPE → {cache_path}")

    data.lap_pe = lap_pe
    return data


def build_cluster_loaders(
    data: Data,
    cfg: dict,
    cache_dir: Path,
) -> tuple[ClusterLoader, ClusterLoader]:
    """Build ClusterLoader instances for GPS train/eval."""
    if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
        raise ImportError(
            "GPS training requires 'pyg-lib' or 'torch-sparse' for ClusterData. "
            "Install matching CUDA wheels in Colab before running --model gps."
        )

    num_parts = int(cfg.get("num_parts", 160))
    cluster_batch_size = int(cfg.get("cluster_batch_size", 1))
    recursive = bool(cfg.get("cluster_recursive", False))

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"ogbn_arxiv_cluster_parts_{num_parts}_r{int(recursive)}.pt"

    cluster_data = ClusterData(
        data=data,
        num_parts=num_parts,
        recursive=recursive,
        save_dir=str(cache_dir),
        filename=filename,
        log=True,
    )

    train_loader = ClusterLoader(
        cluster_data,
        batch_size=cluster_batch_size,
        shuffle=True,
    )
    eval_loader = ClusterLoader(
        cluster_data,
        batch_size=cluster_batch_size,
        shuffle=False,
    )
    return train_loader, eval_loader


def log_gps_attention_context(data: Data, cfg: dict) -> None:
    """Print approximate per-cluster attention context for GPS mini-batching.

    GPS attention is applied per graph in a mini-batch. With ``ClusterLoader``,
    each partition is treated as a separate graph, so ``cluster_batch_size``
    increases throughput but does *not* increase the per-node attention context.
    """
    num_nodes = int(data.num_nodes)
    num_parts = max(1, int(cfg.get("num_parts", 160)))
    cluster_batch_size = max(1, int(cfg.get("cluster_batch_size", 1)))

    approx_nodes_per_cluster = num_nodes / num_parts
    approx_attn_entries = approx_nodes_per_cluster**2

    print(
        "[phase-4] GPS attention context estimate: "
        f"~{approx_nodes_per_cluster:.0f} nodes/cluster "
        f"(~{approx_attn_entries / 1e6:.2f}M pairwise scores per head)."
    )

    if cluster_batch_size > 1:
        print(
            "[phase-4] Note: cluster_batch_size controls throughput only. "
            "Attention remains per cluster (graphs are processed independently)."
        )


def train_epoch_gps(
    model: torch.nn.Module,
    loader: ClusterLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one GPS mini-batch epoch and return (loss, train_acc)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in loader:
        batch = batch.to(device)
        train_mask = batch.train_mask
        if int(train_mask.sum().item()) == 0:
            continue

        optimizer.zero_grad()
        batch_vector = getattr(batch, "batch", None)
        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            lap_pe=batch.lap_pe,
            batch=batch_vector,
        )

        y_train = batch.y[train_mask].view(-1)
        logits = out[train_mask]
        loss = F.cross_entropy(logits, y_train)
        loss.backward()
        optimizer.step()

        n = int(train_mask.sum().item())
        total_loss += float(loss.item()) * n
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == y_train).sum().item())
        total_examples += n

    if total_examples == 0:
        return 0.0, 0.0
    return total_loss / total_examples, total_correct / total_examples


def split_mask_attr(split_name: str) -> str:
    """Map split name to Data mask attribute name."""
    if split_name not in {"train", "valid", "test"}:
        raise ValueError(f"Unsupported split: {split_name}")
    return f"{split_name}_mask"


@torch.no_grad()
def evaluate_gps(
    model: torch.nn.Module,
    loader: ClusterLoader,
    evaluator: Any,
    split_name: str,
    device: torch.device,
) -> float:
    """Evaluate GPS accuracy for a split using mini-batches."""
    model.eval()
    mask_name = split_mask_attr(split_name)
    preds: list[Tensor] = []
    trues: list[Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        batch_vector = getattr(batch, "batch", None)
        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            lap_pe=batch.lap_pe,
            batch=batch_vector,
        )

        mask = getattr(batch, mask_name)
        if int(mask.sum().item()) == 0:
            continue

        preds.append(out[mask].argmax(dim=-1).cpu())
        trues.append(batch.y[mask].view(-1).cpu())

    if not preds:
        empty_cache(device)
        return 0.0

    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(trues, dim=0)
    acc = float(eval_acc(evaluator, y_pred, y_true))
    empty_cache(device)
    return acc


@torch.no_grad()
def evaluate_with_predictions_gps(
    model: torch.nn.Module,
    loader: ClusterLoader,
    evaluator: Any,
    device: torch.device,
) -> tuple[float, Tensor, Tensor]:
    """Evaluate GPS test split and return (acc, y_pred, y_true)."""
    model.eval()
    preds: list[Tensor] = []
    trues: list[Tensor] = []

    for batch in loader:
        batch = batch.to(device)
        batch_vector = getattr(batch, "batch", None)
        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            lap_pe=batch.lap_pe,
            batch=batch_vector,
        )

        mask = batch.test_mask
        if int(mask.sum().item()) == 0:
            continue

        preds.append(out[mask].argmax(dim=-1).cpu())
        trues.append(batch.y[mask].view(-1).cpu())

    if not preds:
        empty_cache(device)
        return 0.0, torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

    y_pred = torch.cat(preds, dim=0)
    y_true = torch.cat(trues, dim=0)
    test_acc = float(eval_acc(evaluator, y_pred, y_true))
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

    # ogbn-arxiv is directed (citation graph). Our baselines and GPS pipeline
    # use an undirected variant for stable message passing and Laplacian PE.
    if cfg["model"] in {"gcn", "gat", "gps"}:
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    cfg.setdefault("in_channels", int(data.x.size(1)))
    cfg.setdefault("num_classes", int(dataset.num_classes))

    # Model instantiation (Phases 2–4)
    if cfg["model"] == "gcn":
        from models.gcn import GCN

        model = GCN(cfg).to(device)
    elif cfg["model"] == "gat":
        from models.gat import GAT

        model = GAT(cfg).to(device)
    elif cfg["model"] == "gps":
        from models.gps import GPS

        model = GPS(cfg).to(device)
    else:
        raise NotImplementedError(
            f"Model '{cfg['model']}' is not implemented in train.py yet. "
            "Current support: --model gcn, --model gat, and --model gps."
        )

    train_loader: ClusterLoader | None = None
    eval_loader: ClusterLoader | None = None
    if cfg["model"] == "gps":
        if not (WITH_PYG_LIB or WITH_TORCH_SPARSE):
            raise ImportError(
                "GPS training requires 'pyg-lib' or 'torch-sparse' for ClusterData. "
                "Install matching CUDA wheels in Colab before running --model gps."
            )
        build_split_masks(data=data, split_idx=split_idx)
        lap_pe_k = int(cfg.get("lap_pe_k", 16))
        data = add_or_load_laplacian_pe(
            data=data,
            root=PROJECT_ROOT / "data",
            lap_pe_k=lap_pe_k,
        )
        log_gps_attention_context(data=data, cfg=cfg)
        train_loader, eval_loader = build_cluster_loaders(
            data=data,
            cfg=cfg,
            cache_dir=PROJECT_ROOT / "data" / "processed" / "cluster_cache",
        )
        print("[phase-4] GPS mini-batch loaders ready (ClusterLoader).")
    else:
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
        if cfg["model"] == "gps":
            if train_loader is None or eval_loader is None:
                raise RuntimeError("GPS loaders were not initialised.")
            train_loss, train_acc = train_epoch_gps(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
            )
            val_acc = evaluate_gps(
                model=model,
                loader=eval_loader,
                evaluator=evaluator,
                split_name="valid",
                device=device,
            )
        else:
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
                    train_loss, train_acc = train_epoch(
                        model, data, split_idx, optimizer
                    )
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

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )

    if cfg["model"] == "gps":
        if eval_loader is None:
            raise RuntimeError("GPS eval loader was not initialised.")
        test_acc, y_pred_test, y_true_test = evaluate_with_predictions_gps(
            model=model,
            loader=eval_loader,
            evaluator=evaluator,
            device=device,
        )
    else:
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

    phase_map = {"gcn": "phase-2", "gat": "phase-3", "gps": "phase-4"}
    phase_label = phase_map[cfg["model"]]
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
