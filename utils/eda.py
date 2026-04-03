"""Phase 1 utilities for ogbn-arxiv dataset loading and EDA metrics."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected


# Fallback mapping used only when OGB mapping CSV is unavailable.
# We keep known class IDs required by the project narrative and use generic labels
# for the remaining indices.
FALLBACK_LABEL_MAP: dict[int, str] = {
    **{idx: f"class_{idx}" for idx in range(40)},
    6: "cs.HC",
    34: "cs.DS",
    4: "cs.CR",
}


def _normalize_arxiv_category(raw: str) -> str:
    """Convert OGB category strings (e.g., 'arxiv cs hc') to 'cs.HC'."""
    tokens = raw.strip().lower().split()
    if len(tokens) >= 3 and tokens[0] == "arxiv":
        field = tokens[-2]
        subfield = tokens[-1].upper()
        return f"{field}.{subfield}"
    return raw.strip()


def load_dataset(
    root: str | Path = "data/",
) -> tuple[Data, dict[str, torch.Tensor], PygNodePropPredDataset, dict[int, str]]:
    """Load ogbn-arxiv and return (data, split_idx, dataset, label_names)."""
    # Compatibility fix for OGB + PyTorch>=2.6:
    # OGB internally uses torch.load for processed graph objects.
    # This project trusts the local OGB download, so we pass
    # weights_only=False during dataset deserialization.
    original_torch_load = torch.load

    def _torch_load_compat(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_compat  # type: ignore[assignment]
    try:
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(Path(root)))
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]

    data = dataset[0]
    split_idx = dataset.get_idx_split()
    label_names = get_label_names(dataset)
    return data, split_idx, dataset, label_names


def get_label_names(dataset: PygNodePropPredDataset) -> dict[int, str]:
    """Load class-index to arXiv category mapping, with fallback labels."""
    mapping_dir = Path(dataset.root) / "mapping"
    mapping_path = mapping_dir / "labelidx2arxivcategeory.csv.gz"

    if not mapping_path.exists():
        return FALLBACK_LABEL_MAP.copy()

    names: dict[int, str] = {}
    with gzip.open(mapping_path, mode="rt", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            # OGB file headers: "label idx", "arxiv category"
            idx = int(row["label idx"])
            names[idx] = _normalize_arxiv_category(row["arxiv category"])

    if len(names) == 0:
        return FALLBACK_LABEL_MAP.copy()

    # Ensure all 40 classes have at least a generic label.
    merged = FALLBACK_LABEL_MAP.copy()
    merged.update(names)
    return merged


def find_class_id(label_names: dict[int, str], target_label: str) -> int | None:
    """Return class index for a label (e.g., 'cs.HC'), if present."""
    norm_target = target_label.strip().lower()
    for idx, name in label_names.items():
        norm_name = name.strip().lower()
        if norm_name == norm_target or norm_target in norm_name:
            return idx
    return None


def compute_dataset_stats(
    data: Data, split_idx: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Compute basic ogbn-arxiv dataset statistics for Phase 1."""
    y = data.y.view(-1)
    num_classes = int(y.max().item() + 1)
    class_counts = torch.bincount(y, minlength=num_classes)

    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.size(1)),
        "feature_dim": int(data.x.size(1)),
        "num_classes": num_classes,
        "split_sizes": {
            "train": int(split_idx["train"].numel()),
            "valid": int(split_idx["valid"].numel()),
            "test": int(split_idx["test"].numel()),
        },
        "class_distribution": {
            str(i): int(class_counts[i].item()) for i in range(num_classes)
        },
    }


def compute_degree_vectors(
    data: Data,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (in_degree, out_degree, undirected_degree) for each node."""
    src, dst = data.edge_index
    out_degree = degree(src, num_nodes=data.num_nodes)
    in_degree = degree(dst, num_nodes=data.num_nodes)
    undirected = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    undirected_degree = degree(undirected[0], num_nodes=data.num_nodes)
    return in_degree, out_degree, undirected_degree


def summarize_degree_stats(
    deg: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, float]:
    """Summarize degree tensor with mean/std/median and selected percentiles."""
    if mask is not None:
        values = deg[mask]
    else:
        values = deg

    if values.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
        }

    values = values.float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "median": float(values.median().item()),
        "p25": float(torch.quantile(values, 0.25).item()),
        "p75": float(torch.quantile(values, 0.75).item()),
    }


def _neighbor_label_counts(
    edge_index_undirected: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Build neighbor label count matrix of shape (N, C)."""
    src, dst = edge_index_undirected
    dst_labels = labels[dst]
    one_hot = F.one_hot(dst_labels, num_classes=num_classes).to(torch.float32)

    counts = torch.zeros((labels.numel(), num_classes), dtype=torch.float32)
    counts.index_add_(0, src, one_hot)
    return counts


def compute_node_neighbor_entropy(data: Data, labels: torch.Tensor) -> torch.Tensor:
    """Compute Shannon entropy of neighbor label distribution per node."""
    labels = labels.view(-1).to(torch.long)
    num_classes = int(labels.max().item() + 1)

    edge_index_undirected = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    counts = _neighbor_label_counts(edge_index_undirected, labels, num_classes)
    totals = counts.sum(dim=1, keepdim=True)
    probs = counts / totals.clamp_min(1.0)
    ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    ent[totals.view(-1) == 0] = 0.0
    return ent


def compute_neighbor_label_entropy(
    data: Data, labels: torch.Tensor, node_mask: torch.Tensor
) -> float:
    """Average neighbor label entropy for nodes selected by node_mask."""
    ent = compute_node_neighbor_entropy(data, labels)
    if node_mask.numel() == 0 or node_mask.sum() == 0:
        return 0.0
    return float(ent[node_mask].mean().item())


def compute_per_class_neighbor_entropy(
    data: Data, labels: torch.Tensor
) -> dict[int, float]:
    """Compute average neighbor label entropy for each class."""
    labels = labels.view(-1).to(torch.long)
    ent = compute_node_neighbor_entropy(data, labels)
    num_classes = int(labels.max().item() + 1)

    results: dict[int, float] = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            results[c] = 0.0
        else:
            results[c] = float(ent[mask].mean().item())
    return results


def compute_cross_domain_ratio(data: Data, labels: torch.Tensor) -> dict[int, float]:
    """Compute per-class mean cross-domain citation ratio on undirected edges."""
    labels = labels.view(-1).to(torch.long)
    num_nodes = labels.numel()
    num_classes = int(labels.max().item() + 1)

    # We treat the graph as undirected so each paper's neighborhood captures
    # both citing and cited contexts for interdisciplinary analysis.
    edge_index_undirected = to_undirected(data.edge_index, num_nodes=num_nodes)
    src, dst = edge_index_undirected

    deg = degree(src, num_nodes=num_nodes)
    cross_mask = labels[src] != labels[dst]
    cross_counts = torch.bincount(src[cross_mask], minlength=num_nodes).float()
    ratio_per_node = torch.zeros(num_nodes, dtype=torch.float32)
    nonzero = deg > 0
    ratio_per_node[nonzero] = cross_counts[nonzero] / deg[nonzero]

    ratios: dict[int, float] = {}
    for c in range(num_classes):
        class_mask = labels == c
        if class_mask.sum() == 0:
            ratios[c] = 0.0
        else:
            ratios[c] = float(ratio_per_node[class_mask].mean().item())
    return ratios


def split_class_counts(
    labels: torch.Tensor, split_idx: dict[str, torch.Tensor], class_id: int
) -> dict[str, int]:
    """Count class occurrences in train/valid/test splits."""
    labels = labels.view(-1).to(torch.long)
    result: dict[str, int] = {}
    for split_name in ("train", "valid", "test"):
        idx = split_idx[split_name]
        result[split_name] = int((labels[idx] == class_id).sum().item())
    return result


def summarize_phase1(
    data: Data,
    split_idx: dict[str, torch.Tensor],
    label_names: dict[int, str],
    cs_hc_label: str = "cs.HC",
) -> dict[str, Any]:
    """Build a complete Phase 1 summary dictionary for JSON export."""
    labels = data.y.view(-1).to(torch.long)
    base_stats = compute_dataset_stats(data, split_idx)
    in_deg, out_deg, undeg = compute_degree_vectors(data)
    cross_ratio = compute_cross_domain_ratio(data, labels)
    per_class_entropy = compute_per_class_neighbor_entropy(data, labels)

    cs_hc_id = find_class_id(label_names, cs_hc_label)
    cs_ds_id = find_class_id(label_names, "cs.DS")
    cs_cr_id = find_class_id(label_names, "cs.CR")

    summary: dict[str, Any] = {
        **base_stats,
        "label_names": {str(k): v for k, v in label_names.items()},
        "degree_stats": {
            "in_degree": summarize_degree_stats(in_deg),
            "out_degree": summarize_degree_stats(out_deg),
            "undirected_degree": summarize_degree_stats(undeg),
        },
        "per_class_neighbor_entropy": {
            str(k): float(v) for k, v in per_class_entropy.items()
        },
        "cross_domain_citation_ratio": {
            str(k): float(v) for k, v in cross_ratio.items()
        },
    }

    if cs_hc_id is not None:
        cs_hc_mask = labels == cs_hc_id
        summary["cs_hc"] = {
            "class_id": int(cs_hc_id),
            "split_counts": split_class_counts(labels, split_idx, cs_hc_id),
            "neighbor_label_entropy": float(per_class_entropy[cs_hc_id]),
            "cross_domain_citation_ratio": float(cross_ratio[cs_hc_id]),
            "degree_stats_undirected": summarize_degree_stats(undeg, cs_hc_mask),
            "degree_stats_in": summarize_degree_stats(in_deg, cs_hc_mask),
            "degree_stats_out": summarize_degree_stats(out_deg, cs_hc_mask),
        }

    summary["comparison_classes"] = {}
    for name, cls_id in (("cs.DS", cs_ds_id), ("cs.CR", cs_cr_id)):
        if cls_id is not None:
            summary["comparison_classes"][name] = {
                "class_id": int(cls_id),
                "neighbor_label_entropy": float(per_class_entropy[cls_id]),
                "cross_domain_citation_ratio": float(cross_ratio[cls_id]),
            }

    return summary


def save_dataset_stats(stats: dict[str, Any], output_path: str | Path) -> None:
    """Persist dataset statistics JSON to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
