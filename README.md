# ogbn-arxiv: Graph Transformer vs GNN Baselines

**EEL 6878 — Modeling and AI | Final Project**

This repository compares **GCN**, **GAT**, and **GPS (Graph Transformer)** on the [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) citation network (169K nodes, 1.1M edges, 40 classes).

**Research question:** does global attention (GPS) disproportionately benefit interdisciplinary categories like `cs.HC`?

---

## Environment

- Python **3.11+**
- PyTorch / PyTorch Geometric / OGB
- Matplotlib / Seaborn / scikit-learn

Two supported execution environments:

1. **Local (Apple M1 Max, MPS backend available)**
   - Use `--device cpu` for **GCN/GAT** (PyG sparse/scatter ops are typically faster and more stable on CPU locally).
   - Reserve MPS for GPS dense attention workloads in Phase 4.
2. **Colab (CUDA: H100/A100/T4 fallback)**
   - Use `--device auto` or `--device cuda`.
   - Main workflow notebook: `notebooks/colab_train_and_compare.ipynb`.

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Repository Structure

```text
project/
├── configs/                 # YAML hyperparameters per model
│   ├── gcn.yaml
│   ├── gat.yaml
│   └── gps.yaml
├── src/
│   └── gt_vs_gnn/
│       ├── models/          # GCN, GAT, GPS model definitions
│       └── utils/           # Device, EDA, metrics, plotting helpers
├── scripts/
│   ├── train.py             # Training entry point (GCN/GAT/GPS implemented)
│   └── compare_results.py   # Local comparison tables/plots/F1 metrics
├── notebooks/
│   └── colab_train_and_compare.ipynb
├── results/                 # Metrics, checkpoints, plots
├── docs/
│   ├── IMPLEMENTATION_GUIDE.md
│   └── CHANGELOG.md
├── pyproject.toml
└── README.md
```

---

## Quick Start

### 1) Verify device detection

```bash
python -m gt_vs_gnn.utils.device
```

### 2) Train GCN (local CPU recommended)

```bash
python scripts/train.py --model gcn --device cpu
```

### 3) Train GAT (local CPU recommended)

```bash
python scripts/train.py --model gat --device cpu
```

### 4) Train with CLI overrides

```bash
python scripts/train.py --model gat --device cpu --epochs 300 --lr 0.0005
```

### 5) Train GPS (Colab CUDA recommended)

```bash
python scripts/train.py --model gps --device auto
```

> GPS uses Laplacian positional encoding + ClusterLoader mini-batching. Ensure `pyg-lib` or `torch-sparse` is installed in your runtime.

### 6) Generate comparison tables and plots locally

```bash
python scripts/compare_results.py
```

This reads saved `results/<model>/metrics.json` and `per_class_acc.json`
artifacts, then writes report-ready outputs to `results/comparisons/`.
It is intended to run on the local M1 Max without Colab.

---

## Training Outputs

Each run writes artifacts under `results/<model>/`, including:

- `best_model.pt`
- `metrics.json`
- `per_class_acc.json`
- `test_predictions.npz`
- `training_curves.png`

Comparison artifacts are written under `results/comparisons/`, including:

- `overall_metrics.csv`
- `per_class_accuracy.csv`
- `prediction_metrics.csv`
- `summary.json`
- comparison plots (`*.png`)

---

## Project Status (Implementation Phases)

| Phase | Status | Goal |
|---|---|---|
| 0 | ✅ Done | Scaffold, environment, device detection |
| 1 | ✅ Done | Dataset loading & EDA |
| 2 | ✅ Done | GCN baseline (~71%) |
| 3 | ✅ Done | GAT baseline (~72%) |
| 4 | ✅ Done | GPS / Graph Transformer prototype (~70%) |
| 5 | ✅ Done | Per-class comparative analysis |
| 6 | ↩️ Deferred | Attention & embedding visualization |
| 7 | ⏳ In Progress | Report & submission |

For detailed deliverables and risk mitigation, see `docs/IMPLEMENTATION_GUIDE.md`.

Phase 6 was deferred because the current GPS implementation uses
ClusterLoader mini-batching, which limits attention to cluster-local context
and does not expose a clean full-graph attention analysis path. The final
report should instead focus on the completed aggregate, per-class, F1, and
cross-domain analyses in `results/comparisons/`.
