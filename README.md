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
   - Main workflow notebook: `notebooks/02_train_colab.ipynb`.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Repository Structure

```text
project/
├── configs/                 # YAML hyperparameters per model
│   ├── gcn.yaml
│   ├── gat.yaml
│   └── gps.yaml
├── models/
│   ├── gcn.py               # Phase 2 baseline
│   ├── gat.py               # Phase 3 baseline
│   └── gps.py               # Phase 4 baseline
├── scripts/
│   └── train.py             # Training entry point (GCN/GAT/GPS implemented)
├── utils/
│   ├── device.py            # Device selection + sanity check + cache management
│   ├── eda.py               # Dataset loading and EDA helpers
│   ├── metrics.py           # OGB eval + per-class metrics + JSON saving
│   └── viz.py               # Training curves and analysis plots
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_colab.ipynb
│   └── 03_colab_evaluate_results.ipynb
├── results/                 # Metrics, checkpoints, plots
├── IMPLEMENTATION_GUIDE.md  # Phase-by-phase execution plan
└── CHANGELOG.md
```

---

## Quick Start

### 1) Verify device detection

```bash
python utils/device.py
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

---

## Training Outputs

Each run writes artifacts under `results/<model>/`, including:

- `best_model.pt`
- `metrics.json`
- `per_class_acc.json`
- `training_curves.png`

---

## Project Status (Implementation Phases)

| Phase | Status | Goal |
|---|---|---|
| 0 | ✅ Done | Scaffold, environment, device detection |
| 1 | ✅ Done | Dataset loading & EDA |
| 2 | ✅ Done | GCN baseline (~71%) |
| 3 | ✅ Done | GAT baseline (~73%) |
| 4 | ⏳ In Progress | GPS / Graph Transformer (~79%) |
| 5 | ⬜ Planned | Per-class comparative analysis |
| 6 | ⬜ Planned | Attention & embedding visualization |
| 7 | ⬜ Planned | Report & submission |

For detailed deliverables and risk mitigation, see `IMPLEMENTATION_GUIDE.md`.
