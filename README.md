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
   - Use `--device cpu` for **GCN/GAT**. PyG sparse/scatter ops are typically faster and more stable on CPU locally.
   - GPS can run on CPU if `pyg-lib` or `torch-sparse` is installed, but the intended GPS run is on CUDA.
2. **Colab (CUDA: H100/A100/T4 fallback)**
   - Use `--device cuda` for the GPS experiment.
   - Main workflow notebook: `notebooks/colab_train_and_compare.ipynb`.

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

The OGB dataset is not committed to this repository. It is downloaded
automatically into `data/` the first time `train.py` or the dataset loader runs.

### 1) Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 2) Verify device detection

```bash
python -m gt_vs_gnn.utils.device
```

### 3) Train GCN and GAT (local CPU recommended)

```bash
python scripts/train.py --model gcn --device cpu
python scripts/train.py --model gat --device cpu
```

### 4) Train GPS (Colab CUDA recommended)

```bash
python scripts/train.py --model gps --device cuda
```

> GPS uses Laplacian positional encoding + ClusterLoader mini-batching. Ensure `pyg-lib` or `torch-sparse` is installed in the runtime before running GPS.

### 5) Generate comparison tables and plots

```bash
python scripts/compare_results.py
```

This reads saved `results/<model>/metrics.json`, `per_class_acc.json`, and
`results/dataset_stats.json`, then writes report-ready outputs to
`results/comparisons/`.

If you want to regenerate comparisons from the included result files without
loading/downloading the OGB dataset, run:

```bash
python scripts/compare_results.py --skip-dataset
```

### Optional: custom training run

Use CLI overrides when testing alternate hyperparameters:

```bash
python scripts/train.py --model gat --device cpu --epochs 300 --lr 0.0005
```

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
| 2 | ✅ Done | GCN baseline (72.00%) |
| 3 | ✅ Done | GAT baseline (71.75%) |
| 4 | ✅ Done | GPS / Graph Transformer prototype (68.97%) |
| 5 | ✅ Done | Per-class comparative analysis |
| 6 | ↩️ Deferred | Attention & embedding visualization |
| 7 | ✅ Done | Report & submission |
