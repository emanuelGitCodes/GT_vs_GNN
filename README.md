# ogbn-arxiv: Graph Transformer vs GNN Baselines

**EEL 6878 — Modeling and AI | Final Project**

Comparing GCN, GAT, and GPS (Graph Transformer) on the [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv) citation network (169K nodes, 1.1M edges, 40 classes).

**Research question:** Does global attention (GPS) disproportionately benefit interdisciplinary categories like `cs.HC`?

---

## Environment

Python 3.9+ · PyTorch 2.8 · PyTorch Geometric 2.6 · OGB 1.3 · Apple M1 Max (MPS backend)

```bash
pip install -r requirements.txt
```

---

## Repo Structure

```
project/
├── configs/          # Hyperparameter YAML files per model
│   ├── gcn.yaml
│   ├── gat.yaml
│   └── gps.yaml
├── models/           # Model definitions (added in Phases 2–4)
│   ├── gcn.py
│   ├── gat.py
│   └── gps.py
├── utils/
│   ├── device.py     # MPS → CUDA → CPU selector + sanity check
│   ├── metrics.py    # OGB evaluator wrapper, per-class accuracy
│   └── viz.py        # Plotting utilities (Phases 5–6)
├── notebooks/        # EDA, attention viz, t-SNE (Phases 1, 6)
├── scripts/
│   └── train.py      # Training entry point
├── results/          # Saved metrics, plots, checkpoints
│   ├── gcn/
│   ├── gat/
│   └── gps/
└── IMPLEMENTATION_GUIDE.md
```

---

## Quick Start

```bash
# Verify device detection
python utils/device.py

# Train GCN (Phase 2+)
python scripts/train.py --model gcn

# Train GAT with CLI overrides
python scripts/train.py --model gat --epochs 300 --lr 0.0005

# Train GPS
python scripts/train.py --model gps
```

---

## Implementation Phases

| Phase | Status | Goal |
|-------|--------|------|
| 0 | ✅ Done | Scaffold, environment, device detection |
| 1 | ⬜ | Dataset loading & EDA |
| 2 | ⬜ | GCN baseline (~71%) |
| 3 | ⬜ | GAT (~73%) |
| 4 | ⬜ | GPS / Graph Transformer (~79%) |
| 5 | ⬜ | Per-class comparative analysis |
| 6 | ⬜ | Attention & embedding visualization |
| 7 | ⬜ | Report & submission |
