# ogbn-arxiv: Graph Transformer vs GNN Baselines — Implementation Guide

> OpenCode working doc for EEL 6878 Final Project.
> All code targets Apple M1 Max (MPS backend). Single-developer execution.

---

## Phase 0: Project Scaffold & Environment

**Goal:** Reproducible environment, clean repo structure, device detection utility.

- [ ] Create conda/venv environment with pinned dependencies
  - `torch`, `torch_geometric`, `ogb`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorboard` (optional)
- [ ] Repo structure:
  ```
  project/
  ├── configs/          # hyperparameter YAML/dict files per model
  ├── models/           # gcn.py, gat.py, gps.py
  ├── utils/            # device.py, metrics.py, viz.py
  ├── notebooks/        # EDA, attention viz, t-SNE
  ├── scripts/          # train.py, evaluate.py
  ├── results/          # saved metrics, plots, checkpoints
  └── README.md
  ```
- [ ] `utils/device.py` — device selector (MPS → CUDA → CPU), with a sanity-check tensor op
- [ ] `scripts/train.py` — skeleton with argparse: `--model {gcn,gat,gps}`, `--epochs`, `--lr`, etc.
- [ ] Confirm `torch.backends.mps.is_available()` returns `True` and a small matmul runs clean

**Deliverable:** `pip install -e .` or `pip install -r requirements.txt` works, device detection passes.

---

## Phase 1: Dataset Loading & EDA

**Goal:** Load ogbn-arxiv, understand class distribution, identify cs.HC characteristics.

- [ ] Load dataset via `ogb.nodeproppred.PygNodePropPredDataset`
- [ ] Extract and log: node count, edge count, feature dim, label count, split sizes
- [ ] Class distribution histogram (all 40 classes) — flag class imbalance
- [ ] cs.HC deep dive:
  - Number of cs.HC nodes in train/val/test
  - Neighbor label entropy for cs.HC vs other classes (quantifies "interdisciplinary-ness")
  - Degree distribution for cs.HC vs dataset average
- [ ] Citation pattern analysis: for cs.HC nodes, what fraction of neighbors belong to other categories? Compare against a well-contained class like cs.DS or cs.CR

**Suggestion:** Compute a simple "cross-domain citation ratio" per class — this becomes a key explanatory variable when you later correlate it with per-class accuracy deltas between GPS and GCN/GAT. It strengthens the narrative beyond just reporting numbers.

**Deliverable:** EDA notebook with plots + a `dataset_stats.json` summary.

---

## Phase 2: GCN Baseline

**Goal:** Working GCN, validated against OGB leaderboard (~71%).

- [ ] `models/gcn.py` — 3-layer GCN, hidden=256, ReLU, dropout=0.5
- [ ] Integrate into `train.py` with:
  - Adam, lr=1e-3, weight_decay=5e-4
  - Early stopping (patience=50) on validation accuracy
  - OGB Evaluator for test accuracy
- [ ] Train, log train/val curves, report test accuracy
- [ ] Save best checkpoint + per-class accuracy breakdown to `results/gcn/`
- [ ] Sanity check: if accuracy is significantly below ~71%, debug before moving on

**MPS note:** GCN on ogbn-arxiv should fit comfortably in memory. If you hit MPS-specific bugs (e.g., scatter ops), fall back to CPU for that op or file a minimal repro.

**Deliverable:** GCN test accuracy + `results/gcn/per_class_acc.json`.

---

## Phase 3: GAT

**Goal:** Working GAT, validated against OGB leaderboard (~73%).

- [ ] `models/gat.py` — 3-layer GAT, hidden=256, 8 heads (hidden), 1 head (output), dropout=0.5
- [ ] Same training protocol as Phase 2
- [ ] Compare per-class accuracy against GCN — note any cs.HC movement
- [ ] Save checkpoint + metrics to `results/gat/`

**Heads up:** GAT with 8 heads on 169K nodes will use more memory than GCN. Monitor with `torch.mps.current_allocated_memory()`. If tight, reduce heads to 4 and document the change.

**Deliverable:** GAT test accuracy + `results/gat/per_class_acc.json`.

---

## Phase 4: GPS (Graph Transformer)

**Goal:** Working GPS, target ~79% accuracy.

- [ ] `models/gps.py` — 4 GPS layers, hidden=256, 8 Transformer heads, GatedGCN local MPNN
- [ ] Laplacian Positional Encoding (LapPE) with 16 eigenvectors
  - Use `torch_geometric.transforms.AddLaplacianEigenvectorPE`
  - This is a **preprocessing step** — compute once and cache
- [ ] LapPE gotcha: eigenvector sign ambiguity. PyG handles this, but verify embeddings are deterministic across runs
- [ ] Same training protocol, but watch for:
  - Memory: global self-attention is O(N²) — on 169K nodes, **full-batch global attention is infeasible**
  - You'll need mini-batching (e.g., `torch_geometric.loader.ClusterLoader` or `NeighborLoader`) or GPS's built-in subgraph sampling
- [ ] If memory is a wall, options (in order of preference):
  1. Use `GraphGPS` from PyG with their recommended batching
  2. Reduce to ~64 hidden dim + 4 heads as a proof of concept
  3. Subsample the graph (document the tradeoff)
- [ ] Save checkpoint + metrics to `results/gps/`

**Suggestion:** Start with a small subgraph (~10K nodes) to validate the GPS pipeline end-to-end before scaling up. This avoids burning time on OOM debugging before you know the model code is correct.

**Deliverable:** GPS test accuracy + `results/gps/per_class_acc.json`.

---

## Phase 5: Per-Class Analysis & Comparative Evaluation

**Goal:** Generate the core results table and per-class comparison.

- [ ] Load all three `per_class_acc.json` files
- [ ] Aggregate accuracy table: GCN vs GAT vs GPS (overall + cs.HC)
- [ ] Per-class accuracy bar chart (grouped by model, all 40 classes)
- [ ] Delta plot: `(GPS_acc - GCN_acc)` per class, sorted — highlight where GPS gains/loses most
- [ ] Correlation analysis: cross-domain citation ratio (from Phase 1) vs GPS accuracy gain
  - If this correlation is strong, it's the strongest evidence for your hypothesis
  - If it's weak, that's also a valid finding — report it honestly

**Suggestion:** Also compute per-class F1 in addition to accuracy. With 40 imbalanced classes, accuracy alone can be misleading for minority classes like cs.HC. The OGB evaluator uses accuracy, so report both.

**Deliverable:** Comparative results notebook + publication-ready plots in `results/`.

---

## Phase 6: Attention & Embedding Visualization

**Goal:** Interpretability analysis for GPS.

### Attention Visualization
- [ ] Extract attention weights from GPS Transformer heads for a sample of cs.HC nodes (~20-50 papers)
- [ ] For each sampled cs.HC node, rank top-K attended nodes by attention weight
- [ ] Compute: what fraction of top-K attended nodes are from *different* categories?
- [ ] Heatmap or chord diagram: cs.HC attention distribution across categories

### t-SNE Embedding Comparison
- [ ] Extract final-layer embeddings from all three models (on test set)
- [ ] t-SNE projection, colored by category
- [ ] Qualitative comparison: cluster separation, cs.HC positioning
- [ ] Optional: silhouette score per model as a quantitative cluster quality metric

**Suggestion:** For t-SNE, fix `random_state` and use the same perplexity across models so differences reflect embeddings, not hyperparameters. Also consider UMAP as an alternative — it tends to preserve global structure better, which is exactly what you're arguing GPS captures.

**Deliverable:** Attention viz notebook + t-SNE/UMAP plots in `results/`.

---

## Phase 7: Report & Submission

**Goal:** Final paper with all figures and analysis.

- [ ] Compile results into paper structure (Introduction through Conclusion from proposal)
- [ ] Insert all plots and tables
- [ ] Write analysis: does the data support the hypothesis?
- [ ] Proofread, check references, verify all numbers match saved metrics
- [ ] Package code repo for submission (clean up notebooks, add docstrings)

---

## Risk Mitigation Notes

| Risk | Mitigation |
|---|---|
| GPS OOM on MPS with 169K nodes | Mini-batch with ClusterLoader; reduce hidden dim; subsample graph |
| MPS backend bugs (scatter, sparse ops) | Pin PyTorch nightly or fall back to CPU for affected ops |
| GPS accuracy below expected ~79% | Check LapPE, verify batching doesn't break global attention, compare against PyG GPS example code |
| cs.HC sample too small for significance | Report sample size, use confidence intervals, don't overclaim |
| Eigenvector computation slow on large graph | Precompute + cache; use sparse eigensolver (`scipy.sparse.linalg.eigsh`) |

---

## Quick Reference: Key Imports

```python
# Dataset
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

# Models
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import GPSConv, GatedGraphConv  # or build GPS manually

# Positional Encoding
from torch_geometric.transforms import AddLaplacianEigenvectorPE

# Batching (if needed for GPS)
from torch_geometric.loader import ClusterLoader, ClusterData

# Viz
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
```
