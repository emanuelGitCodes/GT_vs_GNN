# Changelog

## [2026-04-03]

### Added
- Implemented 3-layer GCN baseline model with BatchNorm/ReLU/dropout and full logits output in `models/gcn.py` (Phase 2).
- Implemented 3-layer multi-head GAT baseline model (8 hidden heads, 1 output head) in `models/gat.py` (Phase 3).
- Added Colab training workflow notebook `notebooks/02_train_colab.ipynb` to clone from GitHub, run CUDA training, sync results to Drive, and optionally push metrics/plots to GitHub (Phase 2–3 enablement).
- Added Colab comparison notebook `notebooks/03_colab_evaluate_results.ipynb` for overall/per-class model analysis and delta plotting (Phase 5).

### Changed
- Upgraded `scripts/train.py` from scaffold to full training pipeline with dataset loading, model dispatch, training/eval loops, early stopping, checkpointing, per-class export, and training-curve saving for GCN/GAT (Phases 2–3).
- Added device override support (`--device {auto,mps,cuda,cpu}`) and config passthrough in `scripts/train.py` for local/Colab backend control (Phase 2 infrastructure).
- Converted ogbn-arxiv edges to undirected in `scripts/train.py` for stable/accurate full-batch GCN/GAT normalization (Phases 2–3).
- Implemented training-curve plotting in `utils/viz.py` (`plot_training_curves`) and integrated artifact saving to `results/<model>/` (Phase 2).
- Extended `utils/device.py` with explicit device preference handling and backend-aware memory reporting used by runtime logs (Phase 2 infrastructure).
- Updated model exports in `models/__init__.py` to include `GCN` and `GAT` (Phases 2–3).
- Updated configs with explicit input/output dimensions in `configs/gcn.yaml` and `configs/gat.yaml` (Phases 2–3).

### Fixed
- Resolved severe GCN underperformance by correcting graph preprocessing to undirected adjacency before message passing in `scripts/train.py`; restored expected baseline behavior (~71% test range) (Phase 2).
