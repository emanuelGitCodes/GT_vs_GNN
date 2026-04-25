# Changelog

## [2026-04-24]

### Changed
- Reorganized reusable code into a `src/gt_vs_gnn/` package with `models/` and `utils/` subpackages.
- Renamed the active Colab workflow notebook to `notebooks/colab_train_and_compare.ipynb`.
- Moved planning/history docs into `docs/`.
- Streamlined the final workflow around `notebooks/colab_train_and_compare.ipynb`, `scripts/train.py`, and `scripts/compare_results.py`.
- Updated `README.md` and `docs/IMPLEMENTATION_GUIDE.md` to reflect the cleaned workflow, generated comparison artifacts, and deferred Phase 6 interpretability scope.

### Removed
- Removed obsolete notebooks `notebooks/01_eda.ipynb` and `notebooks/03_colab_evaluate_results.ipynb`; their active functionality is covered by `src/gt_vs_gnn/utils/eda.py`, saved EDA artifacts, and `scripts/compare_results.py`.
- Removed deferred Phase 6 placeholder visualization stubs from `src/gt_vs_gnn/utils/viz.py`.

## [2026-04-05]

### Changed
- Re-tuned Phase 4 GPS defaults in `configs/gps.yaml` (`num_parts=64`, `cluster_batch_size=1`, `lr=5e-4`, `weight_decay=1e-4`, `patience=100`) to increase per-cluster attention context and reduce premature early stopping (Phase 4).
- Added `log_gps_attention_context` in `scripts/train.py` to print estimated per-cluster attention scope and clarify that `cluster_batch_size` improves throughput but not attention context size (Phase 4).
- Switched GPS normalization in `models/gps.py` from `batch_norm` to `layer_norm` to improve mini-batch Transformer stability under ClusterLoader training (Phase 4).
- Updated Phase 4 GPS config in `configs/gps.yaml` to `num_parts=32`, `lr=1e-3`, and added `max_grad_norm` plus ReduceLROnPlateau scheduler settings for more stable optimization and larger attention context (Phase 4).
- Extended the GPS training path in `scripts/train.py` with configurable gradient clipping, ReduceLROnPlateau stepping on validation accuracy, and learning-rate logging every 10 epochs (Phase 4).

## [2026-04-04]

### Added
- Implemented Phase 4 GPS model in `models/gps.py` with 4-layer `GPSConv` stack, `GatedGraphConv` local message passing, and Laplacian PE-aware input projection for ogbn-arxiv node classification (Phase 4).

### Changed
- Integrated GPS training path in `scripts/train.py` with LapPE caching, split-mask construction, ClusterLoader mini-batching, GPS-specific train/eval loops, and phase-aware logging/output handling (Phase 4).
- Updated `configs/gps.yaml` with explicit ogbn-arxiv dimensions and ClusterLoader controls (`num_parts`, `cluster_batch_size`, `cluster_recursive`) for reproducible Phase 4 runs (Phase 4).
- Switched Phase 4 GPS local message passing in `models/gps.py` from `GatedGraphConv` to `GCNConv` to improve optimization stability for ogbn-arxiv node classification (Phase 4).
- Tuned Phase 4 defaults in `configs/gps.yaml` (`cluster_batch_size=4`, `dropout=0.3`, `patience=30`) to increase mini-batch attention context and reduce early underfitting (Phase 4).
- Exported `GPS` in `models/__init__.py` so all three architectures are available from the package namespace (Phase 4).
- Updated `README.md` to reflect GPS implementation status, add GPS training command guidance, and refresh phase tracker state (Phase 4).
- Updated `notebooks/02_train_colab.ipynb` to install PyG sampling dependencies (`pyg_lib`/sparse extensions), enable GPS training cell, and include GPS artifacts in result push helper usage (Phase 4 enablement).

### Fixed
- Removed hardcoded GitHub PAT usage from `notebooks/02_train_colab.ipynb` by reading `GH_PAT` from Colab secrets (`userdata.get("GH_PAT")`) to avoid committing plaintext credentials (Phase 1 infrastructure hygiene).

## [2026-04-03]

### Added
- Implemented 3-layer GCN baseline model with BatchNorm/ReLU/dropout and full logits output in `models/gcn.py` (Phase 2).
- Implemented 3-layer multi-head GAT baseline model (8 hidden heads, 1 output head) in `models/gat.py` (Phase 3).
- Added Colab training workflow notebook `notebooks/02_train_colab.ipynb` to clone from GitHub, run CUDA training, sync results to Drive, and optionally push metrics/plots to GitHub (Phase 2–3 enablement).
- Added Colab comparison notebook `notebooks/03_colab_evaluate_results.ipynb` for overall/per-class model analysis and delta plotting (Phase 5).

### Changed
- Reworked Phase 3 GAT architecture in `models/gat.py` to use per-head hidden sizing plus BatchNorm and residual connections for more stable optimization on ogbn-arxiv (Phase 3).
- Tuned default GAT hyperparameters in `configs/gat.yaml` (hidden_dim per-head, dropout/lr/weight_decay/patience) to better match the revised architecture capacity and convergence behavior (Phase 3).
- Updated graph preprocessing logic in `scripts/train.py` to apply `to_undirected` only for GCN while preserving directed citation edges for GAT runs (Phases 2–3).
- Pinned Colab install in `notebooks/02_train_colab.ipynb` to `torch==2.5.1`, `torchvision==0.20.1`, and `torchaudio==2.5.1` (CUDA 12.1) to keep OGB dataset loading compatible across Colab sessions (Phase 1).
- Aligned core framework pins in `requirements.txt` to PyTorch 2.5.1 / torchvision 0.20.1 / torchaudio 2.5.1 for consistent environment resolution with notebook workflows (Phase 1).
- Updated `notebooks/02_train_colab.ipynb` repository settings cell to authenticate private-clone access via Colab secret `GH_PAT` and explicit repo owner defaults (Phase 1).
- Hardened Colab clone/pull cell in `notebooks/02_train_colab.ipynb` with non-git directory detection and guarded project root entry to prevent false `Working directory: /` states (Phase 1).
- Refreshed `README.md` with accurate local/Colab environment guidance, current model implementation scope (GCN/GAT in `scripts/train.py`, GPS planned), updated quick-start commands, and phase-status table alignment (Phases 0–5 context).
- Upgraded `scripts/train.py` from scaffold to full training pipeline with dataset loading, model dispatch, training/eval loops, early stopping, checkpointing, per-class export, and training-curve saving for GCN/GAT (Phases 2–3).
- Added device override support (`--device {auto,mps,cuda,cpu}`) and config passthrough in `scripts/train.py` for local/Colab backend control (Phase 2 infrastructure).
- Converted ogbn-arxiv edges to undirected in `scripts/train.py` for stable/accurate full-batch GCN/GAT normalization (Phases 2–3).
- Implemented training-curve plotting in `utils/viz.py` (`plot_training_curves`) and integrated artifact saving to `results/<model>/` (Phase 2).
- Extended `utils/device.py` with explicit device preference handling and backend-aware memory reporting used by runtime logs (Phase 2 infrastructure).
- Updated model exports in `models/__init__.py` to include `GCN` and `GAT` (Phases 2–3).
- Updated configs with explicit input/output dimensions in `configs/gcn.yaml` and `configs/gat.yaml` (Phases 2–3).

### Fixed
- Restored undirected edge preprocessing for GAT in `scripts/train.py` (same as GCN) after directed-edge training caused major Phase 3 accuracy regression on Colab.
- Fixed Colab git commit failure (`exit status 128`) in `notebooks/02_train_colab.ipynb` by configuring `git config user.name/user.email` inside `push_results_to_github` before committing (Phase 1).
- Removed interactive username/PAT prompts from `push_results_to_github` and reused `GH_USER` + `GH_PAT` loaded in Cell 2, making Cell 11 one-step and non-interactive for secret-based auth (Phase 1).
- Fixed incorrect `REPO_DIR = "/content/codebase"` in `notebooks/02_train_colab.ipynb` Cell 2; corrected to `"/content/GT_vs_GNN"` to match the actual cloned repo folder name (Phase 1).
- Added PyTorch version guard in Cell 5 of `notebooks/02_train_colab.ipynb` that raises a clear `RuntimeError` if the runtime was not restarted after the Cell 1 pip install, preventing silent use of the wrong PyTorch version (Phase 1).
- Added Cell 5b in `notebooks/02_train_colab.ipynb` to wipe `data/ogbn_arxiv/processed/` before dataset load, eliminating stale `.pt` cache files written under PyTorch>=2.6 that cause `UnpicklingError` when read back under 2.5.1 (Phase 1).
- Fixed `UnpicklingError` during `PygNodePropPredDataset` initialization on Colab by avoiding PyTorch>=2.6 `weights_only=True` default behavior via version pinning in `notebooks/02_train_colab.ipynb` (Phase 1).
- Fixed Colab setup failure mode where git commands ran against a non-repository path and notebook continued to `/`, by adding explicit runtime errors in `notebooks/02_train_colab.ipynb` (Phase 1).
- Resolved severe GCN underperformance by correcting graph preprocessing to undirected adjacency before message passing in `scripts/train.py`; restored expected baseline behavior (~71% test range) (Phase 2).
