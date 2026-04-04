# Changelog

## [2026-04-03]

### Added
- Implemented 3-layer GCN baseline model with BatchNorm/ReLU/dropout and full logits output in `models/gcn.py` (Phase 2).
- Implemented 3-layer multi-head GAT baseline model (8 hidden heads, 1 output head) in `models/gat.py` (Phase 3).
- Added Colab training workflow notebook `notebooks/02_train_colab.ipynb` to clone from GitHub, run CUDA training, sync results to Drive, and optionally push metrics/plots to GitHub (Phase 2–3 enablement).
- Added Colab comparison notebook `notebooks/03_colab_evaluate_results.ipynb` for overall/per-class model analysis and delta plotting (Phase 5).

### Changed
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
- Fixed Colab git commit failure (`exit status 128`) in `notebooks/02_train_colab.ipynb` by configuring `git config user.name/user.email` inside `push_results_to_github` before committing (Phase 1).
- Removed interactive username/PAT prompts from `push_results_to_github` and reused `GH_USER` + `GH_PAT` loaded in Cell 2, making Cell 11 one-step and non-interactive for secret-based auth (Phase 1).
- Fixed incorrect `REPO_DIR = "/content/codebase"` in `notebooks/02_train_colab.ipynb` Cell 2; corrected to `"/content/GT_vs_GNN"` to match the actual cloned repo folder name (Phase 1).
- Added PyTorch version guard in Cell 5 of `notebooks/02_train_colab.ipynb` that raises a clear `RuntimeError` if the runtime was not restarted after the Cell 1 pip install, preventing silent use of the wrong PyTorch version (Phase 1).
- Added Cell 5b in `notebooks/02_train_colab.ipynb` to wipe `data/ogbn_arxiv/processed/` before dataset load, eliminating stale `.pt` cache files written under PyTorch>=2.6 that cause `UnpicklingError` when read back under 2.5.1 (Phase 1).
- Fixed `UnpicklingError` during `PygNodePropPredDataset` initialization on Colab by avoiding PyTorch>=2.6 `weights_only=True` default behavior via version pinning in `notebooks/02_train_colab.ipynb` (Phase 1).
- Fixed Colab setup failure mode where git commands ran against a non-repository path and notebook continued to `/`, by adding explicit runtime errors in `notebooks/02_train_colab.ipynb` (Phase 1).
- Resolved severe GCN underperformance by correcting graph preprocessing to undirected adjacency before message passing in `scripts/train.py`; restored expected baseline behavior (~71% test range) (Phase 2).
