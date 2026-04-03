---
description: Primary development agent for the ogbn-arxiv GNN vs Graph Transformer project. Handles all implementation — model code, training scripts, data pipelines, and utilities.
mode: primary
model: openai/gpt-5.3-codex
temperature: 0
permissions:
  edit: allow
  bash:
    allow:
      - "python *"
      - "pip *"
      - "conda *"
      - "pytest *"
      - "git *"
      - "ls *"
      - "cat *"
      - "mkdir *"
      - "cp *"
      - "mv *"
      - "rm *"
      - "find *"
      - "grep *"
      - "head *"
      - "tail *"
      - "wc *"
---

# Project Context

You are working on an EEL 6878 final project: comparing GCN, GAT, and GPS (Graph Transformer) on the ogbn-arxiv citation network (169K nodes, 1.1M edges, 40 classes). The research question is whether global attention (GPS) disproportionately benefits interdisciplinary categories like cs.HC.

# Stack

- Python 3.11+, PyTorch, PyTorch Geometric, OGB
- **Two environments:**
  - **Local:** Apple M1 Max, 64GB unified RAM, MPS backend. MPS has no native sparse scatter/gather kernels — GCN and GAT run on CPU (`--device cpu`). MPS is reserved for GPS dense attention ops.
  - **Colab:** NVIDIA H100 (or T4/A100 fallback), CUDA backend. All models run on CUDA. Training is driven by `notebooks/02_train_colab.ipynb`, which clones the repo, runs `scripts/train.py`, and mirrors results to Google Drive.
- `--device auto` selects CUDA > MPS > CPU. `--device cpu` forces CPU (used locally for GCN/GAT).
- Matplotlib / Seaborn for visualization
- scikit-learn for t-SNE and metrics

# Coding Standards

- Type hints on all function signatures.
- Docstrings (one-liner OK for simple functions).
- Device-agnostic code: always use a `device` variable from `utils/device.py`, never hardcode `"mps"` or `"cuda"`.
- Configs as dictionaries or dataclasses, not magic numbers in training loops.
- Save all metrics as JSON to `results/<model_name>/`.
- Use `pathlib.Path` over `os.path`.
- Print training progress every 10 epochs, not every epoch.

# MPS Gotchas (Local Only)

- MPS has no native kernel for irregular sparse scatter/gather — GCN/GAT are ~5× slower on MPS than CPU. Always use `--device cpu` for GCN/GAT locally.
- Some PyG scatter ops fail on MPS. If you hit this, wrap the op with a CPU fallback and add a comment.
- `torch.mps.current_allocated_memory()` for memory monitoring.
- Always call `torch.mps.empty_cache()` after evaluation loops when on MPS.
- GPS Transformer attention (dense matmul) is the one workload MPS accelerates well.

# Repo Structure

```
project/
├── configs/
├── models/         # gcn.py, gat.py, gps.py
├── utils/          # device.py, metrics.py, viz.py
├── notebooks/      # 01_eda.ipynb, 02_train_colab.ipynb
├── scripts/        # train.py, evaluate.py
├── results/
├── IMPLEMENTATION_GUIDE.md
└── CHANGELOG.md
```

# Workflow

1. Before writing code, briefly state what you're building and which phase of the IMPLEMENTATION_GUIDE.md it maps to.
2. Write complete, runnable files — no placeholder `# TODO` blocks unless explicitly asked.
3. After writing a file, suggest how to test it (a command or a quick smoke test).

# Changelog (REQUIRED)

After completing any task, you MUST update `CHANGELOG.md` in the project root. Follow this format:

```markdown
## [YYYY-MM-DD]

### Added / Changed / Fixed / Removed
- Brief description of what was done and why
- Reference the file(s) modified
- Note which IMPLEMENTATION_GUIDE.md phase this relates to (if applicable)
```

Rules:
- Append to the top of the file (newest first), below the `# Changelog` header.
- One `## [date]` section per session. Group changes under `### Added`, `### Changed`, `### Fixed`, or `### Removed` as appropriate.
- Keep entries concise — one line per change.
- If `CHANGELOG.md` doesn't exist yet, create it with a `# Changelog` header.
- Do this as the LAST step before finishing, not mid-task.
