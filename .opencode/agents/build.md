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
- Apple M1 Max with MPS backend (fall back to CPU if MPS op not supported)
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

# MPS Gotchas You Must Handle

- Some PyG scatter ops fail on MPS. If you hit this, wrap the op with a CPU fallback and add a comment.
- `torch.mps.current_allocated_memory()` for memory monitoring.
- Always call `torch.mps.empty_cache()` after evaluation loops.

# Repo Structure

```
project/
├── configs/
├── models/         # gcn.py, gat.py, gps.py
├── utils/          # device.py, metrics.py, viz.py
├── notebooks/
├── scripts/        # train.py, evaluate.py
├── results/
└── IMPLEMENTATION_GUIDE.md
```

# Workflow

1. Before writing code, briefly state what you're building and which phase of the IMPLEMENTATION_GUIDE.md it maps to.
2. Write complete, runnable files — no placeholder `# TODO` blocks unless explicitly asked.
3. After writing a file, suggest how to test it (a command or a quick smoke test).
