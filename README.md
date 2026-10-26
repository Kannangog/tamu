# Konnon Drawing Model (v0.1)

A complete, laptop-first implementation guide for **Konnon Drawing Model v0.1**.

> **Goal:** Train Konnon's own compact neural model to generate new stroke sequences, render them as editable SVG, and evaluate the results — without using an image-generation API.  
> **Target Machine:** Ryzen 5 5000-series CPU · 8 GB RAM · integrated Radeon graphics · Windows laptop · zero-budget starting point.

---

## Architecture Overview

| Component | Choice in v0.1 |
|---|---|
| **Representation** | Fixed canvas (256×256); action tokens `MOVE`, `LINE`, `PEN_UP`, `END`, `PAD` plus quantized x/y targets (32 bins). |
| **Model** | Compact `DrawingGRU` sequence model in PyTorch (`hidden_dim=128`, `embedding_dim=32`). |
| **Training** | Next-action prediction with masked cross-entropy. Pad positions and non-coordinate actions (`PEN_UP`, `END`, `PAD`) do not contribute to coordinate loss. |
| **Rendering** | Deterministic Python SVG renderer producing valid, editable W3C SVG paths (`M x y L x y`). |
| **Data** | Synthetic geometric shapes (triangles, rectangles, crosses, zigzags) with jitter; plus responsible intake pipeline for Google Quick, Draw! |
| **Evaluation** | 6 Release Gates (0 to 5): Data/Renderer, Training convergence, Generation vs Random baseline, Held-out generalization, SVG export & editability, Governance. |

---

## Repository Structure

```
konnon-drawing-model/
├── .gitignore
├── README.md
├── pyproject.toml
├── configs/
│   └── train_config.json
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── docs/
│   └── evaluation_report.md
├── src/
│   └── konnon_draw/
│       ├── __init__.py
│       ├── schema.py          # Action vocabulary, quantization & validation
│       ├── renderer.py        # Deterministic SVG renderer
│       ├── data_gen.py        # Synthetic dataset generator
│       ├── encoding.py        # Tokenizer: action & coordinate encode/decode
│       ├── model.py           # PyTorch DrawingGRU architecture
│       ├── training.py        # Masked loss training loop with checkpointing
│       ├── sampling.py        # Constrained autoregressive generation & contact sheet
│       ├── evaluation.py      # Release gates verification & metrics
│       └── quickdraw.py       # Responsible intake pipeline for Quick, Draw!
├── scripts/
│   └── render_demo.py         # Standalone SVG smoke test
├── tests/
│   ├── test_schema.py
│   ├── test_renderer.py
│   ├── test_encoding.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_sampling.py
│   ├── test_evaluation.py
│   └── test_quickdraw.py
├── logs/
│   ├── training_log.csv
│   └── experiment_log.csv
└── outputs/
    ├── samples/               # Exported SVGs, JSONs, and contact_sheet.html
    └── checkpoints/           # best_model.pt and checkpoint_last.pt
```

---

## Quickstart Guide

### 1. Environment Setup

```powershell
# Create virtual environment with Python 3.11
uv venv --python 3.11 .venv
.\.venv\Scripts\Activate.ps1

# Install package in editable mode with dependencies
uv pip install -e .
```

### 2. Smoke Test the SVG Renderer

```powershell
python scripts/render_demo.py
```
Outputs `outputs/samples/renderer_demo.svg`.

### 3. Run Unit Tests

```powershell
pytest -v
```

### 4. Generate the Synthetic Dataset

```powershell
python -m konnon_draw.data_gen --n 600 --seed 7
```
Writes 600 validated examples to `data/processed/toy_drawings.jsonl`.

### 5. Train the DrawingGRU Model

```powershell
python -m konnon_draw.training --epochs 20 --batch-size 8 --lr 0.001 --hidden-dim 128
```
Saves best model to `outputs/checkpoints/best_model.pt` and logs to `logs/training_log.csv`.

### 6. Autoregressively Sample New Drawings

```powershell
python -m konnon_draw.sampling --num-samples 64 --temperature 0.8
```
Generates 64 SVG drawings and an HTML contact sheet `outputs/samples/contact_sheet.html`.

### 7. Run Evaluation & Release Gates

```powershell
python -m konnon_draw.evaluation
```
Evaluates Gates 0–5, checks metrics against the unconstrained random baseline, appends to `logs/experiment_log.csv`, and writes `docs/evaluation_report.md`.
