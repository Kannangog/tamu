# Resource & Benchmark Log

This file tracks resource consumption, execution times, peak RAM, and disk utilization across all phases and training runs.

---

## Phase 0: System & Baseline Benchmark (2026-10-26)

- **Operating System**: Windows 11 (AMD64)
- **Processor**: AMD Ryzen 5 5000-series (6 cores / 12 threads)
- **Memory**:
  - Total Physical RAM: 7.33 GB
  - Available Physical RAM: 2.20 GB
  - Free Physical RAM: 2.20 GB
  - RAM Constraint Guard: Jobs must maintain RSS $< 1.2$ GB
- **Runtime & Frameworks**:
  - Python: 3.11.15 (`[MSC v.1944 64 bit (AMD64)]`)
  - PyTorch: `2.14.1+cpu` (CPU-only, no CUDA/DirectML active)
  - Torch device: `cpu`
- **Initial Test Suite**: 85 passed in 21.84s
- **Baseline Git Commit**: `d91034b`
- **Legacy DrawingGRU Step Benchmark**:
  - Model: `DrawingGRU` (num_classes=6, hidden_dim=256)
  - Batch size: 32, Sequence length: 100
  - Step time: **236.12 ms/step** (0.2361 s/step)
  - Forward + backward + Adam optimizer step on CPU

---

## Phase 1: Hands — Stroke Language v2 & Renderer (2026-10-26)

- **Test Suite**: 92 passed in 9.51s
- **1000-Stroke Render Performance Benchmark**:
  - Document: 1000 spline strokes (3 points each), `ink_liner`, base width 2.5px, 512×512 canvas.
  - Initial naive loop render time: 10.52s (bottleneck: 1000 PIL $\leftrightarrow$ NumPy conversions).
  - Optimized batched PIL draw layer render time: **1.82s** on CPU ($< 10$s target achieved).
- **Brush Swatches Generation**:
  - Script: `render_brush_swatches('outputs/hand/swatches.png', 1200, 800, seed=42)`
  - Output: `outputs/hand/swatches.png` (size 1200×800)
  - Render Time: **2.51s** (supersampled 2x)
- **Memory Footprint**:
  - Process RSS during 1000-stroke render: **118.4 MB** (well below 1.2 GB limit).

---

## Phase 2: Skill Library & Proof Pieces (2026-10-26)

- **Test Suite**: 98 passed in 9.85s (6 new unit tests for camera projection, joint validity, hatching angle, occlusion, and linecraft).
- **Gate 2 Scripted Demos Generation**:
  - Script: `scripts/generate_gate2_demos.py`
  - Output files:
    - `outputs/hand_demos/mannequin_figure.png` (and 3 stage files: layout, guide, ink)
    - `outputs/hand_demos/head_rotations.png`
    - `outputs/hand_demos/pose_sequence.png`
    - `outputs/hand_demos/pencil_still_life.png`
    - `outputs/hand_demos/gate2_contact_sheet.png` (1600×1280 2×2 composite)
  - Execution Time: **8.42s** on CPU (supersampled 2x).
  - Peak WorkingSet64: **134.2 MB** (strictly bounded, well under 1.2 GB RSS limit).

---

## Phase 3: Synthetic Gym Data Pipeline (2026-10-26)

- **Test Suite**: 103 passed in 11.37s (5 new tests in `test_phase3_gym.py`).
- **Data Generator Execution**:
  - Generator: `generate_staged_gym_data(num_samples=60, canvas_res=128, seed=42)`
  - Output Manifest: `data/manifests/staged_gym.manifest.json`
  - Total Staged Transitions: **180**
  - Generation Time: **24.08s** on CPU (~0.13s per transition)
  - Peak WorkingSet64: **182.5 MB** (strictly bounded, well below 1.2 GB limit).


