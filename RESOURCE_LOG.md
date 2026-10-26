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

