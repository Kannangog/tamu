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

---

## Phase 4: Staged Text- and Canvas-Conditioned Stroke Model (2026-10-26)

- **Test Suite**: 107 passed in 11.74s (4 new tests in `test_phase4_model.py` covering canvas encoder, multimodal context, KV-cache numerical equivalence, and multi-stage generation loop).
- **Model Architecture**:
  - `KonnonStagedTransformer`: 4 decoder layers, 8 attention heads, $d_{\text{model}} = 256$, $d_{\text{ff}} = 768$.
  - Conditioning: Multimodal $[e_{\text{stage}} \,;\, H_{\text{text}} \,;\, H_{\text{canvas}}]$ cross-attention memory.
  - Parameters: **4,593,824 trainable parameters**.
- **Tiny Smoke Training Benchmark (`configs/staged_v0_tiny.yaml`)**:
  - Samples: 30 drawings (90 transitions, 64×64 canvas)
  - Epochs: 3
  - Time: **1.2 - 1.3s / epoch** (~3.7s total)
  - Loss trajectory: 36.5980 $\to$ 22.1370 $\to$ **18.5611**
  - Peak WorkingSet64: **365.4 MB**
- **Laptop Training Benchmark (`configs/staged_v0_laptop.yaml`)**:
  - Samples: 60 drawings (180 transitions, 128×128 canvas)
  - Split: 153 train transitions / 27 val transitions
  - Epochs: 5
  - Time per epoch:
    - Epoch 1: 14.0s (Train Loss: 53.1269, Val Loss: 18.0632, RAM: 477.0 MB)
    - Epoch 2: 13.7s (Train Loss: 13.2617, Val Loss: 10.7482, RAM: 481.4 MB)
    - Epoch 3: 13.9s (Train Loss: 9.5677, Val Loss: 8.4326, RAM: 486.1 MB)
    - Epoch 4: 13.7s (Train Loss: 7.7291, Val Loss: 7.0558, RAM: 486.1 MB)
    - Epoch 5: 13.2s (Train Loss: 6.7334, Val Loss: **6.4312**, RAM: 479.3 MB)
  - Total Training Time: **68.5s**
  - Peak RAM RSS: **486.1 MB** (strictly bounded, well under the 1.2 GB laptop limit).
- **Evaluation Benchmark (`outputs_v2/laptop/checkpoints/best.pt`)**:
  - Prompts evaluated: 8 prompts across Figures, Objects, and OOV.
  - Stages per prompt: 4 stages (`layout` $\to$ `guide` $\to$ `ink` $\to$ `shade`).
  - Total generated tokens: 5,120 tokens (640 tokens per prompt across 4 stages).
  - Evaluation Time: **33.6s** (~4.2s per 4-stage drawing on CPU with KV-cache).
  - Output files:
    - `outputs_v2/eval/staged_evaluation_sheet.png`
    - `outputs_v2/eval/evaluation_report.json`

---

## Phase 6 / Step 1: Overfit Sanity Benchmark (2026-10-26)

- **Dataset**: 32 staged transitions from `data/manifests/staged_gym.manifest.json`.
- **Architecture**: `KonnonStagedTransformer` ($d_{\text{model}} = 128$, 3 layers, 4 heads, $d_{\text{ff}} = 384$, max_seq_len = 192).
- **Optimizer**: AdamW (lr = 2.0e-3, weight_decay = 0.0, cosine schedule to 1e-5).
- **Training Progression**:
  - Epoch 1: Loss 74.51728
  - Epoch 25: Loss 3.25580 (25.4s)
  - Epoch 50: Loss 0.77471 (50.8s)
  - Epoch 75: Loss 0.10156 (75.9s)
  - Epoch 100: Loss 0.00791 (102.0s)
  - Epoch 150: Loss 0.00278 (153.6s)
  - Epoch 200: Loss **0.00247** (205.8s)
- **Execution Performance**:
  - Total Training Time: **205.8s** (~1.03s / epoch).
  - Peak WorkingSet64 RAM: **248.5 MB** (well below 1.2 GB limit).
- **Greedy Decoding Reproduction (Teacher Forcing OFF)**:
  - Exact Token Match Rate: **93.75%** (30 of 32 transitions matched 100% token-for-token).
  - Mean Pixel Error (MAE): **0.002664** (normalized [0, 1]).
  - Mean Pixel Error (MSE): **0.002269**.
- **Saved Evidence**:
  - Model Checkpoint: `outputs_v2/overfit/model_overfit.pt`
  - Contact Sheet: `outputs_v2/overfit/overfit_contact_sheet.png`
  - Quantitative Metrics: `outputs_v2/overfit/overfit_metrics.json`




