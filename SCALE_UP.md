# Konnon Draw General: GPU Scale-Up Guide (SCALE_UP.md)

This document provides exact, reproducible steps for training the **Konnon Draw General** text-to-vector drawing transformer on rented cloud GPUs (NVIDIA RTX 4090, A100 80GB, or H100).

---

## 1. Cloud Provider Comparison & Cost Estimates (Market Estimates)

*Note: Prices below reflect prevailing on-demand / spot market rates across cloud providers and are estimates subject to dynamic capacity fluctuations.*

| Provider | Recommended GPU | VRAM | Hourly Cost (Est.) | Training Config | Expected Wall Time (30 Epochs) | Total Cost Estimate |
|---|---|---|---|---|---|---|
| **RunPod** (Community Cloud) | 1× RTX 4090 | 24 GB | \$0.69 – \$0.74 / hr | Staged Transformer ($d=512$, 8 layers, 45M params) | ~4.5 hours | **\$3.10 – \$3.35** |
| **RunPod** (Secure Cloud) | 1× A100 SXM4 | 80 GB | \$1.89 – \$1.99 / hr | Full Staged ($d=768$, 12 layers, 125M params) | ~8.0 hours | **\$15.10 – \$15.90** |
| **Lambda Labs** (On-Demand) | 1× A100 SXM4 | 80 GB | \$1.89 / hr | Full Staged ($d=768$, 12 layers, 125M params) | ~8.0 hours | **\$15.12** |
| **Lambda Labs** (On-Demand) | 1× H100 PCIe | 80 GB | \$2.49 / hr | Frontier Staged ($d=1024$, 16 layers, 250M params) | ~5.5 hours | **\$13.70** |
| **Vast.ai** (Interruptible/Spot) | 1× RTX 4090 | 24 GB | \$0.35 – \$0.45 / hr | Staged Transformer ($d=512$, 8 layers) | ~5.0 hours | **\$1.75 – \$2.25** |
| **Vast.ai** (On-Demand) | 1× A100 PCIe | 80 GB | \$1.20 – \$1.45 / hr | Full Staged ($d=768$, 12 layers) | ~9.0 hours | **\$10.80 – \$13.05** |

> **Recommendation**: For training the full 125M parameter staged multimodal model on 100k transitions, rent **1× A100 SXM 80GB on Lambda Labs or RunPod** for ~$15–$16 total cost.

---

## 2. Step-by-Step Deployment & Launch Instructions

### Step 1: Launch Instance
Select an instance with:
- OS: Ubuntu 22.04 LTS
- CUDA: 12.1 or 12.2
- Disk: at least 50 GB NVMe storage
- PyTorch pre-installed (e.g., RunPod PyTorch 2.2 template or Lambda Stack)

### Step 2: One-Click Execution via Launch Script
A turnkey launch script is provided at `scripts/launch_gpu_train.sh`:

```bash
# Clone repository
git clone https://github.com/your-org/konnon-drawing-model.git
cd konnon-drawing-model

# Make launch script executable and run
chmod +x scripts/launch_gpu_train.sh
./scripts/launch_gpu_train.sh
```

### Step 3: Manual Execution Commands (Alternative)
```bash
# Create virtualenv and install dependencies with GPU PyTorch
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
uv pip install pyyaml pillow numpy transformers tokenizers psutil accelerate

# Single GPU training (A100 / RTX 4090)
python scripts/train_scaled_gym.py --ablation full --epochs 30 --batch_size 64 --lr 1.0e-3

# Multi-GPU Distributed Data Parallel (4x A100 nodes)
torchrun --nproc_per_node=4 scripts/train_scaled_gym.py --ablation full --epochs 30 --batch_size 256 --lr 1.0e-3
```

### Step 4: Resume Training from Checkpoint (if interrupted)
```bash
python scripts/train_scaled_gym.py --ablation full --epochs 30 --batch_size 64
```

---

## 3. Dataset Scaling Plan

To graduate from limited vocabulary to a general text-to-vector artist:

| Dataset | Volume | License | Role in Training |
|---|---|---|---|
| **Google Quick, Draw!** (All 345 Categories) | 50,000,000 drawings | CC BY 4.0 | Broad object category representation and diverse human stroke dynamics |
| **Synthetic Composition Corpus** | 500,000 generated pairs | MIT (Project Native) | Verified ground truth for spatial prepositions (above, below, inside), counts, and colors |
| **SVG-CoCo / IconPark / Feather** | 120,000 vector files | Permissive (MIT/Apache) | Layered graphic design, fill gradients, and clean icon contour structures |
| **Sketch-e (Captioned Human Sketches)** | 80,000 sketches with natural language captions | Open Academic | Freeform natural language prompts ("a cat sleeping under a tree") |

### Ingestion Command for 345 Quick, Draw! Classes
```bash
python -m konnon_draw.quickdraw --all-categories --per-class 2000 --out-dir data/processed/quickdraw_all
```

---

## 4. Expected Quality Progression by Model Scale

```
[1.5M Tiny CPU]  -->  [8.5M Small Laptop]  -->  [125M Full A100]  -->  [350M Frontier]
    |                         |                        |                      |
- Single shapes           - Primitives + colors     - Multi-stroke scenes  - True photorealistic
- High train loss         - Spatial relations       - Complex compositions   vector rendering
- Frequent syntax errors  - Recognizable doodles    - Textures, hatching   - Dynamic occlusion
```

1. **1.5M Parameters (Tiny / Smoke Test)**:
   - Learns drawing language grammar (BOS, coordinates, PEN_UP, EOS).
   - Draws basic circles, squares, lines in isolation.
   - Cannot reliably coordinate multiple objects or complex prompts.

2. **8.5M Parameters (Small / Laptop Baseline)**:
   - Solid command of spatial coordinates and color palette tokens.
   - Follows count prompts (e.g. "two circles") and basic spatial relations ("a circle above a square").
   - Reproduces Quick, Draw! categories on prompt.
   - Weaknesses: complex multi-clause prompts cause token bleeding or coordinate drift.

3. **125M Parameters (Full GPU Config)**:
   - Deep cross-attention to pretrained text embeddings (e.g. MiniLM / CLIP).
   - Multi-layered compositions with foreground/background separation.
   - Clean hatching and shadow application driven by text context ("at sunset", "in the dark").
   - Low out-of-vocabulary failure rate due to semantic text clustering.

4. **350M–500M Parameters (Frontier Scale)**:
   - Fine cross-contour shading on 3D curved surfaces.
   - Nuanced artistic style emulation (e.g. "charcoal sketch", "fine technical blueprint").
   - Near-zero syntax invalidity; consistent multi-stroke closure.

---

## 5. Staged Multimodal Model v0 Scaling (`configs/staged_v0_full_gpu.yaml`)

To train the **Staged Multimodal Transformer** (canvas-conditioned + text-conditioned in stages):

### Training Launch Command
```bash
python -m konnon_draw.train.train_v2 --config configs/staged_v0_full_gpu.yaml
```

### Full GPU Model Configuration
- **Model Size**: 125,845,504 parameters
- **Architecture**: 12 layers, 16 heads, $d_{\text{model}} = 768$, $d_{\text{ff}} = 3072$
- **Canvas Encoder**: 4-stage Conv2D ($256 \times 256 \to 16 \times 16 = 256$ spatial patches + 1 summary token)
- **Text Conditioning**: Frozen `all-MiniLM-L6-v2` or `openai/clip-vit-base-patch32` text tower
- **Batch Size**: 64 (gradient accumulation steps = 2, effective batch size = 128)
- **Training Time**: ~18 hours on 1× A100 SXM (80GB)
- **Estimated Cloud Cost**: **~$34.00**

