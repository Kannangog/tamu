#!/usr/bin/env bash
# ==============================================================================
# Konnon Draw General & Staged Transformer: Cloud GPU Launch Script
# Compatible with Ubuntu 22.04 LTS on RunPod, Lambda Labs, or Vast.ai
# Targets: NVIDIA RTX 4090 (24GB), A100 (80GB), H100 (80GB)
# ==============================================================================
set -euo pipefail

echo "=== [1/5] Checking GPU Environment ==="
nvidia-smi
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA Available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0)}')"

echo "=== [2/5] Installing Package & Dependencies ==="
# Install fast python package manager if not present
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
fi

uv pip install -e .
uv pip install pyyaml pillow numpy transformers tokenizers psutil accelerate

echo "=== [3/5] Verifying Dataset & Synthetics ==="
if [ ! -f "data/manifests/staged_gym_10k.manifest.json" ]; then
    echo "Generating staged gym training transitions on GPU host..."
    python scripts/generate_scaled_gym.py
else
    echo "Found staged gym dataset manifest on disk."
fi

echo "=== [4/5] Launching Multi-GPU / Single-GPU Training ==="
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "Detected $GPU_COUNT GPU(s)."

if [ "$GPU_COUNT" -gt 1 ]; then
    echo "Launching Distributed Data Parallel (DDP) across $GPU_COUNT GPUs..."
    torchrun --nproc_per_node="$GPU_COUNT" scripts/train_scaled_gym.py \
        --ablation full \
        --epochs 30 \
        --batch_size $((32 * GPU_COUNT)) \
        --lr 1.0e-3
else
    echo "Launching Single-GPU training with mixed precision and gradient accumulation..."
    python scripts/train_scaled_gym.py \
        --ablation full \
        --epochs 30 \
        --batch_size 64 \
        --lr 1.0e-3
fi

echo "=== [5/5] Running Objective Evaluation & Exporting Checkpoints ==="
python scripts/evaluate_real_metrics.py \
    --checkpoint outputs_v2/scaled/full/checkpoint_best.pt \
    --samples 128

echo "=== Training and Evaluation Complete ==="
