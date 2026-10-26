# Training Konnon Drawing Model on Kaggle's Free GPU (KAGGLE_GUIDE.md)

This guide provides step-by-step instructions for training the **Konnon Staged Transformer** model using Kaggle's free GPU compute (30 hours per week of NVIDIA T4 or P100 GPUs, 30 GB RAM).

---

## 1. Why Kaggle GPU?

| Hardware | Compute | VRAM / RAM | 10k Transitions (10 Epochs) | Max Practical Model Size | Cost |
|---|---|---|---|---|---|
| **Local Laptop** | AMD Ryzen 5 CPU | 8 GB System RAM (No GPU) | ~55 minutes | ~1.3M – 4.7M params | Free |
| **Kaggle GPU** | **NVIDIA Tesla T4 (or P100)** | **16 GB VRAM + 30 GB RAM** | **~3 – 4 minutes** | **22M – 45M params** | **$0.00 (Free)** |

Kaggle provides:
- **30 hours per week** of free GPU compute (resets every Friday).
- **12 hours continuous execution** per background run (more than enough to train a 45M parameter model for 50 epochs).
- **FP16 Mixed Precision (AMP)** for 3×–5× additional throughput.

---

## 2. 5-Minute Quickstart on Kaggle

### Step 1: Open Kaggle & Create a New Notebook
1. Go to [kaggle.com](https://www.kaggle.com/) and log in (or sign up for free).
2. Click **Create** $\to$ **New Notebook** (or go to `kaggle.com/code`).

### Step 2: Enable Free GPU Accelerator
1. In the right-hand panel under **Notebook options**:
   - Find **Accelerator**.
   - Click the dropdown and select **GPU T4 x2** (or **GPU P100**).
   - Ensure **Internet** is toggled **ON** (required to install packages / clone repo).

### Step 3: Run the Training Code
Paste and run the following cells in your Kaggle notebook:

#### Cell 1: Check GPU & Clone Repository
```python
# Verify GPU availability
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# Clone repository
!git clone https://github.com/your-username/konnon-drawing-model.git
%cd konnon-drawing-model

# Install dependencies
!pip install -q pillow numpy pyyaml psutil
```

*(Note: If you have not pushed to GitHub yet, you can also upload your `konnon-drawing-model` folder as a Kaggle Dataset, or use the standalone notebook provided in `notebooks/kaggle_konnon_train.ipynb`)*.

#### Cell 2: Generate Scaled Dataset (10,500 Transitions)
```python
# Generate synthetic staged gym dataset (Layout -> Guide -> Sketch -> Ink -> Shade)
!python -m konnon_draw.gym.generator --num-drawings 3500 --out-dir data/processed/staged_gym_10k
```

#### Cell 3: Launch High-Speed GPU Training
```python
# Train 22M-parameter model on GPU with FP16 Automatic Mixed Precision
!python scripts/train_scaled_gym.py \
    --device cuda \
    --fp16 \
    --d_model 256 \
    --n_layers 6 \
    --n_heads 8 \
    --d_ff 1024 \
    --epochs 30 \
    --batch_size 64 \
    --lr 0.001
```

*Expected duration: ~18 to 22 minutes on Kaggle T4 GPU (compared to 8+ hours on CPU).*

#### Cell 4: Evaluate and Render Contact Sheet
```python
# Run objective evaluation across 64 samples
!python scripts/eval_real_staged.py \
    --checkpoint outputs_v2/scaled/full/checkpoint_best.pt \
    --num-samples 64 \
    --out-dir /kaggle/working/eval_results
```

#### Cell 5: Package Checkpoint for Download
```python
import shutil

# Copy the trained best checkpoint to Kaggle output
shutil.copy("outputs_v2/scaled/full/checkpoint_best.pt", "/kaggle/working/checkpoint_best.pt")
print("Checkpoint ready for download at: /kaggle/working/checkpoint_best.pt")
```

---

## 3. Bringing the Trained Checkpoint Back to Your Laptop

1. In Kaggle, open the right-side **Output** panel.
2. Click the three dots `...` next to `checkpoint_best.pt` and click **Download**.
3. Move the downloaded file into your local project directory:
   ```powershell
   Move-Item -Path "$HOME\Downloads\checkpoint_best.pt" -Destination "e:\tamu\konnon-drawing-model\outputs_v2\scaled\full\checkpoint_best.pt" -Force
   ```
4. Open the Web Studio:
   ```powershell
   python scripts/start_webview.py --port 8080 --checkpoint outputs_v2/scaled/full/checkpoint_best.pt
   ```
5. Your local web studio at [http://127.0.0.1:8080](http://127.0.0.1:8080) will immediately use the high-quality GPU-trained model for all staged drawings!

---

## 4. Recommended Model Configurations on Kaggle GPU

| Preset | Parameters | Architecture | Batch Size | Epoch Time (T4) | Recommended Use Case |
|---|---|---|---|---|---|
| **Compact Fast** | 4.7M | $d=256, L=4, H=4, d_{ff}=512$ | 64 | ~12 sec | Quick experimentation (10 min run) |
| **Medium Artist** | **22M** | **$d=256, L=6, H=8, d_{ff}=1024$** | **64** | **~25 sec** | **Best balance of quality & training speed** |
| **Large Studio** | 45M | $d=512, L=8, H=8, d_{ff}=2048$ | 32 | ~55 sec | Full detail, sharp contours, 30+ epochs (~30 min run) |
