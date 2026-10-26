"""STEP 5: Resumable Staged Model Training on Scaled Gym (10k) with Ablations.

Features:
- Resumable training loop saving latest and best checkpoints
- Per-epoch metric tracking:
  * Overall cross-entropy loss (train & val)
  * Per-token group cross-entropy (structure, coords, style, EOS)
  * Natural EOS termination rate on a fixed 32-sample validation probe
  * Wall clock seconds per epoch
  * Peak process RAM RSS (MB)
- Plateau detection (run at least 10 epochs, stop if val loss plateaus for 3)
- Multi-condition ablation runner:
  (a) full: multimodal text + canvas conditioning
  (b) no_canvas: canvas visual context replaced by blank canvas
  (c) no_text: text embedding context replaced by empty prompt
"""

import argparse
import copy
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from konnon_draw.gym.dataset import StagedDrawingDataset
from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.model.text_encoder import FrozenTextEncoder
from konnon_draw.model.sampling import sample_stage_tokens
from konnon_draw.v2.tokens import (
    VOCAB_SIZE,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
)
from konnon_draw.general.vocab import (
    SPECIAL_COMMAND_OFFSET,
    NUM_COORD_BINS,
    X_OFFSET,
    Y_OFFSET,
    NUM_PRESSURE_BINS,
    PRESSURE_OFFSET,
    NUM_COLOR_BINS,
    COLOR_OFFSET,
    NUM_WIDTH_BINS,
    WIDTH_OFFSET,
    NUM_LAYERS,
    LAYER_OFFSET,
    NUM_BRUSHES,
    BRUSH_OFFSET,
)


TOKEN_GROUPS = {
    "structure": list(range(3, SPECIAL_COMMAND_OFFSET)),
    "coord_x": list(range(X_OFFSET, X_OFFSET + NUM_COORD_BINS)),
    "coord_y": list(range(Y_OFFSET, Y_OFFSET + NUM_COORD_BINS)),
    "pressure": list(range(PRESSURE_OFFSET, PRESSURE_OFFSET + NUM_PRESSURE_BINS)),
    "color": list(range(COLOR_OFFSET, COLOR_OFFSET + NUM_COLOR_BINS)),
    "width": list(range(WIDTH_OFFSET, WIDTH_OFFSET + NUM_WIDTH_BINS)),
    "layer": list(range(LAYER_OFFSET, LAYER_OFFSET + NUM_LAYERS)),
    "brush": list(range(BRUSH_OFFSET, BRUSH_OFFSET + NUM_BRUSHES)),
    "eos": [EOS_TOKEN],
}

TOKEN_TO_GROUP: Dict[int, str] = {}
for g_name, t_ids in TOKEN_GROUPS.items():
    for tid in t_ids:
        TOKEN_TO_GROUP[tid] = g_name


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_group_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Computes mean cross-entropy per token group."""
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)

    valid_mask = flat_targets != PAD_TOKEN
    val_logits = flat_logits[valid_mask]
    val_targets = flat_targets[valid_mask]

    log_probs = F.log_softmax(val_logits, dim=-1)
    target_log_probs = log_probs.gather(1, val_targets.unsqueeze(1)).squeeze(1)
    token_losses = -target_log_probs

    group_sums: Dict[str, float] = {k: 0.0 for k in TOKEN_GROUPS}
    group_counts: Dict[str, int] = {k: 0 for k in TOKEN_GROUPS}

    val_targets_np = val_targets.cpu().numpy()
    token_losses_np = token_losses.detach().cpu().numpy()

    for tid, loss_val in zip(val_targets_np, token_losses_np):
        g = TOKEN_TO_GROUP.get(int(tid))
        if g in group_sums:
            group_sums[g] += float(loss_val)
            group_counts[g] += 1

    group_means: Dict[str, float] = {}
    for g in TOKEN_GROUPS:
        if group_counts[g] > 0:
            group_means[g] = float(round(group_sums[g] / group_counts[g], 4))
        else:
            group_means[g] = 0.0

    return group_means


def evaluate_probe_natural_eos(
    model: KonnonStagedTransformer,
    text_encoder: FrozenTextEncoder,
    probe_items: List[Dict[str, Any]],
    ablation: str = "full",
    device: str = "cpu",
    canvas_res: int = 128,
    max_tokens: int = 160,
) -> float:
    """Evaluates natural EOS termination rate across fixed validation probe."""
    model.eval()
    natural_eos_count = 0

    with torch.no_grad():
        for item in probe_items:
            caption = item["caption"]
            stage_idx = int(item["stage_idx"])

            if ablation == "no_text":
                prompt_to_use = ""
            else:
                prompt_to_use = caption

            if ablation == "no_canvas":
                canvas_tensor = torch.ones(1, 3, canvas_res, canvas_res, device=device)
            else:
                # Load canvas
                c_path = item.get("canvas_path")
                if c_path and os.path.exists(c_path):
                    from PIL import Image
                    c_img = Image.open(c_path).convert("RGB")
                    c_arr = np.array(c_img, dtype=np.uint8)
                    canvas_tensor = torch.from_numpy(c_arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                else:
                    canvas_tensor = torch.ones(1, 3, canvas_res, canvas_res)
                canvas_tensor = canvas_tensor.to(device)

            text_emb = text_encoder.encode_text([prompt_to_use], device=device)
            multimodal_ctx = model.build_multimodal_context(
                text_embeddings=text_emb,
                canvas=canvas_tensor,
                stage_idx=torch.tensor([stage_idx], device=device, dtype=torch.long),
            )

            # Sample with length prior
            tokens = sample_stage_tokens(
                model=model,
                context=multimodal_ctx,
                max_tokens=max_tokens,
                temperature=0.0,  # greedy
                device=device,
                min_stage_tokens=16,
                length_prior_alpha=0.04,
            )

            # Check natural EOS termination
            if EOS_TOKEN in tokens and len(tokens) < max_tokens:
                natural_eos_count += 1

    rate = float(round((natural_eos_count / max(1, len(probe_items))) * 100.0, 2))
    return rate


def train_scaled_model(
    data_dir: str = "data/processed/staged_gym_10k",
    out_dir: str = "outputs_v2/scaled",
    ablation: str = "full",
    batch_size: int = 32,
    max_epochs: int = 12,
    min_epochs: int = 10,
    lr: float = 1.0e-3,
    eos_weight: float = 4.0,
    seed: int = 42,
    resume: bool = True,
    device: Optional[str] = None,
    d_model: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    d_ff: int = 384,
    fp16: Optional[bool] = None,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    process = psutil.Process()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 is None:
        fp16 = True if str(device).startswith("cuda") else False

    out_p = Path(out_dir) / ablation
    out_p.mkdir(parents=True, exist_ok=True)
    ckpt_latest_path = out_p / "checkpoint_latest.pt"
    ckpt_best_path = out_p / "checkpoint_best.pt"
    log_json_path = out_p / f"training_log_{ablation}.json"

    print("=" * 70)
    print(f"STEP 5: TRAINING STAGED MODEL [{ablation.upper()}] (resumable, min {min_epochs} ep)")
    print(f"Device: {device} (FP16: {fp16}) | Data: {data_dir} | Batch Size: {batch_size} | LR: {lr} | EOS Weight: {eos_weight}")
    print(f"Architecture: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_ff={d_ff}")
    print("=" * 70)

    # 1. Load Data
    d_path = Path(data_dir)
    train_items = load_jsonl(d_path / "train.jsonl")
    val_items = load_jsonl(d_path / "val.jsonl")

    canvas_res = 128
    max_seq_len = 160

    # Apply ablation overrides to data if needed
    if ablation == "no_text":
        for it in train_items:
            it["caption"] = ""
        for it in val_items:
            it["caption"] = ""

    train_ds = StagedDrawingDataset(train_items, canvas_res=canvas_res, max_tokens=max_seq_len)
    val_ds = StagedDrawingDataset(val_items, canvas_res=canvas_res, max_tokens=max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Fixed 32-sample validation probe for natural EOS rate tracking
    probe_items = val_items[:32]

    # 2. Initialize Model & Text Tower
    text_encoder = FrozenTextEncoder(model_type="builtin", d_model=d_model, device=device)
    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        canvas_res=canvas_res,
        dropout=0.05,
    ).to(device)

    # Loss Criterion with EOS loss weighting
    loss_weights = torch.ones(VOCAB_SIZE, device=device)
    loss_weights[EOS_TOKEN] = eos_weight
    criterion = nn.CrossEntropyLoss(weight=loss_weights, ignore_index=PAD_TOKEN)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1.0e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    start_epoch = 1
    best_val_loss = float("inf")
    history: List[Dict[str, Any]] = []
    patience = 3
    epochs_no_improve = 0

    # Resume Checkpoint
    if resume and ckpt_latest_path.exists():
        print(f"Resuming from existing checkpoint: {ckpt_latest_path}")
        ckpt = torch.load(ckpt_latest_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])
        print(f"Resumed at epoch {start_epoch}. Previous best val loss: {best_val_loss:.4f}")

    if start_epoch > max_epochs:
        print(f"Training already completed through epoch {max_epochs}.")
        return {"history": history, "best_val_loss": best_val_loss}

    print("\nEpoch | Train Loss | Val Loss | Nat EOS % | Wall (s) | RAM (MB) | LR")
    print("-" * 68)

    text_cache: Dict[str, torch.Tensor] = {}

    def get_cached_text_embeddings(caps: List[str]) -> torch.Tensor:
        uncached = [c for c in set(caps) if c not in text_cache]
        if uncached:
            with torch.no_grad():
                uncached_emb = text_encoder.encode_text(uncached, device=device)
                for c, e in zip(uncached, uncached_emb):
                    text_cache[c] = e
        return torch.stack([text_cache[c] for c in caps], dim=0)

    for epoch in range(start_epoch, max_epochs + 1):
        t_start = time.time()
        model.train()
        train_loss_accum = 0.0
        train_batches = 0

        for batch in train_loader:
            captions = batch["caption"]
            canvases = batch["canvas"].to(device)
            tokens = batch["tokens"].to(device)
            stages = batch["stage_idx"].to(device)

            if ablation == "no_canvas":
                canvases = torch.ones_like(canvases)
            if ablation == "no_text":
                captions = ["" for _ in captions]

            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            optimizer.zero_grad()
            text_emb = get_cached_text_embeddings(captions)

            with torch.cuda.amp.autocast(enabled=fp16):
                ctx = model.build_multimodal_context(
                    text_embeddings=text_emb,
                    canvas=canvases,
                    stage_idx=stages,
                )
                logits, _, _ = model(inputs, context=ctx)
                loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss_accum += float(loss.item())
            train_batches += 1
            if train_batches % 75 == 0 or train_batches == len(train_loader):
                print(f"  [Epoch {epoch:2d}] Batch {train_batches:3d}/{len(train_loader):3d} | Loss: {loss.item():.4f} | RAM: {process.memory_info().rss/(1024*1024):.1f} MB", flush=True)

        scheduler.step()
        train_loss = train_loss_accum / max(1, train_batches)

        # Validation Pass
        model.eval()
        val_loss_accum = 0.0
        val_batches = 0
        all_val_logits: List[torch.Tensor] = []
        all_val_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in val_loader:
                captions = batch["caption"]
                canvases = batch["canvas"].to(device)
                tokens = batch["tokens"].to(device)
                stages = batch["stage_idx"].to(device)

                if ablation == "no_canvas":
                    canvases = torch.ones_like(canvases)
                if ablation == "no_text":
                    captions = ["" for _ in captions]

                inputs = tokens[:, :-1]
                targets = tokens[:, 1:]

                text_emb = get_cached_text_embeddings(captions)
                with torch.cuda.amp.autocast(enabled=fp16):
                    ctx = model.build_multimodal_context(
                        text_embeddings=text_emb,
                        canvas=canvases,
                        stage_idx=stages,
                    )
                    logits, _, _ = model(inputs, context=ctx)
                    v_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

                if torch.isfinite(v_loss):
                    val_loss_accum += float(v_loss.item())
                    val_batches += 1

                if len(all_val_logits) < 4:  # Sample subset for group losses
                    all_val_logits.append(logits)
                    all_val_targets.append(targets)

        val_loss = val_loss_accum / max(1, val_batches)

        # Per-token group cross-entropy breakdown
        if all_val_logits:
            sample_logits = torch.cat(all_val_logits, dim=0)
            sample_targets = torch.cat(all_val_targets, dim=0)
            group_losses = compute_group_losses(sample_logits, sample_targets)
        else:
            group_losses = {}

        # Natural EOS rate on fixed probe
        probe_eos_rate = evaluate_probe_natural_eos(
            model=model,
            text_encoder=text_encoder,
            probe_items=probe_items,
            ablation=ablation,
            device=device,
            canvas_res=canvas_res,
            max_tokens=max_seq_len,
        )

        t_epoch = time.time() - t_start
        curr_ram = process.memory_info().rss / (1024 * 1024)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "natural_eos_rate": probe_eos_rate,
            "group_losses": group_losses,
            "wall_clock_s": round(t_epoch, 2),
            "peak_ram_mb": round(curr_ram, 1),
            "lr": current_lr,
        }
        history.append(epoch_record)

        print(f"{epoch:5d} | {train_loss:10.5f} | {val_loss:8.5f} | {probe_eos_rate:8.2f}% | {t_epoch:7.1f}s | {curr_ram:7.1f} | {current_lr:.2e}", flush=True)

        # Save Latest Checkpoint
        latest_ckpt_data = {
            "epoch": epoch,
            "ablation": ablation,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history,
            "config": {
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "canvas_res": canvas_res,
                "vocab_size": VOCAB_SIZE,
            },
        }
        torch.save(latest_ckpt_data, ckpt_latest_path)

        # Check Best Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_ckpt_data = copy.deepcopy(latest_ckpt_data)
            best_ckpt_data["best_val_loss"] = best_val_loss
            torch.save(best_ckpt_data, ckpt_best_path)
        else:
            epochs_no_improve += 1

        # Save JSON history
        with open(log_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ablation": ablation,
                    "best_val_loss": best_val_loss,
                    "final_probe_eos_rate": probe_eos_rate,
                    "history": history,
                },
                f,
                indent=2,
            )

        # Early Stopping / Plateau check (after min_epochs)
        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"\nValidation loss plateaued for {patience} consecutive epochs after epoch {epoch}. Stopping early.")
            break

    print(f"\nTraining [{ablation}] Complete. Best Val Loss: {best_val_loss:.5f}")
    return {"history": history, "best_val_loss": best_val_loss, "final_probe_eos_rate": probe_eos_rate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", type=str, default="full", choices=["full", "no_canvas", "no_text"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'cpu')")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=384)
    parser.add_argument("--fp16", action="store_true", help="Force FP16 mixed precision")
    parser.add_argument("--no_fp16", action="store_true", help="Disable FP16 mixed precision")
    args = parser.parse_args()

    fp16_val = False if args.no_fp16 else (True if args.fp16 else None)

    train_scaled_model(
        ablation=args.ablation,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resume=not args.no_resume,
        device=args.device,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        fp16=fp16_val,
    )
