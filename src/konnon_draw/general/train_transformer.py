"""Config-driven training engine for Konnon Draw General transformer decoder.

Features:
- Config-driven (YAML) execution: tiny, small, full_gpu.
- Cross-entropy autoregressive token loss with label padding mask.
- AdamW optimizer with cosine learning rate schedule and warmup.
- Gradient accumulation and optional mixed precision (bfloat16 / fp16).
- Peak RAM tracking, seconds-per-epoch timing, and checkpoint saving.
- Experiment history logging (JSON).
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from .dataset import create_curriculum_datasets
from .text_encoder import FrozenTextEncoder
from .transformer_decoder import KonnonTransformerDecoder
from .vocab import PAD_TOKEN, VOCAB_SIZE


def get_current_process_ram_mb() -> float:
    """Returns current process RAM usage in Megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss / (1024 * 1024))
    except Exception:
        try:
            import ctypes
            from ctypes import wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ('cb', wintypes.DWORD),
                    ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t),
                    ('WorkingSetSize', ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return float(pmc.WorkingSetSize / (1024 * 1024))
        except Exception:
            return 0.0


def train_one_epoch(
    model: nn.Module,
    text_encoder: FrozenTextEncoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: str,
    grad_clip: float = 1.0,
    grad_accum_steps: int = 1,
    autocast_dtype: Optional[torch.dtype] = None,
) -> float:
    model.train()
    text_encoder.eval()

    total_loss = 0.0
    total_tokens = 0
    optimizer.zero_grad()

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, reduction="sum")

    for batch_idx, batch in enumerate(dataloader):
        text_ids = batch["text_input_ids"].to(device)
        text_mask = batch["text_attention_mask"].to(device)
        draw_input = batch["drawing_input_ids"].to(device)
        draw_target = batch["drawing_target_ids"].to(device)
        draw_mask = batch["drawing_mask"].to(device)

        # 1. Text Context
        with torch.no_grad():
            text_out = text_encoder.tower(text_ids, text_mask)
            context = text_out["last_hidden_state"]

        # 2. Decoder Forward with Autocast
        use_autocast = (autocast_dtype is not None and device == "cuda")
        with torch.autocast(device_type=device, dtype=autocast_dtype or torch.float32, enabled=use_autocast):
            logits, _ = model(
                input_ids=draw_input,
                context=context,
                context_mask=text_mask,
            )
            # Flatten for cross-entropy
            B, T, V = logits.shape
            loss = loss_fn(logits.view(-1, V), draw_target.view(-1))
            num_valid = int(draw_mask.sum().item())
            loss_scaled = loss / max(1, num_valid) / grad_accum_steps

        # 3. Backward
        loss_scaled.backward()

        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.item())
        total_tokens += max(1, num_valid)

    return total_loss / max(1, total_tokens)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    text_encoder: FrozenTextEncoder,
    dataloader: DataLoader,
    device: str,
) -> float:
    model.eval()
    text_encoder.eval()

    total_loss = 0.0
    total_tokens = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, reduction="sum")

    for batch in dataloader:
        text_ids = batch["text_input_ids"].to(device)
        text_mask = batch["text_attention_mask"].to(device)
        draw_input = batch["drawing_input_ids"].to(device)
        draw_target = batch["drawing_target_ids"].to(device)
        draw_mask = batch["drawing_mask"].to(device)

        text_out = text_encoder.tower(text_ids, text_mask)
        context = text_out["last_hidden_state"]

        logits, _ = model(
            input_ids=draw_input,
            context=context,
            context_mask=text_mask,
        )
        B, T, V = logits.shape
        loss = loss_fn(logits.view(-1, V), draw_target.view(-1))
        num_valid = int(draw_mask.sum().item())

        total_loss += float(loss.item())
        total_tokens += max(1, num_valid)

    return total_loss / max(1, total_tokens)


def train_general_model(config_path: Union[str, Path], resume: bool = False) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "general")
    device = cfg.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available; using CPU")
        device = "cpu"

    print("=" * 60, flush=True)
    print(f"STARTING KONNON DRAW GENERAL TRAINING: {exp_name}", flush=True)
    print(f"Device: {device} | Stage: {cfg.get('stage', 1)}", flush=True)
    print("=" * 60, flush=True)

    # 1. Output & Checkpoint Directory
    out_dir = Path(cfg["checkpoints"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build Datasets & Loaders
    stage = int(cfg.get("stage", 1))
    data_cfg = cfg.get("data", {})
    print("Building curriculum datasets...", flush=True)
    train_ds, val_ds = create_curriculum_datasets(
        stage=stage,
        num_synthetic=data_cfg.get("num_synthetic", 800),
        max_quickdraw_per_class=data_cfg.get("max_quickdraw_per_class", 100),
        val_split=data_cfg.get("val_split", 0.15),
    )
    print(f"Dataset split: {len(train_ds)} train samples, {len(val_ds)} val samples", flush=True)

    batch_size = int(cfg["training"]["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3. Text Encoder
    text_cfg = cfg.get("text_encoder", {})
    text_encoder = FrozenTextEncoder(
        model_type=text_cfg.get("model_type", "builtin"),
        d_model=text_cfg.get("d_model", 256),
        device=device,
    )

    # 4. Transformer Decoder
    model_cfg = cfg["model"]
    model = KonnonTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 8),
        d_ff=model_cfg.get("d_ff", 1024),
        max_seq_len=model_cfg.get("max_seq_len", 256),
        dropout=model_cfg.get("dropout", 0.05),
    ).to(device)

    num_params = model.num_parameters()
    print(f"Initialized KonnonTransformerDecoder: {num_params:,} trainable parameters", flush=True)

    # 5. Optimizer & Cosine Schedule
    train_cfg = cfg["training"]
    lr = float(train_cfg.get("learning_rate", 5e-4))
    min_lr = float(train_cfg.get("min_lr", 5e-5))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    max_epochs = int(train_cfg.get("max_epochs", 10))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum = int(train_cfg.get("grad_accum_steps", 1))

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = (len(train_loader) // grad_accum) * max_epochs
    warmup_steps = max(10, int(0.05 * total_steps))

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 6. Checkpoint Resume
    start_epoch = 1
    best_val_loss = float("inf")
    history = []
    latest_pt = out_dir / "latest.pt"

    if resume and latest_pt.exists():
        print(f"Resuming from checkpoint: {latest_pt}")
        ckpt = torch.load(latest_pt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        history = ckpt.get("history", [])

    # Mixed precision setup
    mp_type = train_cfg.get("mixed_precision", "no")
    autocast_dtype = torch.bfloat16 if mp_type == "bfloat16" else (torch.float16 if mp_type == "fp16" else None)

    # 7. Training Loop
    print("\nEpoch | Train Loss | Val Loss | LR       | Time (s) | RAM (MB)", flush=True)
    print("-" * 62, flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            text_encoder=text_encoder,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum,
            autocast_dtype=autocast_dtype,
        )
        val_loss = evaluate_epoch(
            model=model,
            text_encoder=text_encoder,
            dataloader=val_loader,
            device=device,
        )
        elapsed = time.time() - t0
        ram_mb = get_current_process_ram_mb()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {current_lr:.2e} | {elapsed:7.1f}s | {ram_mb:7.1f}", flush=True)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "learning_rate": current_lr,
            "epoch_time_s": round(elapsed, 2),
            "ram_mb": round(ram_mb, 1),
        }
        history.append(epoch_record)

        # Save latest checkpoint
        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg,
            "history": history,
        }
        torch.save(ckpt_payload, latest_pt)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_payload["best_val_loss"] = best_val_loss
            best_pt = out_dir / "best.pt"
            torch.save(ckpt_payload, best_pt)
            print(f"  * Saved new best checkpoint: {best_pt} (val_loss = {val_loss:.4f})")

    # Save final history
    hist_path = out_dir / "history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[DONE] Training complete! Best validation loss: {best_val_loss:.4f}")
    return {
        "best_val_loss": best_val_loss,
        "final_history": history,
        "best_checkpoint": str(out_dir / "best.pt"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Konnon Draw General transformer decoder")
    parser.add_argument("--config", type=str, default="configs/general_tiny.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_general_model(args.config, resume=args.resume)
