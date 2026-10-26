"""Config-driven training engine for Staged Text- and Canvas-Conditioned Stroke Transformer."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml

from ..gym.dataset import StagedDrawingDataset, generate_staged_gym_data
from ..model.stroke_transformer import KonnonStagedTransformer
from ..model.text_encoder import FrozenTextEncoder
from ..v2.tokens import PAD_TOKEN, VOCAB_SIZE


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
    model: KonnonStagedTransformer,
    text_encoder: FrozenTextEncoder,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Any,
    device: str,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    for batch in dataloader:
        captions = batch["caption"]
        canvases = batch["canvas"].to(device)
        tokens = batch["tokens"].to(device)
        stages = batch["stage_idx"].to(device)

        # Autoregressive teacher forcing
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]

        optimizer.zero_grad()

        with torch.no_grad():
            text_emb = text_encoder.encode_text(captions, device=device)

        context = model.build_multimodal_context(
            text_embeddings=text_emb,
            canvas=canvases,
            stage_idx=stages,
        )

        logits, _, _ = model(inputs, context=context)

        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        if not torch.isfinite(loss):
            continue

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(1, total_batches)


def evaluate(
    model: KonnonStagedTransformer,
    text_encoder: FrozenTextEncoder,
    dataloader: DataLoader,
    device: str,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    with torch.no_grad():
        for batch in dataloader:
            captions = batch["caption"]
            canvases = batch["canvas"].to(device)
            tokens = batch["tokens"].to(device)
            stages = batch["stage_idx"].to(device)

            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            text_emb = text_encoder.encode_text(captions, device=device)
            context = model.build_multimodal_context(
                text_embeddings=text_emb,
                canvas=canvases,
                stage_idx=stages,
            )

            logits, _, _ = model(inputs, context=context)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            if torch.isfinite(loss):
                total_loss += float(loss.item())
                total_batches += 1

    return total_loss / max(1, total_batches)


def train_staged_model(config_path: Union[str, Path]) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = cfg.get("device", "cpu")
    m_cfg = cfg.get("model", {})
    t_cfg = cfg.get("training", {})
    d_cfg = cfg.get("data", {})
    c_cfg = cfg.get("checkpoints", {})

    out_dir = Path(c_cfg.get("out_dir", "outputs_v2/checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"STARTING KONNON DRAW STAGED V0 TRAINING: {cfg.get('experiment_name', 'v0')}")
    print(f"Device: {device}")
    print("=" * 60)

    # 1. Dataset Generation & Loading
    num_samples = d_cfg.get("num_samples", 60)
    canvas_res = d_cfg.get("canvas_res", 128)
    items = generate_staged_gym_data(num_samples=num_samples, canvas_res=canvas_res, seed=42)

    val_frac = d_cfg.get("val_split", 0.15)
    n_val = max(1, int(len(items) * val_frac))
    n_train = len(items) - n_val

    dataset = StagedDrawingDataset(items, canvas_res=canvas_res, max_tokens=m_cfg.get("max_seq_len", 160))
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    batch_size = t_cfg.get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset split: {n_train} train transitions, {n_val} val transitions")

    # 2. Text Encoder & Staged Model
    text_encoder = FrozenTextEncoder(
        model_type=cfg.get("text_encoder", {}).get("model_type", "builtin"),
        d_model=m_cfg.get("d_model", 256),
        device=device,
    )

    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=m_cfg.get("d_model", 256),
        n_layers=m_cfg.get("n_layers", 4),
        n_heads=m_cfg.get("n_heads", 8),
        d_ff=m_cfg.get("d_ff", 768),
        max_seq_len=m_cfg.get("max_seq_len", 160),
        canvas_res=canvas_res,
        dropout=m_cfg.get("dropout", 0.05),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized KonnonStagedTransformer: {n_params:,} trainable parameters")

    # 3. Optimizer & Scheduler
    lr = float(t_cfg.get("learning_rate", 8e-4))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=t_cfg.get("weight_decay", 0.01))
    max_epochs = t_cfg.get("max_epochs", 5)
    total_steps = len(train_loader) * max_epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=lr * 0.1)

    best_val_loss = float("inf")
    history = []

    print("\nEpoch | Train Loss | Val Loss | LR       | Time (s) | RAM (MB)")
    print("-" * 62)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            text_encoder=text_encoder,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            grad_clip=t_cfg.get("grad_clip", 1.0),
        )
        val_loss = evaluate(model, text_encoder, val_loader, device=device)
        scheduler.step()

        elapsed = time.time() - t0
        ram = get_current_process_ram_mb()
        curr_lr = optimizer.param_groups[0]["lr"]

        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {curr_lr:8.2e} | {elapsed:6.1f}s | {ram:8.1f}", flush=True)

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": curr_lr,
            "time_s": elapsed,
            "ram_mb": ram,
        }
        history.append(rec)

        # Save Checkpoints
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "config": cfg,
        }
        torch.save(ckpt_data, out_dir / "latest.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt_data, out_dir / "best.pt")
            print(f"  * Saved new best checkpoint: {out_dir / 'best.pt'} (val_loss = {val_loss:.4f})")

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\n[DONE] Training complete! Best validation loss: {best_val_loss:.4f}")
    return {"best_val_loss": best_val_loss, "history": history}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Konnon Draw Staged Transformer Model")
    parser.add_argument("--config", type=str, default="configs/staged_v0_tiny.yaml")
    args = parser.parse_args()
    train_staged_model(args.config)
