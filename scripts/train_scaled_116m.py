"""Scales and Trains the 116.4M Parameter Staged Visual Stroke Transformer.

Architecture:
- d_model = 768, n_layers = 12, n_heads = 12, d_ff = 3072 (116.4M Parameters)
- max_seq_len = 512 (prevents stroke dropping on complex drawings)
- canvas_res = 128 (convolutional visual patch conditioning)
- vocab_size = 604 discrete coordinate and control tokens
- Text conditioning: Frozen CLIP / MiniLM semantic tower or built-in semantic encoder

Performance & Training:
- Automatic Mixed Precision (FP16 AMP)
- Gradient Accumulation for effective batch size 128
- Cosine Annealing with Warmup & Gradient Clipping (1.0)
- End-of-epoch visual inference validation (saves test_epoch_{n}.png)
- Resumable checkpointing (checkpoint_best.pt, checkpoint_latest.pt)
"""

import argparse
import copy
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.model.text_encoder import FrozenTextEncoder
from konnon_draw.model.sampling import sample_stage_tokens, generate_staged_drawing
from konnon_draw.v2.tokens import (
    VOCAB_SIZE,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    token_sequence_to_doc,
)
from konnon_draw.v2.renderer import render_document_v2


class StagedDrawingDataset(Dataset):
    """Loads staged vector transitions with pre-rendered raster canvas conditioning."""

    def __init__(
        self,
        jsonl_path: Path,
        max_seq_len: int = 512,
        canvas_res: int = 128,
    ):
        self.max_seq_len = max_seq_len
        self.canvas_res = canvas_res
        self.records: List[Dict[str, Any]] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        # Blank white fallback
        self.blank_canvas = torch.ones(3, canvas_res, canvas_res, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        tokens = item["tokens"]
        caption = item.get("caption", item.get("prompt", ""))
        stage_idx = int(item.get("stage_idx", 0))

        # Truncate / pad tokens
        if len(tokens) > self.max_seq_len:
            seq = tokens[:self.max_seq_len - 1] + [EOS_TOKEN]
        else:
            seq = tokens

        input_ids = seq[:-1]
        target_ids = seq[1:]

        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [PAD_TOKEN] * pad_len
            target_ids = target_ids + [PAD_TOKEN] * pad_len

        # Load raster canvas image
        canvas_path = item.get("canvas_path")
        if canvas_path and os.path.exists(canvas_path):
            try:
                img = Image.open(canvas_path).convert("RGB")
                if img.size != (self.canvas_res, self.canvas_res):
                    img = img.resize((self.canvas_res, self.canvas_res), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
                canvas_t = torch.from_numpy(arr)
            except Exception:
                canvas_t = self.blank_canvas.clone()
        else:
            canvas_t = self.blank_canvas.clone()

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "stage_idx": torch.tensor(stage_idx, dtype=torch.long),
            "canvas": canvas_t,
            "caption": caption,
            "raw_len": len(seq),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "target_ids": torch.stack([b["target_ids"] for b in batch]),
        "stage_idx": torch.stack([b["stage_idx"] for b in batch]),
        "canvas": torch.stack([b["canvas"] for b in batch]),
        "captions": [b["caption"] for b in batch],
        "raw_lens": [b["raw_len"] for b in batch],
    }


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_116m_model(
    data_dir: str = "data/processed/quickdraw_staged_50k",
    out_dir: str = "outputs_v2/scaled_116m",
    batch_size: int = 16,
    grad_accum: int = 8,
    epochs: int = 10,
    lr: float = 3e-4,
    eos_weight: float = 3.5,
    device: Optional[str] = None,
    fp16: Optional[bool] = None,
    d_model: int = 768,
    n_layers: int = 12,
    n_heads: int = 12,
    d_ff: int = 3072,
    max_seq_len: int = 512,
    canvas_res: int = 128,
    text_model: str = "builtin",
    seed: int = 42,
    resume: bool = True,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    process = psutil.Process()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 is None:
        fp16 = True if str(device).startswith("cuda") else False

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    ckpt_best_path = out_p / "checkpoint_best.pt"
    ckpt_latest_path = out_p / "checkpoint_latest.pt"

    print("=" * 70)
    print("STEP 5: TRAINING SCALED 116.4M STAGED DRAWING TRANSFORMER")
    print(f"Device: {device} (FP16: {fp16}) | Batch Size: {batch_size} (Effective: {batch_size * grad_accum})")
    print(f"Architecture: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"Sequence Length: {max_seq_len} | Canvas: {canvas_res}x{canvas_res} | Text: {text_model}")
    print("=" * 70)

    # 1. Dataset & Loaders
    d_path = Path(data_dir)
    train_ds = StagedDrawingDataset(d_path / "train.jsonl", max_seq_len=max_seq_len, canvas_res=canvas_res)
    val_ds = StagedDrawingDataset(d_path / "val.jsonl", max_seq_len=max_seq_len, canvas_res=canvas_res)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2 if str(device).startswith("cuda") else 0,
        pin_memory=True if str(device).startswith("cuda") else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2 if str(device).startswith("cuda") else 0,
    )

    print(f"Loaded Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

    # 2. Instantiate Text Encoder & Model
    text_encoder = FrozenTextEncoder(device=device, model_type=text_model, d_model=d_model)
    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        canvas_res=canvas_res,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # 3. Loss with EOS Boost
    weight_tensor = torch.ones(VOCAB_SIZE, device=device)
    weight_tensor[EOS_TOKEN] = eos_weight
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=PAD_TOKEN)

    # 4. Optimizer, Scaler, Scheduler
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01, fused=True if str(device).startswith("cuda") else False)
    except Exception:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    scaler = torch.amp.GradScaler('cuda', enabled=fp16)

    total_steps = (len(train_loader) // grad_accum) * epochs
    warmup_steps = min(500, max(50, total_steps // 10))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    # Resume from checkpoint if available
    if resume and ckpt_latest_path.exists():
        try:
            print(f"Resuming training from {ckpt_latest_path}...")
            ckpt = torch.load(ckpt_latest_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scaler.load_state_dict(ckpt["scaler_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", float("inf"))
            history = ckpt.get("history", [])
            print(f"Resumed at Epoch {start_epoch} (Best Val Loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"Failed to resume ({e}), starting fresh.")

    # 5. Training Loop
    print("\nEpoch | Train Loss | Val Loss | Nat EOS % | Wall (s) | RAM (MB) | LR")
    print("-" * 68)

    for epoch in range(start_epoch, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_tokens_sum = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            target_ids = batch["target_ids"].to(device, non_blocking=True)
            stage_idx = batch["stage_idx"].to(device, non_blocking=True)
            canvas = batch["canvas"].to(device, non_blocking=True)
            captions = batch["captions"]

            with torch.no_grad():
                text_emb = text_encoder.encode_text(captions, device=device)

            with torch.amp.autocast('cuda', enabled=fp16):
                context = model.build_multimodal_context(
                    text_embeddings=text_emb,
                    canvas=canvas,
                    stage_idx=stage_idx,
                )
                logits = model(input_ids, context=context)
                loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))
                loss_scaled = loss / grad_accum

            scaler.scale(loss_scaled).backward()

            train_loss_sum += float(loss.item()) * int((target_ids != PAD_TOKEN).sum().item())
            train_tokens_sum += int((target_ids != PAD_TOKEN).sum().item())

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                cur_ram = process.memory_info().rss / (1024 * 1024)
                print(f"  [Epoch {epoch:2d}] Batch {step+1:4d}/{len(train_loader):4d} | Loss: {loss.item():.4f} | RAM: {cur_ram:.1f} MB")

        train_loss = train_loss_sum / max(1, train_tokens_sum)

        # 6. Validation Loop
        model.eval()
        val_loss_sum = 0.0
        val_tokens_sum = 0
        natural_eos_count = 0
        total_eval_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                target_ids = batch["target_ids"].to(device, non_blocking=True)
                stage_idx = batch["stage_idx"].to(device, non_blocking=True)
                canvas = batch["canvas"].to(device, non_blocking=True)
                captions = batch["captions"]

                text_emb = text_encoder.encode_text(captions, device=device)
                with torch.amp.autocast('cuda', enabled=fp16):
                    context = model.build_multimodal_context(text_embeddings=text_emb, canvas=canvas, stage_idx=stage_idx)
                    logits = model(input_ids, context=context)
                    loss = loss_fn(logits.view(-1, VOCAB_SIZE), target_ids.view(-1))

                val_loss_sum += float(loss.item()) * int((target_ids != PAD_TOKEN).sum().item())
                val_tokens_sum += int((target_ids != PAD_TOKEN).sum().item())

                # Sample natural EOS probe on first batch
                if total_eval_samples < 32:
                    preds = logits.argmax(dim=-1)
                    for p_seq, raw_l in zip(preds, batch["raw_lens"]):
                        if EOS_TOKEN in p_seq[:raw_l].tolist():
                            natural_eos_count += 1
                        total_eval_samples += 1

        val_loss = val_loss_sum / max(1, val_tokens_sum)
        eos_rate = round((natural_eos_count / max(1, total_eval_samples)) * 100.0, 2)
        wall_time = round(time.time() - t_epoch_start, 1)
        cur_ram = process.memory_info().rss / (1024 * 1024)
        cur_lr = scheduler.get_last_lr()[0]

        print(f" {epoch:4d} | {train_loss:10.5f} | {val_loss:9.5f} | {eos_rate:8.2f}% | {wall_time:7.1f}s | {cur_ram:8.1f} | {cur_lr:.2e}")

        # 7. Live Visual Evaluation at End of Epoch
        try:
            sample_res = generate_staged_drawing(
                model=model,
                text_encoder=text_encoder,
                prompt="a drawing of a cat",
                stages=["layout", "guide", "ink", "shade"],
                canvas_res=canvas_res,
                out_res=512,
                device=device,
                seed=42 + epoch,
            )
            sample_img = sample_res["final_image"]
            sample_img.save(str(out_p / f"test_epoch_{epoch}.png"))
            sample_img.save(str(out_p / "test_latest.png"))
            print(f"  🎨 Visual Test Drawing Exported: {out_p / f'test_epoch_{epoch}.png'}")
        except Exception as e:
            print(f"  Visual evaluation warning: {e}")

        # 8. Checkpoints
        ckpt_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "val_loss": val_loss,
            "train_loss": train_loss,
            "config": {
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": n_heads,
                "d_ff": d_ff,
                "max_seq_len": max_seq_len,
                "canvas_res": canvas_res,
                "vocab_size": VOCAB_SIZE,
                "text_model": text_model,
            },
        }

        torch.save(ckpt_state, ckpt_latest_path)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_state["best_val_loss"] = best_val_loss
            torch.save(ckpt_state, ckpt_best_path)
            print(f"  ⭐ NEW BEST CHECKPOINT SAVED (Val Loss: {best_val_loss:.5f}) -> {ckpt_best_path}")

    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE! Best Validation Loss: {best_val_loss:.5f}")
    print(f"Exported Model: {ckpt_best_path}")
    print("=" * 70)

    return {"best_val_loss": best_val_loss, "checkpoint": str(ckpt_best_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/quickdraw_staged_50k")
    parser.add_argument("--out_dir", default="outputs_v2/scaled_116m")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--canvas_res", type=int, default=128)
    parser.add_argument("--text_model", default="builtin")
    args = parser.parse_args()

    train_116m_model(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        device=args.device,
        fp16=args.fp16,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        canvas_res=args.canvas_res,
        text_model=args.text_model,
    )
