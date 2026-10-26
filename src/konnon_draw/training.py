import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .encoding import NO_COORD, encode_one
from .model import ACTION_VOCAB, COORD_VOCAB, DrawingGRU
from .schema import Action


class DrawingDataset(Dataset):
    def __init__(self, examples: List[dict], max_len: int = 128):
        self.max_len = max_len
        self.encoded_samples = []
        for ex in examples:
            ids, xs, ys = encode_one(ex["actions"])
            if len(ids) < 2:
                continue
            # Truncate if exceeds max_len
            ids = ids[:max_len]
            xs = xs[:max_len]
            ys = ys[:max_len]
            self.encoded_samples.append((ids, xs, ys))

    def __len__(self) -> int:
        return len(self.encoded_samples)

    def __getitem__(self, idx: int):
        ids, xs, ys = self.encoded_samples[idx]
        # input is 0..T-1, target is 1..T
        in_ids = ids[:-1]
        in_xs = xs[:-1]
        in_ys = ys[:-1]

        tgt_ids = ids[1:]
        tgt_xs = xs[1:]
        tgt_ys = ys[1:]

        return (
            torch.tensor(in_ids, dtype=torch.long),
            torch.tensor(in_xs, dtype=torch.long),
            torch.tensor(in_ys, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            torch.tensor(tgt_xs, dtype=torch.long),
            torch.tensor(tgt_ys, dtype=torch.long),
        )


def collate_drawings(batch):
    in_ids, in_xs, in_ys, tgt_ids, tgt_xs, tgt_ys = zip(*batch)
    max_in_len = max(x.size(0) for x in in_ids)

    b_in_ids = torch.full((len(batch), max_in_len), Action.PAD, dtype=torch.long)
    b_in_xs = torch.full((len(batch), max_in_len), NO_COORD, dtype=torch.long)
    b_in_ys = torch.full((len(batch), max_in_len), NO_COORD, dtype=torch.long)

    b_tgt_ids = torch.full((len(batch), max_in_len), Action.PAD, dtype=torch.long)
    b_tgt_xs = torch.full((len(batch), max_in_len), NO_COORD, dtype=torch.long)
    b_tgt_ys = torch.full((len(batch), max_in_len), NO_COORD, dtype=torch.long)

    for i in range(len(batch)):
        l = in_ids[i].size(0)
        b_in_ids[i, :l] = in_ids[i]
        b_in_xs[i, :l] = in_xs[i]
        b_in_ys[i, :l] = in_ys[i]

        b_tgt_ids[i, :l] = tgt_ids[i]
        b_tgt_xs[i, :l] = tgt_xs[i]
        b_tgt_ys[i, :l] = tgt_ys[i]

    return b_in_ids, b_in_xs, b_in_ys, b_tgt_ids, b_tgt_xs, b_tgt_ys


def compute_loss(
    action_logits: torch.Tensor,
    x_logits: torch.Tensor,
    y_logits: torch.Tensor,
    tgt_ids: torch.Tensor,
    tgt_xs: torch.Tensor,
    tgt_ys: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Action loss ignores Action.PAD
    act_loss = F.cross_entropy(
        action_logits.view(-1, ACTION_VOCAB),
        tgt_ids.view(-1),
        ignore_index=Action.PAD,
    )

    # Coordinate loss calculated only where target action is MOVE (1) or LINE (2)
    coord_mask = (tgt_ids == Action.MOVE) | (tgt_ids == Action.LINE)
    if coord_mask.sum() > 0:
        x_loss = F.cross_entropy(x_logits[coord_mask], tgt_xs[coord_mask])
        y_loss = F.cross_entropy(y_logits[coord_mask], tgt_ys[coord_mask])
        coord_loss = (x_loss + y_loss) / 2.0
    else:
        coord_loss = torch.tensor(0.0, device=action_logits.device)

    total_loss = act_loss + coord_loss
    return total_loss, act_loss, coord_loss


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss_accum = 0.0
    act_loss_accum = 0.0
    coord_loss_accum = 0.0
    num_batches = 0

    for in_ids, in_xs, in_ys, tgt_ids, tgt_xs, tgt_ys in dataloader:
        in_ids = in_ids.to(device)
        in_xs = in_xs.to(device)
        in_ys = in_ys.to(device)
        tgt_ids = tgt_ids.to(device)
        tgt_xs = tgt_xs.to(device)
        tgt_ys = tgt_ys.to(device)

        optimizer.zero_grad()
        act_logits, x_logits, y_logits, _ = model(in_ids, in_xs, in_ys)
        loss, act_loss, coord_loss = compute_loss(
            act_logits, x_logits, y_logits, tgt_ids, tgt_xs, tgt_ys
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss_accum += loss.item()
        act_loss_accum += act_loss.item()
        coord_loss_accum += coord_loss.item()
        num_batches += 1

    return (
        total_loss_accum / max(1, num_batches),
        act_loss_accum / max(1, num_batches),
        coord_loss_accum / max(1, num_batches),
    )


@torch.no_grad()
def evaluate_loss(model, dataloader, device):
    model.eval()
    total_loss_accum = 0.0
    act_loss_accum = 0.0
    coord_loss_accum = 0.0
    num_batches = 0

    for in_ids, in_xs, in_ys, tgt_ids, tgt_xs, tgt_ys in dataloader:
        in_ids = in_ids.to(device)
        in_xs = in_xs.to(device)
        in_ys = in_ys.to(device)
        tgt_ids = tgt_ids.to(device)
        tgt_xs = tgt_xs.to(device)
        tgt_ys = tgt_ys.to(device)

        act_logits, x_logits, y_logits, _ = model(in_ids, in_xs, in_ys)
        loss, act_loss, coord_loss = compute_loss(
            act_logits, x_logits, y_logits, tgt_ids, tgt_xs, tgt_ys
        )

        total_loss_accum += loss.item()
        act_loss_accum += act_loss.item()
        coord_loss_accum += coord_loss.item()
        num_batches += 1

    return (
        total_loss_accum / max(1, num_batches),
        act_loss_accum / max(1, num_batches),
        coord_loss_accum / max(1, num_batches),
    )


def train(
    data_path: str = "data/processed/toy_drawings.jsonl",
    hidden_dim: int = 128,
    embedding_dim: int = 32,
    batch_size: int = 8,
    max_len: int = 128,
    lr: float = 0.001,
    epochs: int = 20,
    seed: int = 42,
    checkpoint_dir: str = "outputs/checkpoints",
    log_dir: str = "logs",
):
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Load dataset
    with open(data_path, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    # Fixed train / validation split (80 / 20)
    indices = list(range(len(examples)))
    random.seed(seed)
    random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_ex = [examples[i] for i in indices[:split_idx]]
    val_ex = [examples[i] for i in indices[split_idx:]]

    train_ds = DrawingDataset(train_ex, max_len=max_len)
    val_ds = DrawingDataset(val_ex, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_drawings
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_drawings
    )

    model = DrawingGRU(embedding_dim=embedding_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    training_log_path = Path(log_dir) / "training_log.csv"
    with open(training_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_act_loss",
            "train_coord_loss",
            "val_loss",
            "val_act_loss",
            "val_coord_loss",
        ])

    best_val_loss = float("inf")
    print(f"Starting training on {len(train_ds)} train examples, {len(val_ds)} val examples")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_act, tr_coord = train_epoch(model, train_loader, optimizer, device)
        v_loss, v_act, v_coord = evaluate_loss(model, val_loader, device)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {tr_loss:.4f} (act: {tr_act:.4f}, coord: {tr_coord:.4f}) | "
            f"Val Loss: {v_loss:.4f} (act: {v_act:.4f}, coord: {v_coord:.4f})"
        )

        with open(training_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_act, tr_coord, v_loss, v_act, v_coord])

        # Save checkpoint
        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr_loss,
            "val_loss": v_loss,
            "config": {
                "hidden_dim": hidden_dim,
                "embedding_dim": embedding_dim,
                "max_len": max_len,
                "batch_size": batch_size,
                "lr": lr,
                "seed": seed,
            },
        }

        # Periodic recovery checkpoint
        torch.save(checkpoint_payload, Path(checkpoint_dir) / "checkpoint_last.pt")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(checkpoint_payload, Path(checkpoint_dir) / "best_model.pt")
            print(f"  --> Saved new best model (val_loss: {v_loss:.4f})")

    print(f"Training completed. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train the DrawingGRU model.")
    parser.add_argument("--data", default="data/processed/toy_drawings.jsonl", help="JSONL data file")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GRU hidden dimension")
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--max-len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints", help="Checkpoints directory")
    parser.add_argument("--log-dir", default="logs", help="Logs directory")
    args = parser.parse_args()

    train(
        data_path=args.data,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        max_len=args.max_len,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
