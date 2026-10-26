"""Training pipeline for ConditionedDrawingGRU on real Quick, Draw! data."""

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .conditioned_encoding import (
    REAL_COORD_VOCAB,
    REAL_NO_COORD,
    encode_conditioned_one,
)
from .conditioned_model import ACTION_VOCAB, ConditionedDrawingGRU
from .quickdraw import CATEGORY_TO_ID, DEFAULT_CATEGORIES
from .schema import Action


class ConditionedDrawingDataset(Dataset):
    def __init__(self, examples: List[dict], max_len: int = 96):
        self.max_len = max_len
        self.samples = []
        for ex in examples:
            cat = ex["category"]
            ids, xs, ys, cat_id = encode_conditioned_one(ex["actions"], cat)
            if len(ids) < 2:
                continue
            ids = ids[:max_len]
            xs = xs[:max_len]
            ys = ys[:max_len]
            self.samples.append((ids, xs, ys, cat_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        ids, xs, ys, cat_id = self.samples[idx]
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
            torch.tensor(cat_id, dtype=torch.long),
        )


def collate_conditioned(batch):
    in_ids, in_xs, in_ys, tgt_ids, tgt_xs, tgt_ys, cat_ids = zip(*batch)
    max_len = max(x.size(0) for x in in_ids)

    b_in_ids = torch.full((len(batch), max_len), Action.PAD, dtype=torch.long)
    b_in_xs = torch.full((len(batch), max_len), REAL_NO_COORD, dtype=torch.long)
    b_in_ys = torch.full((len(batch), max_len), REAL_NO_COORD, dtype=torch.long)

    b_tgt_ids = torch.full((len(batch), max_len), Action.PAD, dtype=torch.long)
    b_tgt_xs = torch.full((len(batch), max_len), REAL_NO_COORD, dtype=torch.long)
    b_tgt_ys = torch.full((len(batch), max_len), REAL_NO_COORD, dtype=torch.long)

    for i in range(len(batch)):
        l = in_ids[i].size(0)
        b_in_ids[i, :l] = in_ids[i]
        b_in_xs[i, :l] = in_xs[i]
        b_in_ys[i, :l] = in_ys[i]

        b_tgt_ids[i, :l] = tgt_ids[i]
        b_tgt_xs[i, :l] = tgt_xs[i]
        b_tgt_ys[i, :l] = tgt_ys[i]

    return b_in_ids, b_in_xs, b_in_ys, b_tgt_ids, b_tgt_xs, b_tgt_ys, torch.stack(cat_ids)


def compute_conditioned_loss(
    act_logits: torch.Tensor,
    x_logits: torch.Tensor,
    y_logits: torch.Tensor,
    tgt_ids: torch.Tensor,
    tgt_xs: torch.Tensor,
    tgt_ys: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Action loss ignores Action.PAD
    act_loss = F.cross_entropy(
        act_logits.view(-1, ACTION_VOCAB),
        tgt_ids.view(-1),
        ignore_index=Action.PAD,
    )

    # Coordinate loss calculated only where target action is MOVE or LINE
    coord_mask = (tgt_ids == Action.MOVE) | (tgt_ids == Action.LINE)
    if coord_mask.sum() > 0:
        x_loss = F.cross_entropy(x_logits[coord_mask], tgt_xs[coord_mask])
        y_loss = F.cross_entropy(y_logits[coord_mask], tgt_ys[coord_mask])
        coord_loss = (x_loss + y_loss) / 2.0
    else:
        coord_loss = torch.tensor(0.0, device=act_logits.device)

    total_loss = act_loss + coord_loss
    return total_loss, act_loss, coord_loss


def train_conditioned_model(
    train_path: str = "data/processed/real_drawings_train.jsonl",
    val_path: str = "data/processed/real_drawings_val.jsonl",
    hidden_dim: int = 160,
    embedding_dim: int = 32,
    category_dim: int = 32,
    batch_size: int = 16,
    max_len: int = 96,
    lr: float = 0.0015,
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

    with open(train_path, "r", encoding="utf-8") as f:
        train_ex = [json.loads(line) for line in f if line.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_ex = [json.loads(line) for line in f if line.strip()]

    train_ds = ConditionedDrawingDataset(train_ex, max_len=max_len)
    val_ds = ConditionedDrawingDataset(val_ex, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_conditioned)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_conditioned)

    model = ConditionedDrawingGRU(
        embedding_dim=embedding_dim,
        category_dim=category_dim,
        hidden_dim=hidden_dim,
        coord_vocab=REAL_COORD_VOCAB,
        num_categories=len(DEFAULT_CATEGORIES),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    log_file = Path(log_dir) / "real_training_log.csv"
    with open(log_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_act_loss", "train_coord_loss", "val_loss", "val_act_loss", "val_coord_loss"])

    best_val_loss = float("inf")
    print(f"Starting conditioned training on {len(train_ds)} real drawings across {len(DEFAULT_CATEGORIES)} categories...")

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss_accum, t_act_accum, t_coord_accum = 0.0, 0.0, 0.0
        n_b = 0
        for in_ids, in_xs, in_ys, tgt_ids, tgt_xs, tgt_ys, cat_ids in train_loader:
            optimizer.zero_grad()
            act_l, x_l, y_l, _ = model(in_ids, in_xs, in_ys, cat_ids)
            loss, act_l_val, coord_l_val = compute_conditioned_loss(act_l, x_l, y_l, tgt_ids, tgt_xs, tgt_ys)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            t_loss_accum += loss.item()
            t_act_accum += act_l_val.item()
            t_coord_accum += coord_l_val.item()
            n_b += 1

        tr_loss = t_loss_accum / max(1, n_b)
        tr_act = t_act_accum / max(1, n_b)
        tr_coord = t_coord_accum / max(1, n_b)

        # Validation pass
        model.eval()
        v_loss_accum, v_act_accum, v_coord_accum = 0.0, 0.0, 0.0
        v_b = 0
        with torch.no_grad():
            for in_ids, in_xs, in_ys, tgt_ids, tgt_xs, tgt_ys, cat_ids in val_loader:
                act_l, x_l, y_l, _ = model(in_ids, in_xs, in_ys, cat_ids)
                loss, act_l_val, coord_l_val = compute_conditioned_loss(act_l, x_l, y_l, tgt_ids, tgt_xs, tgt_ys)
                v_loss_accum += loss.item()
                v_act_accum += act_l_val.item()
                v_coord_accum += coord_l_val.item()
                v_b += 1

        v_loss = v_loss_accum / max(1, v_b)
        v_act = v_act_accum / max(1, v_b)
        v_coord = v_coord_accum / max(1, v_b)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {tr_loss:.4f} (act: {tr_act:.4f}, coord: {tr_coord:.4f}) | "
            f"Val Loss: {v_loss:.4f} (act: {v_act:.4f}, coord: {v_coord:.4f})"
        )

        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_act, tr_coord, v_loss, v_act, v_coord])

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr_loss,
            "val_loss": v_loss,
            "config": {
                "hidden_dim": hidden_dim,
                "embedding_dim": embedding_dim,
                "category_dim": category_dim,
                "coord_vocab": REAL_COORD_VOCAB,
                "max_len": max_len,
                "batch_size": batch_size,
                "categories": DEFAULT_CATEGORIES,
                "seed": seed,
            },
        }

        torch.save(checkpoint, Path(checkpoint_dir) / "real_checkpoint_last.pt")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(checkpoint, Path(checkpoint_dir) / "real_best_model.pt")
            print(f"  --> Saved new best real-stroke model (val_loss: {v_loss:.4f})")

    print(f"Conditioned model training complete. Best val loss: {best_val_loss:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train ConditionedDrawingGRU on real Quick, Draw! data.")
    parser.add_argument("--train-data", default="data/processed/real_drawings_train.jsonl")
    parser.add_argument("--val-data", default="data/processed/real_drawings_val.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_conditioned_model(
        train_path=args.train_data,
        val_path=args.val_data,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
