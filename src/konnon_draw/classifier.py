"""Automated Sketch Classifier for Recognizability Evaluation.

Trains a lightweight sequence classifier on real Quick, Draw! strokes
to provide an objective benchmark for category recognizability.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .conditioned_encoding import (
    NUM_CATEGORIES,
    REAL_COORD_VOCAB,
    REAL_NO_COORD,
    encode_conditioned_one,
)
from .quickdraw import CATEGORY_TO_ID, DEFAULT_CATEGORIES
from .schema import Action


class SketchClassifierDataset(Dataset):
    def __init__(self, examples: List[dict], max_len: int = 96):
        self.data = []
        for ex in examples:
            cat_name = ex["category"]
            cat_id = CATEGORY_TO_ID[cat_name]
            ids, xs, ys, _ = encode_conditioned_one(ex["actions"], cat_name)
            ids = ids[:max_len]
            xs = xs[:max_len]
            ys = ys[:max_len]
            self.data.append((ids, xs, ys, cat_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, xs, ys, cat_id = self.data[idx]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            torch.tensor(cat_id, dtype=torch.long),
        )


def collate_classifier_batch(batch):
    ids, xs, ys, cat_ids = zip(*batch)
    max_len = max(x.size(0) for x in ids)

    b_ids = torch.full((len(batch), max_len), Action.PAD, dtype=torch.long)
    b_xs = torch.full((len(batch), max_len), REAL_NO_COORD, dtype=torch.long)
    b_ys = torch.full((len(batch), max_len), REAL_NO_COORD, dtype=torch.long)

    for i in range(len(batch)):
        l = ids[i].size(0)
        b_ids[i, :l] = ids[i]
        b_xs[i, :l] = xs[i]
        b_ys[i, :l] = ys[i]

    return b_ids, b_xs, b_ys, torch.stack(cat_ids)


class SketchClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 96, num_classes: int = NUM_CATEGORIES):
        super().__init__()
        self.action_emb = nn.Embedding(5, embedding_dim, padding_idx=0)
        self.x_emb = nn.Embedding(REAL_COORD_VOCAB, embedding_dim)
        self.y_emb = nn.Embedding(REAL_COORD_VOCAB, embedding_dim)
        self.gru = nn.GRU(embedding_dim * 3, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, action_ids, x_ids, y_ids):
        z = torch.cat([
            self.action_emb(action_ids),
            self.x_emb(x_ids),
            self.y_emb(y_ids),
        ], dim=-1)
        h, _ = self.gru(z)
        # Global max pooling over sequence length
        pooled = torch.max(h, dim=1)[0]
        logits = self.classifier(pooled)
        return logits


def train_sketch_classifier(
    train_file: str = "data/processed/real_drawings_train.jsonl",
    val_file: str = "data/processed/real_drawings_val.jsonl",
    test_file: str = "data/processed/real_drawings_test.jsonl",
    save_path: str = "outputs/checkpoints/sketch_classifier.pt",
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 0.002,
) -> float:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    with open(train_file, "r", encoding="utf-8") as f:
        train_ex = [json.loads(line) for line in f if line.strip()]
    with open(val_file, "r", encoding="utf-8") as f:
        val_ex = [json.loads(line) for line in f if line.strip()]
    with open(test_file, "r", encoding="utf-8") as f:
        test_ex = [json.loads(line) for line in f if line.strip()]

    train_ds = SketchClassifierDataset(train_ex)
    val_ds = SketchClassifierDataset(val_ex)
    test_ds = SketchClassifierDataset(test_ex)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_classifier_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_classifier_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_classifier_batch)

    model = SketchClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        for b_ids, b_xs, b_ys, labels in train_loader:
            optimizer.zero_grad()
            logits = model(b_ids, b_xs, b_ys)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Evaluate on validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for b_ids, b_xs, b_ys, labels in val_loader:
                logits = model(b_ids, b_xs, b_ys)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # Evaluate on held-out test
    model.load_state_dict(torch.load(save_path, map_location="cpu", weights_only=False))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b_ids, b_xs, b_ys, labels in test_loader:
            logits = model(b_ids, b_xs, b_ys)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"Sketch Classifier trained! Held-out test accuracy: {test_acc:.1%} (saved to {save_path})")
    return test_acc


def predict_sketch_category(actions: List[dict], classifier_path: str = "outputs/checkpoints/sketch_classifier.pt") -> Tuple[str, float]:
    model = SketchClassifier()
    model.load_state_dict(torch.load(classifier_path, map_location="cpu", weights_only=False))
    model.eval()

    ids, xs, ys, _ = encode_conditioned_one(actions, "apple")
    b_ids = torch.tensor([ids], dtype=torch.long)
    b_xs = torch.tensor([xs], dtype=torch.long)
    b_ys = torch.tensor([ys], dtype=torch.long)

    with torch.no_grad():
        logits = model(b_ids, b_xs, b_ys)
        probs = F.softmax(logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    return DEFAULT_CATEGORIES[pred_idx], confidence
