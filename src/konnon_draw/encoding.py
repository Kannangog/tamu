"""Encode drawings to (action, x, y) token lists and batch them with shifted targets."""
import hashlib
import json
from pathlib import Path

import torch

from .schema import (Action, ID_TO_NAME, NAME_TO_ID, NO_COORD,
                     dequantize_coord, quantize_coord, validate_actions)


def encode_one(actions, add_start=True):
    ids, xs, ys = [], [], []
    if add_start:
        ids.append(int(Action.START)); xs.append(NO_COORD); ys.append(NO_COORD)
    for a in actions:
        kind = a["type"]
        ids.append(NAME_TO_ID[kind])
        if kind in ("MOVE", "LINE"):
            xs.append(quantize_coord(a["x"])); ys.append(quantize_coord(a["y"]))
        else:
            xs.append(NO_COORD); ys.append(NO_COORD)
    return ids, xs, ys


def decode_one(ids, xs, ys):
    """Inverse of encode_one. Skips START/PAD, stops at END, always returns an END-terminated list."""
    actions = []
    for a, x, y in zip(ids, xs, ys):
        a = int(a)
        if a == int(Action.END):
            break
        if a not in ID_TO_NAME:  # PAD / START
            continue
        kind = ID_TO_NAME[a]
        if kind in ("MOVE", "LINE"):
            actions.append({"type": kind, "x": dequantize_coord(int(x)), "y": dequantize_coord(int(y))})
        else:
            actions.append({"type": kind})
    actions.append({"type": "END"})
    return actions


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_version(path):
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:10]


def class_list(examples):
    return sorted({ex.get("category", "unknown") for ex in examples})


def encode_examples(examples, max_len=128, classes=None):
    """Return (encoded, skipped). Item: dict(id, category, cls, ids, xs, ys). Invalid/too-long examples are skipped."""
    classes = classes or class_list(examples)
    out, skipped = [], 0
    for ex in examples:
        try:
            validate_actions(ex["actions"])
        except ValueError:
            skipped += 1
            continue
        ids, xs, ys = encode_one(ex["actions"])
        if len(ids) > max_len:
            skipped += 1
            continue
        cat = ex.get("category", "unknown")
        out.append({"id": ex["id"], "category": cat, "cls": classes.index(cat), "ids": ids, "xs": xs, "ys": ys})
    return out, skipped


def collate(items):
    """Pad to equal length; inputs are tokens 0..T-1, targets are tokens 1..T; last item is class ids."""
    L = max(max(len(i["ids"]) for i in items), 2)
    B = len(items)
    A = torch.full((B, L), int(Action.PAD), dtype=torch.long)
    X = torch.full((B, L), NO_COORD, dtype=torch.long)
    Y = torch.full((B, L), NO_COORD, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it["ids"])
        A[b, :n] = torch.tensor(it["ids"]); X[b, :n] = torch.tensor(it["xs"]); Y[b, :n] = torch.tensor(it["ys"])
    C = torch.tensor([it["cls"] for it in items], dtype=torch.long)
    return A[:, :-1], X[:, :-1], Y[:, :-1], A[:, 1:], X[:, 1:], Y[:, 1:], C
