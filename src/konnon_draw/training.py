"""Teacher-forced next-action training with masked loss, checkpoints and an experiment log."""
import argparse
import csv
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoding import class_list, collate, dataset_version, encode_examples, load_jsonl
from .model import DrawingGRU
from .sampling import generate, save_sample
from .schema import COORD_BINS, SCHEMA_VERSION, Action, is_valid

LOG_FIELDS = ["date", "git_commit", "dataset_version", "seed", "model", "hidden_dim", "batch_size",
              "max_len", "learning_rate", "epoch", "train_loss", "val_loss", "train_action_loss",
              "train_coord_loss", "val_action_loss", "val_coord_loss", "val_action_acc",
              "valid_rate", "completion_rate", "epoch_seconds", "notes"]


@dataclass
class TrainConfig:
    data: str = "data/processed/objects.jsonl"
    out_dir: str = "outputs"
    log_dir: str = "logs"
    seed: int = 1
    val_frac: float = 0.15
    embedding_dim: int = 48
    hidden_dim: int = 256
    batch_size: int = 32
    max_len: int = 160
    lr: float = 2e-3
    lr_final_frac: float = 0.1
    coord_smooth: float = 1.0
    weight_decay: float = 1e-2
    clip: float = 1.0
    epochs: int = 40
    checkpoint_every: int = 5
    sample_every: int = 5
    eval_samples: int = 4  # per class
    notes: str = ""


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def get_peak_ram_mb():
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class PMC(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                            ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                            ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                            ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                            ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.PeakWorkingSetSize / (1024 * 1024)
        else:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass
    return 0.0


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "n/a"


def split_items(items, val_frac, seed):
    """Split by example with a fixed seed; validation is never used for gradient updates."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    n_val = max(1, int(round(len(items) * val_frac)))
    val = [items[i] for i in order[:n_val]]
    train = [items[i] for i in order[n_val:]]
    return train, val


def soft_coord_ce(logits, targets, sigma):
    """Cross-entropy against a Gaussian over neighbouring bins, so near-misses are penalised less."""
    if sigma <= 0:
        return F.cross_entropy(logits, targets)
    bins = torch.arange(COORD_BINS, dtype=torch.float32)
    q = torch.exp(-0.5 * ((bins[None, :] - targets[:, None].float()) / sigma) ** 2)
    q = q / q.sum(1, keepdim=True)
    return -(q * F.log_softmax(logits, -1)[:, :COORD_BINS]).sum(1).mean()


def batch_loss(model, batch, coord_smooth=1.0):
    """Masked losses. Action loss ignores PAD; coord loss only where the target is MOVE/LINE."""
    in_a, in_x, in_y, t_a, t_x, t_y, cls = batch
    a_log, x_log, h, _ = model(in_a, in_x, in_y, cls)
    y_log = model.predict_y(h, t_x)  # y head conditioned on the true x (teacher forcing)
    valid = t_a != int(Action.PAD)
    coord = (t_a == int(Action.MOVE)) | (t_a == int(Action.LINE))
    action_loss = F.cross_entropy(a_log[valid], t_a[valid])
    n_coord = int(coord.sum())
    if n_coord:
        coord_loss = 0.5 * (soft_coord_ce(x_log[coord], t_x[coord], coord_smooth) + soft_coord_ce(y_log[coord], t_y[coord], coord_smooth))
    else:
        coord_loss = action_loss * 0.0
    correct = int((a_log.argmax(-1)[valid] == t_a[valid]).sum())
    return action_loss, coord_loss, int(valid.sum()), n_coord, correct


def run_epoch(model, items, batch_size, optimizer=None, clip=1.0, shuffle_seed=None, coord_smooth=1.0):
    training = optimizer is not None
    model.train(training)
    order = list(range(len(items)))
    if training:
        random.Random(shuffle_seed).shuffle(order)  # shuffle train only
    tot = dict(a=0.0, c=0.0, na=0, nc=0, correct=0)
    with torch.set_grad_enabled(training):
        for s in range(0, len(order), batch_size):
            batch = collate([items[i] for i in order[s:s + batch_size]])
            al, cl, na, nc, correct = batch_loss(model, batch, coord_smooth)
            loss = al + cl
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss: check inputs, targets and learning rate")
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            tot["a"] += al.item() * na; tot["na"] += na
            tot["c"] += cl.item() * nc; tot["nc"] += nc; tot["correct"] += correct
    a = tot["a"] / max(tot["na"], 1)
    c = tot["c"] / max(tot["nc"], 1)
    return {"action_loss": a, "coord_loss": c, "loss": a + c, "action_acc": tot["correct"] / max(tot["na"], 1)}


def quick_generation_check(model, cfg, n_per_class, n_classes):
    res = [generate(model, c, 10_000 + i, cfg.max_len, 0.5) for c in range(n_classes) for i in range(n_per_class)]
    valid = sum(is_valid(r["actions"]) for r in res) / len(res)
    done = sum(r["natural_end"] for r in res) / len(res)
    return valid, done, res


def save_checkpoint(path, model, optimizer, epoch, cfg, ds_version, val_loss, classes):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "epoch": epoch, "config": asdict(cfg), "model_config": model.config,
                "dataset_version": ds_version, "seed": cfg.seed, "val_loss": val_loss,
                "schema_version": SCHEMA_VERSION, "classes": classes}, path)


def load_model(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DrawingGRU(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    out_dir, log_dir = Path(cfg.out_dir), Path(cfg.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)
    examples = load_jsonl(cfg.data)
    classes = class_list(examples)
    items, skipped = encode_examples(examples, cfg.max_len, classes)
    train_items, val_items = split_items(items, cfg.val_frac, cfg.seed)
    ds_version = dataset_version(cfg.data)
    (out_dir / "split.json").write_text(json.dumps({
        "seed": cfg.seed, "val_frac": cfg.val_frac, "dataset_version": ds_version,
        "train_ids": [i["id"] for i in train_items], "val_ids": [i["id"] for i in val_items]}))
    print(f"Data: {len(train_items)} train / {len(val_items)} val (skipped {skipped}); classes {classes}; dataset {ds_version}")

    model = DrawingGRU(len(classes), cfg.embedding_dim, cfg.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    import math
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda e: cfg.lr_final_frac + (1 - cfg.lr_final_frac) * 0.5 * (1 + math.cos(math.pi * min(e, cfg.epochs) / cfg.epochs)))
    print(f"Model: DrawingGRU, {model.num_parameters():,} parameters, CPU")

    log_path = log_dir / "train_log.csv"
    new_log = not log_path.exists()
    best = float("inf")
    with log_path.open("a", newline="", encoding="utf-8") as lf:
        writer = csv.DictWriter(lf, fieldnames=LOG_FIELDS)
        if new_log:
            writer.writeheader()
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()
            tr = run_epoch(model, train_items, cfg.batch_size, optimizer, cfg.clip, cfg.seed * 1000 + epoch, cfg.coord_smooth)
            sched.step()
            va = run_epoch(model, val_items, cfg.batch_size, coord_smooth=cfg.coord_smooth)
            valid_rate, completion_rate, samples = quick_generation_check(model, cfg, cfg.eval_samples, len(classes))
            secs = time.time() - t0
            writer.writerow({
                "date": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
                "dataset_version": ds_version, "seed": cfg.seed, "model": "DrawingGRU",
                "hidden_dim": cfg.hidden_dim, "batch_size": cfg.batch_size, "max_len": cfg.max_len,
                "learning_rate": cfg.lr, "epoch": epoch, "train_loss": f"{tr['loss']:.4f}",
                "val_loss": f"{va['loss']:.4f}", "train_action_loss": f"{tr['action_loss']:.4f}",
                "train_coord_loss": f"{tr['coord_loss']:.4f}", "val_action_loss": f"{va['action_loss']:.4f}",
                "val_coord_loss": f"{va['coord_loss']:.4f}", "val_action_acc": f"{va['action_acc']:.4f}",
                "valid_rate": f"{valid_rate:.3f}", "completion_rate": f"{completion_rate:.3f}",
                "epoch_seconds": f"{secs:.1f}", "notes": cfg.notes})
            lf.flush()
            ram_mb = get_peak_ram_mb()
            print(f"epoch {epoch:3d} | train {tr['loss']:.3f} | val {va['loss']:.3f} "
                  f"(act {va['action_loss']:.3f}, coord {va['coord_loss']:.3f}, acc {va['action_acc']:.2f}) "
                  f"| valid {valid_rate:.2f} done {completion_rate:.2f} | {secs:.1f}s | peak RAM {ram_mb:.1f} MB")
            if va["loss"] < best:
                best = va["loss"]
                save_checkpoint(out_dir / "checkpoints" / "best.pt", model, optimizer, epoch, cfg, ds_version, best, classes)
            save_checkpoint(out_dir / "checkpoints" / "last.pt", model, optimizer, epoch, cfg, ds_version, va["loss"], classes)
            if epoch % cfg.checkpoint_every == 0:
                save_checkpoint(out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg,
                                ds_version, va["loss"], classes)
            if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
                for i, r in enumerate(samples):  # same fixed seeds every time
                    save_sample(out_dir / "samples" / f"epoch_{epoch:03d}", f"{classes[r['class_id']]}_{i}", r)
    print(f"Best val loss {best:.4f}. Peak RAM: {get_peak_ram_mb():.1f} MB. Checkpoints in {out_dir / 'checkpoints'}; log at {log_path}")
    return best


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the Konnon drawing GRU")
    for f, v in asdict(TrainConfig()).items():
        p.add_argument("--" + f.replace("_", "-"), type=type(v), default=v)
    args = p.parse_args(argv)
    train(TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()}))


if __name__ == "__main__":
    main()
