"""Per-category evaluation: validity, completion, diversity, novelty, kNN recognizability, export, baseline."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .encoding import load_jsonl
from .raster import draw_tile, rasterize
from .renderer import actions_to_paths, render_svg
from .sampling import generate, random_baseline, save_sample
from .schema import is_valid


def stroke_stats(actions):
    paths = actions_to_paths(actions)
    length = sum(float(np.hypot(b[0] - a[0], b[1] - a[1])) for p in paths for a, b in zip(p, p[1:]))
    n = len(actions)
    return {"action_count": n, "path_count": len(paths), "path_length": length,
            "pen_up_ratio": sum(a["type"] == "PEN_UP" for a in actions) / n}


def contact_sheet(by_class, path, cols=8, tile=128):
    """One labelled block per category. Red border = invalid, orange = length-limit fallback."""
    band, rows_h, y = 22, [], 0
    for _, res in by_class:
        rows_h.append(band + ((len(res) + cols - 1) // cols) * tile)
    sheet = Image.new("RGB", (cols * tile, sum(rows_h)), "white")
    d = ImageDraw.Draw(sheet)
    for (name, res), h in zip(by_class, rows_h):
        d.text((6, y + 5), name, fill="#111827")
        for i, r in enumerate(res):
            x0, y0 = (i % cols) * tile, y + band + (i // cols) * tile
            color = "#d1d5db"
            if is_valid(r["actions"]):
                sheet.paste(draw_tile(r["actions"], tile - 4).convert("RGB"), (x0 + 2, y0 + 2))
                if r["repaired"]:
                    color = "#f59e0b"
            else:
                color = "#ef4444"
            d.rectangle([x0, y0, x0 + tile - 1, y0 + tile - 1], outline=color, width=1)
        y += h
    sheet.save(path)


def _dists(q, bank):
    """Pairwise L2 via the dot-product identity (memory-safe: no (N, M, D) tensor)."""
    d2 = (q ** 2).sum(1)[:, None] + (bank ** 2).sum(1)[None, :] - 2.0 * q @ bank.T
    return np.sqrt(np.maximum(d2, 0.0))


def _pairwise_mean(mat):
    if len(mat) < 2:
        return 0.0
    d = _dists(mat, mat)
    return float(d.sum() / (len(mat) * (len(mat) - 1)))


def _knn_labels(q, bank, bank_labels, k=5):
    idx = np.argsort(_dists(q, bank), axis=1)[:, :k]
    return [int(np.bincount(bank_labels[r]).argmax()) for r in idx]


def evaluate(checkpoint, data, out_dir, n_per_class=16, seed=1000, temperature=0.4, log_path="logs/train_log.csv"):
    from .training import load_model
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(checkpoint)
    classes, max_len = ckpt["classes"], ckpt["config"]["max_len"]
    examples = {e["id"]: e for e in load_jsonl(data)}
    split_file = Path(ckpt["config"]["out_dir"]) / "split.json"
    split = json.loads(split_file.read_text())
    train_ex = [examples[i] for i in split["train_ids"]]
    val_ex = [examples[i] for i in split["val_ids"]]
    audit_ok = sum(is_valid(e["actions"]) and bool(render_svg(e["actions"])) for e in examples.values())

    train_vecs = np.stack([rasterize(e["actions"]) for e in train_ex])
    train_lab = np.array([classes.index(e["category"]) for e in train_ex])
    val_vecs = np.stack([rasterize(e["actions"]) for e in val_ex])
    val_lab = np.array([classes.index(e["category"]) for e in val_ex])
    ref_knn = float(np.mean(np.array(_knn_labels(val_vecs, train_vecs, train_lab)) == val_lab))
    val_nn = float(_dists(val_vecs, train_vecs).min(1).mean())

    per_class, by_class, all_res, consistent, n_valid_total = {}, [], [], 0, 0
    for ci, name in enumerate(classes):
        res = [generate(model, ci, seed + i, max_len, temperature) for i in range(n_per_class)]
        for i, r in enumerate(res):
            r["category"] = name; r["valid"] = is_valid(r["actions"])
            save_sample(out_dir / "samples", f"{name}_{i:03d}", r)
            if r["valid"]:
                saved = json.loads((out_dir / "samples" / f"{name}_{i:03d}.json").read_text())
                consistent += (out_dir / "samples" / f"{name}_{i:03d}.svg").read_text() == render_svg(saved["actions"])
        valid = [r for r in res if r["valid"]]
        n_valid_total += len(valid)
        vecs = np.stack([rasterize(r["actions"]) for r in valid]) if valid else np.zeros((0, train_vecs.shape[1]))
        stats = [stroke_stats(r["actions"]) for r in valid]
        entry = {
            "validity": len(valid) / n_per_class,
            "natural_end": sum(r["natural_end"] for r in res) / n_per_class,
            "fallback_used": sum(r["repaired"] for r in res) / n_per_class,
            "mean_actions": float(np.mean([s["action_count"] for s in stats])) if stats else 0.0,
            "mean_paths": float(np.mean([s["path_count"] for s in stats])) if stats else 0.0,
            "diversity": _pairwise_mean(vecs),
        }
        if len(vecs):
            entry["knn_recognized_as_requested"] = float(np.mean(np.array(_knn_labels(vecs, train_vecs, train_lab)) == ci))
            entry["nn_dist_to_train"] = float(_dists(vecs, train_vecs).min(1).mean())
        mask = train_lab == ci
        entry["reference_train_diversity"] = _pairwise_mean(train_vecs[mask][:n_per_class])
        per_class[name] = entry
        by_class.append((name, res)); all_res += res
    contact_sheet(by_class, out_dir / "contact_sheet.png")

    n_all = len(all_res)
    base = sum(is_valid(random_baseline(seed + i, max_len)) for i in range(n_all)) / n_all
    nan_free = None
    if Path(log_path).exists():
        rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
        nan_free = all("nan" not in r["train_loss"].lower() and "nan" not in r["val_loss"].lower() for r in rows)
    recog = [e["knn_recognized_as_requested"] for e in per_class.values() if "knn_recognized_as_requested" in e]
    m = {
        "checkpoint": str(checkpoint), "epoch": ckpt["epoch"], "classes": classes, "temperature": temperature,
        "samples_per_class": n_per_class, "first_seed": seed, "best_val_loss": ckpt["val_loss"],
        "render_validity_rate": n_valid_total / n_all,
        "completion_rate_natural_end": sum(r["natural_end"] for r in all_res) / n_all,
        "random_baseline_validity_rate": base,
        "knn_recognizability_generated": float(np.mean(recog)) if recog else 0.0,
        "knn_recognizability_reference_heldout_real": ref_knn,
        "heldout_to_train_nn_dist": val_nn, "per_class": per_class,
        "gates": {
            "0_data_renderer": f"{audit_ok}/{len(examples)} examples parse and render (run pytest for unit tests)",
            "1_training": "no NaN in log" if nan_free else ("NaN found" if nan_free is False else "log not found"),
            "2_generation": "PASS" if n_valid_total / n_all > base and n_valid_total / n_all > 0.9 else "REVIEW",
            "3_generalization": "held-out split: val loss, kNN reference and NN distance reported; unseen categories not tested",
            "4_export": f"{consistent}/{n_valid_total} saved JSON re-render to identical SVG",
            "5_governance": "manual review required before any public release",
        },
        "notes": ("kNN recognizability = nearest-neighbour vote of rasterised samples against training drawings; "
                  "a proxy for 'does it look like the requested category', not a human rating."),
    }
    (out_dir / "report.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    lines = ["# Konnon Drawing Model - evaluation", "",
             f"- checkpoint: {checkpoint} (epoch {ckpt['epoch']}), temperature {temperature}, {n_per_class} samples/class",
             f"- render validity: {m['render_validity_rate']:.3f} (random baseline {base:.3f})",
             f"- natural completion: {m['completion_rate_natural_end']:.3f}",
             f"- kNN recognizability: generated {m['knn_recognizability_generated']:.3f} vs held-out real {ref_knn:.3f}",
             "", "| category | valid | recognized | diversity (ref) | NN dist |", "|---|---|---|---|---|"]
    for k, e in per_class.items():
        lines.append(f"| {k} | {e['validity']:.2f} | {e.get('knn_recognized_as_requested', 0):.2f} | "
                     f"{e['diversity']:.2f} ({e['reference_train_diversity']:.2f}) | {e.get('nn_dist_to_train', 0):.2f} |")
    lines += ["", "## Gates"] + [f"- {k}: {v}" for k, v in m["gates"].items()]
    lines += ["", f"Note: {m['notes']}", "Contact sheet: contact_sheet.png (red = invalid, orange = length-limit fallback)."]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return m


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a checkpoint")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--data", default="data/processed/objects.jsonl")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--n", type=int, default=16, help="samples per category")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--log", default="logs/train_log.csv")
    a = p.parse_args(argv)
    evaluate(a.checkpoint, a.data, a.out, a.n, a.seed, a.temperature, a.log)
    print(Path(a.out, "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
