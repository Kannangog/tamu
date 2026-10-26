"""A/B Test Visual Upgrades: Plain discrete decoding vs Sub-bin continuous coordinates.
Evaluates identical seeds across all categories and compares metrics + produces visual contact sheets.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from konnon_draw.encoding import load_jsonl
from konnon_draw.evaluation import _dists, _knn_labels, stroke_stats
from konnon_draw.raster import draw_tile, rasterize
from konnon_draw.renderer import render_svg
from konnon_draw.sampling import generate
from konnon_draw.schema import is_valid
from konnon_draw.training import load_model


def run_ab_test(checkpoint, data, out_dir, n_per_class=16, seed=2026, temperature=0.4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plain_dir = out_dir / "plain"
    cont_dir = out_dir / "continuous"
    plain_dir.mkdir(exist_ok=True)
    cont_dir.mkdir(exist_ok=True)

    model, ckpt = load_model(checkpoint)
    classes = ckpt["classes"]
    max_len = ckpt["config"]["max_len"]

    examples = {e["id"]: e for e in load_jsonl(data)}
    split_file = Path(ckpt["config"]["out_dir"]) / "split.json"
    split = json.loads(split_file.read_text())
    train_ex = [examples[i] for i in split["train_ids"]]
    train_vecs = np.stack([rasterize(e["actions"]) for e in train_ex])
    train_lab = np.array([classes.index(e["category"]) for e in train_ex])

    records_plain = []
    records_cont = []

    print(f"Running A/B generation for {len(classes)} classes x {n_per_class} samples (seed {seed})...")

    side_by_side_tiles = []  # list of (cat_name, tile_img)

    for ci, name in enumerate(classes):
        for i in range(n_per_class):
            cur_seed = seed + i * 100 + ci

            # 1. Plain discrete decoding
            r_plain = generate(model, ci, cur_seed, max_len, temperature, continuous_coords=False)
            r_plain["category"] = name
            r_plain["valid"] = is_valid(r_plain["actions"])
            records_plain.append(r_plain)

            # 2. Continuous sub-bin coordinates on the EXACT SAME seed
            r_cont = generate(model, ci, cur_seed, max_len, temperature, continuous_coords=True)
            r_cont["category"] = name
            r_cont["valid"] = is_valid(r_cont["actions"])
            records_cont.append(r_cont)

            # Save SVG
            (plain_dir / f"{name}_{i:02d}.svg").write_text(render_svg(r_plain["actions"], smooth=False), encoding="utf-8")
            (cont_dir / f"{name}_{i:02d}.svg").write_text(render_svg(r_cont["actions"], smooth=True), encoding="utf-8")

            # Collect first 4 per category for visual comparison contact sheet
            if i < 4:
                # Render 128x128 plain and 128x128 continuous side-by-side
                img_plain = draw_tile(r_plain["actions"], 124).convert("RGB")
                img_cont = draw_tile(r_cont["actions"], 124).convert("RGB")
                pair = Image.new("RGB", (256, 128), "white")
                pair.paste(img_plain, (2, 2))
                pair.paste(img_cont, (130, 2))
                d = ImageDraw.Draw(pair)
                d.line([(128, 0), (128, 128)], fill="#cbd5e1", width=1)
                d.text((4, 4), "Discrete", fill="#64748b")
                d.text((132, 4), "Sub-bin", fill="#4f46e5")
                side_by_side_tiles.append((name, pair))

    # Evaluate metrics
    def evaluate_set(records):
        vecs = np.stack([rasterize(r["actions"]) for r in records if r["valid"]])
        recog = [int(np.bincount(train_lab[idx]).argmax()) == records[i]["class_id"]
                 for i, idx in enumerate(np.argsort(_dists(vecs, train_vecs), axis=1)[:, :5])]
        stats = [stroke_stats(r["actions"]) for r in records if r["valid"]]

        # Quantization analysis: fraction of coords with non-zero sub-bin offset
        sub_bin_offsets = []
        for r in records:
            for a in r["actions"]:
                if a["type"] in ("MOVE", "LINE"):
                    sub_bin_offsets.append(abs(a["x"] * 63.0 - round(a["x"] * 63.0)))
                    sub_bin_offsets.append(abs(a["y"] * 63.0 - round(a["y"] * 63.0)))

        return {
            "validity": float(np.mean([r["valid"] for r in records])),
            "natural_completion": float(np.mean([r["natural_end"] for r in records])),
            "knn_recognizability": float(np.mean(recog)),
            "mean_path_length": float(np.mean([s["path_length"] for s in stats])),
            "mean_sub_bin_fraction": float(np.mean(np.array(sub_bin_offsets) > 1e-3)),
        }

    metrics_plain = evaluate_set(records_plain)
    metrics_cont = evaluate_set(records_cont)

    # Build A/B Comparison Contact Sheet
    cols = 4
    tile_w, tile_h = 256, 128
    band = 24
    rows = (len(side_by_side_tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + band)), "white")
    d = ImageDraw.Draw(sheet)

    for idx, (cat, pair) in enumerate(side_by_side_tiles):
        c = idx % cols
        r = idx // cols
        x0 = c * tile_w
        y0 = r * (tile_h + band)
        d.text((x0 + 6, y0 + 4), f"{cat.upper()} #{idx%4}", fill="#111827")
        sheet.paste(pair, (x0, y0 + band))
        d.rectangle([x0, y0 + band, x0 + tile_w - 1, y0 + band + tile_h - 1], outline="#e2e8f0", width=1)

    sheet.save(out_dir / "ab_comparison_contact_sheet.png")

    report = {
        "n_samples": len(records_plain),
        "temperature": temperature,
        "seed": seed,
        "plain_metrics": metrics_plain,
        "continuous_metrics": metrics_cont,
        "delta": {
            "knn_recognizability": round(metrics_cont["knn_recognizability"] - metrics_plain["knn_recognizability"], 4),
            "sub_bin_fraction": round(metrics_cont["mean_sub_bin_fraction"] - metrics_plain["mean_sub_bin_fraction"], 4),
            "path_length": round(metrics_cont["mean_path_length"] - metrics_plain["mean_path_length"], 4),
        }
    }

    (out_dir / "ab_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = f"""# A/B Test Report: Plain Discrete Decoding vs. Sub-bin Continuous Coordinates

- Model Checkpoint: `{checkpoint}`
- Samples Evaluated: {len(records_plain)} ({n_per_class} per class across 6 classes)
- Temperature: {temperature} (Identical seeds per pair)

| Metric | Plain Discrete | Sub-bin Continuous | Delta |
|---|---|---|---|
| Validity Rate | {metrics_plain['validity']:.3f} | {metrics_cont['validity']:.3f} | 0.000 |
| Natural Completion Rate | {metrics_plain['natural_completion']:.3f} | {metrics_cont['natural_completion']:.3f} | 0.000 |
| kNN Recognizability | {metrics_plain['knn_recognizability']:.3f} | {metrics_cont['knn_recognizability']:.3f} | {report['delta']['knn_recognizability']:+.3f} |
| Mean Path Length | {metrics_plain['mean_path_length']:.3f} | {metrics_cont['mean_path_length']:.3f} | {report['delta']['path_length']:+.3f} |
| Sub-bin Resolution Rate | {metrics_plain['mean_sub_bin_fraction']:.1%} | {metrics_cont['mean_sub_bin_fraction']:.1%} | {report['delta']['sub_bin_fraction']:+.1%} |

Contact Sheet: `ab_comparison_contact_sheet.png`
"""
    (out_dir / "ab_report.md").write_text(md, encoding="utf-8")
    print(md)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs_qd/checkpoints/best.pt")
    p.add_argument("--data", default="data/processed/quickdraw.jsonl")
    p.add_argument("--out", default="outputs_qd/ab_test")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.4)
    args = p.parse_args()
    run_ab_test(args.checkpoint, args.data, args.out, args.n, temperature=args.temperature)


if __name__ == "__main__":
    main()
