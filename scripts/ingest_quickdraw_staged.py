"""Ingests real-world Google Quick, Draw! vector sketches and converts them into
multi-stage (Layout -> Guide -> Ink -> Shade) training data with visual conditioning.

Features:
- Streams public Google Cloud Storage simplified .ndjson files (no local disk waste)
- Filters for high quality (recognized == True, stroke count between 4 and 60)
- Chronological stroke decomposition into 4 natural artist drafting stages
- Renders raster canvases (128x128) for cross-stage visual conditioning
- Generates rich natural language prompts with varied descriptive templates
- Strict split by drawing ID: 90% train, 5% val, 5% test (0% data leakage)
- Emits manifest to data/manifests/quickdraw_staged_50k.manifest.json
"""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from konnon_draw.v2.schema import (
    CanvasConfigV2,
    DocumentV2,
    StrokeActionV2,
    PointV2,
)
from konnon_draw.artist_skills import create_standard_layers
from konnon_draw.v2.renderer import render_document_v2
from konnon_draw.v2.tokens import (
    doc_to_token_sequence,
    EOS_TOKEN,
    PAD_TOKEN,
)


GCS_BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"

DEFAULT_CATEGORIES = [
    # Animals
    "cat", "dog", "bird", "fish", "horse", "elephant", "rabbit", "butterfly", "lion", "bear",
    # Vehicles & Transport
    "airplane", "bicycle", "car", "bus", "sailboat", "train", "truck", "motorbike",
    # Architecture & Furniture & Objects
    "house", "chair", "table", "bed", "clock", "guitar", "camera", "cup", "tree", "flower",
    # Anatomy & Faces
    "face", "eye",
]

PROMPT_TEMPLATES = [
    "a drawing of a {category}",
    "a clean sketch of a {category}",
    "a line art drawing of a {category}",
    "an illustration of a {category}",
    "a detailed vector sketch of a {category}",
    "a hand-drawn sketch of a {category}",
    "a fine art drawing of a {category}",
]

STAGE_NAMES = ["layout", "guide", "ink", "shade"]


def normalize_strokes(strokes: List[List[List[int]]]) -> Optional[List[List[Tuple[float, float]]]]:
    """Centers and scales 0-255 QuickDraw coordinates to normalized [0.05, 0.95] float coordinates."""
    pts_all = [(x, y) for s in strokes for x, y in zip(s[0], s[1])]
    if not pts_all:
        return None
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y
    scale = max(w, h)
    if scale <= 0:
        return None

    cx = (max_x + min_x) / 2.0
    cy = (max_y + min_y) / 2.0

    norm_strokes = []
    for s in strokes:
        pts = []
        last_pt = None
        for x, y in zip(s[0], s[1]):
            nx = float(round(0.5 + (x - cx) / scale * 0.88, 4))
            ny = float(round(0.5 + (y - cy) / scale * 0.88, 4))
            # Clamp to [0.02, 0.98]
            nx = max(0.02, min(0.98, nx))
            ny = max(0.02, min(0.98, ny))
            pt = (nx, ny)
            if pt != last_pt:
                pts.append(pt)
                last_pt = pt
        if len(pts) >= 2:
            norm_strokes.append(pts)

    if len(norm_strokes) < 2:
        return None
    return norm_strokes


def download_quickdraw_category(
    category: str,
    max_samples: int = 1500,
    raw_dir: Path = Path("data/raw/quickdraw"),
) -> Optional[Path]:
    """Streams Quick, Draw! .ndjson directly from Google Cloud Storage."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest_path = raw_dir / f"{category}.ndjson"
    if dest_path.exists() and dest_path.stat().st_size > 1024:
        return dest_path

    encoded_cat = category.replace(" ", "%20")
    url = GCS_BASE.format(encoded_cat)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (KonnonDraw/0.2)"})

    print(f"  Downloading [{category}] from GCS ({max_samples} samples)...")
    valid_count = 0
    try:
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "w", encoding="utf-8") as out_f:
            for line in resp:
                try:
                    item = json.loads(line.decode("utf-8"))
                    # Filter for recognized quality sketches with valid stroke count
                    if item.get("recognized", False) and 4 <= len(item.get("drawing", [])) <= 60:
                        out_f.write(json.dumps(item) + "\n")
                        valid_count += 1
                        if valid_count >= max_samples:
                            break
                except Exception:
                    continue
    except Exception as e:
        print(f"  Warning: failed to fetch [{category}] ({e}). Skipping.")
        if dest_path.exists():
            dest_path.unlink()
        return None

    return dest_path


def convert_category_to_staged(
    ndjson_path: Path,
    category: str,
    canvases_dir: Path,
    blank_canvas_path: str,
    max_drawings: int = 1000,
    max_seq_len: int = 512,
    canvas_res: int = 128,
    start_drawing_idx: int = 0,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], int]:
    """Converts normalized QuickDraw strokes into 4-stage transitions with visual canvases."""
    transitions = []
    rng = random.Random(seed)
    drawing_count = 0

    with open(ndjson_path, "r", encoding="utf-8") as f:
        for line in f:
            if drawing_count >= max_drawings:
                break
            try:
                item = json.loads(line)
            except Exception:
                continue

            raw_strokes = item.get("drawing", [])
            strokes = normalize_strokes(raw_strokes)
            if not strokes:
                continue

            drawing_idx = start_drawing_idx + drawing_count
            drawing_id = f"qd_{category}_{drawing_idx:05d}"
            prompt = rng.choice(PROMPT_TEMPLATES).format(category=category.replace("_", " "))

            n_strokes = len(strokes)
            # Stage boundaries: 25%, 55%, 85%, 100%
            b1 = max(1, int(round(n_strokes * 0.25)))
            b2 = max(b1 + 1, int(round(n_strokes * 0.55)))
            b3 = max(b2 + 1, int(round(n_strokes * 0.85)))
            boundaries = [b1, b2, b3, n_strokes]

            cum_doc = DocumentV2(layers=create_standard_layers())
            prev_canvas_path = blank_canvas_path
            prev_stroke_idx = 0

            for stg_idx, (stg_name, end_idx) in enumerate(zip(STAGE_NAMES, boundaries)):
                stage_strokes = strokes[prev_stroke_idx:end_idx]
                if not stage_strokes:
                    continue

                brush = "pencil" if stg_name == "layout" else ("ink_liner" if stg_name in ("guide", "ink") else "charcoal")
                layer_id = "underdrawing" if stg_name == "layout" else ("line_art" if stg_name in ("guide", "ink") else "shading")
                color = "#94a3b8" if stg_name == "layout" else ("#475569" if stg_name == "guide" else "#111827")
                width = 1.5 if stg_name == "layout" else (2.5 if stg_name == "guide" else 3.2)
                opacity = 0.5 if stg_name == "layout" else (0.8 if stg_name == "guide" else 1.0)

                stage_doc = DocumentV2(layers=create_standard_layers())
                for s in stage_strokes:
                    pts = [PointV2(x=x, y=y, p=0.8) for x, y in s]
                    cmd = StrokeActionV2(
                        points=pts,
                        brush_id=brush,
                        color=color,
                        base_width=width,
                        opacity=opacity,
                        layer_id=layer_id,
                        stage=stg_name,
                    )
                    stage_doc.commands.append(cmd)
                    cum_doc.commands.append(cmd)

                # Tokenize stage commands
                tokens = doc_to_token_sequence(stage_doc, max_tokens=max_seq_len)
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len - 1] + [EOS_TOKEN]

                # Render current canvas for next stage conditioning
                sample_id = f"{drawing_id}_s{stg_idx}_{stg_name}"
                next_canvas_path = str(canvases_dir / f"{sample_id}.png")
                rendered_img = render_document_v2(cum_doc, width=canvas_res, height=canvas_res, supersample=1)
                rendered_img.save(next_canvas_path)

                transitions.append({
                    "sample_id": sample_id,
                    "drawing_id": drawing_id,
                    "category": category,
                    "prompt": prompt,
                    "caption": prompt,
                    "stage_name": stg_name,
                    "stage_idx": stg_idx,
                    "canvas_path": prev_canvas_path,
                    "tokens": tokens,
                    "num_tokens": len(tokens),
                })

                prev_canvas_path = next_canvas_path
                prev_stroke_idx = end_idx

            drawing_count += 1

    return transitions, drawing_count


def ingest_all_quickdraw_staged(
    categories: Optional[List[str]] = None,
    drawings_per_cat: int = 1000,
    out_dir: str = "data/processed/quickdraw_staged_50k",
    manifest_path: str = "data/manifests/quickdraw_staged_50k.manifest.json",
    canvas_res: int = 128,
    max_seq_len: int = 512,
    seed: int = 42,
) -> Dict[str, Any]:
    t0 = time.time()
    if categories is None:
        categories = DEFAULT_CATEGORIES

    out_p = Path(out_dir)
    canvases_dir = out_p / "canvases"
    canvases_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path("data/raw/quickdraw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Blank canvas
    blank_white = Image.new("RGB", (canvas_res, canvas_res), color=(255, 255, 255))
    blank_canvas_path = str(canvases_dir / "blank_128.png")
    blank_white.save(blank_canvas_path)

    all_transitions: List[Dict[str, Any]] = []
    total_drawings = 0

    print("=" * 70)
    print(f"INGESTING QUICK, DRAW! STAGED DATASET ({len(categories)} categories, {drawings_per_cat}/cat)")
    print(f"Output directory: {out_p}")
    print(f"Canvas resolution: {canvas_res}x{canvas_res} | Max Seq Len: {max_seq_len}")
    print("=" * 70)

    for cat_idx, cat in enumerate(categories):
        ndjson_p = download_quickdraw_category(cat, max_samples=drawings_per_cat + 200, raw_dir=raw_dir)
        if not ndjson_p or not ndjson_p.exists() or ndjson_p.stat().st_size == 0:
            print(f"  [{cat_idx+1:02d}/{len(categories):02d}] {cat:<12} -> SKIPPED")
            continue
        cat_transitions, n_drawings = convert_category_to_staged(
            ndjson_path=ndjson_p,
            category=cat,
            canvases_dir=canvases_dir,
            blank_canvas_path=blank_canvas_path,
            max_drawings=drawings_per_cat,
            max_seq_len=max_seq_len,
            canvas_res=canvas_res,
            start_drawing_idx=total_drawings,
            seed=seed + cat_idx * 17,
        )
        all_transitions.extend(cat_transitions)
        total_drawings += n_drawings
        print(f"  [{cat_idx+1:02d}/{len(categories):02d}] {cat:<12} -> {n_drawings} drawings ({len(cat_transitions)} transitions)")

    # Split strictly by drawing_id (90/5/5)
    drawing_ids = sorted(list(set(t["drawing_id"] for t in all_transitions)))
    rng = random.Random(seed)
    rng.shuffle(drawing_ids)

    n_total_drawings = len(drawing_ids)
    n_train = int(round(n_total_drawings * 0.90))
    n_val = int(round(n_total_drawings * 0.05))

    train_ids = set(drawing_ids[:n_train])
    val_ids = set(drawing_ids[n_train:n_train + n_val])
    test_ids = set(drawing_ids[n_train + n_val:])

    train_data = [t for t in all_transitions if t["drawing_id"] in train_ids]
    val_data = [t for t in all_transitions if t["drawing_id"] in val_ids]
    test_data = [t for t in all_transitions if t["drawing_id"] in test_ids]

    def save_jsonl(records: List[Dict[str, Any]], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    save_jsonl(train_data, out_p / "train.jsonl")
    save_jsonl(val_data, out_p / "val.jsonl")
    save_jsonl(test_data, out_p / "test.jsonl")

    total_time = round(time.time() - t0, 2)
    manifest = {
        "dataset_name": "QuickDraw Staged Scaled Dataset",
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "license": "CC BY 4.0 (Google Creative Lab)",
        "num_categories": len(categories),
        "categories": categories,
        "total_drawings": total_drawings,
        "total_transitions": len(all_transitions),
        "splits": {
            "train": {"drawings": len(train_ids), "transitions": len(train_data)},
            "val": {"drawings": len(val_ids), "transitions": len(val_data)},
            "test": {"drawings": len(test_ids), "transitions": len(test_data)},
        },
        "canvas_resolution": [canvas_res, canvas_res],
        "max_seq_len": max_seq_len,
        "elapsed_seconds": total_time,
    }

    man_p = Path(manifest_path)
    man_p.parent.mkdir(parents=True, exist_ok=True)
    with open(man_p, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    print("\n" + "=" * 70)
    print("SUCCESS: QUICK, DRAW! STAGED INGESTION COMPLETE!")
    print(f"Total Base Drawings: {total_drawings}")
    print(f"Total Transitions:   {len(all_transitions)}")
    print(f"  Train Transitions: {len(train_data)}")
    print(f"  Val Transitions:   {len(val_data)}")
    print(f"  Test Transitions:  {len(test_data)}")
    print(f"Manifest written to: {man_p}")
    print(f"Total Wall Clock Time: {total_time}s")
    print("=" * 70 + "\n")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--drawings_per_cat", type=int, default=1000)
    parser.add_argument("--out_dir", default="data/processed/quickdraw_staged_50k")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--canvas_res", type=int, default=128)
    args = parser.parse_args()

    ingest_all_quickdraw_staged(
        categories=args.categories,
        drawings_per_cat=args.drawings_per_cat,
        out_dir=args.out_dir,
        max_seq_len=args.max_seq_len,
        canvas_res=args.canvas_res,
    )
