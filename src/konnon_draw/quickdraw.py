"""Quick, Draw! (Google, CC BY 4.0 - verify current terms) -> Konnon action JSONL.

Run on a machine with internet access:
    python -m konnon_draw.quickdraw --categories bicycle bed chair --per-class 3000
then train with:  python -m konnon_draw.training --data data/processed/quickdraw.jsonl
Only the first N lines of each category file are streamed, not the whole file.
"""
import argparse
import json
import urllib.request
from datetime import date
from pathlib import Path

from .schema import COORD_BINS, dequantize_coord, quantize_coord, validate_actions

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
LICENSE = "CC BY 4.0 (Google Creative Lab) - attribution required; confirm at github.com/googlecreativelab/quickdraw-dataset"


def download(category, raw_dir, n_lines):
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{category}.ndjson"
    if dest.exists():
        return dest
    req = urllib.request.Request(BASE.format(category.replace(" ", "%20")), headers={"User-Agent": "konnon-draw/0.2"})
    with urllib.request.urlopen(req, timeout=60) as r, dest.open("wb") as out:
        for i, line in enumerate(r):
            out.write(line)
            if i + 1 >= n_lines:
                break
    return dest


def convert_drawing(strokes, max_len=160):
    """strokes: [[xs],[ys]] per stroke, simplified 0-255 coords. Returns action list or None to reject."""
    pts_all = [(x, y) for s in strokes for x, y in zip(s[0], s[1])]
    if not pts_all:
        return None
    xs, ys = [p[0] for p in pts_all], [p[1] for p in pts_all]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    scale = max(w, h)
    if scale <= 0:
        return None
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    actions = []
    for s in strokes:
        pts, last = [], None
        for x, y in zip(s[0], s[1]):  # centre, fit to 0.04..0.96, drop repeats within one bin
            nx = 0.5 + (x - cx) / scale * 0.92
            ny = 0.5 + (y - cy) / scale * 0.92
            q = (quantize_coord(nx), quantize_coord(ny))
            if q != last:
                pts.append((dequantize_coord(q[0]), dequantize_coord(q[1])))
                last = q
        if len(pts) < 2:  # single dots cannot be expressed as MOVE+LINE
            continue
        actions.append({"type": "MOVE", "x": pts[0][0], "y": pts[0][1]})
        actions += [{"type": "LINE", "x": x, "y": y} for x, y in pts[1:]]
        actions.append({"type": "PEN_UP"})
    if not actions:
        return None
    actions.pop()  # trailing PEN_UP
    actions.append({"type": "END"})
    if len(actions) + 1 > max_len:  # +1 for START
        return None
    validate_actions(actions)
    return actions


def build_dataset(categories, per_class, max_len, raw_dir, out_path, fetch=True):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    counts, seen = {}, {}
    with out_path.open("w", encoding="utf-8") as f:
        for cat in categories:
            path = download(cat, raw_dir, per_class * 3) if fetch else Path(raw_dir) / f"{cat}.ndjson"
            kept = seen_c = 0
            with path.open(encoding="utf-8") as src:
                for line in src:
                    if kept >= per_class:
                        break
                    rec = json.loads(line); seen_c += 1
                    if not rec.get("recognized", True):
                        continue
                    acts = convert_drawing(rec["drawing"], max_len)
                    if acts is None:
                        continue
                    f.write(json.dumps({"id": f"{cat}_{rec.get('key_id', kept)}", "category": cat, "actions": acts}) + "\n")
                    kept += 1
            counts[cat], seen[cat] = kept, seen_c
    manifest = {"source": BASE, "license": LICENSE, "access_date": date.today().isoformat(),
                "categories": categories, "kept": counts, "lines_read": seen,
                "transforms": ["recognized==True only", f"bbox centred, longest side -> 0.92 of canvas, {COORD_BINS} bins",
                               "consecutive points in the same bin merged", "single-point strokes dropped",
                               f"drawings longer than {max_len} tokens rejected"],
                "split_note": "training.py splits by example; Quick, Draw! has no artist id, so splits may share style."}
    mp = Path("data/manifests") / (out_path.stem + ".manifest.json")
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return counts, mp


def main(argv=None):
    p = argparse.ArgumentParser(description="Download + convert Quick, Draw! categories")
    p.add_argument("--categories", nargs="+", default=["bicycle", "bed", "chair"])
    p.add_argument("--per-class", type=int, default=3000)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--raw-dir", default="data/raw/quickdraw")
    p.add_argument("--out", default="data/processed/quickdraw.jsonl")
    p.add_argument("--no-fetch", action="store_true", help="use files already in --raw-dir")
    a = p.parse_args(argv)
    counts, mp = build_dataset(a.categories, a.per_class, a.max_len, a.raw_dir, a.out, fetch=not a.no_fetch)
    print("Kept:", counts, "\nManifest:", mp)


if __name__ == "__main__":
    main()
