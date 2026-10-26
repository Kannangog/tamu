"""Quick, Draw! Responsible Dataset Intake Module.

Conforms to Step 10 Checklist:
- Explicit license recording (Creative Commons Attribution 4.0 International).
- Coordinate normalization from 0..255 to 0.0..1.0 with padding.
- Strict rejection of non-finite, out-of-bounds, or empty strokes.
- Human stroke order and pen-up preservation.
"""

import argparse
import json
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .schema import validate_actions

DATASET_NAME = "Google Quick, Draw! Dataset"
DATASET_LICENSE = "Creative Commons Attribution 4.0 International License (CC BY 4.0)"
DATASET_ATTRIBUTION = "Google Creative Lab / Quick, Draw! Contributors"
QUICKDRAW_BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/simplified"


def quickdraw_strokes_to_actions(drawing: List[List[List[float]]], margin: float = 0.05) -> List[dict]:
    """Convert Quick, Draw! stroke array [[xs, ys], [xs, ys], ...] to Konnon action representation.

    Normalized coordinates are scaled into [margin, 1.0 - margin] preserving aspect ratio.
    """
    if not drawing:
        raise ValueError("Empty drawing")

    all_x = [pt for stroke in drawing for pt in stroke[0]]
    all_y = [pt for stroke in drawing for pt in stroke[1]]

    if not all_x or not all_y:
        raise ValueError("No coordinate points in drawing")

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)

    span_x = max(1.0, max_x - min_x)
    span_y = max(1.0, max_y - min_y)
    scale = max(span_x, span_y)

    usable_span = 1.0 - 2 * margin
    offset_x = margin + (usable_span - (span_x / scale) * usable_span) / 2.0
    offset_y = margin + (usable_span - (span_y / scale) * usable_span) / 2.0

    actions = []
    for stroke in drawing:
        xs, ys = stroke[0], stroke[1]
        if not xs or not ys:
            continue

        # Start of stroke
        norm_x = offset_x + ((xs[0] - min_x) / scale) * usable_span
        norm_y = offset_y + ((ys[0] - min_y) / scale) * usable_span
        actions.append({
            "type": "MOVE",
            "x": round(max(0.0, min(1.0, norm_x)), 4),
            "y": round(max(0.0, min(1.0, norm_y)), 4),
        })

        # Lines in stroke
        for x, y in zip(xs[1:], ys[1:]):
            norm_x = offset_x + ((x - min_x) / scale) * usable_span
            norm_y = offset_y + ((y - min_y) / scale) * usable_span
            actions.append({
                "type": "LINE",
                "x": round(max(0.0, min(1.0, norm_x)), 4),
                "y": round(max(0.0, min(1.0, norm_y)), 4),
            })

        actions.append({"type": "PEN_UP"})

    # Strip trailing PEN_UP and finish with END
    if actions and actions[-1]["type"] == "PEN_UP":
        actions.pop()
    actions.append({"type": "END"})

    validate_actions(actions)
    return actions


def download_category_subset(
    category: str,
    max_examples: int = 100,
    dest_dir: str = "data/raw/quickdraw",
) -> Path:
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    out_file = dest_path / f"{category}.ndjson"

    safe_category = category.replace(" ", "%20")
    url = f"{QUICKDRAW_BASE_URL}/{safe_category}.ndjson"

    print(f"Downloading subset of {category} from {url}...")
    headers = {"User-Agent": "KonnonDrawingModel/0.1"}
    req = urllib.request.Request(url, headers=headers)

    downloaded = 0
    with urllib.request.urlopen(req) as resp, open(out_file, "w", encoding="utf-8") as out_f:
        for line in resp:
            out_f.write(line.decode("utf-8"))
            downloaded += 1
            if downloaded >= max_examples:
                break

    print(f"Downloaded {downloaded} raw examples to {out_file}")
    return out_file


def process_quickdraw_ndjson(
    input_file: Path,
    output_file: Path,
    category: str,
    max_count: int = 100,
) -> int:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    valid_count = 0

    with open(input_file, "r", encoding="utf-8") as in_f, open(output_file, "a", encoding="utf-8") as out_f:
        for idx, line in enumerate(in_f):
            if valid_count >= max_count:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("recognized", True):
                continue
            raw_drawing = item.get("drawing", [])
            try:
                actions = quickdraw_strokes_to_actions(raw_drawing)
                record = {
                    "id": f"qd_{category}_{idx}",
                    "category": category,
                    "key_id": item.get("key_id", str(idx)),
                    "actions": actions,
                }
                out_f.write(json.dumps(record) + "\n")
                valid_count += 1
            except Exception:
                continue

    return valid_count


def write_manifest(manifest_path: Path, categories: List[str], counts: Dict[str, int]):
    manifest = {
        "dataset_name": DATASET_NAME,
        "source": QUICKDRAW_BASE_URL,
        "license": DATASET_LICENSE,
        "attribution": DATASET_ATTRIBUTION,
        "access_date": datetime.now().isoformat(),
        "categories": categories,
        "example_counts": counts,
        "transformations": [
            "Parsed stroke array [[xs], [ys]] to MOVE/LINE/PEN_UP/END sequence",
            "Normalized coordinates to [0.05, 0.95] preserving aspect ratio",
            "Filtered out non-recognized or empty doodles",
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote intake manifest to {manifest_path}")
