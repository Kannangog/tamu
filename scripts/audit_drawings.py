import json
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from konnon_draw.renderer import render_svg
from konnon_draw.schema import validate_actions

out_dir = Path("outputs/audit_qd_48")
out_dir.mkdir(parents=True, exist_ok=True)

by_cat = defaultdict(list)
with open("data/processed/quickdraw.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        by_cat[rec["category"]].append(rec)

random.seed(42)
sampled = []
for cat in sorted(by_cat.keys()):
    items = random.sample(by_cat[cat], 8)
    sampled.extend(items)

print(f"Auditing {len(sampled)} drawings across {len(by_cat)} categories...")
errors = []
stroke_counts = []
for i, item in enumerate(sampled):
    cat = item["category"]
    actions = item["actions"]
    try:
        validate_actions(actions)
        svg = render_svg(actions)
        root = ET.fromstring(svg)
        paths = root.findall("{http://www.w3.org/2000/svg}path")
        if len(paths) == 0:
            errors.append(f"{item['id']}: 0 path elements rendered")
        stroke_counts.append(len(paths))
        out_file = out_dir / f"{i:02d}_{cat}_{item['id']}.svg"
        out_file.write_text(svg, encoding="utf-8")
    except Exception as e:
        errors.append(f"{item['id']}: {e}")

print(f"Rendered {len(sampled)} drawings into {out_dir}. Errors: {len(errors)}")
if errors:
    for err in errors[:5]:
        print("  Error:", err)
else:
    avg_strokes = sum(stroke_counts) / len(stroke_counts)
    print(f"All 48 audited Quick, Draw! converted drawings are structurally valid and rendered cleanly.")
    print(f"Average stroke count: {avg_strokes:.1f} (min: {min(stroke_counts)}, max: {max(stroke_counts)})")
