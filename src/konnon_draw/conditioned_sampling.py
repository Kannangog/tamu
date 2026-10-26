"""Conditioned Autoregressive Generation & Multi-Category Gallery Generator."""

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .conditioned_encoding import (
    REAL_COORD_BINS,
    REAL_COORD_VOCAB,
    REAL_NO_COORD,
    decode_conditioned_one,
)
from .conditioned_model import ConditionedDrawingGRU
from .quickdraw import CATEGORY_TO_ID, DEFAULT_CATEGORIES, ID_TO_CATEGORY
from .renderer import render_svg
from .schema import Action, dequantize_coord, quantize_coord, validate_actions


def sample_conditioned_drawing(
    model: ConditionedDrawingGRU,
    category: str,
    temperature: float = 0.7,
    max_len: int = 96,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[List[dict], bool, Optional[str]]:
    """Autoregressively generate a drawing conditioned on a target category name."""
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    cat_id = CATEGORY_TO_ID.get(category, 0)
    cat_id_t = torch.tensor([cat_id], dtype=torch.long, device=device)

    # Initial start point near canvas center
    sx = 0.35 + 0.3 * torch.rand(1).item()
    sy = 0.35 + 0.3 * torch.rand(1).item()
    sx_q = quantize_coord(sx, bins=REAL_COORD_BINS)
    sy_q = quantize_coord(sy, bins=REAL_COORD_BINS)

    cur_ids = [int(Action.MOVE)]
    cur_xs = [sx_q]
    cur_ys = [sy_q]

    actions = [
        {
            "type": "MOVE",
            "x": dequantize_coord(sx_q, bins=REAL_COORD_BINS),
            "y": dequantize_coord(sy_q, bins=REAL_COORD_BINS),
            "xq": sx_q,
            "yq": sy_q,
            "coord_bins": REAL_COORD_BINS,
        }
    ]

    in_path = True
    hidden = None

    with torch.no_grad():
        for step in range(1, max_len):
            in_ids_t = torch.tensor([cur_ids], dtype=torch.long, device=device)
            in_xs_t = torch.tensor([cur_xs], dtype=torch.long, device=device)
            in_ys_t = torch.tensor([cur_ys], dtype=torch.long, device=device)

            act_l, x_l, y_l, _ = model(in_ids_t, in_xs_t, in_ys_t, cat_id_t)

            next_act_logits = act_l[0, -1, :].clone()
            # Rule: Exclude PAD
            next_act_logits[Action.PAD] = -float("inf")

            # Avoid premature termination in early strokes
            if step < 4:
                next_act_logits[Action.END] = -float("inf")

            if temperature > 0:
                act_probs = F.softmax(next_act_logits / temperature, dim=-1)
                next_act = torch.multinomial(act_probs, 1).item()
            else:
                next_act = torch.argmax(next_act_logits).item()

            # Repair: LINE without active path becomes MOVE
            if next_act == Action.LINE and not in_path:
                next_act = Action.MOVE

            if next_act == Action.END:
                actions.append({"type": "END"})
                cur_ids.append(int(Action.END))
                cur_xs.append(REAL_NO_COORD)
                cur_ys.append(REAL_NO_COORD)
                break

            elif next_act == Action.PEN_UP:
                if not in_path:
                    continue
                actions.append({"type": "PEN_UP"})
                cur_ids.append(int(Action.PEN_UP))
                cur_xs.append(REAL_NO_COORD)
                cur_ys.append(REAL_NO_COORD)
                in_path = False

            elif next_act in (Action.MOVE, Action.LINE):
                next_x_l = x_l[0, -1, :REAL_COORD_BINS].clone()
                next_y_l = y_l[0, -1, :REAL_COORD_BINS].clone()

                if temperature > 0:
                    x_probs = F.softmax(next_x_l / temperature, dim=-1)
                    y_probs = F.softmax(next_y_l / temperature, dim=-1)
                    xq = torch.multinomial(x_probs, 1).item()
                    yq = torch.multinomial(y_probs, 1).item()
                else:
                    xq = torch.argmax(next_x_l).item()
                    yq = torch.argmax(next_y_l).item()

                kind = "MOVE" if next_act == Action.MOVE else "LINE"
                actions.append({
                    "type": kind,
                    "x": dequantize_coord(xq, bins=REAL_COORD_BINS),
                    "y": dequantize_coord(yq, bins=REAL_COORD_BINS),
                    "xq": xq,
                    "yq": yq,
                    "coord_bins": REAL_COORD_BINS,
                })
                cur_ids.append(next_act)
                cur_xs.append(xq)
                cur_ys.append(yq)
                in_path = True

    if actions[-1]["type"] != "END":
        actions.append({"type": "END"})

    try:
        validate_actions(actions)
        return actions, True, None
    except Exception as e:
        return actions, False, str(e)


def generate_multicategory_gallery(
    checkpoint_path: str = "outputs/checkpoints/real_best_model.pt",
    output_dir: str = "outputs/real_samples",
    samples_per_category: int = 16,
    temperature: float = 0.7,
    base_seed: int = 500,
) -> Dict[str, any]:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = ConditionedDrawingGRU(
        embedding_dim=cfg.get("embedding_dim", 32),
        category_dim=cfg.get("category_dim", 32),
        hidden_dim=cfg.get("hidden_dim", 160),
        coord_vocab=cfg.get("coord_vocab", REAL_COORD_VOCAB),
        num_categories=len(DEFAULT_CATEGORIES),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    gallery_data = {c: [] for c in DEFAULT_CATEGORIES}
    total_samples = 0
    valid_samples = 0

    print(f"Generating {samples_per_category} samples per category across {DEFAULT_CATEGORIES}...")
    for c_idx, cat in enumerate(DEFAULT_CATEGORIES):
        cat_dir = out_path / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        for i in range(samples_per_category):
            seed = base_seed + c_idx * 1000 + i
            actions, is_valid, err = sample_conditioned_drawing(
                model=model,
                category=cat,
                temperature=temperature,
                seed=seed,
            )

            svg_content = None
            if is_valid:
                try:
                    svg_content = render_svg(actions)
                    valid_samples += 1
                except Exception as e:
                    is_valid = False
                    err = f"Render error: {e}"

            total_samples += 1
            sample_id = f"{cat}_{i:02d}"
            json_file = cat_dir / f"{sample_id}.json"
            svg_file = cat_dir / f"{sample_id}.svg"

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump({
                    "id": sample_id,
                    "category": cat,
                    "seed": seed,
                    "is_valid": is_valid,
                    "action_count": len(actions),
                    "actions": actions,
                }, f, indent=2)

            if svg_content:
                svg_file.write_text(svg_content, encoding="utf-8")

            # Calculate relative path for HTML
            rel_svg = f"{cat}/{sample_id}.svg" if svg_content else None
            gallery_data[cat].append({
                "id": sample_id,
                "category": cat,
                "seed": seed,
                "is_valid": is_valid,
                "action_count": len(actions),
                "rel_svg": rel_svg,
            })

    # Build interactive HTML gallery
    html_file = out_path / "gallery.html"
    build_html_gallery(gallery_data, html_file)
    print(f"Multi-category generation complete. {valid_samples}/{total_samples} valid. Gallery: {html_file}")
    return {
        "total": total_samples,
        "valid": valid_samples,
        "valid_rate": valid_samples / total_samples,
        "gallery_html": str(html_file),
    }


def build_html_gallery(gallery_data: Dict[str, List[dict]], output_file: Path):
    sections = []
    category_icons = {
        "apple": "🍎",
        "cat": "🐱",
        "fish": "🐟",
        "star": "⭐",
        "tree": "🌲",
    }

    for cat, items in gallery_data.items():
        icon = category_icons.get(cat, "🎨")
        cards = []
        for it in items:
            status_color = "#10b981" if it["is_valid"] else "#ef4444"
            status_text = "VALID" if it["is_valid"] else "INVALID"

            if it["rel_svg"]:
                img_html = f'<object data="{it["rel_svg"]}" type="image/svg+xml" class="svg-obj"></object>'
            else:
                img_html = '<div class="placeholder">Render Failed</div>'

            card = f"""
            <div class="card">
                <div class="card-header">
                    <span><strong>{it["id"]}</strong></span>
                    <span style="color: {status_color}; font-size: 11px; font-weight: bold;">{status_text}</span>
                </div>
                <div class="card-body">
                    {img_html}
                </div>
                <div class="card-footer">
                    <span>Actions: {it["action_count"]}</span>
                    <span>Seed: {it["seed"]}</span>
                </div>
            </div>
            """
            cards.append(card)

        cards_html = "\n".join(cards)
        section = f"""
        <div class="category-section" id="cat-{cat}">
            <h2>{icon} Category: {cat.capitalize()} <span class="badge">{len(items)} samples</span></h2>
            <div class="grid">
                {cards_html}
            </div>
        </div>
        """
        sections.append(section)

    body_sections = "\n".join(sections)
    nav_links = " ".join([
        f'<a href="#cat-{c}" class="nav-btn">{category_icons.get(c, "")} {c.capitalize()}</a>'
        for c in gallery_data.keys()
    ])

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Konnon Drawing Model — Real Stroke Object Gallery</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f8fafc;
            color: #0f172a;
            margin: 0;
            padding: 32px 48px;
        }}
        header {{
            margin-bottom: 24px;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 16px;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 28px;
            letter-spacing: -0.5px;
        }}
        p.subtitle {{
            margin: 0;
            color: #64748b;
            font-size: 15px;
        }}
        .nav-bar {{
            display: flex;
            gap: 12px;
            margin: 20px 0;
            position: sticky;
            top: 10px;
            background: rgba(248, 250, 252, 0.95);
            backdrop-filter: blur(8px);
            padding: 10px 0;
            z-index: 10;
        }}
        .nav-btn {{
            text-decoration: none;
            background: white;
            border: 1px solid #cbd5e1;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            color: #334155;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            transition: all 0.15s ease;
        }}
        .nav-btn:hover {{
            background: #1e293b;
            color: white;
            border-color: #1e293b;
        }}
        .category-section {{
            margin-bottom: 40px;
        }}
        h2 {{
            font-size: 20px;
            color: #1e293b;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .badge {{
            font-size: 12px;
            font-weight: normal;
            background: #e2e8f0;
            color: #475569;
            padding: 2px 8px;
            border-radius: 12px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.04);
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 8px 12px;
            background: #f8fafc;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #f1f5f9;
        }}
        .card-body {{
            width: 100%;
            aspect-ratio: 1;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .svg-obj {{
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        .placeholder {{
            color: #94a3b8;
            font-size: 12px;
        }}
        .card-footer {{
            padding: 6px 12px;
            font-size: 11px;
            color: #64748b;
            background: #fafafa;
            border-top: 1px solid #f1f5f9;
            display: flex;
            justify-content: space-between;
        }}
    </style>
</head>
<body>
    <header>
        <h1>Konnon Drawing Model — Real Object Milestone Gallery</h1>
        <p class="subtitle">Autonomous stroke generation from Google Quick, Draw! real-human doodles across 5 distinct categories.</p>
    </header>

    <div class="nav-bar">
        {nav_links}
    </div>

    {body_sections}
</body>
</html>
"""
    output_file.write_text(html_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate multi-category real object gallery.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/real_best_model.pt")
    parser.add_argument("--output-dir", default="outputs/real_samples")
    parser.add_argument("--samples-per-cat", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=500)
    args = parser.parse_args()

    generate_multicategory_gallery(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        samples_per_category=args.samples_per_cat,
        temperature=args.temperature,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()
