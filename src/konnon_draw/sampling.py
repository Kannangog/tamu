import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .encoding import NO_COORD, dequantize_coord, quantize_coord
from .model import ACTION_VOCAB, COORD_VOCAB, DrawingGRU
from .renderer import render_svg
from .schema import Action, validate_actions


def generate_sequence(
    model: DrawingGRU,
    max_len: int = 64,
    temperature: float = 0.8,
    start_coord: Optional[Tuple[float, float]] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tuple[List[dict], bool, Optional[str]]:
    """Generate a single drawing sequence autoregressively.

    Returns:
        (actions, is_valid, error_message)
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()

    # Step 1: Start context - initial MOVE
    if start_coord is not None:
        sx, sy = start_coord
    else:
        # Sample starting point near center region
        sx = 0.3 + 0.4 * torch.rand(1).item()
        sy = 0.3 + 0.4 * torch.rand(1).item()

    sx_q = quantize_coord(sx)
    sy_q = quantize_coord(sy)

    cur_ids = [int(Action.MOVE)]
    cur_xs = [sx_q]
    cur_ys = [sy_q]

    actions = [
        {
            "type": "MOVE",
            "x": dequantize_coord(sx_q),
            "y": dequantize_coord(sy_q),
            "xq": sx_q,
            "yq": sy_q,
        }
    ]

    in_path = True
    ended_naturally = False

    with torch.no_grad():
        for step in range(1, max_len):
            in_ids_t = torch.tensor([cur_ids], dtype=torch.long, device=device)
            in_xs_t = torch.tensor([cur_xs], dtype=torch.long, device=device)
            in_ys_t = torch.tensor([cur_ys], dtype=torch.long, device=device)

            act_logits, x_logits, y_logits, _ = model(in_ids_t, in_xs_t, in_ys_t)

            next_act_logits = act_logits[0, -1, :].clone()
            # Rule: Exclude PAD from generation
            next_act_logits[Action.PAD] = -float("inf")

            # Avoid immediate termination on first step
            if step < 2:
                next_act_logits[Action.END] = -float("inf")

            if temperature > 0:
                act_probs = F.softmax(next_act_logits / temperature, dim=-1)
                next_act = torch.multinomial(act_probs, 1).item()
            else:
                next_act = torch.argmax(next_act_logits).item()

            # Rule: Constrained decoding / repair illegal transitions
            if next_act == Action.LINE and not in_path:
                next_act = Action.MOVE

            if next_act == Action.END:
                actions.append({"type": "END"})
                cur_ids.append(int(Action.END))
                cur_xs.append(NO_COORD)
                cur_ys.append(NO_COORD)
                ended_naturally = True
                break

            elif next_act == Action.PEN_UP:
                if not in_path:
                    # Ignore consecutive PEN_UP
                    continue
                actions.append({"type": "PEN_UP"})
                cur_ids.append(int(Action.PEN_UP))
                cur_xs.append(NO_COORD)
                cur_ys.append(NO_COORD)
                in_path = False

            elif next_act in (Action.MOVE, Action.LINE):
                next_x_logits = x_logits[0, -1, :32].clone()  # Bins 0..31
                next_y_logits = y_logits[0, -1, :32].clone()

                if temperature > 0:
                    x_probs = F.softmax(next_x_logits / temperature, dim=-1)
                    y_probs = F.softmax(next_y_logits / temperature, dim=-1)
                    xq = torch.multinomial(x_probs, 1).item()
                    yq = torch.multinomial(y_probs, 1).item()
                else:
                    xq = torch.argmax(next_x_logits).item()
                    yq = torch.argmax(next_y_logits).item()

                kind = "MOVE" if next_act == Action.MOVE else "LINE"
                actions.append({
                    "type": kind,
                    "x": dequantize_coord(xq),
                    "y": dequantize_coord(yq),
                    "xq": xq,
                    "yq": yq,
                })
                cur_ids.append(next_act)
                cur_xs.append(xq)
                cur_ys.append(yq)
                in_path = True

    # Rule: Enforce maximum action count and fallback END
    if actions[-1]["type"] != "END":
        actions.append({"type": "END"})

    # Validate sequence
    try:
        validate_actions(actions)
        return actions, True, None
    except Exception as e:
        return actions, False, str(e)


def generate_samples_batch(
    checkpoint_path: str = "outputs/checkpoints/best_model.pt",
    output_dir: str = "outputs/samples",
    num_samples: int = 64,
    base_seed: int = 100,
    temperature: float = 0.8,
    max_len: int = 64,
) -> dict:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint and model
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {"hidden_dim": 128, "embedding_dim": 32})
    model = DrawingGRU(
        embedding_dim=config.get("embedding_dim", 32),
        hidden_dim=config.get("hidden_dim", 128),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_results = []
    valid_count = 0
    ended_count = 0

    print(f"Generating {num_samples} samples into {output_dir}...")
    for i in range(num_samples):
        seed = base_seed + i
        actions, is_valid, err = generate_sequence(
            model=model,
            max_len=max_len,
            temperature=temperature,
            seed=seed,
        )

        svg_content = None
        if is_valid:
            try:
                svg_content = render_svg(actions)
                valid_count += 1
            except Exception as e:
                is_valid = False
                err = f"Render error: {e}"

        ended = actions[-1]["type"] == "END" and len(actions) < max_len
        if ended:
            ended_count += 1

        sample_name = f"sample_{i:03d}"
        json_file = out_path / f"{sample_name}.json"
        svg_file = out_path / f"{sample_name}.svg"

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump({
                "id": sample_name,
                "seed": seed,
                "is_valid": is_valid,
                "error": err,
                "action_count": len(actions),
                "actions": actions,
            }, f, indent=2)

        if svg_content:
            svg_file.write_text(svg_content, encoding="utf-8")

        sample_results.append({
            "id": sample_name,
            "seed": seed,
            "is_valid": is_valid,
            "error": err,
            "action_count": len(actions),
            "svg_file": str(svg_file.name) if svg_content else None,
        })

    # Generate Contact Sheet HTML
    contact_sheet_path = out_path / "contact_sheet.html"
    create_contact_sheet(sample_results, contact_sheet_path)

    stats = {
        "total": num_samples,
        "valid": valid_count,
        "valid_rate": valid_count / num_samples,
        "completion_rate": ended_count / num_samples,
        "contact_sheet": str(contact_sheet_path),
    }
    print(f"Sampling done: Valid {valid_count}/{num_samples} ({stats['valid_rate']:.1%}), Contact sheet: {contact_sheet_path}")
    return stats


def create_contact_sheet(results: List[dict], output_file: Path):
    cards = []
    for r in results:
        status_color = "#10b981" if r["is_valid"] else "#ef4444"
        status_text = "VALID" if r["is_valid"] else f"INVALID: {html.escape(str(r['error']))}"

        if r["svg_file"]:
            media_html = f'<object data="{r["svg_file"]}" type="image/svg+xml" class="svg-preview"></object>'
        else:
            media_html = '<div class="svg-placeholder">No SVG Rendered</div>'

        card = f"""
        <div class="card">
            <div class="card-header">
                <strong>{r["id"]}</strong> (seed: {r["seed"]})
                <span style="color: {status_color}; font-weight: bold; font-size: 11px;">{status_text}</span>
            </div>
            <div class="card-body">
                {media_html}
            </div>
            <div class="card-footer">
                Actions: {r["action_count"]}
            </div>
        </div>
        """
        cards.append(card)

    cards_html = "\n".join(cards)
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Konnon Drawing Model - Generated Samples Contact Sheet</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: #f8fafc;
            color: #1e293b;
            padding: 24px;
            margin: 0;
        }}
        h1 {{
            font-size: 24px;
            margin-bottom: 8px;
        }}
        .summary {{
            color: #64748b;
            margin-bottom: 24px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 8px 12px;
            background: #f1f5f9;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .card-body {{
            width: 100%;
            aspect-ratio: 1;
            background: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid #e2e8f0;
        }}
        .svg-preview {{
            width: 100%;
            height: 100%;
            pointer-events: none;
        }}
        .svg-placeholder {{
            color: #94a3b8;
            font-size: 13px;
        }}
        .card-footer {{
            padding: 6px 12px;
            font-size: 11px;
            color: #64748b;
        }}
    </style>
</head>
<body>
    <h1>Konnon Drawing Model v0.1 — Sample Inspection</h1>
    <div class="summary">Generated {len(results)} samples with deterministic seeds.</div>
    <div class="grid">
        {cards_html}
    </div>
</body>
</html>
"""
    output_file.write_text(html_content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate sample drawings with DrawingGRU.")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--output-dir", default="outputs/samples", help="Directory for output files")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=100, help="Base random seed")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--max-len", type=int, default=64, help="Max sequence length")
    args = parser.parse_args()

    generate_samples_batch(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        base_seed=args.seed,
        temperature=args.temperature,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
