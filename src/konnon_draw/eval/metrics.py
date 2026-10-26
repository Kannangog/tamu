"""Evaluation metrics, benchmark harness, and contact sheet generation for staged drawing model."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..v2.schema import DocumentV2
from ..v2.renderer import render_document_v2
from ..model.stroke_transformer import KonnonStagedTransformer
from ..model.text_encoder import FrozenTextEncoder
from ..model.sampling import generate_staged_drawing
from ..v2.tokens import VOCAB_SIZE


def create_staged_contact_sheet(
    eval_records: List[Dict[str, Any]],
    out_path: Path,
    tile_res: int = 256,
) -> Image.Image:
    """Arranges staged progression (stage 1..N) side-by-side for each prompt."""
    num_prompts = len(eval_records)
    # Each row shows: [Stage 1, Stage 2, Stage 3, Stage 4, Final Render]
    cols = 5
    rows = num_prompts

    sheet_w = cols * tile_res
    sheet_h = rows * (tile_res + 36)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (241, 245, 249))
    draw = ImageDraw.Draw(sheet)

    for r_idx, rec in enumerate(eval_records):
        y_top = r_idx * (tile_res + 36)
        prompt = rec["prompt"]
        stage_imgs = rec.get("stage_images", [])

        # Display up to 4 stages + 1 final
        display_imgs = stage_imgs[:4]
        if len(display_imgs) < 4:
            display_imgs += [display_imgs[-1]] * (4 - len(display_imgs))
        display_imgs.append(rec["final_image"])

        for c_idx, img in enumerate(display_imgs):
            x_left = c_idx * tile_res
            resized = img.resize((tile_res, tile_res), Image.Resampling.LANCZOS)
            sheet.paste(resized, (x_left, y_top))

        # Bottom label bar
        draw.rectangle([0, y_top + tile_res, sheet_w, y_top + tile_res + 36], fill=(15, 23, 42))
        lbl = f"[{rec['group'].upper()}] \"{prompt}\" | Tokens: {rec['total_tokens']}"
        draw.text((12, y_top + tile_res + 10), lbl[:110], fill=(248, 250, 252))

    sheet.save(out_path)
    return sheet


def evaluate_staged_model(
    checkpoint_path: Union[str, Path],
    prompts_path: Union[str, Path] = "src/konnon_draw/eval/prompts_eval.json",
    out_dir: str = "outputs_v2/eval",
    device: str = "cpu",
    temperature: float = 0.5,
    top_k: int = 25,
) -> Dict[str, Any]:
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    m_cfg = cfg.get("model", {})

    text_encoder = FrozenTextEncoder(
        model_type=cfg.get("text_encoder", {}).get("model_type", "builtin"),
        d_model=m_cfg.get("d_model", 256),
        device=device,
    )

    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=m_cfg.get("d_model", 256),
        n_layers=m_cfg.get("n_layers", 4),
        n_heads=m_cfg.get("n_heads", 8),
        d_ff=m_cfg.get("d_ff", 768),
        max_seq_len=m_cfg.get("max_seq_len", 160),
        canvas_res=cfg.get("data", {}).get("canvas_res", 128),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    results = []
    print(f"Evaluating {len(prompts_list)} staged prompts...")

    for item in prompts_list:
        prompt = item["prompt"]
        group = item["group"]

        gen_res = generate_staged_drawing(
            model=model,
            text_encoder=text_encoder,
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            device=device,
            seed=42,
        )

        stage_imgs = [h["canvas_image"] for h in gen_res["stage_history"]]
        total_toks = sum(len(h["tokens"]) for h in gen_res["stage_history"])

        rec = {
            "prompt": prompt,
            "group": group,
            "total_tokens": total_toks,
            "num_stages": len(gen_res["stage_history"]),
            "final_image": gen_res["final_image"],
            "stage_images": stage_imgs,
        }
        results.append(rec)
        print(f"[{group.upper():8s}] \"{prompt[:35]}...\" -> {len(gen_res['stage_history'])} stages, {total_toks} tokens")

    # Composite contact sheet
    sheet_path = out_p / "staged_evaluation_sheet.png"
    create_staged_contact_sheet(results, sheet_path)
    print(f"\nSaved staged evaluation sheet: {sheet_path}")

    # Report JSON (omit image objects)
    json_results = [
        {"prompt": r["prompt"], "group": r["group"], "total_tokens": r["total_tokens"], "num_stages": r["num_stages"]}
        for r in results
    ]
    report = {
        "checkpoint": str(checkpoint_path),
        "prompts_evaluated": len(results),
        "results": json_results,
    }
    with open(out_p / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="outputs_v2/laptop/checkpoints/best.pt")
    parser.add_argument("--out-dir", type=str, default="outputs_v2/eval")
    args = parser.parse_args()
    evaluate_staged_model(args.checkpoint, out_dir=args.out_dir)
