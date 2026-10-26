"""Evaluation suite for Konnon Draw General text-to-vector drawing model.

Evaluates:
1. Rule-based prompt-following (counts, colors, spatial positions, shapes).
2. Critic visual alignment scoring.
3. Quick, Draw! category recognizability.
4. Out-of-vocabulary failure mode analysis.
5. Contact sheet generation.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from ..renderer_v2 import render_document_v2
from ..schema_v2 import DocumentV2
from .critic import ImageTextCritic
from .sampling_general import generate_best_of_n, generate_from_prompt
from .text_encoder import FrozenTextEncoder
from .transformer_decoder import KonnonTransformerDecoder
from .vocab import VOCAB_SIZE


BENCHMARK_PROMPTS = [
    # 1. Shape Primitives & Colors
    {"prompt": "a red circle", "group": "primitives"},
    {"prompt": "a blue square", "group": "primitives"},
    {"prompt": "a green triangle", "group": "primitives"},
    {"prompt": "a yellow star", "group": "primitives"},
    # 2. Counts
    {"prompt": "two red circles", "group": "counts"},
    {"prompt": "three blue squares", "group": "counts"},
    {"prompt": "two green stars", "group": "counts"},
    # 3. Spatial Compositions
    {"prompt": "a red circle above a blue square", "group": "spatial"},
    {"prompt": "a yellow star left of a green triangle", "group": "spatial"},
    {"prompt": "a blue square below a red circle", "group": "spatial"},
    # 4. Quick, Draw! Categories
    {"prompt": "a drawing of a bicycle", "group": "categories"},
    {"prompt": "a simple sketch of a car", "group": "categories"},
    {"prompt": "a drawing of a chair", "group": "categories"},
    {"prompt": "a drawing of a house", "group": "categories"},
    {"prompt": "a drawing of a table", "group": "categories"},
    {"prompt": "a drawing of a bed", "group": "categories"},
    # 5. Out-of-Vocabulary (Unseen)
    {"prompt": "an astronaut floating in space", "group": "oov"},
    {"prompt": "a cybernetic dragon breathing fire", "group": "oov"},
]


def create_contact_sheet(
    images_with_labels: List[Tuple[Image.Image, str]],
    cols: int = 4,
    tile_size: int = 256,
    bg_color: Tuple[int, int, int] = (245, 245, 245),
) -> Image.Image:
    """Arranges labelled images into a clean contact sheet."""
    n = len(images_with_labels)
    rows = int(math.ceil(n / float(cols)))

    sheet_w = cols * tile_size
    sheet_h = rows * (tile_size + 36)
    sheet = Image.new("RGB", (sheet_w, sheet_h), bg_color)
    draw = ImageDraw.Draw(sheet)

    for idx, (img, label) in enumerate(images_with_labels):
        r = idx // cols
        c = idx % cols
        x = c * tile_size
        y = r * (tile_size + 36)

        resized = img.resize((tile_size, tile_size))
        sheet.paste(resized, (x, y))

        # Label banner
        draw.rectangle([x, y + tile_size, x + tile_size, y + tile_size + 36], fill=(15, 23, 42))
        truncated_label = label if len(label) < 28 else label[:25] + "..."
        draw.text((x + 8, y + tile_size + 10), truncated_label, fill=(248, 250, 252))

    return sheet


def evaluate_general_benchmark(
    checkpoint_path: Union[str, Path],
    out_dir: str = "outputs_general/eval",
    device: str = "cpu",
    temperature: float = 0.5,
    top_k: int = 25,
) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EVALUATING KONNON DRAW GENERAL BENCHMARK")
    print(f"Checkpoint: {checkpoint_path}")
    print("=" * 60)

    # Load Checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    text_cfg = cfg.get("text_encoder", {})

    text_encoder = FrozenTextEncoder(
        model_type=text_cfg.get("model_type", "builtin"),
        d_model=text_cfg.get("d_model", 256),
        device=device,
    )

    model = KonnonTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 8),
        d_ff=model_cfg.get("d_ff", 1024),
        max_seq_len=model_cfg.get("max_seq_len", 256),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    critic = ImageTextCritic(text_encoder=text_encoder)

    results = []
    contact_sheet_items = []
    group_scores: Dict[str, List[float]] = {}

    for idx, item in enumerate(BENCHMARK_PROMPTS):
        prompt = item["prompt"]
        group = item["group"]

        # Sample best of 3 candidates
        res = generate_best_of_n(
            model=model,
            text_encoder=text_encoder,
            prompt=prompt,
            n_candidates=3,
            temperature=temperature,
            top_k=top_k,
            device=device,
            critic=critic,
        )

        doc = res["doc"]
        score = res["best_critic_score"]
        rule_score = critic.evaluate_rule_fidelity(doc, prompt)

        img = render_document_v2(doc, width=256, height=256, supersample=1, seed=42)
        contact_sheet_items.append((img, f"[{group}] {prompt}"))

        if group not in group_scores:
            group_scores[group] = []
        group_scores[group].append(score)

        record = {
            "prompt": prompt,
            "group": group,
            "critic_score": score,
            "rule_score": rule_score,
            "n_tokens": res["n_tokens"],
        }
        results.append(record)
        print(f"[{group.upper():10s}] \"{prompt}\" -> Critic: {score:.3f} | Rule: {rule_score:.3f}")

    # Generate Composite Contact Sheet
    cs_img = create_contact_sheet(contact_sheet_items, cols=4, tile_size=256)
    cs_path = out_path / "benchmark_contact_sheet.png"
    cs_img.save(cs_path)
    print(f"\nSaved contact sheet: {cs_path}")

    # Calculate Group Means
    summary = {}
    for grp, scores in group_scores.items():
        summary[grp] = {
            "mean_critic_score": round(float(np.mean(scores)), 4),
            "num_prompts": len(scores),
        }

    overall_mean = float(np.mean([r["critic_score"] for r in results]))
    summary["overall_mean_critic_score"] = round(overall_mean, 4)

    eval_report = {
        "checkpoint": str(checkpoint_path),
        "summary": summary,
        "results": results,
    }

    report_path = out_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2)

    print(f"Saved evaluation report: {report_path}")
    print(f"Overall Benchmark Mean Critic Score: {overall_mean:.4f}")
    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Konnon Draw General benchmark")
    parser.add_argument("--checkpoint", type=str, default="outputs_general/small/checkpoints/best.pt")
    parser.add_argument("--out-dir", type=str, default="outputs_general/eval")
    args = parser.parse_args()

    evaluate_general_benchmark(args.checkpoint, out_dir=args.out_dir)
