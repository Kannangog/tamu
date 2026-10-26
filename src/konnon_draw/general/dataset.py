"""Curriculum dataset, streaming dataloaders, and augmentations for Konnon Draw General.

Curriculum Design:
- Stage 1 (Primitives & Rules): Synthetic shapes, colors, counts, and spatial relations.
  * Why: Teaches the transformer tokenizer grammar, coordinate spatial bounds, and color/layer tokens with 100% verifiable ground truth.
- Stage 2 (Object Semantic Concepts): Quick, Draw! human doodles with category captions.
  * Why: Exposes the model to diverse human drawing styles, multi-stroke structures, and real-world semantic shapes.
- Stage 3 (Compositional Mixed): Joint mixture of spatial compositions, shaded forms, and real categories.
  * Why: Prevents catastrophic forgetting of spatial rules while grounding complex scene layouts.

Augmentations:
- Stroke order permutation (draw order invariance).
- Horizontal flip with text token coordinate reflection (left <-> right).
- Affine coordinate jitter (scale in [0.95, 1.05], shift +/- 0.02).
"""

import json
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..encoding_v2 import quickdraw_to_v2
from ..schema_v2 import DocumentV2, PointV2, StrokeActionV2
from .synthetic_data import generate_synthetic_dataset, generate_synthetic_example
from .text_encoder import BuiltinSemanticTokenizer
from .vocab import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    TOKEN_LINE,
    TOKEN_MOVE,
    TOKEN_PEN_UP,
    VOCAB_SIZE,
    X_OFFSET,
    Y_OFFSET,
    doc_to_token_sequence,
    token_sequence_to_doc,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# ---------------------------------------------------------
# Augmentations
# ---------------------------------------------------------

def augment_drawing(
    doc: DocumentV2,
    prompt: str,
    permute_strokes: bool = True,
    hflip: bool = True,
    jitter: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[DocumentV2, str]:
    """Applies stroke-order permutation, horizontal flip, and affine jitter."""
    if rng is None:
        rng = random.Random()

    strokes = [c for c in doc.commands if getattr(c, "type", "") == "stroke"]
    fills = [c for c in doc.commands if getattr(c, "type", "") == "fill"]

    # 1. Stroke order permutation (within strokes)
    if permute_strokes and len(strokes) > 1 and rng.random() < 0.5:
        rng.shuffle(strokes)

    # 2. Horizontal Flip
    do_flip = hflip and (rng.random() < 0.5)
    new_prompt = prompt
    if do_flip:
        if "left of" in prompt:
            new_prompt = prompt.replace("left of", "right of")
        elif "right of" in prompt:
            new_prompt = prompt.replace("right of", "left of")

    # 3. Affine Jitter parameters
    scale = rng.uniform(0.96, 1.04) if jitter else 1.0
    dx = rng.uniform(-0.02, 0.02) if jitter else 0.0
    dy = rng.uniform(-0.02, 0.02) if jitter else 0.0

    def transform_pt(x: float, y: float) -> Tuple[float, float]:
        nx = (1.0 - x) if do_flip else x
        nx = 0.5 + (nx - 0.5) * scale + dx
        ny = 0.5 + (y - 0.5) * scale + dy
        return max(0.02, min(0.98, nx)), max(0.02, min(0.98, ny))

    aug_commands = []

    for f in fills:
        new_path = [transform_pt(px, py) for px, py in f.path]
        f.path = new_path
        aug_commands.append(f)

    for st in strokes:
        new_pts = []
        for p in st.points:
            nx, ny = transform_pt(p.x, p.y)
            new_pts.append(PointV2(x=nx, y=ny, p=p.p, tilt=p.tilt))
        st.points = new_pts
        aug_commands.append(st)

    new_doc = DocumentV2(canvas=doc.canvas, layers=doc.layers, commands=aug_commands)
    return new_doc, new_prompt


# ---------------------------------------------------------
# Manifest Creation
# ---------------------------------------------------------

def ensure_general_manifest(manifest_path: Optional[Path] = None):
    """Writes metadata manifest for all datasets used in Konnon Draw General."""
    if manifest_path is None:
        manifest_path = PROJECT_ROOT / "data" / "manifests" / "konnon_general.manifest.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_data = {
        "project": "konnon_draw_general",
        "description": "Multi-stage dataset corpus for text-to-vector drawing",
        "access_date": "2026-10-26",
        "sources": [
            {
                "name": "Google Quick, Draw!",
                "license": "CC BY 4.0",
                "attribution": "Google Creative Lab / Quick, Draw! Dataset",
                "url": "https://github.com/googlecreativelab/quickdraw-dataset",
                "role": "Stage 2 category pretraining and human stroke dynamics",
            },
            {
                "name": "Artist Hand Synthetic Primitives",
                "license": "MIT (Project Native)",
                "attribution": "Konnon Draw Project",
                "role": "Stage 1 verified shape/color/spatial composition curriculum",
            },
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)


# ---------------------------------------------------------
# Streaming / Map-Style PyTorch Dataset
# ---------------------------------------------------------

class GeneralDrawingDataset(Dataset):
    """Dataset providing (text prompt, drawing action token sequence) pairs."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Optional[BuiltinSemanticTokenizer] = None,
        max_text_len: int = 32,
        max_drawing_len: int = 256,
        augment: bool = True,
    ):
        self.samples = samples
        self.tokenizer = tokenizer or BuiltinSemanticTokenizer()
        self.max_text_len = max_text_len
        self.max_drawing_len = max_drawing_len
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        prompt = item["prompt"]
        doc = item["doc"]

        if self.augment:
            doc, prompt = augment_drawing(doc, prompt)

        # 1. Text tokenization
        text_enc = self.tokenizer.encode(prompt, max_length=self.max_text_len)

        # 2. Drawing tokenization
        drawing_tokens = doc_to_token_sequence(doc, max_tokens=self.max_drawing_len)

        # Truncate / pad drawing tokens
        if len(drawing_tokens) > self.max_drawing_len:
            drawing_tokens = drawing_tokens[:self.max_drawing_len - 1] + [EOS_TOKEN]

        seq_len = len(drawing_tokens)
        pad_len = self.max_drawing_len - seq_len

        # Input is tokens[:-1], Target is tokens[1:]
        tokens_tensor = torch.tensor(drawing_tokens, dtype=torch.long)
        input_ids = torch.cat([tokens_tensor[:-1], torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)])
        target_ids = torch.cat([tokens_tensor[1:], torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)])

        drawing_mask = torch.cat([torch.ones(seq_len - 1, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])

        return {
            "text_input_ids": text_enc["input_ids"],
            "text_attention_mask": text_enc["attention_mask"],
            "drawing_input_ids": input_ids,
            "drawing_target_ids": target_ids,
            "drawing_mask": drawing_mask,
            "prompt": prompt,
        }


# ---------------------------------------------------------
# Curriculum Dataset Builders
# ---------------------------------------------------------

def load_quickdraw_category_samples(
    jsonl_path: Path,
    max_per_class: int = 200,
    categories: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Loads Quick, Draw! items and formats them as (prompt, DocumentV2) pairs."""
    items = []
    if not jsonl_path.exists():
        return items

    class_counts: Dict[str, int] = {}
    templates = [
        "a drawing of a {cat}",
        "a simple sketch of a {cat}",
        "a doodle of a {cat}",
        "a {cat}",
    ]

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            cat = d.get("category", "drawing")
            if categories and cat not in categories:
                continue

            count = class_counts.get(cat, 0)
            if count >= max_per_class:
                continue

            actions = d.get("actions", [])
            # Convert legacy v1 actions to v2 document
            from ..schema_v2 import v1_to_v2
            doc = v1_to_v2(actions, default_brush="ink_liner", base_width=3.0)

            prompt = random.choice(templates).format(cat=cat)
            items.append({"prompt": prompt, "doc": doc, "category": cat})
            class_counts[cat] = count + 1

    return items


def create_curriculum_datasets(
    stage: int = 1,
    num_synthetic: int = 1200,
    max_quickdraw_per_class: int = 150,
    qd_path: Optional[Path] = None,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[GeneralDrawingDataset, GeneralDrawingDataset]:
    """Builds train and validation datasets based on the curriculum stage."""
    ensure_general_manifest()
    random.seed(seed)
    all_samples = []

    # Stage 1: Synthetic shapes and spatial compositions
    if stage in (1, 3):
        synth_samples = generate_synthetic_dataset(num_samples=num_synthetic, start_seed=seed * 100)
        all_samples.extend(synth_samples)

    # Stage 2: Quick, Draw! categories
    if stage in (2, 3):
        if qd_path is None:
            qd_path = PROJECT_ROOT / "data" / "processed" / "quickdraw.jsonl"
        qd_samples = load_quickdraw_category_samples(
            qd_path,
            max_per_class=max_quickdraw_per_class,
            categories=["bicycle", "car", "chair", "table", "house", "bed"],
        )
        all_samples.extend(qd_samples)

    random.shuffle(all_samples)
    val_count = max(20, int(len(all_samples) * val_split))
    train_samples = all_samples[val_count:]
    val_samples = all_samples[:val_count]

    train_ds = GeneralDrawingDataset(train_samples, augment=True)
    val_ds = GeneralDrawingDataset(val_samples, augment=False)

    return train_ds, val_ds
