"""Staged PyTorch Dataset and data pipeline generator for canvas-conditioned stroke modeling."""

import json
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from ..v2.schema import DocumentV2
from ..v2.renderer import render_document_v2
from ..v2.tokens import doc_to_token_sequence, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN
from .stage_generator import (
    STAGE_ORDER,
    generate_staged_human_drawing,
    generate_staged_object_drawing,
    permute_strokes_within_stages,
)


class StagedDrawingDataset(Dataset):
    """PyTorch Dataset yielding (caption, prev_canvas_tensor, target_tokens, stage_idx)."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        canvas_res: int = 128,
        max_tokens: int = 192,
        augment_stroke_order: bool = True,
    ):
        self.samples = samples
        self.canvas_res = canvas_res
        self.max_tokens = max_tokens
        self.augment = augment_stroke_order

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        caption = item["caption"]
        stage_idx = int(item["stage_idx"])

        # Canvas tensor: shape (3, H, W) normalized [0, 1]
        if "canvas_arr" in item and item["canvas_arr"] is not None:
            canvas_arr = item["canvas_arr"]  # np.ndarray uint8 (H, W, 3)
        elif "canvas_path" in item:
            canvas_img = Image.open(item["canvas_path"]).convert("RGB")
            if canvas_img.size != (self.canvas_res, self.canvas_res):
                canvas_img = canvas_img.resize((self.canvas_res, self.canvas_res), Image.Resampling.BILINEAR)
            canvas_arr = np.array(canvas_img, dtype=np.uint8)
        else:
            canvas_arr = np.ones((self.canvas_res, self.canvas_res, 3), dtype=np.uint8) * 255

        canvas_t = torch.from_numpy(canvas_arr).permute(2, 0, 1).float() / 255.0

        # Token sequence for this stage's new strokes
        tokens = list(item["tokens"])
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens - 1] + [EOS_TOKEN]
        else:
            tokens = tokens + [PAD_TOKEN] * (self.max_tokens - len(tokens))

        tokens_t = torch.tensor(tokens, dtype=torch.long)

        return {
            "caption": caption,
            "canvas": canvas_t,
            "tokens": tokens_t,
            "stage_idx": stage_idx,
            "category": item["category"],
        }


def generate_staged_gym_data(
    out_dir: Union[str, Path] = "data/processed/staged_gym",
    manifest_path: Union[str, Path] = "data/manifests/staged_gym.manifest.json",
    num_samples: int = 400,
    canvas_res: int = 128,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generates synthetic staged dataset with pre-rendered canvas conditioning."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    man_p = Path(manifest_path)
    man_p.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    categories = ["stand", "walk", "run", "crouch", "reach", "sit", "box", "chair", "bicycle", "house"]

    all_dataset_items = []
    print(f"Generating {num_samples} staged gym drawings (canvas res: {canvas_res}x{canvas_res})...")

    # Blank canvas for Stage 0 (white canvas)
    blank_white = np.ones((canvas_res, canvas_res, 3), dtype=np.uint8) * 255

    for i in range(num_samples):
        cat = categories[i % len(categories)]
        s_seed = seed + i * 17

        if cat in ("stand", "walk", "run", "crouch", "reach", "sit"):
            yaw = float(rng.uniform(-45.0, 45.0))
            pitch = float(rng.uniform(-10.0, 15.0))
            data = generate_staged_human_drawing(pose_name=cat, yaw_deg=yaw, pitch_deg=pitch, seed=s_seed)
        else:
            data = generate_staged_object_drawing(object_type=cat, seed=s_seed)

        caption = data["caption"]
        stages = data["stages"]

        prev_canvas_arr = blank_white

        for stg_idx, stg in enumerate(stages):
            stg_name = stg["stage_name"]
            stg_cmds = stg["commands"]
            doc_so_far = stg["doc_so_far"]

            # Token sequence for this stage's new strokes
            stage_doc = DocumentV2(commands=stg_cmds)
            tokens = doc_to_token_sequence(stage_doc, max_tokens=192)

            item_record = {
                "sample_id": f"{i}_{stg_idx}",
                "drawing_id": f"drawing_{i}",
                "category": cat,
                "caption": caption,
                "stage_idx": stg_idx,
                "stage_name": stg_name,
                "canvas_arr": prev_canvas_arr,
                "tokens": tokens,
            }
            all_dataset_items.append(item_record)

            # Render canvas after this stage to serve as condition for next stage
            canvas_img = render_document_v2(doc_so_far, width=canvas_res, height=canvas_res, supersample=1, seed=s_seed)
            prev_canvas_arr = np.array(canvas_img, dtype=np.uint8)

    print(f"Total staged training transitions generated: {len(all_dataset_items)}")

    # Write Manifest
    manifest_data = {
        "dataset_name": "Konnon Draw Staged Gym",
        "version": "1.0",
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "license": "CC0 1.0 Universal (Self-Generated Synthetic)",
        "source": "src/konnon_draw/gym/stage_generator.py",
        "num_base_drawings": num_samples,
        "num_stage_transitions": len(all_dataset_items),
        "canvas_resolution": [canvas_res, canvas_res],
        "categories": categories,
        "stages": STAGE_ORDER,
        "transforms": [
            "Loomis and 3D articulated kinematic projection",
            "Pinhole camera orbit with random yaw/pitch",
            "Progressive rasterization of prior canvas",
            "Tokenization via discrete v2 vocabulary (V=604)",
        ],
    }

    with open(man_p, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Saved manifest: {man_p}")
    return all_dataset_items


def split_staged_dataset_by_drawing(
    items: List[Dict[str, Any]],
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits staged transitions strictly by drawing ID so all stages of a drawing stay in the same split."""
    drawing_to_items: Dict[str, List[Dict[str, Any]]] = {}
    for it in items:
        did = it.get("drawing_id", it["sample_id"].split("_")[0])
        if did not in drawing_to_items:
            drawing_to_items[did] = []
        drawing_to_items[did].append(it)

    unique_drawings = sorted(list(drawing_to_items.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_drawings)

    n_total = len(unique_drawings)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    if n_val < 1 and val_frac > 0 and n_total > 1:
        n_val = 1
    if n_train + n_val >= n_total and n_total > 2:
        n_train = n_total - n_val - 1

    train_ids = set(unique_drawings[:n_train])
    val_ids = set(unique_drawings[n_train : n_train + n_val])
    test_ids = set(unique_drawings[n_train + n_val :])

    train_items = [it for did in train_ids for it in drawing_to_items[did]]
    val_items = [it for did in val_ids for it in drawing_to_items[did]]
    test_items = [it for did in test_ids for it in drawing_to_items[did]]

    return train_items, val_items, test_items
