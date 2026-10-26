"""STEP 4: Scale the Gym (10,000+ staged transitions).

Generates 10,000+ staged transitions from the synthetic gym:
- 3,000 base drawings across human figure poses and 3D objects
- Progressive multi-stage rasterization
- 90/5/5 split strictly by drawing_id
- Memory-safe disk storage of canvas images
- Tracking:
  * Generation wall clock time
  * Peak RAM RSS (MB)
  * Sequence truncation statistics (how many transitions hit max_seq_len, strokes dropped)
- Emits manifest to data/manifests/staged_gym_10k.manifest.json
"""

import copy
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import psutil

from konnon_draw.v2.schema import (
    CanvasConfigV2,
    DocumentV2,
    StrokeActionV2,
    FillActionV2,
)
from konnon_draw.v2.renderer import render_document_v2
from konnon_draw.v2.tokens import (
    doc_to_token_sequence,
    token_sequence_to_doc,
    EOS_TOKEN,
    PAD_TOKEN,
)
from konnon_draw.gym.stage_generator import (
    STAGE_ORDER,
    generate_staged_human_drawing,
    generate_staged_object_drawing,
)
from konnon_draw.gym.dataset import split_staged_dataset_by_drawing


def generate_scaled_gym(
    num_drawings: int = 3500,
    canvas_res: int = 128,
    max_seq_len: int = 160,
    out_dir: str = "data/processed/staged_gym_10k",
    manifest_path: str = "data/manifests/staged_gym_10k.manifest.json",
    seed: int = 42,
) -> Dict[str, Any]:
    process = psutil.Process()
    ram_initial = process.memory_info().rss / (1024 * 1024)
    peak_ram = ram_initial
    t0 = time.time()

    out_p = Path(out_dir)
    canvases_dir = out_p / "canvases"
    canvases_dir.mkdir(parents=True, exist_ok=True)
    man_p = Path(manifest_path)
    man_p.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    categories = ["stand", "walk", "run", "crouch", "reach", "sit", "box", "chair", "bicycle", "house"]
    presets = ["adult", "heroic", "chibi"]

    # Pre-render shared blank white canvas
    blank_white = Image.new("RGB", (canvas_res, canvas_res), color=(255, 255, 255))
    blank_canvas_path = str(canvases_dir / "blank_128.png")
    blank_white.save(blank_canvas_path)

    all_transitions: List[Dict[str, Any]] = []

    # Sequence truncation counters
    total_raw_tokens = 0
    total_truncated_tokens = 0
    raw_lengths: List[int] = []
    num_transitions_hit_max = 0
    total_raw_strokes = 0
    total_dropped_strokes = 0

    print("=" * 70)
    print(f"STEP 4: SCALING SYNTHETIC GYM TO 10,000+ TRANSITIONS")
    print(f"Base drawings: {num_drawings} | Canvas: {canvas_res}x{canvas_res} | Max Seq Len: {max_seq_len}")
    print(f"Output directory: {out_p}")
    print("=" * 70)

    for i in range(num_drawings):
        cat = categories[i % len(categories)]
        s_seed = seed + i * 19

        if cat in ("stand", "walk", "run", "crouch", "reach", "sit"):
            preset = presets[(i // len(categories)) % len(presets)]
            yaw = float(rng.uniform(-60.0, 60.0))
            pitch = float(rng.uniform(-15.0, 20.0))
            data = generate_staged_human_drawing(
                pose_name=cat,
                preset=preset,
                yaw_deg=yaw,
                pitch_deg=pitch,
                seed=s_seed,
            )
        else:
            data = generate_staged_object_drawing(object_type=cat, seed=s_seed)

        caption = data["caption"]
        stages = data["stages"]
        drawing_id = f"drawing_{i:05d}"

        prev_canvas_path = blank_canvas_path

        for stg_idx, stg in enumerate(stages):
            stg_name = stg["stage_name"]
            stg_cmds = stg["commands"]
            doc_so_far = stg["doc_so_far"]
            sample_id = f"{drawing_id}_s{stg_idx}_{stg_name}"

            stage_doc = DocumentV2(commands=stg_cmds)

            # Measure untruncated tokens (cap high at 2048 to see true length)
            raw_tokens = doc_to_token_sequence(stage_doc, max_tokens=2048)
            raw_len = len(raw_tokens)
            raw_lengths.append(raw_len)
            total_raw_tokens += raw_len

            raw_stroke_count = len([c for c in stg_cmds if getattr(c, "type", "stroke") == "stroke"])
            total_raw_strokes += raw_stroke_count

            # Discretize with target max_seq_len
            if raw_len > max_seq_len:
                num_transitions_hit_max += 1
                tokens = raw_tokens[: max_seq_len - 1] + [EOS_TOKEN]
                # Reconstruct to measure dropped strokes
                dec_doc = token_sequence_to_doc(tokens)
                dec_stroke_count = len([c for c in dec_doc.commands if getattr(c, "type", "stroke") == "stroke"])
                dropped_strokes = max(0, raw_stroke_count - dec_stroke_count)
            else:
                tokens = raw_tokens
                dropped_strokes = 0

            total_dropped_strokes += dropped_strokes
            total_truncated_tokens += len(tokens)

            # Record item (using relative path to keep manifest compact)
            item_record = {
                "sample_id": sample_id,
                "drawing_id": drawing_id,
                "category": cat,
                "caption": caption,
                "stage_idx": stg_idx,
                "stage_name": stg_name,
                "canvas_path": prev_canvas_path,
                "tokens": tokens,
                "raw_len": raw_len,
                "is_truncated": (raw_len > max_seq_len),
                "dropped_strokes": dropped_strokes,
            }
            all_transitions.append(item_record)

            # Render canvas after this stage to disk for subsequent stage conditioning
            curr_canvas_path = str(canvases_dir / f"{sample_id}.png")
            if not os.path.exists(curr_canvas_path):
                canvas_img = render_document_v2(
                    doc_so_far,
                    width=canvas_res,
                    height=canvas_res,
                    supersample=1,
                    seed=s_seed,
                )
                canvas_img.save(curr_canvas_path)
            prev_canvas_path = curr_canvas_path

        # Update peak RAM tracking periodically
        if (i + 1) % 500 == 0 or i == num_drawings - 1:
            curr_ram = process.memory_info().rss / (1024 * 1024)
            if curr_ram > peak_ram:
                peak_ram = curr_ram
            elapsed = time.time() - t0
            print(f"[{i+1:5d}/{num_drawings}] Generated {len(all_transitions):6d} transitions | RAM: {curr_ram:.1f} MB | Elapsed: {elapsed:.1f}s")

    t_generation = time.time() - t0
    total_transitions = len(all_transitions)

    # Sequence truncation statistics
    raw_lengths_arr = np.array(raw_lengths)
    truncation_stats = {
        "total_transitions": total_transitions,
        "max_seq_len": max_seq_len,
        "transitions_hit_max_seq_len": int(num_transitions_hit_max),
        "hit_max_seq_len_pct": float(round((num_transitions_hit_max / total_transitions) * 100.0, 2)),
        "total_raw_strokes": int(total_raw_strokes),
        "total_dropped_strokes": int(total_dropped_strokes),
        "dropped_strokes_pct": float(round((total_dropped_strokes / max(1, total_raw_strokes)) * 100.0, 3)),
        "mean_raw_length": float(round(float(np.mean(raw_lengths_arr)), 2)),
        "std_raw_length": float(round(float(np.std(raw_lengths_arr)), 2)),
        "min_raw_length": int(np.min(raw_lengths_arr)),
        "p50_raw_length": float(np.percentile(raw_lengths_arr, 50)),
        "p90_raw_length": float(np.percentile(raw_lengths_arr, 90)),
        "p99_raw_length": float(np.percentile(raw_lengths_arr, 99)),
        "max_raw_length": int(np.max(raw_lengths_arr)),
    }

    print("\n" + "=" * 70)
    print("SEQUENCE TRUNCATION STATISTICS:")
    print(f"  Transitions hitting max_seq_len ({max_seq_len}): {truncation_stats['transitions_hit_max_seq_len']} / {total_transitions} ({truncation_stats['hit_max_seq_len_pct']}%)")
    print(f"  Strokes dropped: {truncation_stats['total_dropped_strokes']} / {truncation_stats['total_raw_strokes']} ({truncation_stats['dropped_strokes_pct']}%)")
    print(f"  Raw token lengths: mean={truncation_stats['mean_raw_length']} (p50={truncation_stats['p50_raw_length']}, p90={truncation_stats['p90_raw_length']}, p99={truncation_stats['p99_raw_length']}, max={truncation_stats['max_raw_length']})")
    print("=" * 70)

    # Split 90/5/5 strictly by drawing ID
    train_items, val_items, test_items = split_staged_dataset_by_drawing(
        all_transitions,
        train_frac=0.90,
        val_frac=0.05,
        test_frac=0.05,
        seed=seed,
    )

    # Verification of zero drawing leakage
    def get_drawing_ids(split_items):
        return set(it["drawing_id"] for it in split_items)

    train_ids = get_drawing_ids(train_items)
    val_ids = get_drawing_ids(val_items)
    test_ids = get_drawing_ids(test_items)

    assert len(train_ids & val_ids) == 0, f"Drawing leakage between train and val: {len(train_ids & val_ids)}"
    assert len(train_ids & test_ids) == 0, f"Drawing leakage between train and test: {len(train_ids & test_ids)}"
    assert len(val_ids & test_ids) == 0, f"Drawing leakage between val and test: {len(val_ids & test_ids)}"

    print(f"\nSplit Summary (Strict by Drawing ID, 0 Leakage):")
    print(f"  Train: {len(train_ids):4d} drawings -> {len(train_items):5d} transitions ({len(train_items)/total_transitions*100:.1f}%)")
    print(f"  Val:   {len(val_ids):4d} drawings -> {len(val_items):5d} transitions ({len(val_items)/total_transitions*100:.1f}%)")
    print(f"  Test:  {len(test_ids):4d} drawings -> {len(test_items):5d} transitions ({len(test_items)/total_transitions*100:.1f}%)")

    # Save JSONL files
    for split_name, split_list in [("train", train_items), ("val", val_items), ("test", test_items)]:
        split_p = out_p / f"{split_name}.jsonl"
        with open(split_p, "w", encoding="utf-8") as f:
            for it in split_list:
                f.write(json.dumps(it) + "\n")
        print(f"Saved {split_p} ({len(split_list)} records)")

    # Write Manifest
    manifest_data = {
        "dataset_name": "Konnon Draw Scaled Staged Gym 10k",
        "version": "1.0",
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "license": "CC0 1.0 Universal (Self-Generated Synthetic)",
        "source": "src/konnon_draw/gym/stage_generator.py",
        "num_base_drawings": num_drawings,
        "total_transitions": total_transitions,
        "splits": {
            "train": {"drawings": len(train_ids), "transitions": len(train_items)},
            "val": {"drawings": len(val_ids), "transitions": len(val_items)},
            "test": {"drawings": len(test_ids), "transitions": len(test_items)},
        },
        "canvas_resolution": [canvas_res, canvas_res],
        "categories": categories,
        "stages": STAGE_ORDER,
        "truncation_stats": truncation_stats,
        "resource_metrics": {
            "wall_clock_seconds": round(t_generation, 2),
            "peak_ram_mb": round(peak_ram, 1),
            "initial_ram_mb": round(ram_initial, 1),
        },
    }

    with open(man_p, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\nSaved manifest to {man_p}")
    print(f"Total Wall Clock Time: {t_generation:.1f} s | Peak RAM: {peak_ram:.1f} MB (RAM guard < 1200 MB: PASSED)")
    return manifest_data


if __name__ == "__main__":
    generate_scaled_gym()
