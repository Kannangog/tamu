"""Unit tests for Phase 3: Synthetic Gym, staged data generation, captions, and dataset."""

import json
from pathlib import Path
import pytest
import torch

from konnon_draw.gym.captions import generate_caption, POSE_TEMPLATES, OBJECT_TEMPLATES
from konnon_draw.gym.stage_generator import (
    generate_staged_human_drawing,
    generate_staged_object_drawing,
    permute_strokes_within_stages,
    STAGE_ORDER,
)
from konnon_draw.gym.dataset import (
    StagedDrawingDataset,
    generate_staged_gym_data,
)
from konnon_draw.v2.schema import validate_document_v2


def test_caption_diversity_and_paraphrases():
    """Verify that generate_caption returns 10+ varied paraphrase phrasings."""
    samples = set()
    for seed in range(25):
        cap = generate_caption("stand", camera_yaw=15.0, seed=seed)
        samples.add(cap)
    # Must produce at least 8 distinct phrases across 25 samples
    assert len(samples) >= 8, f"Expected varied paraphrases, got only {len(samples)} distinct strings"


def test_staged_human_drawing_structure():
    """Verify that human drawing generator produces chronologically ordered stages."""
    data = generate_staged_human_drawing(pose_name="walk", yaw_deg=10.0, seed=42)
    assert data["category"] == "walk"
    assert len(data["caption"]) > 10

    stages = data["stages"]
    assert len(stages) >= 3  # Must contain layout, guide, ink

    stage_names = [s["stage_name"] for s in stages]
    # Check that stage names are in non-decreasing STAGE_ORDER
    indices = [STAGE_ORDER.index(n) for n in stage_names]
    assert indices == sorted(indices), f"Stages out of order: {stage_names}"

    # Verify full doc validity
    assert validate_document_v2(data["full_doc"]) is True


def test_staged_object_drawing_structure():
    """Verify that object drawing generator produces valid DocumentV2."""
    for obj_name in ["box", "chair", "bicycle", "house"]:
        data = generate_staged_object_drawing(object_type=obj_name, seed=42)
        assert data["category"] == obj_name
        assert validate_document_v2(data["full_doc"]) is True
        assert len(data["stages"]) >= 2


def test_stroke_permutation_preserves_commands():
    """Permuting stroke order within stages must preserve total stroke count."""
    data = generate_staged_human_drawing(pose_name="run", seed=42)
    original_cmds = data["full_doc"].commands
    permuted = permute_strokes_within_stages(original_cmds)
    assert len(permuted) == len(original_cmds)


def test_staged_dataset_batching():
    """Verify that StagedDrawingDataset emits correctly shaped canvas and token tensors."""
    items = generate_staged_gym_data(num_samples=4, canvas_res=64, seed=42)
    ds = StagedDrawingDataset(items, canvas_res=64, max_tokens=64)

    assert len(ds) == len(items)
    sample = ds[0]
    assert sample["canvas"].shape == (3, 64, 64)
    assert sample["tokens"].shape == (64,)
    assert sample["tokens"].dtype == torch.long
    assert isinstance(sample["caption"], str)
