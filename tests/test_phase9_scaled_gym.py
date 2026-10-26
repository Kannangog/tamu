"""Unit tests for Step 4 / Gate 9: Scaled Synthetic Gym generation and truncation statistics."""

from pathlib import Path
import pytest
import json

from konnon_draw.gym.dataset import split_staged_dataset_by_drawing


def test_split_staged_dataset_proportions_and_disjointness():
    """Verify split_staged_dataset_by_drawing adheres to proportions and has zero drawing leakage."""
    # Synthetic mock items with 100 drawings, 3 stages each = 300 transitions
    mock_items = []
    for d in range(100):
        for s in range(3):
            mock_items.append({
                "sample_id": f"drawing_{d:04d}_s{s}",
                "drawing_id": f"drawing_{d:04d}",
                "stage_idx": s,
                "tokens": [1, 2, 3],
            })

    train, val, test = split_staged_dataset_by_drawing(
        mock_items,
        train_frac=0.90,
        val_frac=0.05,
        test_frac=0.05,
        seed=42,
    )

    train_drawings = set(it["drawing_id"] for it in train)
    val_drawings = set(it["drawing_id"] for it in val)
    test_drawings = set(it["drawing_id"] for it in test)

    assert len(train_drawings & val_drawings) == 0, "Train and Val share drawings!"
    assert len(train_drawings & test_drawings) == 0, "Train and Test share drawings!"
    assert len(val_drawings & test_drawings) == 0, "Val and Test share drawings!"

    assert len(train_drawings) == 90
    assert len(val_drawings) == 5
    assert len(test_drawings) == 5

    assert len(train) == 270
    assert len(val) == 15
    assert len(test) == 15


def test_manifest_verification_when_present():
    """Verify manifest schema if scaled gym generation has completed."""
    man_path = Path("data/manifests/staged_gym_10k.manifest.json")
    if not man_path.exists():
        pytest.skip("Scaled gym manifest not yet created.")

    with open(man_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_transitions"] >= 10000, f"Expected >= 10000 transitions, got {data['total_transitions']}"
    assert "truncation_stats" in data
    assert "splits" in data
    assert data["splits"]["train"]["transitions"] > data["splits"]["val"]["transitions"]
