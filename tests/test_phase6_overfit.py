"""Unit test for Step 1 / Gate 6: Overfit sanity verification."""

import json
from pathlib import Path
import pytest
import torch

from konnon_draw.v2.tokens import VOCAB_SIZE, BOS_TOKEN, EOS_TOKEN
from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.model.text_encoder import FrozenTextEncoder
from konnon_draw.model.sampling import sample_stage_tokens


def test_overfit_metrics_exist():
    """Verify that overfit evidence files exist and match requirements."""
    metrics_path = Path("outputs_v2/overfit/overfit_metrics.json")
    assert metrics_path.exists(), "overfit_metrics.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["num_transitions"] == 32
    assert data["final_loss"] < 0.05
    assert data["exact_match_rate"] >= 0.70  # measured 93.75%
    assert data["mean_pixel_mae"] < 0.01    # measured 0.00266
