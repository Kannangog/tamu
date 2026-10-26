"""Unit tests for Step 5 / Gate 10: Training metrics, group cross-entropy breakdown, and length prior."""

import pytest
import torch
import torch.nn.functional as F

from konnon_draw.v2.tokens import (
    VOCAB_SIZE,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    TOKEN_MOVE,
    TOKEN_LINE,
    TOKEN_SET_LAYER,
)
from konnon_draw.general.vocab import (
    X_OFFSET,
    Y_OFFSET,
    PRESSURE_OFFSET,
    COLOR_OFFSET,
    WIDTH_OFFSET,
    LAYER_OFFSET,
    BRUSH_OFFSET,
)
from scripts.train_scaled_gym import compute_group_losses, TOKEN_GROUPS, TOKEN_TO_GROUP


def test_token_to_group_mapping_coverage():
    """Verify that all functional token types belong to their expected group."""
    assert TOKEN_TO_GROUP[TOKEN_MOVE] == "structure"
    assert TOKEN_TO_GROUP[TOKEN_LINE] == "structure"
    assert TOKEN_TO_GROUP[TOKEN_SET_LAYER] == "structure"
    assert TOKEN_TO_GROUP[X_OFFSET + 10] == "coord_x"
    assert TOKEN_TO_GROUP[Y_OFFSET + 20] == "coord_y"
    assert TOKEN_TO_GROUP[PRESSURE_OFFSET + 5] == "pressure"
    assert TOKEN_TO_GROUP[COLOR_OFFSET + 2] == "color"
    assert TOKEN_TO_GROUP[WIDTH_OFFSET + 4] == "width"
    assert TOKEN_TO_GROUP[LAYER_OFFSET + 1] == "layer"
    assert TOKEN_TO_GROUP[BRUSH_OFFSET + 0] == "brush"
    assert TOKEN_TO_GROUP[EOS_TOKEN] == "eos"


def test_compute_group_losses_calculation():
    """Verify compute_group_losses returns expected dictionary without NaN."""
    # Synthetic logits: shape (B=2, L=5, V=604)
    logits = torch.randn(2, 5, VOCAB_SIZE)
    # Synthetic targets: mix of structure, coord, layer, eos, pad
    targets = torch.tensor([
        [TOKEN_MOVE, X_OFFSET + 10, Y_OFFSET + 20, EOS_TOKEN, PAD_TOKEN],
        [LAYER_OFFSET + 1, BRUSH_OFFSET + 0, COLOR_OFFSET + 2, EOS_TOKEN, PAD_TOKEN],
    ], dtype=torch.long)

    group_means = compute_group_losses(logits, targets)

    assert "structure" in group_means
    assert "coord_x" in group_means
    assert "coord_y" in group_means
    assert "layer" in group_means
    assert "brush" in group_means
    assert "eos" in group_means

    for g, val in group_means.items():
        assert val >= 0.0, f"Group loss for {g} must be non-negative, got {val}"
        assert not torch.isnan(torch.tensor(val)), f"Group loss for {g} is NaN"
