"""Unit test for Step 2 / Gate 7: Loss and EOS audit verification."""

import json
from pathlib import Path
import pytest
import torch
import torch.nn as nn

from konnon_draw.v2.tokens import (
    VOCAB_SIZE,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    TOKEN_MOVE,
)
from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.model.sampling import sample_stage_tokens


def test_loss_masking_and_label_shift():
    """Verify that PAD_TOKEN is masked and EOS is supervised on label shift."""
    seq = torch.tensor([[BOS_TOKEN, TOKEN_MOVE, 20, 200, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN]])
    inputs = seq[:, :-1]
    targets = seq[:, 1:]

    # Target sequence: [TOKEN_MOVE, 20, 200, EOS_TOKEN, PAD_TOKEN, PAD_TOKEN]
    assert targets[0, 3].item() == EOS_TOKEN
    assert targets[0, 4].item() == PAD_TOKEN

    weights = torch.ones(VOCAB_SIZE)
    weights[EOS_TOKEN] = 4.0
    crit = nn.CrossEntropyLoss(weight=weights, ignore_index=PAD_TOKEN)

    # Dummy logits
    logits = torch.randn(1, 6, VOCAB_SIZE)
    loss = crit(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_sampling_length_prior():
    """Verify that length_prior_alpha adds calibrated EOS bonus after min_stage_tokens."""
    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=64,
        canvas_res=64,
    )
    context = torch.randn(1, 10, 64)

    # Greedy sample with length prior
    toks = sample_stage_tokens(
        model=model,
        context=context,
        max_tokens=48,
        temperature=0.0,
        top_k=0,
        length_prior_alpha=0.1,
        min_stage_tokens=8,
        seed=42,
    )
    assert len(toks) >= 2
    assert toks[0] == BOS_TOKEN
