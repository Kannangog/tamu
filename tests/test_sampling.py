import pytest
import torch
from konnon_draw.model import DrawingGRU
from konnon_draw.sampling import generate_sequence
from konnon_draw.schema import validate_actions


def test_sampling_produces_valid_actions():
    model = DrawingGRU(embedding_dim=32, hidden_dim=64)
    model.eval()

    actions, is_valid, err = generate_sequence(
        model=model,
        max_len=32,
        temperature=0.8,
        seed=123,
    )

    assert is_valid, f"Validation failed: {err}"
    assert len(actions) >= 2
    assert actions[0]["type"] == "MOVE"
    assert actions[-1]["type"] == "END"
    assert validate_actions(actions)


def test_sampling_rejection_and_fallback():
    model = DrawingGRU(embedding_dim=32, hidden_dim=64)
    model.eval()

    # Small max_len to test fallback policy
    actions, is_valid, err = generate_sequence(
        model=model,
        max_len=5,
        temperature=1.0,
        seed=456,
    )

    assert actions[-1]["type"] == "END"
    assert is_valid
