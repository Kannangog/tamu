"""Tests for collaborative prefix drawing."""
import pytest
import torch
from konnon_draw.model import DrawingGRU
from konnon_draw.sampling import generate_with_prefix
from konnon_draw.schema import is_valid


@pytest.fixture
def dummy_model():
    torch.manual_seed(42)
    return DrawingGRU(num_classes=2, embedding_dim=16, hidden_dim=32)


def test_generate_with_prefix_single_stroke(dummy_model):
    prefix = [
        {"type": "MOVE", "x": 0.2, "y": 0.3},
        {"type": "LINE", "x": 0.6, "y": 0.7},
    ]
    res = generate_with_prefix(dummy_model, class_id=0, prefix_actions=prefix, seed=10, max_len=40)

    actions = res["actions"]
    assert is_valid(actions), "Resulting actions must be syntactically valid"
    assert actions[-1]["type"] == "END", "Drawing must terminate with END"

    # Prefix must be preserved verbatim at start of actions
    assert actions[0]["type"] == "MOVE"
    assert actions[0]["x"] == 0.2 and actions[0]["y"] == 0.3
    assert actions[1]["type"] == "LINE"
    assert actions[1]["x"] == 0.6 and actions[1]["y"] == 0.7
    assert res["user_stroke_count"] == 2


def test_generate_with_prefix_two_strokes(dummy_model):
    prefix = [
        {"type": "MOVE", "x": 0.1, "y": 0.2},
        {"type": "LINE", "x": 0.3, "y": 0.4},
        {"type": "PEN_UP"},
        {"type": "MOVE", "x": 0.5, "y": 0.6},
        {"type": "LINE", "x": 0.7, "y": 0.8},
    ]
    res = generate_with_prefix(dummy_model, class_id=1, prefix_actions=prefix, seed=99, max_len=50)

    actions = res["actions"]
    assert is_valid(actions)
    assert len(actions) >= len(prefix) + 1  # prefix + at least END

    for i, expected in enumerate(prefix):
        assert actions[i]["type"] == expected["type"]
        if "x" in expected:
            assert actions[i]["x"] == expected["x"]
            assert actions[i]["y"] == expected["y"]


def test_generate_with_prefix_empty_fallback(dummy_model):
    res = generate_with_prefix(dummy_model, class_id=0, prefix_actions=[], seed=5, max_len=30)
    assert is_valid(res["actions"])
    assert res["actions"][-1]["type"] == "END"


def test_generate_with_prefix_continuous_coords(dummy_model):
    prefix = [
        {"type": "MOVE", "x": 0.12345, "y": 0.23456},
        {"type": "LINE", "x": 0.67891, "y": 0.78912},
    ]
    res = generate_with_prefix(dummy_model, class_id=0, prefix_actions=prefix, seed=12, max_len=40,
                                continuous_coords=True)
    actions = res["actions"]
    assert is_valid(actions)
    # Exact floats preserved for user prefix
    assert actions[0]["x"] == 0.12345
    assert actions[0]["y"] == 0.23456
    assert actions[1]["x"] == 0.67891
    assert actions[1]["y"] == 0.78912
