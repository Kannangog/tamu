import pytest
from konnon_draw.evaluation import compute_stroke_statistics, generate_random_baseline


def test_stroke_statistics():
    actions = [
        {"type": "MOVE", "x": 0.0, "y": 0.0},
        {"type": "LINE", "x": 0.0, "y": 1.0},
        {"type": "LINE", "x": 1.0, "y": 1.0},
        {"type": "PEN_UP"},
        {"type": "MOVE", "x": 0.5, "y": 0.5},
        {"type": "END"},
    ]

    stats = compute_stroke_statistics(actions)
    assert stats["action_count"] == 6
    assert stats["path_count"] == 2
    assert stats["pen_up_ratio"] == 1 / 6
    assert pytest.approx(stats["total_path_length"], 1e-4) == 2.0


def test_random_baseline():
    valid_rate, completion_rate = generate_random_baseline(num_samples=20, max_len=30, seed=42)
    # Random unconstrained actions will have a significantly low validity rate
    assert 0.0 <= valid_rate <= 1.0
    assert 0.0 <= completion_rate <= 1.0
