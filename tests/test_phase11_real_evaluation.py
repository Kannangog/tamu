"""Unit tests for Step 6 / Gate 11: Real objective evaluation metrics."""

import pytest
from konnon_draw.v2.schema import DocumentV2, StrokeActionV2, PointV2
from scripts.evaluate_real_metrics import check_closed_shape, evaluate_sample_metrics


def test_check_closed_shape():
    """Verify closed contour detection within epsilon."""
    # Closed triangle: start and end are close
    closed_stroke = StrokeActionV2(
        points=[
            PointV2(0.2, 0.2),
            PointV2(0.8, 0.2),
            PointV2(0.5, 0.8),
            PointV2(0.21, 0.21),
        ],
        brush_id="ink_liner",
        layer_id="line_art",
    )
    assert check_closed_shape(closed_stroke, eps=0.08) is True

    # Open line: start and end are far
    open_stroke = StrokeActionV2(
        points=[
            PointV2(0.1, 0.1),
            PointV2(0.9, 0.9),
        ],
        brush_id="ink_liner",
        layer_id="line_art",
    )
    assert check_closed_shape(open_stroke, eps=0.08) is False


def test_evaluate_sample_metrics_clean_rollout():
    """Verify evaluate_sample_metrics returns bounded rates and expected keys."""
    doc = DocumentV2(commands=[
        StrokeActionV2(
            points=[PointV2(0.5, 0.1), PointV2(0.5, 0.9)],
            brush_id="pencil",
            layer_id="underdrawing",
            stage="layout",
        )
    ])
    tokens = [1, 11, 593, 12, 599, 4, 128, 256, 2]  # contains EOS
    metrics = evaluate_sample_metrics(doc, category="stand", stage_idx=0, tokens=tokens, max_tokens=160)

    assert metrics["natural_eos"] is True
    assert metrics["valid_stage"] is True
    assert 0.0 <= metrics["closed_shape_rate"] <= 1.0
    assert 0.0 <= metrics["occlusion_correctness"] <= 1.0
    assert isinstance(metrics["prompt_following"], bool)
    assert metrics["stroke_count"] == 1
