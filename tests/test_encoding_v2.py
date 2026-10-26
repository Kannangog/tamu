"""Tests for v2 token encoding, decoding, QuickDraw conversion, and learning extensions."""

import numpy as np
import pytest
from PIL import Image

from konnon_draw.schema_v2 import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
    validate_document_v2,
)
from konnon_draw.encoding_v2 import (
    compute_critic_score,
    decode_tokens_to_v2,
    encode_v2_to_tokens,
    quickdraw_to_v2,
    stroke_order_augmentation,
)


def test_token_roundtrip_strokes_and_fills():
    """Verify that encode_v2 -> decode_v2 preserves geometry within quantization tolerance."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1024, height=1024, background="#ffffff"),
        layers=[
            LayerV2(id="bg", name="BG"),
            LayerV2(id="main", name="Main", opacity=0.8, blend_mode="multiply"),
        ],
    )
    # Add stroke
    pts = [PointV2(0.125, 0.250, p=0.4), PointV2(0.500, 0.750, p=0.9), PointV2(0.875, 0.625, p=0.7)]
    doc.commands.append(
        StrokeActionV2(
            points=pts,
            brush_id="pencil",
            color="#ef4444",
            base_width=4.0,
            layer_id="main",
            blend_mode="multiply",
        )
    )
    # Add fill
    doc.commands.append(
        FillActionV2(
            path=[(0.2, 0.2), (0.6, 0.2), (0.4, 0.6)],
            fill_type="hatching",
            color="#3b82f6",
            layer_id="main",
        )
    )

    tokens = encode_v2_to_tokens(doc, coord_bins=256)
    assert len(tokens) > 5
    assert tokens[0]["type"] == "SET_CANVAS"
    assert tokens[-1]["type"] == "END_DOC"

    # Decode back
    reconstructed = decode_tokens_to_v2(tokens, coord_bins=256)
    assert validate_document_v2(reconstructed) is True
    assert len(reconstructed.commands) == 2

    # Verify stroke points are preserved within 1 bin tolerance (1/255 ~ 0.004)
    rec_stroke = reconstructed.commands[0]
    assert rec_stroke.brush_id == "pencil"
    assert len(rec_stroke.points) == len(pts)
    for orig_p, rec_p in zip(pts, rec_stroke.points):
        assert abs(orig_p.x - rec_p.x) < 0.005
        assert abs(orig_p.y - rec_p.y) < 0.005
        assert abs(orig_p.p - rec_p.p) < 0.1


def test_quickdraw_to_v2_conversion():
    """Verify raw QuickDraw vector conversion produces valid DocumentV2 with synthetic taper."""
    # Synthetic 2-stroke doodle (e.g. circle + line)
    raw_qd = [
        [[10, 50, 100, 150, 100, 50, 10], [50, 10, 50, 100, 150, 100, 50]],
        [[100, 200], [100, 200]],
    ]
    doc = quickdraw_to_v2(raw_qd, default_brush="ink_liner", add_shadow=True)
    assert validate_document_v2(doc) is True
    assert len(doc.commands) == 3  # 1 ground shadow + 2 strokes
    # Check normalization to [0, 1]
    for c in doc.commands:
        if c.type == "stroke":
            for p in c.points:
                assert 0.0 <= p.x <= 1.0
                assert 0.0 <= p.y <= 1.0
                assert 0.2 <= p.p <= 1.0  # tapered


def test_stroke_order_augmentation():
    """Augmentation shuffles strokes while preserving layer hierarchy."""
    doc = DocumentV2()
    for i in range(5):
        pts = [PointV2(0.1 * i, 0.1), PointV2(0.1 * i + 0.05, 0.2)]
        doc.commands.append(StrokeActionV2(points=pts, color=f"#00000{i}"))

    aug = stroke_order_augmentation(doc, seed=42)
    assert len(aug.commands) == 5
    # Colors should be permuted
    orig_colors = [c.color for c in doc.commands]
    aug_colors = [c.color for c in aug.commands]
    assert orig_colors != aug_colors
    assert set(orig_colors) == set(aug_colors)


def test_critic_score():
    """Critic score computes perceptual similarity."""
    doc = DocumentV2()
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(0.2, 0.5), PointV2(0.8, 0.5)],
            base_width=10.0,
        )
    )
    # Render reference image
    from konnon_draw.renderer_v2 import render_document_v2
    ref_img = render_document_v2(doc, width=128, height=128, supersample=1)

    # Identical doc should give near 1.0 score
    score_identical = compute_critic_score(doc, ref_img, eval_size=128)
    assert score_identical > 0.95

    # Blank doc should give lower score
    blank_doc = DocumentV2()
    score_blank = compute_critic_score(blank_doc, ref_img, eval_size=128)
    assert score_blank < score_identical
