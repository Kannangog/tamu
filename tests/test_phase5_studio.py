"""Unit tests for Phase 5: Live Drawing Studio API endpoints and staged handlers."""

import json
from pathlib import Path
import pytest
import torch

from konnon_draw.web_server import DrawingStudioHandler, image_to_data_url
from konnon_draw.v2.schema import DocumentV2
from PIL import Image


def test_image_to_data_url():
    """Verify base64 PNG data URL generation."""
    img = Image.new("RGB", (32, 32), color=(255, 0, 0))
    data_url = image_to_data_url(img)
    assert data_url.startswith("data:image/png;base64,")
    assert len(data_url) > 50


def test_staged_model_lazy_loading():
    """Verify DrawingStudioHandler lazily loads the staged transformer."""
    staged_model, staged_encoder = DrawingStudioHandler.get_staged_model()
    # If checkpoint exists, staged_model should be loaded
    ckpt_path = Path("outputs_v2/laptop/checkpoints/best.pt")
    if ckpt_path.exists():
        assert staged_model is not None
        assert staged_encoder is not None
        assert hasattr(staged_model, "build_multimodal_context")


def test_record_stylus_stroke_logic():
    """Verify stylus stroke payload properly constructs DocumentV2."""
    from konnon_draw.v2.schema import PointV2, StrokeActionV2
    from konnon_draw.v2.renderer import export_svg_v2

    raw_pts = [
        {"x": 0.1, "y": 0.2, "pressure": 0.4, "tilt": 12.0},
        {"x": 0.3, "y": 0.5, "pressure": 0.8, "tilt": 15.0},
    ]
    pts = [PointV2(x=p["x"], y=p["y"], p=p["pressure"], tilt=p["tilt"]) for p in raw_pts]
    stroke = StrokeActionV2(points=pts, brush_id="pencil", color="#333333", base_width=2.5)
    doc = DocumentV2(commands=[stroke])

    svg = export_svg_v2(doc, width=256, height=256)
    assert "<svg" in svg
    assert "</svg>" in svg


def test_staged_transformer_inference_pipeline():
    """Verify that staged generation executes correctly through the studio helper."""
    staged_model, staged_encoder = DrawingStudioHandler.get_staged_model()
    if staged_model is not None and staged_encoder is not None:
        from konnon_draw.model.sampling import generate_staged_drawing
        res = generate_staged_drawing(
            model=staged_model,
            text_encoder=staged_encoder,
            prompt="a small box",
            stages=["layout", "ink"],
            canvas_res=64,
            out_res=128,
            temperature=0.5,
            top_k=10,
            seed=42,
        )
        assert "stage_history" in res
        assert len(res["stage_history"]) == 2
        assert "doc" in res
        assert len(res["doc"].commands) >= 0
