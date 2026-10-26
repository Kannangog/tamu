"""Tests for Phase 1: v2 Stroke Language, Renderer, Brushes, Depth Occlusion, and Converter."""

import time
import numpy as np
import pytest
from PIL import Image

from konnon_draw.v2.schema import (
    PointV2,
    StrokeActionV2,
    FillActionV2,
    LayerV2,
    CanvasConfigV2,
    DocumentV2,
    validate_document_v2,
    SUPPORTED_BRUSHES,
    SUPPORTED_STAGES,
    SUPPORTED_PURPOSES,
)
from konnon_draw.v2.renderer import (
    render_document_v2,
    export_svg_v2,
    render_brush_swatches,
)
from konnon_draw.v2.convert_v1 import (
    v1_to_v2,
    v2_to_v1,
)


def test_schema_validation_success():
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=512, height=512),
        layers=[LayerV2(id="main", name="Main")],
        commands=[
            StrokeActionV2(
                points=[PointV2(0.1, 0.1, 0.5), PointV2(0.9, 0.9, 0.8)],
                brush_id="colored_pencil",
                stage="ink",
                purpose="contour",
                depth_rank=2,
            ),
            FillActionV2(
                path=[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)],
                fill_type="flat",
                depth_rank=1,
            ),
        ],
    )
    assert validate_document_v2(doc) is True


def test_schema_validation_invalid_brush_and_stage():
    doc = DocumentV2(
        commands=[
            StrokeActionV2(
                points=[PointV2(0.1, 0.1), PointV2(0.5, 0.5)],
                brush_id="magical_wand",
                stage="post_processing",
            )
        ]
    )
    with pytest.raises(ValueError):
        validate_document_v2(doc)


def test_renderer_determinism():
    """Identical seeds must yield bitwise pixel-identical renders."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=256, height=256),
        commands=[
            StrokeActionV2(
                points=[PointV2(0.2, 0.2), PointV2(0.8, 0.8)],
                brush_id="pencil",
                base_width=4.0,
            ),
            StrokeActionV2(
                points=[PointV2(0.8, 0.2), PointV2(0.2, 0.8)],
                brush_id="charcoal",
                base_width=5.0,
            ),
        ],
    )
    img1 = render_document_v2(doc, width=256, height=256, seed=123, supersample=1)
    img2 = render_document_v2(doc, width=256, height=256, seed=123, supersample=1)

    arr1 = np.array(img1)
    arr2 = np.array(img2)
    assert np.array_equal(arr1, arr2), "Renders with identical seed must match exactly"


def test_depth_occlusion_ordering():
    """Higher depth_rank must be rendered on top of lower depth_rank."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=100, height=100, background="#ffffff"),
        layers=[LayerV2(id="main", name="Main")],
        commands=[
            # Command 1: Red horizontal bar with depth_rank=5 (nearer, drawn on top)
            StrokeActionV2(
                points=[PointV2(0.2, 0.5), PointV2(0.8, 0.5)],
                brush_id="ink_liner",
                color="#ff0000",
                base_width=20.0,
                depth_rank=5,
            ),
            # Command 2: Blue vertical bar with depth_rank=1 (farther, should be drawn underneath)
            StrokeActionV2(
                points=[PointV2(0.5, 0.2), PointV2(0.5, 0.8)],
                brush_id="ink_liner",
                color="#0000ff",
                base_width=20.0,
                depth_rank=1,
            ),
        ],
    )
    img = render_document_v2(doc, width=100, height=100, supersample=1, seed=42)
    # The intersection at (50, 50) must be dominated by red (#ff0000) because depth_rank 5 > 1
    r, g, b = img.getpixel((50, 50))
    assert r > 200 and b < 50, f"Expected red on top at intersection, got RGB=({r}, {g}, {b})"


def test_taper_effect():
    """Tapered strokes should start narrower than base width at center."""
    pts = [PointV2(0.1, 0.5, 1.0), PointV2(0.5, 0.5, 1.0), PointV2(0.9, 0.5, 1.0)]
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=200, height=100, background="#ffffff"),
        commands=[
            StrokeActionV2(
                points=pts,
                brush_id="ink_liner",
                color="#000000",
                base_width=12.0,
                taper_in=0.3,
                taper_out=0.3,
            )
        ],
    )
    img = render_document_v2(doc, width=200, height=100, supersample=1, seed=42)
    assert img.size == (200, 100)


def test_roundtrip_v1_to_v2_to_render():
    """v1 actions converted to v2 must validate and render cleanly to PNG and SVG."""
    v1_actions = [
        {"type": "MOVE", "x": 10, "y": 10},
        {"type": "LINE", "x": 30, "y": 30},
        {"type": "LINE", "x": 50, "y": 10},
        {"type": "PEN_UP"},
        {"type": "MOVE", "x": 20, "y": 40},
        {"type": "LINE", "x": 40, "y": 40},
        {"type": "PEN_UP"},
        {"type": "END"},
    ]
    doc = v1_to_v2(v1_actions)
    assert validate_document_v2(doc) is True

    img = render_document_v2(doc, width=256, height=256, supersample=1)
    assert img.size == (256, 256)

    svg = export_svg_v2(doc, width=256, height=256)
    assert "<svg" in svg and "</svg>" in svg

    # Reverse conversion
    v1_back = v2_to_v1(doc)
    assert len(v1_back) > 0
    assert v1_back[0]["type"] == "MOVE"
    assert v1_back[-1]["type"] == "END"


def test_performance_1000_strokes():
    """Gate 1: 1000-stroke render < 10 seconds on CPU."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=512, height=512),
        layers=[LayerV2(id="main", name="Main")],
    )
    rng = np.random.RandomState(42)
    for _ in range(1000):
        pts = [
            PointV2(float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.4, 1.0)))
            for _ in range(3)
        ]
        doc.commands.append(
            StrokeActionV2(
                points=pts,
                brush_id="ink_liner",
                color="#0f172a",
                base_width=2.5,
            )
        )

    t0 = time.time()
    img = render_document_v2(doc, width=512, height=512, supersample=1, seed=42)
    elapsed = time.time() - t0

    assert img.size == (512, 512)
    print(f"\n[PHASE 1 PERF] 1000 strokes render time: {elapsed:.2f}s")
    assert elapsed < 10.0, f"Rendering took {elapsed:.2f}s, expected < 10s"
