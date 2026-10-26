"""Tests for renderer_v2: brushes, fills, layer blending, determinism, performance, and SVG export."""

import time
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
)
from konnon_draw.renderer_v2 import (
    blend_layers,
    export_svg_v2,
    interpolate_points,
    render_document_v2,
)


def _sample_stroke(brush="ink_liner", blend="normal") -> StrokeActionV2:
    pts = [
        PointV2(0.1, 0.1, p=0.2),
        PointV2(0.3, 0.5, p=0.9),
        PointV2(0.6, 0.4, p=0.7),
        PointV2(0.9, 0.8, p=0.3),
    ]
    return StrokeActionV2(
        points=pts,
        brush_id=brush,
        color="#1e3a8a",
        opacity=0.85,
        base_width=6.0,
        blend_mode=blend,
    )


def test_renderer_determinism():
    """Identical seeds must produce pixel-exact identical images."""
    doc = DocumentV2()
    doc.commands.append(_sample_stroke("pencil"))
    doc.commands.append(_sample_stroke("charcoal"))

    img1 = render_document_v2(doc, width=256, height=256, supersample=1, seed=12345)
    img2 = render_document_v2(doc, width=256, height=256, supersample=1, seed=12345)

    arr1 = np.asarray(img1)
    arr2 = np.asarray(img2)
    assert np.array_equal(arr1, arr2), "Rendering with fixed seed must be 100% deterministic"


def test_all_brushes_render():
    """Verify that all 6 brush types render cleanly without error."""
    brushes = ["ink_liner", "pencil", "charcoal", "airbrush", "marker", "watercolor_wash"]
    doc = DocumentV2()
    for b in brushes:
        doc.commands.append(_sample_stroke(brush=b))

    img = render_document_v2(doc, width=256, height=256, supersample=1)
    assert img.size == (256, 256)
    # Check that ink was deposited (not pure blank white)
    arr = np.asarray(img)
    assert np.any(arr < 250), "Brushes must render visible pixels"


def test_all_fill_types_render():
    """Verify that all region fill types render correctly."""
    fill_types = ["flat", "linear_gradient", "radial_gradient", "hatching", "cross_hatching", "stipple"]
    poly = [(0.2, 0.2), (0.8, 0.2), (0.7, 0.7), (0.3, 0.8)]

    for ft in fill_types:
        doc = DocumentV2()
        doc.commands.append(
            FillActionV2(
                path=poly,
                fill_type=ft,
                color="#ef4444",
                color_end="#3b82f6",
                opacity=0.8,
                angle_deg=30.0,
                spacing=6.0,
                density=0.08,
            )
        )
        img = render_document_v2(doc, width=256, height=256, supersample=1)
        arr = np.asarray(img)
        assert np.any(arr < 250), f"Fill type {ft} must produce visible pixels"


def test_blend_modes():
    """Verify multiply, screen, and overlay blend modes."""
    base = Image.new("RGBA", (100, 100), (200, 100, 50, 255))
    overlay = Image.new("RGBA", (100, 100), (50, 150, 200, 180))

    for mode in ["normal", "multiply", "screen", "overlay"]:
        res = blend_layers(base, overlay, mode=mode, opacity=0.8)
        assert res.size == (100, 100)
        arr = np.asarray(res)
        assert arr.shape == (100, 100, 4)


def test_supersampling_resizes_correctly():
    """Supersampling 2x returns image of requested dimensions."""
    doc = DocumentV2()
    doc.commands.append(_sample_stroke("ink_liner"))

    img = render_document_v2(doc, width=128, height=128, supersample=2)
    assert img.size == (128, 128)


def test_svg_export():
    """SVG export produces valid XML string with layers and polygons."""
    doc = DocumentV2()
    doc.commands.append(_sample_stroke("ink_liner"))
    doc.commands.append(
        FillActionV2(
            path=[(0.1, 0.1), (0.4, 0.1), (0.3, 0.4)],
            fill_type="flat",
            color="#10b981",
        )
    )

    svg = export_svg_v2(doc, width=512, height=512)
    assert svg.startswith("<svg")
    assert "</svg>" in svg
    assert '<polygon points=' in svg
    assert 'viewBox="0 0 512 512"' in svg


def test_rendering_performance_1000_strokes():
    """Hard rule: 1000-stroke drawing renders in a few seconds on CPU."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=512, height=512),
        layers=[LayerV2(id="main", name="Main")],
    )

    rng = np.random.RandomState(42)
    for _ in range(1000):
        # Stroke with 4 points
        x = rng.uniform(0.1, 0.9, size=4)
        y = rng.uniform(0.1, 0.9, size=4)
        p = rng.uniform(0.3, 1.0, size=4)
        pts = [PointV2(x[i], y[i], p=p[i]) for i in range(4)]
        doc.commands.append(
            StrokeActionV2(
                points=pts,
                brush_id="ink_liner",
                color="#111827",
                base_width=2.0,
            )
        )

    t0 = time.time()
    img = render_document_v2(doc, width=512, height=512, supersample=1, seed=42)
    elapsed = time.time() - t0

    assert img.size == (512, 512)
    print(f"\n[PERF] 1000 strokes render time: {elapsed:.2f} s")
    assert elapsed < 6.0, f"Rendering took {elapsed:.2f} s, expected under 6 seconds"
