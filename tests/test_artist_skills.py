"""Unit tests for the Artist Skill Library (Phase 3)."""

import math
import pytest
from konnon_draw.schema_v2 import DocumentV2, validate_document_v2
from konnon_draw.artist_skills import (
    apply_atmospheric_perspective,
    apply_line_weight_by_light,
    create_standard_layers,
    generate_cast_shadow_polygon,
    generate_directional_hatching,
    generate_palette_harmony,
    generate_stipple_shading,
    make_arc,
    make_ellipse,
    make_gradient_sky,
    make_perspective_grid,
    make_proportion_guides,
    make_sketchy_lines,
    make_symmetric_stroke,
    make_wobbly_contour,
    rule_of_thirds_points,
)


def test_standard_layers():
    layers = create_standard_layers()
    assert len(layers) >= 5
    layer_ids = {l.id for l in layers}
    assert "background" in layer_ids
    assert "shading" in layer_ids
    assert "line_art" in layer_ids


def test_rule_of_thirds():
    pts = rule_of_thirds_points()
    assert len(pts) == 4
    for x, y in pts:
        assert 0.0 < x < 1.0
        assert 0.0 < y < 1.0


def test_ellipse_construction():
    st = make_ellipse(center=(0.5, 0.5), rx=0.2, ry=0.1, angle_deg=30.0, num_points=24)
    assert len(st.points) == 25  # closed
    assert st.brush_id == "ink_liner"
    # First and last points should coincide for closed ellipse
    assert math.hypot(st.points[0].x - st.points[-1].x, st.points[0].y - st.points[-1].y) < 1e-4


def test_arc_construction():
    st = make_arc(center=(0.5, 0.5), rx=0.2, ry=0.2, start_deg=0.0, end_deg=180.0, num_points=15)
    assert len(st.points) == 15
    assert st.curve_type == "catmull_rom"


def test_wobbly_contour():
    points = [(0.1, 0.1), (0.5, 0.2), (0.9, 0.1)]
    st = make_wobbly_contour(points, wobble_amount=0.005, seed=123)
    assert len(st.points) > len(points)
    for p in st.points:
        assert 0.0 <= p.p <= 1.0


def test_symmetric_stroke():
    st = make_ellipse(center=(0.4, 0.5), rx=0.1, ry=0.1)
    sym = make_symmetric_stroke(st, axis_x=0.5)
    # Center of mirrored stroke should be at 2*0.5 - 0.4 = 0.6
    xs = [p.x for p in sym.points]
    assert abs(sum(xs) / len(xs) - 0.6) < 0.02


def test_perspective_and_proportion_guides():
    p_grid = make_perspective_grid(vanishing_point=(0.5, 0.3), horizon_y=0.3)
    assert len(p_grid) > 5

    prop = make_proportion_guides((0.3, 0.2, 0.7, 0.8), divisions=[0.25, 0.5, 0.75])
    assert len(prop) == 4


def test_line_weight_by_light():
    # Horizontal stroke from left to right
    from konnon_draw.schema_v2 import PointV2, StrokeActionV2
    st = StrokeActionV2(
        points=[PointV2(0.1, 0.5, p=0.5), PointV2(0.9, 0.5, p=0.5)],
        base_width=4.0,
    )
    # Light coming from top (0, -1) -> normal (-0, 1) faces downward (shadow side)
    weighted = apply_line_weight_by_light(st, light_dir=(0.0, -1.0), min_pressure=0.2, max_pressure=1.0)
    for p in weighted.points:
        assert 0.2 <= p.p <= 1.0


def test_sketchy_lines():
    pts = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
    strokes = make_sketchy_lines(pts, count=4, spread=0.005)
    assert len(strokes) == 4
    for s in strokes:
        assert len(s.points) == 3


def test_value_and_form_helpers():
    poly = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    hatch = generate_directional_hatching(poly, angle_deg=45.0, cross=True)
    assert hatch.fill_type == "cross_hatching"

    stipple = generate_stipple_shading(poly, density=0.04)
    assert stipple.fill_type == "stipple"

    shadow = generate_cast_shadow_polygon([(0.3, 0.7), (0.7, 0.7)], light_offset=(0.1, 0.05))
    assert shadow.fill_type == "flat"
    assert len(shadow.path) == 4


def test_color_and_atmosphere():
    comp = generate_palette_harmony("#3b82f6", harmony="complementary")
    assert len(comp) == 2

    tri = generate_palette_harmony("#ef4444", harmony="triadic")
    assert len(tri) == 3

    warm_cool = generate_palette_harmony("#10b981", harmony="warm_shadow_cool_light")
    assert len(warm_cool) == 3

    # Atmospheric perspective: distant color should differ from near color and shift toward sky
    near = apply_atmospheric_perspective("#15803d", depth=0.0, sky_tint_hex="#bfdbfe")
    distant = apply_atmospheric_perspective("#15803d", depth=1.0, sky_tint_hex="#bfdbfe")
    assert near == "#15803d"
    assert distant != near

    sky = make_gradient_sky("#1e3a8a", "#fed7aa")
    assert sky.fill_type == "linear_gradient"


def test_document_validation_with_skills():
    """Document composed entirely from skill helpers passes strict v2 validation."""
    doc = DocumentV2(layers=create_standard_layers())
    doc.commands.append(make_gradient_sky())
    doc.commands.extend(make_perspective_grid())
    doc.commands.append(make_ellipse((0.5, 0.5), 0.2, 0.15))
    doc.commands.append(generate_directional_hatching([(0.3, 0.3), (0.7, 0.3), (0.5, 0.7)]))

    assert validate_document_v2(doc) is True
