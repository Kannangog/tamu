import pytest
import torch
from konnon_draw.renderer import (
    actions_to_bezier_segments,
    points_to_bezier_commands,
    render_svg,
)
from konnon_draw.sampling import generate
from konnon_draw.schema import validate_actions
from konnon_draw.training import load_model


def test_points_to_bezier_curved_segment():
    # 5 points forming a semicircle
    pts = [(0.1, 0.5), (0.2, 0.7), (0.5, 0.8), (0.8, 0.7), (0.9, 0.5)]
    cmds = points_to_bezier_commands(pts, corner_threshold_deg=50.0)

    assert len(cmds) == 4
    # All segments in a smooth arc should be CUBIC
    for cmd in cmds:
        assert cmd[0] == "CUBIC"
        assert len(cmd) == 7  # ('CUBIC', cx1, cy1, cx2, cy2, x, y)


def test_points_to_bezier_sharp_corner():
    # Right-angle corner: (0, 0) -> (0.5, 0) -> (0.5, 0.5)
    pts = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]
    cmds = points_to_bezier_commands(pts, corner_threshold_deg=45.0)

    assert len(cmds) == 2
    # At least one segment should be preserved as LINE due to sharp 90 degree angle
    assert any(cmd[0] == "LINE" for cmd in cmds)


def test_render_svg_smooth_vs_polygonal():
    actions = [
        {"type": "MOVE", "x": 0.1, "y": 0.5},
        {"type": "LINE", "x": 0.2, "y": 0.7},
        {"type": "LINE", "x": 0.5, "y": 0.8},
        {"type": "LINE", "x": 0.8, "y": 0.7},
        {"type": "LINE", "x": 0.9, "y": 0.5},
        {"type": "END"},
    ]

    # Standard polygonal
    svg_raw = render_svg(actions, smooth=False)
    assert "<svg" in svg_raw
    assert " L " in svg_raw
    assert " C " not in svg_raw

    # Smooth Bézier
    svg_smooth = render_svg(actions, smooth=True)
    assert "<svg" in svg_smooth
    assert " C " in svg_smooth


def test_actions_to_bezier_segments_structure():
    actions = [
        {"type": "MOVE", "x": 0.2, "y": 0.2},
        {"type": "LINE", "x": 0.4, "y": 0.3},
        {"type": "LINE", "x": 0.6, "y": 0.5},
        {"type": "LINE", "x": 0.8, "y": 0.8},
        {"type": "END"},
    ]

    strokes = actions_to_bezier_segments(actions, width=256, height=256)
    assert len(strokes) == 1
    assert "start" in strokes[0]
    assert len(strokes[0]["segments"]) == 3


def test_continuous_coordinates_generation():
    model, ckpt = load_model("outputs/checkpoints/best.pt")
    res = generate(model, class_id=0, seed=42, max_len=60, continuous_coords=True)

    actions = res["actions"]
    assert validate_actions(actions)

    # Check for floating point sub-bin precision (not just integer multiples of 1/63)
    coords = [a["x"] for a in actions if a["type"] in ("MOVE", "LINE")]
    has_sub_bin = any(abs(x * 63.0 - round(x * 63.0)) > 1e-3 for x in coords)
    assert has_sub_bin, "Expected sub-bin continuous coordinates"


def test_closed_loop_bezier_joins_cleanly():
    # 8 points approximating a closed circle
    import math
    pts = [
        (0.5 + 0.3 * math.cos(2 * math.pi * i / 8), 0.5 + 0.3 * math.sin(2 * math.pi * i / 8))
        for i in range(8)
    ]
    pts.append(pts[0])  # explicitly closed loop
    cmds = points_to_bezier_commands(pts, corner_threshold_deg=50.0)

    assert len(cmds) == 8
    # In a smooth circular loop, every command should be CUBIC
    assert all(cmd[0] == "CUBIC" for cmd in cmds)
    # Final segment endpoint must match the starting point
    assert abs(cmds[-1][-2] - pts[0][0]) < 1e-5
    assert abs(cmds[-1][-1] - pts[0][1]) < 1e-5


def test_raw_json_unchanged_by_smoothing():
    import copy
    from konnon_draw.sampling import save_sample
    raw_actions = [
        {"type": "MOVE", "x": 0.2, "y": 0.2},
        {"type": "LINE", "x": 0.4, "y": 0.5},
        {"type": "LINE", "x": 0.7, "y": 0.8},
        {"type": "END"},
    ]
    actions_copy = copy.deepcopy(raw_actions)
    # Render with smooth=True
    _ = render_svg(raw_actions, smooth=True)
    # Verify raw actions dict was not modified in-place
    assert raw_actions == actions_copy


def test_corner_above_50_degrees_stays_straight():
    # Two straight segments meeting at an acute angle (~60 degrees deviation)
    pts = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.1)]
    cmds = points_to_bezier_commands(pts, corner_threshold_deg=50.0)
    # A >50 deg turn must not be rounded into a single continuous curve
    assert any(cmd[0] == "LINE" for cmd in cmds)
