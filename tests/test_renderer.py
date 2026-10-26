import pytest
from konnon_draw.renderer import render_svg
from konnon_draw.schema import validate_actions


def test_basic_svg():
    actions = [{"type": "MOVE", "x": 0.1, "y": 0.1}, {"type": "LINE", "x": 0.9, "y": 0.9}, {"type": "END"}]
    svg = render_svg(actions)
    assert "<svg" in svg and "<path" in svg and "M 25.60 25.60" in svg


def test_requires_end():
    with pytest.raises(ValueError):
        validate_actions([{"type": "MOVE", "x": 0.1, "y": 0.1}])


def test_pen_up_splits_paths():
    actions = [{"type": "MOVE", "x": 0.1, "y": 0.1}, {"type": "PEN_UP"},
               {"type": "MOVE", "x": 0.8, "y": 0.8}, {"type": "END"}]
    assert render_svg(actions).count("<path") == 2


def test_line_requires_move():
    with pytest.raises(ValueError):
        render_svg([{"type": "LINE", "x": 0.8, "y": 0.8}, {"type": "END"}])


def test_deterministic():
    a = [{"type": "MOVE", "x": 0.2, "y": 0.3}, {"type": "LINE", "x": 0.6, "y": 0.7}, {"type": "END"}]
    assert render_svg(a) == render_svg(a)


def test_output_is_only_trusted_markup():
    svg = render_svg([{"type": "MOVE", "x": 0.2, "y": 0.3}, {"type": "LINE", "x": 0.6, "y": 0.7}, {"type": "END"}])
    assert "<script" not in svg and svg.count("<path") == 1
