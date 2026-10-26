import pytest
from konnon_draw.renderer import render_svg
from konnon_draw.schema import validate_actions


def test_basic_svg():
    actions = [
        {"type": "MOVE", "x": 0.1, "y": 0.1},
        {"type": "LINE", "x": 0.9, "y": 0.9},
        {"type": "END"},
    ]
    svg = render_svg(actions)
    assert "<svg" in svg
    assert "<path" in svg
    assert "M 25.60 25.60" in svg


def test_requires_end():
    with pytest.raises(ValueError):
        validate_actions([{"type": "MOVE", "x": 0.1, "y": 0.1}])


def test_pen_up_splits_paths():
    actions = [
        {"type": "MOVE", "x": 0.1, "y": 0.1},
        {"type": "PEN_UP"},
        {"type": "MOVE", "x": 0.8, "y": 0.8},
        {"type": "END"},
    ]
    svg = render_svg(actions)
    assert svg.count("<path") == 2


def test_line_requires_move():
    actions = [
        {"type": "LINE", "x": 0.8, "y": 0.8},
        {"type": "END"},
    ]
    with pytest.raises(ValueError):
        render_svg(actions)


def test_multiple_lines_in_single_path():
    actions = [
        {"type": "MOVE", "x": 0.1, "y": 0.1},
        {"type": "LINE", "x": 0.2, "y": 0.3},
        {"type": "LINE", "x": 0.4, "y": 0.5},
        {"type": "END"},
    ]
    svg = render_svg(actions)
    assert svg.count("<path") == 1
    assert "M 25.60 25.60" in svg
    assert "L 51.20 76.80" in svg
    assert "L 102.40 128.00" in svg


def test_quantized_coordinates_rendering():
    actions = [
        {"type": "MOVE", "x": 0.0, "y": 0.0, "xq": 0, "yq": 0},
        {"type": "LINE", "x": 1.0, "y": 1.0, "xq": 31, "yq": 31},
        {"type": "END"},
    ]
    svg = render_svg(actions)
    assert "<svg" in svg
    assert "M 0.00 0.00" in svg
    assert "L 256.00 256.00" in svg
