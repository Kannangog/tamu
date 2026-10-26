import pytest
from konnon_draw.schema import (
    Action,
    CANVAS_SIZE,
    COORD_BINS,
    quantize_coord,
    dequantize_coord,
    validate_actions,
)


def test_quantization_round_trip_extremes():
    assert quantize_coord(0.0) == 0
    assert quantize_coord(1.0) == COORD_BINS - 1
    assert quantize_coord(-0.5) == 0
    assert quantize_coord(1.5) == COORD_BINS - 1

    assert dequantize_coord(0) == 0.0
    assert dequantize_coord(COORD_BINS - 1) == 1.0


def test_empty_actions_raises():
    with pytest.raises(ValueError, match="Drawing must contain actions"):
        validate_actions([])


def test_missing_end_raises():
    with pytest.raises(ValueError, match="Drawing must end with END"):
        validate_actions([{"type": "MOVE", "x": 0.5, "y": 0.5}])


def test_multiple_ends_raises():
    with pytest.raises(ValueError, match="Drawing must contain exactly one END"):
        validate_actions([
            {"type": "MOVE", "x": 0.5, "y": 0.5},
            {"type": "END"},
            {"type": "END"},
        ])


def test_unsupported_action_raises():
    with pytest.raises(ValueError, match="Unsupported action: CIRCLE"):
        validate_actions([
            {"type": "CIRCLE", "x": 0.5, "y": 0.5},
            {"type": "END"},
        ])


def test_out_of_range_coordinates_raises():
    with pytest.raises(ValueError, match="Invalid x for MOVE"):
        validate_actions([
            {"type": "MOVE", "x": 1.5, "y": 0.5},
            {"type": "END"},
        ])

    with pytest.raises(ValueError, match="Invalid y for LINE"):
        validate_actions([
            {"type": "MOVE", "x": 0.5, "y": 0.5},
            {"type": "LINE", "x": 0.5, "y": -0.1},
            {"type": "END"},
        ])


def test_missing_coordinate_key_raises():
    with pytest.raises(ValueError, match="Invalid y for MOVE"):
        validate_actions([
            {"type": "MOVE", "x": 0.5},
            {"type": "END"},
        ])
