"""Tests for v2 schema, color parsing, validation, and v1->v2 conversion."""
import pytest
from konnon_draw.schema_v2 import (
    DocumentV2,
    FillActionV2,
    PointV2,
    StrokeActionV2,
    hex_to_rgb,
    hsv_to_rgb,
    parse_color,
    rgb_to_hex,
    v1_to_v2,
    validate_document_v2,
)


def test_color_parsers():
    # Hex
    r, g, b = hex_to_rgb("#ff0000")
    assert abs(r - 1.0) < 1e-4 and g == 0.0 and b == 0.0
    assert rgb_to_hex(1.0, 0.0, 0.0).lower() == "#ff0000"

    # HSV
    r, g, b = hsv_to_rgb(0, 1.0, 1.0)  # Pure red
    assert abs(r - 1.0) < 1e-4 and g == 0.0 and b == 0.0

    # Parse color versatile
    assert parse_color("#3b82f6") == hex_to_rgb("#3b82f6")
    assert parse_color([255, 0, 0]) == (1.0, 0.0, 0.0)
    assert parse_color({"h": 120, "s": 1.0, "v": 1.0})[1] == 1.0


def test_document_v2_validation():
    doc = DocumentV2()
    # Add a stroke
    doc.commands.append(
        StrokeActionV2(
            brush_id="pencil",
            color="#1e293b",
            base_width=2.5,
            points=[
                PointV2(x=0.1, y=0.1, p=0.5),
                PointV2(x=0.5, y=0.5, p=1.0),
                PointV2(x=0.8, y=0.9, p=0.4),
            ],
        )
    )
    # Add a fill
    doc.commands.append(
        FillActionV2(
            fill_type="linear_gradient",
            path=[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)],
            color="#38bdf8",
            color_end="#6366f1",
        )
    )

    assert validate_document_v2(doc) is True


def test_invalid_document_raises():
    doc = DocumentV2()
    # Stroke with only 1 point
    doc.commands.append(StrokeActionV2(points=[PointV2(x=0.1, y=0.1)]))
    with pytest.raises(ValueError):
        validate_document_v2(doc)


def test_v1_to_v2_conversion():
    v1_actions = [
        {"type": "MOVE", "x": 0.2, "y": 0.2},
        {"type": "LINE", "x": 0.5, "y": 0.3},
        {"type": "LINE", "x": 0.8, "y": 0.8},
        {"type": "PEN_UP"},
        {"type": "MOVE", "x": 0.1, "y": 0.9},
        {"type": "LINE", "x": 0.9, "y": 0.9},
        {"type": "END"},
    ]

    doc_v2 = v1_to_v2(v1_actions, default_brush="ink_liner", base_width=4.0)
    assert validate_document_v2(doc_v2) is True
    assert len(doc_v2.commands) == 2  # Two strokes separated by PEN_UP

    # First stroke has 3 points
    s1 = doc_v2.commands[0]
    assert len(s1.points) == 3
    # Check synthetic pressure bell curve
    assert s1.points[1].p > s1.points[0].p
    assert s1.points[1].p > s1.points[2].p
