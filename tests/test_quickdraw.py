import pytest
from konnon_draw.quickdraw import quickdraw_strokes_to_actions
from konnon_draw.schema import validate_actions


def test_quickdraw_conversion():
    # A single square doodle: 4 points
    # QuickDraw format: [[[x0, x1, x2, x3], [y0, y1, y2, y3]]]
    drawing = [
        [[10, 50, 50, 10], [10, 10, 50, 50]],
    ]

    actions = quickdraw_strokes_to_actions(drawing)
    assert len(actions) == 5  # MOVE, 3 LINEs, END
    assert actions[0]["type"] == "MOVE"
    assert actions[-1]["type"] == "END"
    assert validate_actions(actions)

    for a in actions:
        if a["type"] in ("MOVE", "LINE"):
            assert 0.0 <= a["x"] <= 1.0
            assert 0.0 <= a["y"] <= 1.0


def test_quickdraw_empty_drawing_raises():
    with pytest.raises(ValueError):
        quickdraw_strokes_to_actions([])
