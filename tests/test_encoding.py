import pytest
from konnon_draw.encoding import NO_COORD, decode_one, encode_one
from konnon_draw.schema import Action, dequantize_coord, quantize_coord


def test_encode_decode_round_trip():
    original_actions = [
        {"type": "MOVE", "x": 0.25, "y": 0.5},
        {"type": "LINE", "x": 0.75, "y": 0.5},
        {"type": "PEN_UP"},
        {"type": "MOVE", "x": 0.5, "y": 0.1},
        {"type": "LINE", "x": 0.5, "y": 0.9},
        {"type": "END"},
    ]

    ids, xs, ys = encode_one(original_actions)
    assert len(ids) == len(original_actions)
    assert ids[0] == int(Action.MOVE)
    assert ids[1] == int(Action.LINE)
    assert ids[2] == int(Action.PEN_UP)
    assert ids[3] == int(Action.MOVE)
    assert ids[4] == int(Action.LINE)
    assert ids[5] == int(Action.END)

    assert xs[2] == NO_COORD
    assert ys[2] == NO_COORD
    assert xs[5] == NO_COORD
    assert ys[5] == NO_COORD

    decoded_actions = decode_one(ids, xs, ys)
    assert len(decoded_actions) == len(original_actions)
    for orig, dec in zip(original_actions, decoded_actions):
        assert orig["type"] == dec["type"]
        if orig["type"] in ("MOVE", "LINE"):
            expected_x = dequantize_coord(quantize_coord(orig["x"]))
            expected_y = dequantize_coord(quantize_coord(orig["y"]))
            assert pytest.approx(dec["x"], abs=1e-4) == expected_x
            assert pytest.approx(dec["y"], abs=1e-4) == expected_y


def test_decode_terminates_at_first_end():
    ids = [int(Action.MOVE), int(Action.END), int(Action.LINE)]
    xs = [10, NO_COORD, 20]
    ys = [10, NO_COORD, 20]

    decoded = decode_one(ids, xs, ys)
    assert len(decoded) == 2
    assert decoded[0]["type"] == "MOVE"
    assert decoded[1]["type"] == "END"


def test_decode_appends_end_if_missing():
    ids = [int(Action.MOVE), int(Action.LINE)]
    xs = [5, 15]
    ys = [5, 15]

    decoded = decode_one(ids, xs, ys)
    assert decoded[-1]["type"] == "END"
