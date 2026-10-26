import pytest
from konnon_draw.schema import (COORD_BINS, dequantize_coord, is_valid, quantize_coord, validate_actions)

M = lambda x=0.1, y=0.1: {"type": "MOVE", "x": x, "y": y}
L = lambda x=0.5, y=0.5: {"type": "LINE", "x": x, "y": y}
END, PEN = {"type": "END"}, {"type": "PEN_UP"}


def test_valid_minimal():
    assert validate_actions([M(), L(), END])


@pytest.mark.parametrize("bad", [
    [],                                  # empty
    [M(), L()],                          # no END
    [M(), L(), END, END],                # two ENDs
    [END, M(), L()],                     # END not last
    [L(), END],                          # LINE without MOVE
    [PEN, END],                          # PEN_UP without path
    [M(), M(0.2, 0.2), L(), END],        # MOVE while path open
    [END],                               # no strokes
    [M(1.5, 0.1), L(), END],             # out of range
    [M(float("nan"), 0.1), L(), END],    # non-finite
    [{"type": "MOVE", "x": 0.1}, END],   # missing y
    [{"type": "CURVE"}, END],            # unsupported
])
def test_invalid(bad):
    with pytest.raises(ValueError):
        validate_actions(bad)
    assert not is_valid(bad)


def test_quantize_roundtrip_and_clamp():
    for b in range(COORD_BINS):
        assert quantize_coord(dequantize_coord(b)) == b
    assert quantize_coord(-3) == 0 and quantize_coord(9) == COORD_BINS - 1
