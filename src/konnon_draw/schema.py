"""Action schema contract (v0.1.1). Keep stable while a model is being trained.

v0.2: 64 coordinate bins (was 32).
Changes vs. the original guide: a START token (ID 5) so the model can learn the
first MOVE and its coordinates, and a stricter validator (finite coordinates,
no LINE/PEN_UP without an open path, at least one stroke).
"""
import math
from enum import IntEnum

SCHEMA_VERSION = "0.2.0"
CANVAS_SIZE = 256
COORD_BINS = 64
NO_COORD = COORD_BINS  # reserved x/y token for actions without coordinates


class Action(IntEnum):
    PAD = 0
    MOVE = 1
    LINE = 2
    PEN_UP = 3
    END = 4
    START = 5  # input-only: prepended to every sequence, never rendered


ACTION_VOCAB = len(Action)
COORD_VOCAB = COORD_BINS + 1
ID_TO_NAME = {
    int(Action.MOVE): "MOVE",
    int(Action.LINE): "LINE",
    int(Action.PEN_UP): "PEN_UP",
    int(Action.END): "END",
}
NAME_TO_ID = {v: k for k, v in ID_TO_NAME.items()}


def quantize_coord(value: float) -> int:
    value = max(0.0, min(1.0, float(value)))
    return min(COORD_BINS - 1, int(round(value * (COORD_BINS - 1))))


def dequantize_coord(value: int) -> float:
    value = max(0, min(COORD_BINS - 1, int(value)))
    return value / (COORD_BINS - 1)


def validate_actions(actions):
    """Raise ValueError unless `actions` is a structurally valid drawing."""
    if not actions:
        raise ValueError("Drawing must contain actions")
    if actions[-1].get("type") != "END":
        raise ValueError("Drawing must end with END")
    if sum(a.get("type") == "END" for a in actions) != 1:
        raise ValueError("Drawing must contain exactly one END")
    path_open = False
    strokes = 0
    for a in actions:
        kind = a.get("type")
        if kind not in NAME_TO_ID:
            raise ValueError(f"Unsupported action: {kind}")
        if kind in ("MOVE", "LINE"):
            for key in ("x", "y"):
                if key not in a:
                    raise ValueError(f"Missing {key} for {kind}")
                v = float(a[key])
                if not math.isfinite(v) or not (0.0 <= v <= 1.0):
                    raise ValueError(f"Invalid {key} for {kind}")
        if kind == "MOVE":
            if path_open:
                raise ValueError("MOVE while a path is open (missing PEN_UP)")
            path_open = True
            strokes += 1
        elif kind == "LINE":
            if not path_open:
                raise ValueError("LINE requires a preceding MOVE")
        elif kind == "PEN_UP":
            if not path_open:
                raise ValueError("PEN_UP without an open path")
            path_open = False
    if strokes == 0:
        raise ValueError("Drawing must contain at least one stroke")
    return True


def is_valid(actions) -> bool:
    try:
        validate_actions(actions)
        return True
    except (ValueError, TypeError, AttributeError):
        return False
