from enum import IntEnum

CANVAS_SIZE = 256
COORD_BINS = 32


class Action(IntEnum):
    PAD = 0
    MOVE = 1
    LINE = 2
    PEN_UP = 3
    END = 4


def quantize_coord(value: float) -> int:
    value = max(0.0, min(1.0, float(value)))
    return min(COORD_BINS - 1, int(round(value * (COORD_BINS - 1))))


def dequantize_coord(value: int) -> float:
    value = max(0, min(COORD_BINS - 1, int(value)))
    return value / (COORD_BINS - 1)


def validate_actions(actions):
    if not actions:
        raise ValueError("Drawing must contain actions")
    if actions[-1]["type"] != "END":
        raise ValueError("Drawing must end with END")
    if sum(a["type"] == "END" for a in actions) != 1:
        raise ValueError("Drawing must contain exactly one END")
    for a in actions:
        kind = a["type"]
        if kind not in {"MOVE", "LINE", "PEN_UP", "END"}:
            raise ValueError(f"Unsupported action: {kind}")
        if kind in {"MOVE", "LINE"}:
            for key in ("x", "y"):
                if key not in a or not (0.0 <= float(a[key]) <= 1.0):
                    raise ValueError(f"Invalid {key} for {kind}")
    return True
