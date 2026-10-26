"""Conditioned action and coordinate encoding with 64-bin resolution."""

from typing import Dict, List, Optional, Tuple
from .schema import Action, quantize_coord, dequantize_coord
from .quickdraw import CATEGORY_TO_ID, ID_TO_CATEGORY, DEFAULT_CATEGORIES

REAL_COORD_BINS = 64
REAL_NO_COORD = 64
REAL_COORD_VOCAB = 65  # 0..63 bins + 64 for NO_COORD
NUM_CATEGORIES = len(DEFAULT_CATEGORIES)


def encode_conditioned_one(actions: List[dict], category: str) -> Tuple[List[int], List[int], List[int], int]:
    ids, xs, ys = [], [], []
    mapping = {
        "MOVE": Action.MOVE,
        "LINE": Action.LINE,
        "PEN_UP": Action.PEN_UP,
        "END": Action.END,
    }

    for a in actions:
        kind = a["type"]
        ids.append(int(mapping[kind]))
        if kind in {"MOVE", "LINE"}:
            xs.append(quantize_coord(a["x"], bins=REAL_COORD_BINS))
            ys.append(quantize_coord(a["y"], bins=REAL_COORD_BINS))
        else:
            xs.append(REAL_NO_COORD)
            ys.append(REAL_NO_COORD)

    cat_id = CATEGORY_TO_ID.get(category, 0)
    return ids, xs, ys, cat_id


def decode_conditioned_one(ids: List[int], xs: List[int], ys: List[int], category_id: Optional[int] = None) -> List[dict]:
    reverse = {
        int(Action.MOVE): "MOVE",
        int(Action.LINE): "LINE",
        int(Action.PEN_UP): "PEN_UP",
        int(Action.END): "END",
    }
    actions = []
    for a, x, y in zip(ids, xs, ys):
        a = int(a)
        if a == int(Action.END):
            actions.append({"type": "END"})
            break
        if a not in reverse:
            continue
        kind = reverse[a]
        if kind in {"MOVE", "LINE"}:
            actions.append({
                "type": kind,
                "x": dequantize_coord(int(x), bins=REAL_COORD_BINS),
                "y": dequantize_coord(int(y), bins=REAL_COORD_BINS),
                "xq": int(x),
                "yq": int(y),
                "coord_bins": REAL_COORD_BINS,
            })
        else:
            actions.append({"type": kind})

    if not actions or actions[-1]["type"] != "END":
        actions.append({"type": "END"})

    return actions
