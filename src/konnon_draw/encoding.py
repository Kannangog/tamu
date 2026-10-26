import torch
from .schema import Action, quantize_coord, dequantize_coord

NO_COORD = 32


def encode_one(actions):
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
            xs.append(quantize_coord(a["x"]))
            ys.append(quantize_coord(a["y"]))
        else:
            xs.append(NO_COORD)
            ys.append(NO_COORD)
    return ids, xs, ys


def decode_one(ids, xs, ys):
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
                "x": dequantize_coord(int(x)),
                "y": dequantize_coord(int(y)),
                "xq": int(x),
                "yq": int(y),
            })
        else:
            actions.append({"type": kind})
    if not actions or actions[-1]["type"] != "END":
        actions.append({"type": "END"})
    return actions
