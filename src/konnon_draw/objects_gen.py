"""Procedural everyday-object drawings (bicycle, bed, chair, table, house, car).

Self-made training data: no licence issues, zero download. Each object is built from
parameterised strokes with random proportions, mirroring, scale and shift, so the model
sees real variation. This teaches clean line-drawing structure; it does NOT capture the
variety of human doodles (use quickdraw.py for that).
"""
import argparse
import json
import math
import random
from pathlib import Path

from .data_gen import drawing
from .schema import validate_actions

OUT = Path("data/processed/objects.jsonl")
CATEGORIES = ["bicycle", "bed", "chair", "table", "house", "car"]


def circle(cx, cy, r, rng, n=20):
    a0 = rng.uniform(0, 2 * math.pi)
    pts = [(cx + r * math.cos(a0 + 2 * math.pi * k / n), cy + r * math.sin(a0 + 2 * math.pi * k / n))
           for k in range(n)]
    return pts + [pts[0]]


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]


def bicycle(rng):
    r = rng.uniform(0.15, 0.2)
    yw = 0.68
    xr, xf = 0.5 - rng.uniform(0.25, 0.3), 0.5 + rng.uniform(0.25, 0.3)
    bb = (rng.uniform(0.45, 0.52), yw)                       # bottom bracket
    seat = (bb[0] - rng.uniform(0.06, 0.1), yw - rng.uniform(0.28, 0.34))
    head = (xf - rng.uniform(0.04, 0.07), yw - rng.uniform(0.3, 0.36))
    rear, front = (xr, yw), (xf, yw)
    return [
        circle(xr, yw, r, rng), circle(xf, yw, r, rng),
        [rear, bb, seat, rear],
        [seat, head, bb],
        [front, head],
        [seat, (seat[0] - 0.01, seat[1] - 0.05)],
        [(seat[0] - 0.06, seat[1] - 0.05), (seat[0] + 0.04, seat[1] - 0.05)],
        [head, (head[0] - 0.02, head[1] - 0.07), (head[0] + 0.05, head[1] - 0.08)],
    ]


def bed(rng):
    x0, x1 = rng.uniform(0.1, 0.16), rng.uniform(0.84, 0.9)
    yf = rng.uniform(0.74, 0.8)
    yb = yf - rng.uniform(0.07, 0.1)
    ym = yb - rng.uniform(0.09, 0.13)
    yh = ym - rng.uniform(0.16, 0.24)
    pw = rng.uniform(0.16, 0.22)
    return [
        rect(x0, yb, x1, yb + 0.05),
        rect(x0 + 0.05, ym, x1 - 0.02, yb),
        [(x0, yb + 0.05), (x0, yh), (x0 + 0.05, yh), (x0 + 0.05, yb)],
        [(x1, yb + 0.05), (x1, ym + 0.02), (x1 - 0.02, ym + 0.02)],
        [(x0 + 0.01, yb + 0.05), (x0 + 0.01, yf)],
        [(x1 - 0.01, yb + 0.05), (x1 - 0.01, yf)],
        rect(x0 + 0.07, ym - 0.05, x0 + 0.07 + pw, ym),
        [(x0 + 0.07 + pw + 0.12, yb), (x0 + 0.07 + pw + 0.12, ym)],
    ]


def chair(rng):
    xl, xr = 0.5 - rng.uniform(0.14, 0.2), 0.5 + rng.uniform(0.14, 0.2)
    ys = rng.uniform(0.5, 0.56)
    yt = ys - rng.uniform(0.3, 0.38)
    yf = rng.uniform(0.86, 0.92)
    return [
        [(xl, ys), (xl, yt), (xr, yt), (xr, ys)],
        [(xl, (ys + yt) / 2), (xr, (ys + yt) / 2)],
        [(xl - 0.03, ys), (xr + 0.03, ys), (xr + 0.03, ys + 0.06), (xl - 0.03, ys + 0.06), (xl - 0.03, ys)],
        [(xl - 0.01, ys + 0.06), (xl - 0.01, yf)],
        [(xr + 0.01, ys + 0.06), (xr + 0.01, yf)],
        [(xl + 0.03, ys + 0.06), (xl + 0.03, yf - 0.04)],
        [(xr - 0.03, ys + 0.06), (xr - 0.03, yf - 0.04)],
    ]


def table(rng):
    xl, xr = 0.5 - rng.uniform(0.3, 0.38), 0.5 + rng.uniform(0.3, 0.38)
    yt = rng.uniform(0.35, 0.42)
    yf = rng.uniform(0.84, 0.9)
    th = rng.uniform(0.05, 0.08)
    inset = rng.uniform(0.04, 0.08)
    strokes = [
        rect(xl, yt, xr, yt + th),
        [(xl + inset, yt + th), (xl + inset, yf)],
        [(xr - inset, yt + th), (xr - inset, yf)],
    ]
    if rng.random() < 0.6:
        yc = yf - rng.uniform(0.15, 0.22)
        strokes.append([(xl + inset, yc), (xr - inset, yc)])
    return strokes


def house(rng):
    xl, xr = 0.5 - rng.uniform(0.25, 0.32), 0.5 + rng.uniform(0.25, 0.32)
    yb = rng.uniform(0.82, 0.88)
    yw = yb - rng.uniform(0.3, 0.36)
    peak = (0.5 + rng.uniform(-0.05, 0.05), yw - rng.uniform(0.16, 0.24))
    dw = rng.uniform(0.07, 0.1)
    dx = rng.uniform(0.38, 0.5)
    wx = rng.uniform(xl + 0.05, dx - 0.12) if dx - 0.12 > xl + 0.05 else xl + 0.05
    ws = rng.uniform(0.07, 0.1)
    return [
        rect(xl, yw, xr, yb),
        [(xl - 0.03, yw), peak, (xr + 0.03, yw), (xl - 0.03, yw)],
        rect(dx, yb - rng.uniform(0.16, 0.22), dx + dw * 1.4, yb),
        rect(wx, yw + 0.06, wx + ws, yw + 0.06 + ws),
        rect(xr - 0.05 - ws, yw + 0.06, xr - 0.05, yw + 0.06 + ws),
    ]


def car(rng):
    x0, x1 = rng.uniform(0.08, 0.14), rng.uniform(0.86, 0.92)
    yb = rng.uniform(0.64, 0.68)
    ym = yb - rng.uniform(0.12, 0.16)
    yt = ym - rng.uniform(0.13, 0.18)
    r = rng.uniform(0.07, 0.09)
    wxl, wxr = x0 + rng.uniform(0.18, 0.22), x1 - rng.uniform(0.18, 0.22)
    cl, cr = x0 + rng.uniform(0.24, 0.3), x1 - rng.uniform(0.24, 0.3)
    return [
        [(x0, yb), (x0, ym), (x0 + 0.1, ym), (cl, yt), (cr, yt), (x1 - 0.1, ym), (x1, ym), (x1, yb), (x0, yb)],
        circle(wxl, yb, r, rng, 16), circle(wxr, yb, r, rng, 16),
        [(cl + 0.03, ym), (cl + 0.06, yt + 0.03), (0.5, yt + 0.03), (0.5, ym), (cl + 0.03, ym)],
        [(0.5 + 0.03, ym), (0.5 + 0.03, yt + 0.03), (cr - 0.06, yt + 0.03), (cr - 0.03, ym), (0.5 + 0.03, ym)],
    ]


BUILDERS = {"bicycle": bicycle, "bed": bed, "chair": chair, "table": table, "house": house, "car": car}


def _transform(strokes, rng, flip, scale, dx, dy, jitter=0.003):
    out = []
    for s in strokes:
        pts = []
        for x, y in s:
            if flip:
                x = 1 - x
            x = 0.5 + (x - 0.5) * scale + dx + rng.uniform(-jitter, jitter)
            y = 0.5 + (y - 0.5) * scale + dy + rng.uniform(-jitter, jitter)
            pts.append((max(0.02, min(0.98, x)), max(0.02, min(0.98, y))))
        out.append(pts)
    return out


def make_object(category, rng):
    strokes = BUILDERS[category](rng)
    strokes = _transform(strokes, rng, flip=rng.random() < 0.5 and category in ("bicycle", "bed", "car"),
                         scale=rng.uniform(0.85, 1.0), dx=rng.uniform(-0.04, 0.04), dy=rng.uniform(-0.04, 0.04))
    return drawing(strokes)


def make_examples(per_class=800, seed=11, categories=CATEGORIES):
    rng = random.Random(seed)
    examples = []
    for i in range(per_class):
        for c in categories:
            actions = make_object(c, rng)
            validate_actions(actions)
            examples.append({"id": f"{c}_{i}", "category": c, "actions": actions})
    return examples


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--per-class", type=int, default=800)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ex = make_examples(a.per_class, a.seed)
    with a.out.open("w", encoding="utf-8") as f:
        for e in ex:
            f.write(json.dumps(e) + "\n")
    lens = [len(e["actions"]) for e in ex]
    print(f"Wrote {len(ex)} examples ({len(CATEGORIES)} classes) to {a.out}; actions per drawing: "
          f"mean {sum(lens)/len(lens):.0f}, max {max(lens)}")


if __name__ == "__main__":
    main()
