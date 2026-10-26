"""Synthetic toy dataset: triangle / rectangle / cross / zigzag with jitter."""
import argparse
import json
import random
from pathlib import Path

from .schema import validate_actions

OUT = Path("data/processed/toy_drawings.jsonl")
CATEGORIES = ["triangle", "rectangle", "cross", "zigzag"]


def drawing(points_list):
    actions = []
    for points in points_list:
        if not points:
            continue
        actions.append({"type": "MOVE", "x": points[0][0], "y": points[0][1]})
        for x, y in points[1:]:
            actions.append({"type": "LINE", "x": x, "y": y})
        actions.append({"type": "PEN_UP"})
    if actions and actions[-1]["type"] == "PEN_UP":
        actions.pop()
    actions.append({"type": "END"})
    return actions


def jitter(points, rng, amount=0.025):
    return [
        (
            max(0.03, min(0.97, x + rng.uniform(-amount, amount))),
            max(0.03, min(0.97, y + rng.uniform(-amount, amount))),
        )
        for x, y in points
    ]


def make_examples(n=600, seed=7):
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        cx, cy = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65)
        size = rng.uniform(0.15, 0.3)
        kind = i % 4
        if kind == 0:
            strokes = [[(cx, cy - size), (cx + size, cy + size), (cx - size, cy + size), (cx, cy - size)]]
        elif kind == 1:
            strokes = [[(cx - size, cy - size), (cx + size, cy - size),
                        (cx + size, cy + size), (cx - size, cy + size), (cx - size, cy - size)]]
        elif kind == 2:
            strokes = [[(cx - size, cy - size), (cx + size, cy + size)],
                       [(cx - size, cy + size), (cx + size, cy - size)]]
        else:
            strokes = [[(cx - size, cy), (cx - size / 2, cy - size), (cx, cy + size),
                        (cx + size / 2, cy - size), (cx + size, cy)]]
        actions = drawing([jitter(s, rng) for s in strokes])
        validate_actions(actions)
        examples.append({"id": f"toy_{i}", "category": CATEGORIES[kind], "actions": actions})
    return examples


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    examples = make_examples(args.n, args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
