import argparse
import json
import random
from pathlib import Path

OUT = Path("data/processed/toy_drawings.jsonl")


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


def jitter(points, amount=0.025):
    return [
        (
            max(0.03, min(0.97, x + random.uniform(-amount, amount))),
            max(0.03, min(0.97, y + random.uniform(-amount, amount))),
        )
        for x, y in points
    ]


def make_examples(n=600, seed=7):
    random.seed(seed)
    examples = []
    for i in range(n):
        cx, cy = random.uniform(0.35, 0.65), random.uniform(0.35, 0.65)
        size = random.uniform(0.15, 0.3)
        kind = i % 4
        if kind == 0:
            p = [(cx, cy - size), (cx + size, cy + size), (cx - size, cy + size), (cx, cy - size)]
        elif kind == 1:
            p = [
                (cx - size, cy - size),
                (cx + size, cy - size),
                (cx + size, cy + size),
                (cx - size, cy + size),
                (cx - size, cy - size),
            ]
        elif kind == 2:
            p = [(cx - size, cy - size), (cx + size, cy + size)]
            q = [(cx - size, cy + size), (cx + size, cy - size)]
            examples.append({
                "id": f"toy_{i}",
                "category": "cross",
                "actions": drawing([jitter(p), jitter(q)]),
            })
            continue
        else:
            p = [
                (cx - size, cy),
                (cx - size / 2, cy - size),
                (cx, cy + size),
                (cx + size / 2, cy - size),
                (cx + size, cy),
            ]
        examples.append({
            "id": f"toy_{i}",
            "category": ["triangle", "rectangle", "cross", "zigzag"][kind],
            "actions": drawing([jitter(p)]),
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic toy drawings dataset.")
    parser.add_argument("--out", type=Path, default=OUT, help="Output path for JSONL file")
    parser.add_argument("--n", type=int, default=600, help="Number of examples")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    examples = make_examples(n=args.n, seed=args.seed)
    with args.out.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {args.out}")


if __name__ == "__main__":
    main()
