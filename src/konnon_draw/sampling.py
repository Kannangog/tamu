"""Constrained autoregressive sampling. Illegal transitions are masked, never repaired after the fact
(except an explicit, reported fallback when the length limit is hit)."""
import argparse
import json
import random
from pathlib import Path

import torch

from .encoding import decode_one
from .renderer import render_svg
from .schema import Action, COORD_BINS, NO_COORD, is_valid

A = Action
# Legal next actions given the previous one. START->MOVE only, a stroke needs >=1 LINE,
# and END is only possible after a LINE (so every drawing has at least one real segment).
LEGAL = {
    int(A.START): [int(A.MOVE)],
    int(A.MOVE): [int(A.LINE)],
    int(A.LINE): [int(A.LINE), int(A.PEN_UP), int(A.END)],
    int(A.PEN_UP): [int(A.MOVE)],
}


def _sample(logits, allowed, temperature, top_k, gen):
    """Sample an index from `logits` restricted to `allowed`."""
    idx = torch.tensor(allowed, dtype=torch.long)
    sub = logits[idx] / max(temperature, 1e-6)
    if top_k and top_k < len(allowed):
        kth = torch.topk(sub, top_k).values[-1]
        sub = sub.masked_fill(sub < kth, float("-inf"))
    probs = torch.softmax(sub, dim=-1)
    return int(idx[torch.multinomial(probs, 1, generator=gen)])


def _continuous_coord(logits, selected_bin, num_bins=COORD_BINS, window=2, temp=0.5):
    """Sub-bin continuous coordinate expectation in local neighborhood of selected bin."""
    low = max(0, selected_bin - window)
    high = min(num_bins - 1, selected_bin + window)
    sub = logits[low : high + 1] / max(temp, 1e-4)
    probs = torch.softmax(sub, dim=-1)
    indices = torch.arange(low, high + 1, dtype=torch.float32, device=logits.device)
    cont_bin = torch.sum(indices * probs).item()
    return max(0.0, min(1.0, cont_bin / (num_bins - 1)))


@torch.no_grad()
def generate(model, class_id, seed, max_len=128, temperature=1.0, top_k=None, continuous_coords=False):
    """Return dict(actions, class_id, natural_end, repaired, n_tokens, seed). `max_len` counts START and END."""
    model.eval()
    gen = torch.Generator().manual_seed(int(seed))
    ids, xs, ys = [int(A.START)], [NO_COORD], [NO_COORD]
    cont_xs, cont_ys = [], []
    hidden, natural, forced = None, False, False
    cls = torch.tensor([int(class_id)])
    coord_bins = list(range(COORD_BINS))
    while len(ids) < max_len:
        a_in = torch.tensor([[ids[-1]]]); x_in = torch.tensor([[xs[-1]]]); y_in = torch.tensor([[ys[-1]]])
        a_log, x_log, h, hidden = model(a_in, x_in, y_in, cls, hidden)
        allowed = LEGAL[ids[-1]]
        if len(ids) == max_len - 1:  # last slot is reserved for END
            if int(A.END) not in allowed:
                break
            allowed, forced = [int(A.END)], True  # model never got to choose END: this is a fallback
        a = _sample(a_log[0, -1], allowed, temperature, top_k, gen)
        if a == int(A.END):
            ids.append(a); xs.append(NO_COORD); ys.append(NO_COORD)
            natural = not forced
            break
        if a in (int(A.MOVE), int(A.LINE)):
            x = _sample(x_log[0, -1], coord_bins, temperature, top_k, gen)
            y_log = model.predict_y(h[:, -1], torch.tensor([x]))[0]
            y = _sample(y_log, coord_bins, temperature, top_k, gen)
            if continuous_coords:
                cont_x = _continuous_coord(x_log[0, -1], x)
                cont_y = _continuous_coord(y_log, y)
                cont_xs.append(cont_x)
                cont_ys.append(cont_y)
        else:
            x = y = NO_COORD
        ids.append(a); xs.append(x); ys.append(y)
    repaired = forced
    if not natural and not forced:  # explicit fallback: drop dangling MOVE/PEN_UP, then append END
        repaired = True
        while len(ids) > 1 and ids[-1] != int(A.LINE):
            ids.pop(); xs.pop(); ys.pop()
        ids.append(int(A.END)); xs.append(NO_COORD); ys.append(NO_COORD)
    actions = decode_one(ids, xs, ys)
    if continuous_coords and cont_xs and cont_ys:
        c_idx = 0
        for act in actions:
            if act["type"] in ("MOVE", "LINE") and c_idx < len(cont_xs):
                act["x"] = round(cont_xs[c_idx], 5)
                act["y"] = round(cont_ys[c_idx], 5)
                c_idx += 1
    return {"seed": int(seed), "class_id": int(class_id), "actions": actions, "natural_end": natural,
            "repaired": repaired, "n_tokens": len(ids)}


def random_baseline(seed, max_len=128):
    """Uniform-random action baseline for release gate 2."""
    rng = random.Random(seed)
    actions = []
    for _ in range(max_len):
        kind = rng.choice(["MOVE", "LINE", "PEN_UP", "END"])
        if kind in ("MOVE", "LINE"):
            actions.append({"type": kind, "x": rng.random(), "y": rng.random()})
        else:
            actions.append({"type": kind})
        if kind == "END":
            break
    return actions


def save_sample(out_dir, name, result):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    if is_valid(result["actions"]):
        (out_dir / f"{name}.svg").write_text(render_svg(result["actions"]), encoding="utf-8")


def main(argv=None):
    from .training import load_model
    p = argparse.ArgumentParser(description="Generate drawings from a checkpoint")
    p.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    p.add_argument("--category", default="all", help="a category name or 'all'")
    p.add_argument("--n", type=int, default=8, help="samples per category")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--out", default="outputs/samples/generated")
    args = p.parse_args(argv)
    model, ckpt = load_model(args.checkpoint)
    max_len = args.max_len or ckpt["config"]["max_len"]
    classes = ckpt["classes"]
    wanted = classes if args.category == "all" else [args.category]
    ok = total = 0
    for name in wanted:
        if name not in classes:
            raise SystemExit(f"Unknown category {name!r}. Available: {classes}")
        for i in range(args.n):
            r = generate(model, classes.index(name), args.seed + i, max_len, args.temperature, args.top_k)
            r["category"] = name; r["valid"] = is_valid(r["actions"]); ok += r["valid"]; total += 1
            save_sample(args.out, f"{name}_{i:03d}", r)
    print(f"Wrote {total} samples to {args.out} ({ok} valid)")


if __name__ == "__main__":
    main()
