"""Deterministic SVG renderer. Model output is data; only this trusted code emits markup."""
from .schema import validate_actions


def actions_to_paths(actions):
    """Return a list of paths; each path is a list of normalized (x, y) points."""
    validate_actions(actions)
    paths, current = [], []
    for a in actions:
        kind = a["type"]
        if kind == "MOVE":
            current = [(float(a["x"]), float(a["y"]))]
        elif kind == "LINE":
            current.append((float(a["x"]), float(a["y"])))
        elif kind == "PEN_UP":
            paths.append(current)
            current = []
        elif kind == "END":
            if current:
                paths.append(current)
                current = []
            break
    return paths


def render_svg(actions, width=256, height=256, stroke="#111827", stroke_width=2):
    paths = actions_to_paths(actions)
    elements = []
    for points in paths:
        d = f"M {points[0][0] * width:.2f} {points[0][1] * height:.2f}"
        for x, y in points[1:]:
            d += f" L {x * width:.2f} {y * height:.2f}"
        elements.append(
            f'<path d="{d}" fill="none" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    content = "\n".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f"{content}</svg>"
    )
