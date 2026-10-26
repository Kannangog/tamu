from xml.sax.saxutils import escape
from .schema import validate_actions, dequantize_coord, COORD_BINS, CANVAS_SIZE


def render_svg(actions, width=256, height=256):
    validate_actions(actions)
    paths = []
    current = []
    for a in actions:
        kind = a["type"]
        if kind == "MOVE":
            if current:
                paths.append(current)
            x = dequantize_coord(a["xq"]) if "xq" in a else float(a["x"])
            y = dequantize_coord(a["yq"]) if "yq" in a else float(a["y"])
            current = [(x * width, y * height)]
        elif kind == "LINE":
            if not current:
                raise ValueError("LINE requires a preceding MOVE")
            x = dequantize_coord(a["xq"]) if "xq" in a else float(a["x"])
            y = dequantize_coord(a["yq"]) if "yq" in a else float(a["y"])
            current.append((x * width, y * height))
        elif kind == "PEN_UP":
            if current:
                paths.append(current)
            current = []
        elif kind == "END":
            break
    if current:
        paths.append(current)
    elements = []
    for points in paths:
        if not points:
            continue
        d = f"M {points[0][0]:.2f} {points[0][1]:.2f}"
        for x, y in points[1:]:
            d += f" L {x:.2f} {y:.2f}"
        elements.append(
            f'<path d="{d}" fill="none" stroke="#111827" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    content = "\n".join(elements)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'{content}</svg>'
    )
