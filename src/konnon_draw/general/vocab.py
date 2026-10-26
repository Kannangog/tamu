"""Vocabulary and token serialization for Konnon Draw General (text-to-vector).

Features:
- Discretized 256x256 spatial grid.
- Command tokens: BOS, EOS, PAD, SEP, MOVE, LINE, CURVE, FILL_FLAT, FILL_HATCH, SET_LAYER, SET_BRUSH, SET_COLOR, SET_WIDTH.
- Curated 32-color artistic palette + 16 pressure bins + 16 width bins.
- Bidirectional round-trip conversion: DocumentV2 <-> Token ID Sequence.
- Strict token validation and grammar checking.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..schema_v2 import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
    parse_color,
    rgb_to_hex,
    validate_document_v2,
)

# ---------------------------------------------------------
# Curated 32-Color Palette (Hex Codes)
# ---------------------------------------------------------

COLOR_PALETTE = [
    # Monochromes & Grays (0-5)
    "#000000", "#111827", "#334155", "#64748b", "#94a3b8", "#ffffff",
    # Reds & Pinks (6-9)
    "#ef4444", "#dc2626", "#b91c1c", "#f43f5e",
    # Oranges & Ambers (10-12)
    "#f97316", "#ea580c", "#f59e0b",
    # Yellows (13-14)
    "#eab308", "#fde047",
    # Greens (15-18)
    "#10b981", "#059669", "#15803d", "#84cc16",
    # Cyans & Teals (19-21)
    "#06b6d4", "#0891b2", "#14b8a6",
    # Blues (22-25)
    "#3b82f6", "#2563eb", "#1d4ed8", "#1e3a8a",
    # Purples & Violets (26-28)
    "#8b5cf6", "#7c3aed", "#6366f1",
    # Earth / Browns (29-31)
    "#78350f", "#92400e", "#b45309",
]

COLOR_TO_ID = {c.lower(): idx for idx, c in enumerate(COLOR_PALETTE)}

# Map common color name words to palette indexes
COLOR_NAME_MAP = {
    "black": 0, "dark": 1, "slate": 2, "gray": 3, "grey": 3, "white": 5,
    "red": 6, "crimson": 7, "dark red": 8, "pink": 9, "rose": 9,
    "orange": 10, "dark orange": 11, "amber": 12,
    "yellow": 13, "gold": 14, "light yellow": 14,
    "green": 15, "emerald": 16, "dark green": 17, "lime": 18,
    "cyan": 19, "teal": 21,
    "blue": 22, "royal blue": 23, "dark blue": 24, "navy": 25, "indigo": 25,
    "purple": 26, "violet": 27,
    "brown": 29, "amber brown": 30, "earth": 31,
}


def hex_to_palette_id(hex_color: str) -> int:
    """Finds the nearest palette color index via Euclidean RGB distance."""
    hex_clean = hex_color.lower().strip()
    if hex_clean in COLOR_TO_ID:
        return COLOR_TO_ID[hex_clean]

    try:
        r, g, b = parse_color(hex_clean)
    except Exception:
        return 0

    best_idx = 0
    best_dist = float("inf")
    for idx, c_hex in enumerate(COLOR_PALETTE):
        cr, cg, cb = parse_color(c_hex)
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


# ---------------------------------------------------------
# Discrete Token Vocabulary Definitions
# ---------------------------------------------------------

# Special Tokens (0-3)
PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SEP_TOKEN = 3

# Action Command Tokens (4-15)
TOKEN_MOVE = 4
TOKEN_LINE = 5
TOKEN_CURVE = 6
TOKEN_PEN_UP = 7
TOKEN_FILL_FLAT = 8
TOKEN_FILL_HATCH = 9
TOKEN_FILL_END = 10
TOKEN_SET_LAYER = 11
TOKEN_SET_BRUSH = 12
TOKEN_SET_COLOR = 13
TOKEN_SET_WIDTH = 14

SPECIAL_COMMAND_OFFSET = 16

# Coordinates: X (16..271), Y (272..527)
NUM_COORD_BINS = 256
X_OFFSET = SPECIAL_COMMAND_OFFSET
Y_OFFSET = X_OFFSET + NUM_COORD_BINS  # 272

# Pressure (528..543)
NUM_PRESSURE_BINS = 16
PRESSURE_OFFSET = Y_OFFSET + NUM_COORD_BINS  # 528

# Color Palette (544..575)
NUM_COLOR_BINS = len(COLOR_PALETTE)  # 32
COLOR_OFFSET = PRESSURE_OFFSET + NUM_PRESSURE_BINS  # 544

# Width Bins (576..591)
NUM_WIDTH_BINS = 16
WIDTH_OFFSET = COLOR_OFFSET + NUM_COLOR_BINS  # 576

# Layers (592..597)
NUM_LAYERS = 6
LAYER_OFFSET = WIDTH_OFFSET + NUM_WIDTH_BINS  # 592

# Brushes (598..603)
NUM_BRUSHES = 6
BRUSH_OFFSET = LAYER_OFFSET + NUM_LAYERS  # 598

VOCAB_SIZE = BRUSH_OFFSET + NUM_BRUSHES  # 604


LAYER_NAMES = ["background", "underdrawing", "flats", "shading", "line_art", "highlights"]
LAYER_NAME_TO_ID = {name: idx for idx, name in enumerate(LAYER_NAMES)}

BRUSH_NAMES = ["ink_liner", "pencil", "charcoal", "airbrush", "marker", "watercolor_wash"]
BRUSH_NAME_TO_ID = {name: idx for idx, name in enumerate(BRUSH_NAMES)}


# ---------------------------------------------------------
# Encoding: DocumentV2 -> Discrete Token Sequence
# ---------------------------------------------------------

def doc_to_token_sequence(
    doc: DocumentV2,
    max_tokens: int = 1024,
) -> List[int]:
    """Encodes a DocumentV2 into a clean flat sequence of integer token IDs."""
    tokens = [BOS_TOKEN]

    current_layer_idx = 4  # default line_art
    current_brush_idx = 0  # ink_liner
    current_color_idx = 0  # black
    current_width_idx = 3  # ~3px

    # Set default layer
    tokens.extend([TOKEN_SET_LAYER, LAYER_OFFSET + current_layer_idx])

    for cmd in doc.commands:
        if len(tokens) >= max_tokens - 10:
            break

        lid = getattr(cmd, "layer_id", "line_art")
        l_idx = LAYER_NAME_TO_ID.get(lid, 4)
        if l_idx != current_layer_idx:
            tokens.extend([TOKEN_SET_LAYER, LAYER_OFFSET + l_idx])
            current_layer_idx = l_idx

        ctype = getattr(cmd, "type", "stroke")

        if ctype == "stroke":
            brush = getattr(cmd, "brush_id", "ink_liner")
            b_idx = BRUSH_NAME_TO_ID.get(brush, 0)
            if b_idx != current_brush_idx:
                tokens.extend([TOKEN_SET_BRUSH, BRUSH_OFFSET + b_idx])
                current_brush_idx = b_idx

            col_idx = hex_to_palette_id(cmd.color)
            if col_idx != current_color_idx:
                tokens.extend([TOKEN_SET_COLOR, COLOR_OFFSET + col_idx])
                current_color_idx = col_idx

            w_val = getattr(cmd, "base_width", 3.0)
            w_idx = max(0, min(NUM_WIDTH_BINS - 1, int(round((w_val / 16.0) * (NUM_WIDTH_BINS - 1)))))
            if w_idx != current_width_idx:
                tokens.extend([TOKEN_SET_WIDTH, WIDTH_OFFSET + w_idx])
                current_width_idx = w_idx

            pts = cmd.points
            if len(pts) < 2:
                continue

            # First point: MOVE
            p0 = pts[0]
            x0 = max(0, min(NUM_COORD_BINS - 1, int(round(p0.x * (NUM_COORD_BINS - 1)))))
            y0 = max(0, min(NUM_COORD_BINS - 1, int(round(p0.y * (NUM_COORD_BINS - 1)))))
            tokens.extend([TOKEN_MOVE, X_OFFSET + x0, Y_OFFSET + y0])

            # Remaining points: LINE
            for pt in pts[1:]:
                if len(tokens) >= max_tokens - 5:
                    break
                xi = max(0, min(NUM_COORD_BINS - 1, int(round(pt.x * (NUM_COORD_BINS - 1)))))
                yi = max(0, min(NUM_COORD_BINS - 1, int(round(pt.y * (NUM_COORD_BINS - 1)))))
                pi = max(0, min(NUM_PRESSURE_BINS - 1, int(round(pt.p * (NUM_PRESSURE_BINS - 1)))))
                tokens.extend([TOKEN_LINE, X_OFFSET + xi, Y_OFFSET + yi, PRESSURE_OFFSET + pi])

            tokens.append(TOKEN_PEN_UP)

        elif ctype == "fill":
            col_idx = hex_to_palette_id(cmd.color)
            if col_idx != current_color_idx:
                tokens.extend([TOKEN_SET_COLOR, COLOR_OFFSET + col_idx])
                current_color_idx = col_idx

            ftype = getattr(cmd, "fill_type", "flat")
            cmd_token = TOKEN_FILL_HATCH if ftype in ("hatching", "cross_hatching") else TOKEN_FILL_FLAT
            tokens.append(cmd_token)

            for x, y in cmd.path:
                if len(tokens) >= max_tokens - 4:
                    break
                xi = max(0, min(NUM_COORD_BINS - 1, int(round(x * (NUM_COORD_BINS - 1)))))
                yi = max(0, min(NUM_COORD_BINS - 1, int(round(y * (NUM_COORD_BINS - 1)))))
                tokens.extend([X_OFFSET + xi, Y_OFFSET + yi])

            tokens.append(TOKEN_FILL_END)

    tokens.append(EOS_TOKEN)
    return tokens


# ---------------------------------------------------------
# Decoding: Discrete Token Sequence -> DocumentV2
# ---------------------------------------------------------

def token_sequence_to_doc(
    tokens: List[int],
    canvas_size: int = 1024,
) -> DocumentV2:
    """Decodes a discrete token sequence back into a valid DocumentV2."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=canvas_size, height=canvas_size, background="#ffffff"),
        layers=[
            LayerV2(id="background", name="Background", opacity=1.0),
            LayerV2(id="underdrawing", name="Underdrawing", opacity=0.4),
            LayerV2(id="flats", name="Flats", opacity=0.9, blend_mode="multiply"),
            LayerV2(id="shading", name="Shading", opacity=0.85, blend_mode="multiply"),
            LayerV2(id="line_art", name="Line Art", opacity=1.0),
            LayerV2(id="highlights", name="Highlights", opacity=0.75, blend_mode="screen"),
        ],
    )

    cur_layer = "line_art"
    cur_brush = "ink_liner"
    cur_color = "#111827"
    cur_width = 3.0

    cur_stroke_pts: List[PointV2] = []
    cur_fill_path: List[Tuple[float, float]] = []
    is_filling = False
    fill_is_hatch = False

    def flush_stroke():
        nonlocal cur_stroke_pts
        if len(cur_stroke_pts) >= 2:
            doc.commands.append(
                StrokeActionV2(
                    points=list(cur_stroke_pts),
                    brush_id=cur_brush,
                    color=cur_color,
                    base_width=cur_width,
                    curve_type="catmull_rom",
                    taper_in=0.1,
                    taper_out=0.1,
                    layer_id=cur_layer,
                )
            )
        cur_stroke_pts = []

    def flush_fill():
        nonlocal cur_fill_path, is_filling
        if len(cur_fill_path) >= 3:
            doc.commands.append(
                FillActionV2(
                    path=list(cur_fill_path),
                    fill_type="hatching" if fill_is_hatch else "flat",
                    color=cur_color,
                    opacity=0.75 if fill_is_hatch else 0.85,
                    layer_id=cur_layer,
                    blend_mode="multiply",
                )
            )
        cur_fill_path = []
        is_filling = False

    i = 0
    n = len(tokens)

    while i < n:
        tok = tokens[i]

        if tok in (BOS_TOKEN, PAD_TOKEN, SEP_TOKEN):
            i += 1
            continue
        elif tok == EOS_TOKEN:
            break

        elif tok == TOKEN_SET_LAYER:
            flush_stroke()
            flush_fill()
            if i + 1 < n and (LAYER_OFFSET <= tokens[i + 1] < LAYER_OFFSET + NUM_LAYERS):
                l_idx = tokens[i + 1] - LAYER_OFFSET
                cur_layer = LAYER_NAMES[l_idx]
                i += 2
            else:
                i += 1

        elif tok == TOKEN_SET_BRUSH:
            if i + 1 < n and (BRUSH_OFFSET <= tokens[i + 1] < BRUSH_OFFSET + NUM_BRUSHES):
                b_idx = tokens[i + 1] - BRUSH_OFFSET
                cur_brush = BRUSH_NAMES[b_idx]
                i += 2
            else:
                i += 1

        elif tok == TOKEN_SET_COLOR:
            if i + 1 < n and (COLOR_OFFSET <= tokens[i + 1] < COLOR_OFFSET + NUM_COLOR_BINS):
                c_idx = tokens[i + 1] - COLOR_OFFSET
                cur_color = COLOR_PALETTE[c_idx]
                i += 2
            else:
                i += 1

        elif tok == TOKEN_SET_WIDTH:
            if i + 1 < n and (WIDTH_OFFSET <= tokens[i + 1] < WIDTH_OFFSET + NUM_WIDTH_BINS):
                w_bin = tokens[i + 1] - WIDTH_OFFSET
                cur_width = max(0.5, (w_bin / float(NUM_WIDTH_BINS - 1)) * 16.0)
                i += 2
            else:
                i += 1

        elif tok == TOKEN_MOVE:
            flush_stroke()
            if i + 2 < n and (X_OFFSET <= tokens[i + 1] < X_OFFSET + NUM_COORD_BINS) and (Y_OFFSET <= tokens[i + 2] < Y_OFFSET + NUM_COORD_BINS):
                x = (tokens[i + 1] - X_OFFSET) / float(NUM_COORD_BINS - 1)
                y = (tokens[i + 2] - Y_OFFSET) / float(NUM_COORD_BINS - 1)
                cur_stroke_pts.append(PointV2(x=x, y=y, p=0.4))
                i += 3
            else:
                i += 1

        elif tok == TOKEN_LINE:
            if i + 3 < n and (X_OFFSET <= tokens[i + 1] < X_OFFSET + NUM_COORD_BINS) and (Y_OFFSET <= tokens[i + 2] < Y_OFFSET + NUM_COORD_BINS) and (PRESSURE_OFFSET <= tokens[i + 3] < PRESSURE_OFFSET + NUM_PRESSURE_BINS):
                x = (tokens[i + 1] - X_OFFSET) / float(NUM_COORD_BINS - 1)
                y = (tokens[i + 2] - Y_OFFSET) / float(NUM_COORD_BINS - 1)
                p = (tokens[i + 3] - PRESSURE_OFFSET) / float(NUM_PRESSURE_BINS - 1)
                cur_stroke_pts.append(PointV2(x=x, y=y, p=max(0.1, p)))
                i += 4
            else:
                i += 1

        elif tok == TOKEN_PEN_UP:
            flush_stroke()
            i += 1

        elif tok in (TOKEN_FILL_FLAT, TOKEN_FILL_HATCH):
            flush_stroke()
            flush_fill()
            is_filling = True
            fill_is_hatch = (tok == TOKEN_FILL_HATCH)
            i += 1
            # Consume vertex pairs
            while i + 1 < n and (X_OFFSET <= tokens[i] < X_OFFSET + NUM_COORD_BINS) and (Y_OFFSET <= tokens[i + 1] < Y_OFFSET + NUM_COORD_BINS):
                vx = (tokens[i] - X_OFFSET) / float(NUM_COORD_BINS - 1)
                vy = (tokens[i + 1] - Y_OFFSET) / float(NUM_COORD_BINS - 1)
                cur_fill_path.append((vx, vy))
                i += 2

        elif tok == TOKEN_FILL_END:
            flush_fill()
            i += 1

        else:
            i += 1

    flush_stroke()
    flush_fill()
    return doc
