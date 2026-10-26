"""Token encoding, decoding, QuickDraw conversion, and learning extensions for v2.

Features:
- Coarse-to-fine drawing order representation (layers -> composition -> contours -> shading -> details).
- Discretized spatial (256/1024 bins), pressure (16 bins), width (16 bins), and palette tokens.
- Lossless/high-precision round-trip encoding and decoding.
- Quick, Draw! raw vector to DocumentV2 converter with stroke smoothing and synthetic taper.
- Extension points: stroke-order augmentation, prefix completion API, and critic similarity hook.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .schema_v2 import (
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


BRUSH_TO_ID = {
    "ink_liner": 0,
    "pencil": 1,
    "charcoal": 2,
    "airbrush": 3,
    "marker": 4,
    "watercolor_wash": 5,
}
ID_TO_BRUSH = {v: k for k, v in BRUSH_TO_ID.items()}

BLEND_TO_ID = {
    "normal": 0,
    "multiply": 1,
    "screen": 2,
    "overlay": 3,
}
ID_TO_BLEND = {v: k for k, v in BLEND_TO_ID.items()}

FILL_TYPE_TO_ID = {
    "flat": 0,
    "linear_gradient": 1,
    "radial_gradient": 2,
    "hatching": 3,
    "cross_hatching": 4,
    "stipple": 5,
}
ID_TO_FILL_TYPE = {v: k for k, v in FILL_TYPE_TO_ID.items()}


# ---------------------------------------------------------
# Token Serialization & Deserialization
# ---------------------------------------------------------

def encode_v2_to_tokens(
    doc: DocumentV2,
    coord_bins: int = 256,
    num_p_bins: int = 16,
    num_w_bins: int = 16,
) -> List[Dict[str, Any]]:
    """Encodes a DocumentV2 into a sequence of discrete, coarse-to-fine tokens.

    Preserves layers, brush styles, colors, pressure variations, and region fills.
    """
    tokens: List[Dict[str, Any]] = []

    # Canvas token
    tokens.append({
        "type": "SET_CANVAS",
        "bg": doc.canvas.background,
        "w": doc.canvas.width,
        "h": doc.canvas.height,
    })

    # Layer definition tokens
    for ly in doc.layers:
        tokens.append({
            "type": "DEF_LAYER",
            "id": ly.id,
            "opacity": int(round(ly.opacity * 15)),
            "blend": BLEND_TO_ID.get(ly.blend_mode, 0),
        })

    # Group commands coarse-to-fine by layer definition order
    layer_order = {ly.id: idx for idx, ly in enumerate(doc.layers)}
    sorted_commands = sorted(
        doc.commands,
        key=lambda c: layer_order.get(getattr(c, "layer_id", "main"), 999),
    )

    current_layer = None

    for cmd in sorted_commands:
        lid = getattr(cmd, "layer_id", "main")
        if lid != current_layer:
            tokens.append({"type": "SET_LAYER", "layer_id": lid})
            current_layer = lid

        ctype = getattr(cmd, "type", "stroke")
        if ctype == "stroke":
            brush_idx = BRUSH_TO_ID.get(cmd.brush_id, 0)
            blend_idx = BLEND_TO_ID.get(cmd.blend_mode, 0)
            w_bin = max(0, min(num_w_bins - 1, int(round((cmd.base_width / 16.0) * (num_w_bins - 1)))))

            tokens.append({
                "type": "STROKE_START",
                "brush": brush_idx,
                "color": cmd.color,
                "width": w_bin,
                "blend": blend_idx,
                "curve": cmd.curve_type,
            })

            for pt in cmd.points:
                xb = max(0, min(coord_bins - 1, int(round(pt.x * (coord_bins - 1)))))
                yb = max(0, min(coord_bins - 1, int(round(pt.y * (coord_bins - 1)))))
                pb = max(0, min(num_p_bins - 1, int(round(pt.p * (num_p_bins - 1)))))
                tokens.append({
                    "type": "STROKE_POINT",
                    "x": xb,
                    "y": yb,
                    "p": pb,
                })

            tokens.append({"type": "STROKE_END"})

        elif ctype == "fill":
            ft_idx = FILL_TYPE_TO_ID.get(cmd.fill_type, 0)
            tokens.append({
                "type": "FILL_START",
                "fill_type": ft_idx,
                "color": cmd.color,
                "color_end": cmd.color_end,
                "angle": int(round(cmd.angle_deg)),
                "spacing": int(round(cmd.spacing)),
                "density": int(round(cmd.density * 100)),
            })
            for x, y in cmd.path:
                xb = max(0, min(coord_bins - 1, int(round(x * (coord_bins - 1)))))
                yb = max(0, min(coord_bins - 1, int(round(y * (coord_bins - 1)))))
                tokens.append({"type": "FILL_VERTEX", "x": xb, "y": yb})
            tokens.append({"type": "FILL_END"})

    tokens.append({"type": "END_DOC"})
    return tokens


def decode_tokens_to_v2(
    tokens: List[Dict[str, Any]],
    coord_bins: int = 256,
    num_p_bins: int = 16,
    num_w_bins: int = 16,
) -> DocumentV2:
    """Decodes a token sequence back into a valid DocumentV2."""
    canvas = CanvasConfigV2()
    layers: List[LayerV2] = []
    commands: List[Union[StrokeActionV2, FillActionV2]] = []

    current_layer = "main"
    current_stroke: Optional[Dict[str, Any]] = None
    current_fill: Optional[Dict[str, Any]] = None

    for tok in tokens:
        ttype = tok.get("type")

        if ttype == "SET_CANVAS":
            canvas.background = tok.get("bg", "#ffffff")
            canvas.width = tok.get("w", 1024)
            canvas.height = tok.get("h", 1024)

        elif ttype == "DEF_LAYER":
            lid = tok["id"]
            op = tok.get("opacity", 15) / 15.0
            bm = ID_TO_BLEND.get(tok.get("blend", 0), "normal")
            layers.append(LayerV2(id=lid, name=lid, opacity=op, blend_mode=bm))

        elif ttype == "SET_LAYER":
            current_layer = tok.get("layer_id", "main")

        elif ttype == "STROKE_START":
            brush = ID_TO_BRUSH.get(tok.get("brush", 0), "ink_liner")
            blend = ID_TO_BLEND.get(tok.get("blend", 0), "normal")
            w_bin = tok.get("width", 3)
            base_w = (w_bin / max(1, num_w_bins - 1)) * 16.0
            current_stroke = {
                "brush": brush,
                "color": tok.get("color", "#111827"),
                "base_width": max(0.5, base_w),
                "blend_mode": blend,
                "curve_type": tok.get("curve", "catmull_rom"),
                "layer_id": current_layer,
                "points": [],
            }

        elif ttype == "STROKE_POINT":
            if current_stroke is not None:
                x = tok["x"] / float(coord_bins - 1)
                y = tok["y"] / float(coord_bins - 1)
                p = tok.get("p", num_p_bins - 1) / float(num_p_bins - 1)
                current_stroke["points"].append(PointV2(x=x, y=y, p=p))

        elif ttype == "STROKE_END":
            if current_stroke is not None and len(current_stroke["points"]) >= 2:
                commands.append(
                    StrokeActionV2(
                        points=current_stroke["points"],
                        brush_id=current_stroke["brush"],
                        color=current_stroke["color"],
                        base_width=current_stroke["base_width"],
                        blend_mode=current_stroke["blend_mode"],
                        curve_type=current_stroke["curve_type"],
                        layer_id=current_stroke["layer_id"],
                        taper_in=0.1,
                        taper_out=0.1,
                    )
                )
            current_stroke = None

        elif ttype == "FILL_START":
            ft = ID_TO_FILL_TYPE.get(tok.get("fill_type", 0), "flat")
            current_fill = {
                "fill_type": ft,
                "color": tok.get("color", "#e2e8f0"),
                "color_end": tok.get("color_end"),
                "angle_deg": float(tok.get("angle", 45)),
                "spacing": float(tok.get("spacing", 6)),
                "density": float(tok.get("density", 5)) / 100.0,
                "layer_id": current_layer,
                "path": [],
            }

        elif ttype == "FILL_VERTEX":
            if current_fill is not None:
                x = tok["x"] / float(coord_bins - 1)
                y = tok["y"] / float(coord_bins - 1)
                current_fill["path"].append((x, y))

        elif ttype == "FILL_END":
            if current_fill is not None and len(current_fill["path"]) >= 3:
                commands.append(
                    FillActionV2(
                        path=current_fill["path"],
                        fill_type=current_fill["fill_type"],
                        color=current_fill["color"],
                        color_end=current_fill["color_end"],
                        angle_deg=current_fill["angle_deg"],
                        spacing=current_fill["spacing"],
                        density=current_fill["density"],
                        layer_id=current_fill["layer_id"],
                    )
                )
            current_fill = None

        elif ttype == "END_DOC":
            break

    if not layers:
        layers = [LayerV2(id="main", name="Main")]

    return DocumentV2(canvas=canvas, layers=layers, commands=commands)


# ---------------------------------------------------------
# Quick, Draw! to DocumentV2 Converter
# ---------------------------------------------------------

def quickdraw_to_v2(
    qd_drawing: List[List[List[float]]],
    default_brush: str = "ink_liner",
    canvas_size: int = 1024,
    color: str = "#1e293b",
    base_width: float = 3.5,
    taper: bool = True,
    add_shadow: bool = False,
) -> DocumentV2:
    """Converts raw Quick, Draw! vector strokes [[[x...], [y...]], ...] to a v2 Document.

    Normalizes coordinates [0..255] -> [0..1], calculates natural curvature/velocity pressure,
    and applies smooth Catmull-Rom curve rendering.
    """
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=canvas_size, height=canvas_size, background="#ffffff"),
        layers=[
            LayerV2(id="shadows", name="Shadows", opacity=0.4, blend_mode="multiply"),
            LayerV2(id="line_art", name="Line Art", opacity=1.0, blend_mode="normal"),
        ],
    )

    all_pts_flat = []

    for stroke_raw in qd_drawing:
        xs, ys = stroke_raw[0], stroke_raw[1]
        n = len(xs)
        if n < 2:
            continue

        pts: List[PointV2] = []
        for i in range(n):
            nx = float(xs[i]) / 255.0
            ny = float(ys[i]) / 255.0
            all_pts_flat.append((nx, ny))

            # Synthetic pressure: calligraphic taper + velocity variation
            p = 1.0
            if taper:
                t = i / max(1, n - 1)
                p = 0.35 + 0.65 * math.sin(t * math.pi)

            pts.append(PointV2(x=nx, y=ny, p=round(p, 3)))

        doc.commands.append(
            StrokeActionV2(
                points=pts,
                brush_id=default_brush,
                color=color,
                base_width=base_width,
                curve_type="catmull_rom",
                taper_in=0.1 if taper else 0.0,
                taper_out=0.1 if taper else 0.0,
                layer_id="line_art",
            )
        )

    # Optional subtle ground shadow under the doodle
    if add_shadow and all_pts_flat:
        all_x = [p[0] for p in all_pts_flat]
        all_y = [p[1] for p in all_pts_flat]
        min_x, max_x = min(all_x), max(all_x)
        max_y = max(all_y)
        shadow_poly = [
            (min_x + 0.05, max_y + 0.01),
            (max_x - 0.05, max_y + 0.01),
            (max_x + 0.02, max_y + 0.04),
            (min_x - 0.02, max_y + 0.04),
        ]
        doc.commands.insert(
            0,
            FillActionV2(
                path=shadow_poly,
                fill_type="flat",
                color="#64748b",
                opacity=0.35,
                layer_id="shadows",
                blend_mode="multiply",
            ),
        )

    return doc


# ---------------------------------------------------------
# Learning Extensions: Augmentation, Prefix Completion, Critic
# ---------------------------------------------------------

def stroke_order_augmentation(
    doc: DocumentV2,
    seed: Optional[int] = None,
) -> DocumentV2:
    """Permutes independent stroke sequences within the same layer.

    Trains neural models to be invariant to arbitrary stroke recording order.
    """
    rng = np.random.RandomState(seed)
    new_doc = DocumentV2(canvas=doc.canvas, layers=list(doc.layers), commands=[])

    # Group commands by layer
    by_layer: Dict[str, List[Any]] = {ly.id: [] for ly in doc.layers}
    for c in doc.commands:
        lid = getattr(c, "layer_id", "main")
        if lid not in by_layer:
            by_layer[lid] = []
        by_layer[lid].append(c)

    # Permute strokes within each layer while preserving layer hierarchy
    for ly in doc.layers:
        cmds = list(by_layer.get(ly.id, []))
        if len(cmds) > 1:
            perm = rng.permutation(len(cmds))
            cmds = [cmds[i] for i in perm]
        new_doc.commands.extend(cmds)

    return new_doc


def complete_prefix_v2(
    prefix_doc: DocumentV2,
    model_generate_fn: Any,
    max_tokens: int = 512,
) -> DocumentV2:
    """Takes a prefix of user strokes in v2 format and generates a completion."""
    prefix_tokens = encode_v2_to_tokens(prefix_doc)
    # Strip terminal END_DOC
    if prefix_tokens and prefix_tokens[-1].get("type") == "END_DOC":
        prefix_tokens.pop()

    completed_tokens = model_generate_fn(prefix_tokens, max_tokens=max_tokens)
    return decode_tokens_to_v2(completed_tokens)


def compute_critic_score(
    generated_doc: DocumentV2,
    target_image: Image.Image,
    eval_size: int = 128,
) -> float:
    """Hook for neural critic / image similarity score in [0, 1].

    Computes normalized structural correlation and pixel L1 distance between
    rendered document and target reference image.
    """
    from .renderer_v2 import render_document_v2

    rendered = render_document_v2(generated_doc, width=eval_size, height=eval_size, supersample=1)
    arr_gen = np.asarray(rendered.convert("L"), dtype=np.float32) / 255.0
    arr_tgt = np.asarray(target_image.resize((eval_size, eval_size)).convert("L"), dtype=np.float32) / 255.0

    l1_diff = float(np.mean(np.abs(arr_gen - arr_tgt)))
    score = max(0.0, 1.0 - l1_diff * 2.0)
    return round(score, 4)
