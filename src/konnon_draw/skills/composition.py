"""Composition helpers: Rule-of-thirds, margins, anchors, and standard layer stacks."""

from typing import Any, Dict, List, Optional, Tuple, Union
from ..v2.schema import LayerV2


def create_standard_layers() -> List[LayerV2]:
    """Returns a production-grade 6-layer artist stack."""
    return [
        LayerV2(id="background", name="Background / Sky", opacity=1.0, blend_mode="normal"),
        LayerV2(id="underdrawing", name="Guides & Construction", opacity=0.45, blend_mode="normal"),
        LayerV2(id="flats", name="Flat Colors / Washes", opacity=0.9, blend_mode="multiply"),
        LayerV2(id="shading", name="Shading & Hatching", opacity=0.85, blend_mode="multiply"),
        LayerV2(id="line_art", name="Main Line Art", opacity=1.0, blend_mode="normal"),
        LayerV2(id="highlights", name="Highlights & Atmosphere", opacity=0.75, blend_mode="screen"),
    ]


def rule_of_thirds_points() -> List[Tuple[float, float]]:
    """Returns the four classic Rule-of-Thirds focal power points."""
    return [
        (1.0 / 3.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 3.0),
        (1.0 / 3.0, 2.0 / 3.0),
        (2.0 / 3.0, 2.0 / 3.0),
    ]


def fit_into_bounding_box(
    points: List[Tuple[float, float]],
    target_bbox: Tuple[float, float, float, float],  # (min_x, min_y, max_x, max_y)
    keep_aspect: bool = True,
) -> List[Tuple[float, float]]:
    """Rescales and translates points to fit inside a target bounding box."""
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    src_w = max(1e-5, max_x - min_x)
    src_h = max(1e-5, max_y - min_y)

    tgt_min_x, tgt_min_y, tgt_max_x, tgt_max_y = target_bbox
    tgt_w = tgt_max_x - tgt_min_x
    tgt_h = tgt_max_y - tgt_min_y

    if keep_aspect:
        scale = min(tgt_w / src_w, tgt_h / src_h)
        sx = sy = scale
        offset_x = tgt_min_x + (tgt_w - src_w * scale) * 0.5 - min_x * scale
        offset_y = tgt_min_y + (tgt_h - src_h * scale) * 0.5 - min_y * scale
    else:
        sx = tgt_w / src_w
        sy = tgt_h / src_h
        offset_x = tgt_min_x - min_x * sx
        offset_y = tgt_min_y - min_y * sy

    return [(p[0] * sx + offset_x, p[1] * sy + offset_y) for p in points]
