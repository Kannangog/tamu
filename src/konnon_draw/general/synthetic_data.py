"""Synthetic dataset generator for Konnon Draw General.

Generates ground-truth (caption, DocumentV2) pairs using the Artist Skill Library:
- Shapes: circle, square, rectangle, triangle, star, ellipse.
- Colors: red, blue, green, yellow, orange, purple, black, brown, etc.
- Counts: one, two, three, four.
- Spatial relations: above, below, left_of, right_of, top, bottom, center.
- Multi-object compositions with verifiable structure.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..artist_skills import make_arc, make_ellipse
from ..schema_v2 import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
)
from .vocab import COLOR_NAME_MAP, COLOR_PALETTE


COLOR_NAMES = ["red", "blue", "green", "yellow", "orange", "purple", "black", "brown"]
COLOR_HEX_MAP = {
    "red": "#ef4444",
    "blue": "#3b82f6",
    "green": "#10b981",
    "yellow": "#eab308",
    "orange": "#f97316",
    "purple": "#8b5cf6",
    "black": "#111827",
    "brown": "#78350f",
}


def make_polygon_points(
    center: Tuple[float, float],
    radius: float,
    num_sides: int,
    angle_offset: float = 0.0,
) -> List[PointV2]:
    """Generates regular polygon vertices."""
    cx, cy = center
    pts = []
    for i in range(num_sides + 1):
        ang = angle_offset + (i % num_sides) * (2.0 * math.pi / num_sides)
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang)
        pts.append(PointV2(x=x, y=y, p=0.8))
    return pts


def make_star_points(
    center: Tuple[float, float],
    outer_r: float,
    inner_r: float,
    num_points: int = 5,
) -> List[PointV2]:
    """Generates 5-pointed star vertices."""
    cx, cy = center
    pts = []
    total_pts = num_points * 2
    for i in range(total_pts + 1):
        ang = -math.pi / 2.0 + (i % total_pts) * (math.pi / num_points)
        r = outer_r if (i % 2 == 0) else inner_r
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pts.append(PointV2(x=x, y=y, p=0.85))
    return pts


def make_single_shape(
    shape_type: str,
    color_name: str,
    center: Tuple[float, float],
    size: float = 0.15,
    filled: bool = False,
    layer_id: str = "line_art",
) -> List[Any]:
    """Creates commands for a single geometric primitive."""
    color_hex = COLOR_HEX_MAP.get(color_name, "#111827")
    cmds = []

    if shape_type == "circle":
        st = make_ellipse(center, size, size, num_points=24, color=color_hex, base_width=3.5, layer_id=layer_id)
        cmds.append(st)
        if filled:
            path = [(p.x, p.y) for p in st.points[:-1]]
            cmds.insert(0, FillActionV2(path=path, fill_type="flat", color=color_hex, opacity=0.85, layer_id=layer_id))

    elif shape_type == "square":
        cx, cy = center
        s = size
        pts = [
            PointV2(cx - s, cy - s),
            PointV2(cx + s, cy - s),
            PointV2(cx + s, cy + s),
            PointV2(cx - s, cy + s),
            PointV2(cx - s, cy - s),
        ]
        cmds.append(StrokeActionV2(points=pts, brush_id="ink_liner", color=color_hex, base_width=3.5, layer_id=layer_id))
        if filled:
            path = [(p.x, p.y) for p in pts[:-1]]
            cmds.insert(0, FillActionV2(path=path, fill_type="flat", color=color_hex, opacity=0.85, layer_id=layer_id))

    elif shape_type == "rectangle":
        cx, cy = center
        sx, sy = size * 1.4, size * 0.7
        pts = [
            PointV2(cx - sx, cy - sy),
            PointV2(cx + sx, cy - sy),
            PointV2(cx + sx, cy + sy),
            PointV2(cx - sx, cy + sy),
            PointV2(cx - sx, cy - sy),
        ]
        cmds.append(StrokeActionV2(points=pts, brush_id="ink_liner", color=color_hex, base_width=3.5, layer_id=layer_id))
        if filled:
            path = [(p.x, p.y) for p in pts[:-1]]
            cmds.insert(0, FillActionV2(path=path, fill_type="flat", color=color_hex, opacity=0.85, layer_id=layer_id))

    elif shape_type == "triangle":
        pts = make_polygon_points(center, radius=size * 1.2, num_sides=3, angle_offset=-math.pi / 2.0)
        cmds.append(StrokeActionV2(points=pts, brush_id="ink_liner", color=color_hex, base_width=3.5, layer_id=layer_id))
        if filled:
            path = [(p.x, p.y) for p in pts[:-1]]
            cmds.insert(0, FillActionV2(path=path, fill_type="flat", color=color_hex, opacity=0.85, layer_id=layer_id))

    elif shape_type == "star":
        pts = make_star_points(center, outer_r=size * 1.2, inner_r=size * 0.55)
        cmds.append(StrokeActionV2(points=pts, brush_id="ink_liner", color=color_hex, base_width=3.0, layer_id=layer_id))
        if filled:
            path = [(p.x, p.y) for p in pts[:-1]]
            cmds.insert(0, FillActionV2(path=path, fill_type="flat", color=color_hex, opacity=0.85, layer_id=layer_id))

    return cmds


def generate_synthetic_example(seed: int) -> Dict[str, Any]:
    """Generates a random structured (prompt, DocumentV2) pair."""
    rng = random.Random(seed)
    task_type = rng.choice(["single", "count", "spatial"])

    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1024, height=1024, background="#ffffff"),
        layers=[
            LayerV2(id="background", name="BG"),
            LayerV2(id="line_art", name="Line Art"),
        ],
    )

    shapes = ["circle", "square", "rectangle", "triangle", "star"]
    colors = COLOR_NAMES

    if task_type == "single":
        shape = rng.choice(shapes)
        color = rng.choice(colors)
        center = (rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65))
        filled = rng.choice([False, True])

        prefix = "a filled " if filled else "a "
        prompt = f"{prefix}{color} {shape}"
        cmds = make_single_shape(shape, color, center, size=0.18, filled=filled)
        doc.commands.extend(cmds)

    elif task_type == "count":
        shape = rng.choice(["circle", "square", "star"])
        color = rng.choice(colors)
        count = rng.choice([2, 3])
        count_words = {2: "two", 3: "three"}
        plural = {"circle": "circles", "square": "squares", "star": "stars"}[shape]
        prompt = f"{count_words[count]} {color} {plural}"

        xs = np.linspace(0.25, 0.75, count)
        for x in xs:
            y = rng.uniform(0.40, 0.60)
            cmds = make_single_shape(shape, color, (float(x), y), size=0.10)
            doc.commands.extend(cmds)

    elif task_type == "spatial":
        shape1 = rng.choice(shapes)
        color1 = rng.choice(colors)
        shape2 = rng.choice([s for s in shapes if s != shape1])
        color2 = rng.choice([c for c in colors if c != color1])
        relation = rng.choice(["above", "below", "left of", "right of"])

        prompt = f"a {color1} {shape1} {relation} a {color2} {shape2}"

        if relation == "above":
            c1 = (0.50, 0.32)
            c2 = (0.50, 0.68)
        elif relation == "below":
            c1 = (0.50, 0.68)
            c2 = (0.50, 0.32)
        elif relation == "left of":
            c1 = (0.32, 0.50)
            c2 = (0.68, 0.50)
        else:  # right of
            c1 = (0.68, 0.50)
            c2 = (0.32, 0.50)

        doc.commands.extend(make_single_shape(shape1, color1, c1, size=0.14))
        doc.commands.extend(make_single_shape(shape2, color2, c2, size=0.14))

    return {
        "prompt": prompt,
        "doc": doc,
        "seed": seed,
    }


def generate_synthetic_dataset(num_samples: int = 1000, start_seed: int = 10000) -> List[Dict[str, Any]]:
    """Generates a dataset of synthetic (prompt, doc) pairs."""
    dataset = []
    for i in range(num_samples):
        ex = generate_synthetic_example(seed=start_seed + i)
        dataset.append(ex)
    return dataset
