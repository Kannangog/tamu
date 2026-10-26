"""Dynamic graphic effects: Speed lines, motion arcs, and impact bursts."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..v2.schema import PointV2, StrokeActionV2


def make_speed_lines(
    direction_deg: float = 0.0,
    origin_box: Tuple[float, float, float, float] = (0.2, 0.2, 0.8, 0.8),
    num_lines: int = 12,
    length: float = 0.18,
    base_width: float = 2.0,
    brush_id: str = "ink_liner",
    color: str = "#1e293b",
    layer_id: str = "highlights",
    stage: str = "effect",
    depth_rank: int = 15,
    rng: Optional[np.random.RandomState] = None,
) -> List[StrokeActionV2]:
    """Generates parallel manga/comic speed lines indicating rapid directional velocity."""
    if rng is None:
        rng = np.random.RandomState(42)

    rad = math.radians(direction_deg)
    dx = math.cos(rad)
    dy = math.sin(rad)

    min_x, min_y, max_x, max_y = origin_box
    strokes = []

    for _ in range(num_lines):
        # Pick random origin point along the boundary or volume
        ox = float(rng.uniform(min_x, max_x))
        oy = float(rng.uniform(min_y, max_y))
        line_len = length * float(rng.uniform(0.7, 1.3))

        p1 = PointV2(ox, oy, p=0.1)
        p2 = PointV2(ox + dx * line_len * 0.5, oy + dy * line_len * 0.5, p=1.0)
        p3 = PointV2(ox + dx * line_len, oy + dy * line_len, p=0.0)

        strokes.append(
            StrokeActionV2(
                points=[p1, p2, p3],
                brush_id=brush_id,
                color=color,
                base_width=base_width * float(rng.uniform(0.8, 1.4)),
                layer_id=layer_id,
                stage=stage,
                purpose="accent",
                depth_rank=depth_rank,
            )
        )
    return strokes


def make_motion_arcs(
    center: Tuple[float, float],
    radius: float = 0.15,
    start_angle_deg: float = 30.0,
    arc_span_deg: float = 45.0,
    num_arcs: int = 3,
    brush_id: str = "ink_liner",
    color: str = "#334155",
    layer_id: str = "highlights",
    stage: str = "effect",
    depth_rank: int = 15,
    rng: Optional[np.random.RandomState] = None,
) -> List[StrokeActionV2]:
    """Generates concentric motion wake arcs trailing a moving limb or object."""
    if rng is None:
        rng = np.random.RandomState(42)

    cx, cy = center
    strokes = []

    for k in range(num_arcs):
        r_k = radius * (0.85 + 0.15 * k)
        span = arc_span_deg * (0.8 + 0.4 * float(rng.rand()))
        ang_start = math.radians(start_angle_deg - span * 0.5)
        ang_end = math.radians(start_angle_deg + span * 0.5)

        pts = []
        n_pts = 16
        for i in range(n_pts):
            u = i / float(n_pts - 1)
            ang = ang_start + u * (ang_end - ang_start)
            px = cx + r_k * math.cos(ang)
            py = cy + r_k * math.sin(ang)
            press = math.sin(u * math.pi) * (1.0 - 0.2 * k)
            pts.append(PointV2(px, py, press))

        strokes.append(
            StrokeActionV2(
                points=pts,
                brush_id=brush_id,
                color=color,
                base_width=2.5 - 0.5 * k,
                layer_id=layer_id,
                stage=stage,
                purpose="accent",
                depth_rank=depth_rank,
            )
        )
    return strokes


def make_impact_burst(
    center: Tuple[float, float],
    inner_radius: float = 0.03,
    outer_radius: float = 0.12,
    num_rays: int = 16,
    brush_id: str = "ink_liner",
    color: str = "#0f172a",
    layer_id: str = "highlights",
    stage: str = "effect",
    depth_rank: int = 25,
    rng: Optional[np.random.RandomState] = None,
) -> List[StrokeActionV2]:
    """Generates radial starburst spikes emanating from a strike impact point."""
    if rng is None:
        rng = np.random.RandomState(42)

    cx, cy = center
    strokes = []

    for i in range(num_rays):
        angle = (i / float(num_rays)) * 2.0 * math.pi + float(rng.normal(0, 0.05))
        r_in = inner_radius * float(rng.uniform(0.7, 1.2))
        r_out = outer_radius * float(rng.uniform(0.8, 1.3))

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        p1 = PointV2(cx + r_in * cos_a, cy + r_in * sin_a, p=0.9)
        p2 = PointV2(cx + r_out * cos_a, cy + r_out * sin_a, p=0.1)

        strokes.append(
            StrokeActionV2(
                points=[p1, p2],
                brush_id=brush_id,
                color=color,
                base_width=3.0 * float(rng.uniform(0.7, 1.3)),
                layer_id=layer_id,
                stage=stage,
                purpose="accent",
                depth_rank=depth_rank,
            )
        )
    return strokes
