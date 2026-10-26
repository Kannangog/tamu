"""Linecraft: Line weights, wobble, sketch passes, and confident calligraphic strokes."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..v2.schema import PointV2, StrokeActionV2


def make_ellipse(
    center: Tuple[float, float],
    rx: float,
    ry: float,
    angle_deg: float = 0.0,
    num_points: int = 36,
    pressure: float = 0.8,
    base_width: float = 2.5,
    brush_id: str = "ink_liner",
    color: str = "#111827",
    layer_id: str = "line_art",
    stage: str = "sketch",
    purpose: str = "contour",
    depth_rank: int = 0,
    closed: bool = True,
    wobble: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> StrokeActionV2:
    """Constructs a smooth, rotated ellipse with optional hand wobble."""
    cx, cy = center
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    if rng is None:
        rng = np.random.RandomState(42)

    pts: List[PointV2] = []
    total_steps = num_points + (1 if closed else 0)

    for i in range(total_steps):
        t = (i % num_points) / float(num_points) * 2.0 * math.pi
        ex = rx * math.cos(t)
        ey = ry * math.sin(t)
        if wobble > 1e-4:
            ex += rng.normal(0, wobble * rx)
            ey += rng.normal(0, wobble * ry)

        x = cx + (ex * cos_a - ey * sin_a)
        y = cy + (ex * sin_a + ey * cos_a)
        p = pressure
        if wobble > 1e-4:
            p = max(0.1, min(1.0, pressure + rng.normal(0, 0.1)))
        pts.append(PointV2(x=x, y=y, p=p))

    return StrokeActionV2(
        points=pts,
        brush_id=brush_id,
        color=color,
        base_width=base_width,
        layer_id=layer_id,
        stage=stage,
        purpose=purpose,
        depth_rank=depth_rank,
    )


def make_arc(
    center: Tuple[float, float],
    rx: float,
    ry: float,
    start_deg: float,
    end_deg: float,
    num_points: int = 24,
    base_width: float = 2.5,
    brush_id: str = "ink_liner",
    color: str = "#111827",
    layer_id: str = "line_art",
    stage: str = "sketch",
    purpose: str = "contour",
    depth_rank: int = 0,
) -> StrokeActionV2:
    """Constructs a curved arc between two angles."""
    cx, cy = center
    rad_start = math.radians(start_deg)
    rad_end = math.radians(end_deg)

    pts: List[PointV2] = []
    for i in range(num_points):
        u = i / float(max(1, num_points - 1))
        ang = rad_start + u * (rad_end - rad_start)
        x = cx + rx * math.cos(ang)
        y = cy + ry * math.sin(ang)
        press = 0.5 + 0.5 * math.sin(u * math.pi)
        pts.append(PointV2(x=x, y=y, p=press))

    return StrokeActionV2(
        points=pts,
        brush_id=brush_id,
        color=color,
        base_width=base_width,
        layer_id=layer_id,
        stage=stage,
        purpose=purpose,
        depth_rank=depth_rank,
    )


def make_wobbly_contour(
    points: List[Tuple[float, float]],
    wobble: float = 0.004,
    pressure: float = 0.8,
    base_width: float = 2.5,
    brush_id: str = "ink_liner",
    color: str = "#111827",
    layer_id: str = "line_art",
    stage: str = "ink",
    purpose: str = "contour",
    depth_rank: int = 0,
    part_id: Optional[str] = None,
    rng: Optional[np.random.RandomState] = None,
) -> StrokeActionV2:
    """Adds natural human muscle tremor and organic variation along a contour."""
    if rng is None:
        rng = np.random.RandomState(42)

    pts: List[PointV2] = []
    for x, y in points:
        wx = x + float(rng.normal(0, wobble))
        wy = y + float(rng.normal(0, wobble))
        wp = max(0.1, min(1.0, pressure + float(rng.normal(0, 0.08))))
        pts.append(PointV2(x=wx, y=wy, p=wp))

    return StrokeActionV2(
        points=pts,
        brush_id=brush_id,
        color=color,
        base_width=base_width,
        layer_id=layer_id,
        stage=stage,
        purpose=purpose,
        depth_rank=depth_rank,
        part_id=part_id,
    )


def make_sketchy_lines(
    points: List[Tuple[float, float]],
    num_passes: int = 3,
    jitter: float = 0.005,
    brush_id: str = "pencil",
    color: str = "#334155",
    base_width: float = 2.0,
    layer_id: str = "underdrawing",
    stage: str = "guide",
    purpose: str = "construction",
    depth_rank: int = 0,
    part_id: Optional[str] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[StrokeActionV2]:
    """Generates 2-4 overlapping loose sketchy passes along a path."""
    if rng is None:
        rng = np.random.RandomState(42)

    strokes = []
    for p_idx in range(num_passes):
        pts: List[PointV2] = []
        for x, y in points:
            jx = x + float(rng.normal(0, jitter * (0.8 + 0.3 * p_idx)))
            jy = y + float(rng.normal(0, jitter * (0.8 + 0.3 * p_idx)))
            jp = max(0.2, min(0.9, 0.4 + float(rng.normal(0, 0.15))))
            pts.append(PointV2(x=jx, y=jy, p=jp))

        strokes.append(
            StrokeActionV2(
                points=pts,
                brush_id=brush_id,
                color=color,
                opacity=0.45,
                base_width=base_width,
                layer_id=layer_id,
                stage=stage,
                purpose=purpose,
                depth_rank=depth_rank,
                part_id=part_id,
            )
        )
    return strokes


def make_confident_stroke(
    points: List[Tuple[float, float]],
    base_width: float = 3.5,
    brush_id: str = "ink_liner",
    color: str = "#0f172a",
    layer_id: str = "line_art",
    stage: str = "ink",
    purpose: str = "contour",
    depth_rank: int = 0,
    part_id: Optional[str] = None,
) -> StrokeActionV2:
    """Generates a single, decisive calligraphic stroke with taper-in and taper-out."""
    pts: List[PointV2] = []
    n = len(points)
    for i, (x, y) in enumerate(points):
        u = i / float(max(1, n - 1))
        # Calligraphic taper curve
        press = 0.3 + 0.7 * math.sin(u * math.pi)
        pts.append(PointV2(x=x, y=y, p=press))

    return StrokeActionV2(
        points=pts,
        brush_id=brush_id,
        color=color,
        base_width=base_width,
        taper_in=0.15,
        taper_out=0.15,
        layer_id=layer_id,
        stage=stage,
        purpose=purpose,
        depth_rank=depth_rank,
        part_id=part_id,
    )


def apply_line_weight_by_light(
    stroke: StrokeActionV2,
    light_dir: Tuple[float, float] = (-0.7071, -0.7071),  # Top-left default
    min_pressure: float = 0.25,
    max_pressure: float = 1.0,
) -> StrokeActionV2:
    """Modulates pressure along curve normals: thick on shadow side, thin on light side."""
    pts = stroke.points
    if len(pts) < 2:
        return stroke

    lx, ly = light_dir
    l_mag = math.hypot(lx, ly)
    if l_mag > 1e-5:
        lx, ly = lx / l_mag, ly / l_mag

    weighted_pts = []
    n = len(pts)
    for i in range(n):
        x, y = pts[i].x, pts[i].y
        if i == 0:
            dx = pts[1].x - x
            dy = pts[1].y - y
        elif i == n - 1:
            dx = x - pts[i - 1].x
            dy = y - pts[i - 1].y
        else:
            dx = pts[i + 1].x - pts[i - 1].x
            dy = pts[i + 1].y - pts[i - 1].y

        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = -dy / mag, dx / mag

        # Dot product with light direction
        alignment = nx * lx + ny * ly
        # If facing away from light (shadow side), alignment is negative -> high pressure
        factor = 0.5 - 0.5 * alignment
        p = min_pressure + factor * (max_pressure - min_pressure)
        weighted_pts.append(PointV2(x=x, y=y, p=p, tilt=pts[i].tilt))

    stroke.points = weighted_pts
    return stroke
