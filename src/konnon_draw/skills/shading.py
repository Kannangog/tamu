"""Shading & Value: Light-direction hatching, cast shadows, rim lights, and gradient skies."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..v2.schema import FillActionV2, PointV2, StrokeActionV2


def generate_directional_hatching(
    polygon_vertices: List[Tuple[float, float]],
    light_dir: Tuple[float, float] = (-0.7071, -0.7071),
    spacing: float = 6.0,
    cross_hatch: bool = False,
    color: str = "#1e293b",
    opacity: float = 0.75,
    layer_id: str = "shading",
    stage: str = "shade",
    depth_rank: int = 0,
) -> FillActionV2:
    """Generates directional hatching perpendicular to light direction."""
    lx, ly = light_dir
    light_angle = math.degrees(math.atan2(ly, lx))
    hatch_angle = (light_angle + 90.0) % 180.0

    return FillActionV2(
        path=polygon_vertices,
        fill_type="cross_hatching" if cross_hatch else "hatching",
        color=color,
        opacity=opacity,
        angle_deg=hatch_angle,
        spacing=spacing,
        layer_id=layer_id,
        blend_mode="multiply",
        depth_rank=depth_rank,
    )


def generate_cast_shadow_polygon(
    base_contact_points: List[Tuple[float, float]],
    light_dir: Tuple[float, float] = (-0.7071, -0.7071),
    shadow_length: float = 0.12,
    color: str = "#0f172a",
    opacity: float = 0.5,
    layer_id: str = "shading",
    stage: str = "shade",
    depth_rank: int = -10,
) -> FillActionV2:
    """Projects ground cast shadow polygon from base contact points opposite to light."""
    lx, ly = light_dir
    mag = max(1e-5, math.hypot(lx, ly))
    dx = (-lx / mag) * shadow_length
    dy = (-ly / mag) * shadow_length * 0.4  # Perspective foreshortening on ground plane

    poly = list(base_contact_points)
    projected = [(x + dx, y + dy) for x, y in reversed(base_contact_points)]
    poly.extend(projected)

    return FillActionV2(
        path=poly,
        fill_type="flat",
        color=color,
        opacity=opacity,
        layer_id=layer_id,
        blend_mode="multiply",
        depth_rank=depth_rank,
    )


def generate_stipple_shading(
    polygon_vertices: List[Tuple[float, float]],
    density: float = 0.06,
    color: str = "#111827",
    opacity: float = 0.8,
    layer_id: str = "shading",
    stage: str = "shade",
    depth_rank: int = 0,
) -> FillActionV2:
    """Generates stochastic stipple shading within a boundary polygon."""
    return FillActionV2(
        path=polygon_vertices,
        fill_type="stipple",
        color=color,
        opacity=opacity,
        density=density,
        layer_id=layer_id,
        blend_mode="multiply",
        depth_rank=depth_rank,
    )


def generate_rim_light_strokes(
    contour_points: List[Tuple[float, float]],
    light_dir: Tuple[float, float] = (-0.7071, -0.7071),
    color: str = "#ffffff",
    base_width: float = 2.0,
    layer_id: str = "highlights",
    stage: str = "effect",
    depth_rank: int = 20,
) -> List[StrokeActionV2]:
    """Generates crisp bright highlights along edges directly facing the light source."""
    if len(contour_points) < 2:
        return []

    lx, ly = light_dir
    mag = max(1e-5, math.hypot(lx, ly))
    lx, ly = lx / mag, ly / mag

    highlight_segments: List[List[PointV2]] = []
    current_seg: List[PointV2] = []

    for i in range(len(contour_points) - 1):
        p1 = contour_points[i]
        p2 = contour_points[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        edge_mag = max(1e-5, math.hypot(dx, dy))
        nx = -dy / edge_mag
        ny = dx / edge_mag

        # Facing light: normal points in opposite direction of light propagation
        alignment = -(nx * lx + ny * ly)
        if alignment > 0.4:
            press = min(1.0, 0.4 + alignment * 0.6)
            if not current_seg:
                current_seg.append(PointV2(p1[0], p1[1], press))
            current_seg.append(PointV2(p2[0], p2[1], press))
        else:
            if len(current_seg) >= 2:
                highlight_segments.append(current_seg)
            current_seg = []

    if len(current_seg) >= 2:
        highlight_segments.append(current_seg)

    strokes = []
    for seg in highlight_segments:
        strokes.append(
            StrokeActionV2(
                points=seg,
                brush_id="ink_liner",
                color=color,
                base_width=base_width,
                opacity=0.9,
                layer_id=layer_id,
                stage=stage,
                purpose="accent",
                depth_rank=depth_rank,
            )
        )
    return strokes


def make_gradient_sky(
    top_color: str = "#0f172a",
    horizon_color: str = "#f97316",
    horizon_y: float = 0.65,
    layer_id: str = "background",
) -> FillActionV2:
    """Generates a smooth linear atmospheric sky gradient from zenith to horizon."""
    path = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, horizon_y),
        (0.0, horizon_y),
    ]
    return FillActionV2(
        path=path,
        fill_type="linear_gradient",
        color=top_color,
        color_end=horizon_color,
        angle_deg=90.0,
        opacity=1.0,
        layer_id=layer_id,
        depth_rank=-100,
    )
