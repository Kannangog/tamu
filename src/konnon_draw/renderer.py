"""Deterministic SVG renderer with Cubic Bézier smoothing support.

Model output is data; only this trusted code emits markup.
Supports:
1. Standard polygonal path rendering (M x y L x y ...)
2. Smooth Cubic Bézier curve rendering (M x y C cx1 cy1, cx2 cy2, x y ...) with corner preservation.
3. Bézier segment extraction for live canvas rendering with control handles.
"""

import math
from typing import Dict, List, Optional, Tuple
from .schema import validate_actions


def actions_to_paths(actions: List[dict]) -> List[List[Tuple[float, float]]]:
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
            if current:
                paths.append(current)
            current = []
        elif kind == "END":
            if current:
                paths.append(current)
                current = []
            break
    if current:
        paths.append(current)
    return paths


def points_to_bezier_commands(
    points: List[Tuple[float, float]],
    corner_threshold_deg: float = 50.0,
) -> List[Tuple[str, ...]]:
    """Convert a sequence of points into a mix of LINE and CUBIC Bézier commands.

    Uses Catmull-Rom spline formulation with corner detection:
    - Angles sharper than `corner_threshold_deg` are treated as sharp vertices (LINE).
    - Smooth sequences are converted into C^1 continuous Cubic Bézier curves.
    """
    if len(points) < 3:
        return [("LINE", pt[0], pt[1]) for pt in points[1:]]

    n = len(points)
    # Check if loop is closed (e.g. wheels, closed polygons)
    is_closed = math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 0.05

    cmds = []
    for i in range(n - 1):
        p1 = points[i]
        p2 = points[i + 1]

        # Determine p0 (preceding)
        if i > 0:
            p0 = points[i - 1]
        elif is_closed:
            p0 = points[-2]
        else:
            p0 = (2 * p1[0] - p2[0], 2 * p1[1] - p2[1])

        # Determine p3 (succeeding)
        if i + 2 < n:
            p3 = points[i + 2]
        elif is_closed:
            p3 = points[1]
        else:
            p3 = (2 * p2[0] - p1[0], 2 * p2[1] - p1[1])

        # Detect corner at p1
        v1 = (p1[0] - p0[0], p1[1] - p0[1])
        v2 = (p2[0] - p1[0], p2[1] - p1[1])
        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)

        is_corner = False
        if len1 > 1e-5 and len2 > 1e-5:
            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot) * 180.0 / math.pi
            if angle > corner_threshold_deg:
                is_corner = True

        if is_corner:
            cmds.append(("LINE", p2[0], p2[1]))
        else:
            # Catmull-Rom to Cubic Bézier control points (tau = 0.5 -> tau/3 = 1/6)
            cx1 = p1[0] + (p2[0] - p0[0]) / 6.0
            cy1 = p1[1] + (p2[1] - p0[1]) / 6.0
            cx2 = p2[0] - (p3[0] - p1[0]) / 6.0
            cy2 = p2[1] - (p3[1] - p1[1]) / 6.0
            cmds.append(("CUBIC", cx1, cy1, cx2, cy2, p2[0], p2[1]))

    return cmds


def actions_to_bezier_segments(
    actions: List[dict],
    width: float = 256.0,
    height: float = 256.0,
    corner_threshold_deg: float = 50.0,
) -> List[dict]:
    """Convert actions into structured segments with Bézier control points for interactive canvas rendering."""
    paths = actions_to_paths(actions)
    strokes = []

    for path_idx, points in enumerate(paths):
        if not points:
            continue
        start_pt = {"x": points[0][0] * width, "y": points[0][1] * height}
        cmds = points_to_bezier_commands(points, corner_threshold_deg=corner_threshold_deg)

        segments = []
        for cmd in cmds:
            if cmd[0] == "LINE":
                segments.append({
                    "type": "LINE",
                    "x": cmd[1] * width,
                    "y": cmd[2] * height,
                })
            elif cmd[0] == "CUBIC":
                segments.append({
                    "type": "CUBIC",
                    "cx1": cmd[1] * width,
                    "cy1": cmd[2] * height,
                    "cx2": cmd[3] * width,
                    "cy2": cmd[4] * height,
                    "x": cmd[5] * width,
                    "y": cmd[6] * height,
                })

        strokes.append({
            "stroke_id": path_idx,
            "start": start_pt,
            "segments": segments,
        })

    return strokes


def render_svg(
    actions: List[dict],
    width: int = 256,
    height: int = 256,
    stroke: str = "#111827",
    stroke_width: int = 2,
    smooth: bool = False,
    corner_threshold_deg: float = 50.0,
) -> str:
    """Render deterministic SVG markup.

    Args:
        actions: list of valid action dicts
        width: SVG canvas width in px
        height: SVG canvas height in px
        stroke: CSS stroke color
        stroke_width: stroke line width in px
        smooth: if True, converts continuous curved paths into smooth Cubic Bézier curves (C cx1 cy1, cx2 cy2, x y)
    """
    paths = actions_to_paths(actions)
    elements = []

    for points in paths:
        if not points:
            continue

        if not smooth or len(points) < 3:
            # Standard polygonal rendering
            d = f"M {points[0][0] * width:.2f} {points[0][1] * height:.2f}"
            for x, y in points[1:]:
                d += f" L {x * width:.2f} {y * height:.2f}"
        else:
            # Smooth Cubic Bézier rendering
            d = f"M {points[0][0] * width:.2f} {points[0][1] * height:.2f}"
            cmds = points_to_bezier_commands(points, corner_threshold_deg=corner_threshold_deg)
            for cmd in cmds:
                if cmd[0] == "LINE":
                    d += f" L {cmd[1] * width:.2f} {cmd[2] * height:.2f}"
                elif cmd[0] == "CUBIC":
                    cx1, cy1, cx2, cy2, x, y = [v * width if idx % 2 == 0 else v * height for idx, v in enumerate(cmd[1:])]
                    d += f" C {cx1:.2f} {cy1:.2f}, {cx2:.2f} {cy2:.2f}, {x:.2f} {y:.2f}"

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
