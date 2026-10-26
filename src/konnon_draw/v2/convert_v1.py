"""Bidirectional converter between v1 discrete actions and v2 continuous documents."""

from typing import Any, Dict, List, Optional, Union
from .schema import CanvasConfigV2, DocumentV2, LayerV2, PointV2, StrokeActionV2


def v1_to_v2(
    actions: List[Union[Dict[str, Any], Any]],
    width: int = 1024,
    height: int = 1024,
    default_brush: str = "ink_liner",
    default_color: str = "#111827",
    default_width: float = 3.0,
    base_width: Optional[float] = None,
    **kwargs: Any,
) -> DocumentV2:
    """Converts a sequence of v1 discrete actions (MOVE/LINE/PEN_UP) to a continuous DocumentV2."""
    if base_width is not None:
        default_width = base_width
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=width, height=height),
        layers=[LayerV2(id="main", name="Main")],
    )

    current_stroke_pts: List[PointV2] = []

    def commit_stroke():
        nonlocal current_stroke_pts
        if len(current_stroke_pts) >= 2:
            n = len(current_stroke_pts)
            # Apply synthetic pressure taper: gentle ramp up, sustain, ramp down
            tapered_pts = []
            for i, p in enumerate(current_stroke_pts):
                t = i / max(1, n - 1)
                # Pressure curve: sine envelope peaking at ~0.9, base 0.4
                press = 0.4 + 0.6 * (1.0 - (2.0 * t - 1.0) ** 2)
                tapered_pts.append(PointV2(x=p.x, y=p.y, p=press))

            doc.commands.append(
                StrokeActionV2(
                    points=tapered_pts,
                    brush_id=default_brush,
                    color=default_color,
                    base_width=default_width,
                    curve_type="catmull_rom",
                    taper_in=0.15,
                    taper_out=0.15,
                    stage="sketch",
                    purpose="contour",
                )
            )
        elif len(current_stroke_pts) == 1:
            # Single-point tap: create tiny stroke so it renders as a dot
            p = current_stroke_pts[0]
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(p.x, p.y, 0.8), PointV2(p.x + 0.001, p.y + 0.001, 0.8)],
                    brush_id=default_brush,
                    color=default_color,
                    base_width=default_width,
                    stage="sketch",
                    purpose="accent",
                )
            )
        current_stroke_pts = []

    for act in actions:
        if isinstance(act, dict):
            atype = act.get("type", "")
            raw_x = act.get("x", 0.0)
            raw_y = act.get("y", 0.0)
        else:
            atype = getattr(act, "type", "")
            if hasattr(atype, "name"):
                atype = atype.name
            raw_x = getattr(act, "x", 0.0)
            raw_y = getattr(act, "y", 0.0)

        # Normalize coordinates: if > 1.0, assume 64-bin grid and normalize to [0, 1]
        nx = float(raw_x) / 63.0 if raw_x > 1.0 else float(raw_x)
        ny = float(raw_y) / 63.0 if raw_y > 1.0 else float(raw_y)
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        if atype == "MOVE":
            commit_stroke()
            current_stroke_pts.append(PointV2(x=nx, y=ny, p=0.5))
        elif atype == "LINE":
            if not current_stroke_pts:
                current_stroke_pts.append(PointV2(x=nx, y=ny, p=0.5))
            else:
                current_stroke_pts.append(PointV2(x=nx, y=ny, p=0.8))
        elif atype in ("PEN_UP", "END"):
            commit_stroke()

    commit_stroke()
    return doc


def v2_to_v1(doc: DocumentV2, grid_size: int = 64) -> List[Dict[str, Any]]:
    """Converts a continuous DocumentV2 back to v1 quantized actions (64x64 bins)."""
    actions: List[Dict[str, Any]] = []

    for cmd in doc.commands:
        if isinstance(cmd, StrokeActionV2) or getattr(cmd, "type", "") == "stroke":
            pts = getattr(cmd, "points", [])
            if not pts:
                continue

            # First point is MOVE
            first_x = int(round(pts[0].x * (grid_size - 1)))
            first_y = int(round(pts[0].y * (grid_size - 1)))
            first_x = max(0, min(grid_size - 1, first_x))
            first_y = max(0, min(grid_size - 1, first_y))
            actions.append({"type": "MOVE", "x": first_x, "y": first_y})

            # Subsequent points are LINE
            for p in pts[1:]:
                px = int(round(p.x * (grid_size - 1)))
                py = int(round(p.y * (grid_size - 1)))
                px = max(0, min(grid_size - 1, px))
                py = max(0, min(grid_size - 1, py))
                # Skip duplicate consecutive coordinates
                if px == actions[-1]["x"] and py == actions[-1]["y"]:
                    continue
                actions.append({"type": "LINE", "x": px, "y": py})

            actions.append({"type": "PEN_UP"})

    if actions:
        actions.append({"type": "END"})

    return actions
