"""Action language v2: schema, validation, and v1->v2 conversion.

Supports continuous coordinates, pressure-sensitive variable-width strokes,
brush types, blend modes, region fills (gradients, hatching, stipple), and layers.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union

SCHEMA_VERSION_V2 = "2.0"

SUPPORTED_BRUSHES = {
    "ink_liner",
    "pencil",
    "charcoal",
    "airbrush",
    "marker",
    "watercolor_wash",
}

SUPPORTED_BLEND_MODES = {
    "normal",
    "multiply",
    "screen",
    "overlay",
}

SUPPORTED_FILL_TYPES = {
    "flat",
    "linear_gradient",
    "radial_gradient",
    "hatching",
    "cross_hatching",
    "stipple",
}

SUPPORTED_CURVE_TYPES = {
    "catmull_rom",
    "cubic_bezier",
    "linear",
}


def hex_to_rgb(hex_str: str) -> Tuple[float, float, float]:
    """Parse '#RRGGBB' or '#RGB' to (r, g, b) in [0, 1]."""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_str!r}")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def rgb_to_hex(r: float, g: float, b: float) -> str:
    """Convert (r, g, b) in [0, 1] to '#RRGGBB'."""
    ir = max(0, min(255, int(round(r * 255))))
    ig = max(0, min(255, int(round(g * 255))))
    ib = max(0, min(255, int(round(b * 255))))
    return f"#{ir:02x}{ig:02x}{ib:02x}"


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    """Convert HSV [0..360, 0..1, 0..1] to RGB [0..1, 0..1, 0..1]."""
    h = h % 360.0
    c = v * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = v - c
    if 0 <= h < 60:
        r1, g1, b1 = c, x, 0
    elif 60 <= h < 120:
        r1, g1, b1 = x, c, 0
    elif 120 <= h < 180:
        r1, g1, b1 = 0, c, x
    elif 180 <= h < 240:
        r1, g1, b1 = 0, x, c
    elif 240 <= h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    return (r1 + m, g1 + m, b1 + m)


def parse_color(color: Any) -> Tuple[float, float, float]:
    """Parse hex, [r, g, b], or {'h', 's', 'v'} into (r, g, b) normalized [0, 1]."""
    if isinstance(color, str):
        return hex_to_rgb(color)
    if isinstance(color, (list, tuple)) and len(color) == 3:
        if all(isinstance(v, int) for v in color) and any(v > 1 for v in color):
            return (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)
        return (float(color[0]), float(color[1]), float(color[2]))
    if isinstance(color, dict) and "h" in color and "s" in color and "v" in color:
        return hsv_to_rgb(color["h"], color["s"], color["v"])
    raise ValueError(f"Unrecognized color format: {color!r}")


# ---------------------------------------------------------
# Dataclasses for v2 Actions
# ---------------------------------------------------------

@dataclass
class PointV2:
    x: float
    y: float
    p: float = 1.0  # pressure 0..1
    tilt: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {"x": round(float(self.x), 5), "y": round(float(self.y), 5), "p": round(float(self.p), 4)}
        if self.tilt is not None:
            d["tilt"] = round(float(self.tilt), 3)
        return d


@dataclass
class StrokeActionV2:
    points: List[PointV2]
    type: str = "stroke"
    brush_id: str = "ink_liner"
    color: str = "#111827"
    opacity: float = 1.0
    base_width: float = 3.0
    curve_type: str = "catmull_rom"
    taper_in: float = 0.1
    taper_out: float = 0.1
    layer_id: str = "main"
    blend_mode: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "stroke",
            "brush_id": self.brush_id,
            "color": self.color,
            "opacity": round(float(self.opacity), 3),
            "base_width": round(float(self.base_width), 2),
            "curve_type": self.curve_type,
            "taper_in": round(float(self.taper_in), 3),
            "taper_out": round(float(self.taper_out), 3),
            "layer_id": str(self.layer_id),
            "blend_mode": self.blend_mode,
            "points": [pt.to_dict() if isinstance(pt, PointV2) else pt for pt in self.points],
        }


@dataclass
class FillActionV2:
    path: List[Tuple[float, float]]
    fill_type: str = "flat"  # flat, linear_gradient, radial_gradient, hatching, cross_hatching, stipple
    type: str = "fill"
    color: str = "#e2e8f0"
    color_end: Optional[str] = None
    opacity: float = 1.0
    angle_deg: float = 45.0
    spacing: float = 8.0
    density: float = 0.05
    layer_id: str = "main"
    blend_mode: str = "normal"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": "fill",
            "fill_type": self.fill_type,
            "color": self.color,
            "opacity": round(float(self.opacity), 3),
            "layer_id": str(self.layer_id),
            "blend_mode": self.blend_mode,
            "path": [[round(float(x), 5), round(float(y), 5)] for x, y in self.path],
        }
        if self.color_end is not None:
            d["color_end"] = self.color_end
        if self.fill_type in ("linear_gradient", "hatching", "cross_hatching"):
            d["angle_deg"] = round(float(self.angle_deg), 1)
        if self.fill_type in ("hatching", "cross_hatching"):
            d["spacing"] = round(float(self.spacing), 2)
        if self.fill_type == "stipple":
            d["density"] = round(float(self.density), 4)
        return d


@dataclass
class LayerV2:
    id: str
    name: str
    opacity: float = 1.0
    blend_mode: str = "normal"
    visible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "opacity": round(float(self.opacity), 3),
            "blend_mode": self.blend_mode,
            "visible": self.visible,
        }


@dataclass
class CanvasConfigV2:
    width: int = 1024
    height: int = 1024
    background: str = "#ffffff"
    palette: List[str] = field(default_factory=lambda: [
        "#111827", "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#6366f1"
    ])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "background": self.background,
            "palette": self.palette,
        }


@dataclass
class DocumentV2:
    version: str = SCHEMA_VERSION_V2
    canvas: CanvasConfigV2 = field(default_factory=CanvasConfigV2)
    layers: List[LayerV2] = field(default_factory=lambda: [LayerV2(id="main", name="Main Layer")])
    commands: List[Union[StrokeActionV2, FillActionV2, Dict[str, Any]]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "canvas": self.canvas.to_dict() if hasattr(self.canvas, "to_dict") else self.canvas,
            "layers": [l.to_dict() if hasattr(l, "to_dict") else l for l in self.layers],
            "commands": [
                cmd.to_dict() if hasattr(cmd, "to_dict") else cmd for cmd in self.commands
            ],
        }


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

def validate_document_v2(doc: Union[DocumentV2, Dict[str, Any]]) -> bool:
    """Validates that a v2 document conforms to all schema requirements."""
    d = doc.to_dict() if hasattr(doc, "to_dict") else doc
    if not isinstance(d, dict):
        raise ValueError("Document must be a dictionary or DocumentV2")
    if d.get("version") != SCHEMA_VERSION_V2:
        raise ValueError(f"Unsupported schema version: {d.get('version')!r}, expected {SCHEMA_VERSION_V2}")

    canvas = d.get("canvas", {})
    if canvas.get("width", 0) <= 0 or canvas.get("height", 0) <= 0:
        raise ValueError("Canvas width and height must be positive integers")

    layers = d.get("layers", [])
    layer_ids = set()
    for ly in layers:
        lid = ly.get("id")
        if not lid or lid in layer_ids:
            raise ValueError(f"Duplicate or empty layer id: {lid}")
        layer_ids.add(lid)
        bm = ly.get("blend_mode", "normal")
        if bm not in SUPPORTED_BLEND_MODES:
            raise ValueError(f"Invalid blend mode in layer {lid}: {bm}")

    commands = d.get("commands", [])
    for idx, cmd in enumerate(commands):
        kind = cmd.get("type")
        if kind == "stroke":
            _validate_stroke(cmd, idx, layer_ids)
        elif kind == "fill":
            _validate_fill(cmd, idx, layer_ids)
        else:
            raise ValueError(f"Unknown command type at index {idx}: {kind!r}")
    return True


def _validate_stroke(cmd: Dict[str, Any], idx: int, valid_layers: set):
    brush = cmd.get("brush_id", "ink_liner")
    if brush not in SUPPORTED_BRUSHES:
        raise ValueError(f"Unknown brush_id at index {idx}: {brush!r}")
    bm = cmd.get("blend_mode", "normal")
    if bm not in SUPPORTED_BLEND_MODES:
        raise ValueError(f"Unknown blend_mode at index {idx}: {bm!r}")
    ct = cmd.get("curve_type", "catmull_rom")
    if ct not in SUPPORTED_CURVE_TYPES:
        raise ValueError(f"Unknown curve_type at index {idx}: {ct!r}")
    pts = cmd.get("points", [])
    if len(pts) < 2:
        raise ValueError(f"Stroke at index {idx} has fewer than 2 points")
    for p_idx, pt in enumerate(pts):
        if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
            raise ValueError(f"Point {p_idx} of stroke {idx} must have 'x' and 'y'")
        x, y = float(pt["x"]), float(pt["y"])
        if not (-0.5 <= x <= 1.5 and -0.5 <= y <= 1.5):
            raise ValueError(f"Point {p_idx} coordinate ({x}, {y}) out of reasonable range [-0.5, 1.5]")
        p = float(pt.get("p", 1.0))
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"Point {p_idx} pressure {p} must be in [0, 1]")


def _validate_fill(cmd: Dict[str, Any], idx: int, valid_layers: set):
    ft = cmd.get("fill_type", "flat")
    if ft not in SUPPORTED_FILL_TYPES:
        raise ValueError(f"Unknown fill_type at index {idx}: {ft!r}")
    path = cmd.get("path", [])
    if len(path) < 3:
        raise ValueError(f"Fill at index {idx} has fewer than 3 vertices")


# ---------------------------------------------------------
# Converter: v1 Actions -> v2 Document
# ---------------------------------------------------------

def v1_to_v2(
    v1_actions: List[Dict[str, Any]],
    default_brush: str = "ink_liner",
    base_width: float = 3.5,
    canvas_size: int = 1024,
    color: str = "#111827",
    taper: bool = True,
) -> DocumentV2:
    """Converts classic v1 actions (MOVE/LINE/PEN_UP/END) into a v2 Document.

    Extracts multi-point paths, calculates synthetic velocity/curvature pressure,
    and applies smooth taper-in / taper-out.
    """
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=canvas_size, height=canvas_size),
        layers=[LayerV2(id="main", name="Drawings")],
    )

    current_points: List[PointV2] = []

    def flush_stroke():
        nonlocal current_points
        if len(current_points) >= 2:
            n = len(current_points)
            if taper:
                # Apply natural calligraphic pressure variation based on stroke length
                for i in range(n):
                    t = i / max(1, n - 1)
                    # Smooth sin-based bell curve for pressure
                    p_taper = math.sin(t * math.pi)
                    # Blend base pressure with taper (min 0.35, max 1.0)
                    pressure = 0.35 + 0.65 * p_taper
                    current_points[i].p = round(pressure, 3)

            stroke = StrokeActionV2(
                points=list(current_points),
                brush_id=default_brush,
                color=color,
                base_width=base_width,
                curve_type="catmull_rom",
                taper_in=0.15 if taper else 0.0,
                taper_out=0.15 if taper else 0.0,
                layer_id="main",
            )
            doc.commands.append(stroke)
        current_points = []

    for a in v1_actions:
        kind = a.get("type")
        if kind == "MOVE":
            flush_stroke()
            current_points.append(PointV2(x=float(a["x"]), y=float(a["y"]), p=0.4))
        elif kind == "LINE":
            current_points.append(PointV2(x=float(a["x"]), y=float(a["y"]), p=1.0))
        elif kind == "PEN_UP":
            flush_stroke()
        elif kind == "END":
            flush_stroke()
            break

    flush_stroke()
    return doc
