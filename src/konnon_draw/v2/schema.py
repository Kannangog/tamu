"""Action language v2: schema, validation, and serialization.

Supports continuous coordinates, variable-width strokes, 6+ artist brushes,
depth ranking, stages, purposes, blend modes, region fills, and layers.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

SCHEMA_VERSION_V2 = "2.0"

SUPPORTED_BRUSHES = {
    "ink_liner",
    "pencil",
    "charcoal",
    "airbrush",
    "colored_pencil",
    "marker",
    "watercolor_wash",
}

SUPPORTED_STAGES = {
    "layout",
    "guide",
    "sketch",
    "ink",
    "shade",
    "effect",
}

SUPPORTED_PURPOSES = {
    "construction",
    "contour",
    "hatch",
    "accent",
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
    stage: str = "sketch"
    purpose: str = "contour"
    depth_rank: int = 0
    part_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
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
            "stage": str(self.stage),
            "purpose": str(self.purpose),
            "depth_rank": int(self.depth_rank),
            "points": [pt.to_dict() if isinstance(pt, PointV2) else pt for pt in self.points],
        }
        if self.part_id is not None:
            d["part_id"] = str(self.part_id)
        return d


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
    depth_rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": "fill",
            "fill_type": self.fill_type,
            "color": self.color,
            "opacity": round(float(self.opacity), 3),
            "layer_id": str(self.layer_id),
            "blend_mode": self.blend_mode,
            "depth_rank": int(self.depth_rank),
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
    layers: List[LayerV2] = field(default_factory=lambda: [LayerV2(id="main", name="Main")])
    commands: List[Union[StrokeActionV2, FillActionV2]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "canvas": self.canvas.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
            "commands": [cmd.to_dict() for cmd in self.commands],
        }

    def save(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DocumentV2":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentV2":
        canvas_data = data.get("canvas", {})
        canvas = CanvasConfigV2(
            width=canvas_data.get("width", 1024),
            height=canvas_data.get("height", 1024),
            background=canvas_data.get("background", "#ffffff"),
            palette=canvas_data.get("palette", [
                "#111827", "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#6366f1"
            ]),
        )

        layers = [
            LayerV2(
                id=ld["id"],
                name=ld.get("name", ld["id"]),
                opacity=float(ld.get("opacity", 1.0)),
                blend_mode=ld.get("blend_mode", "normal"),
                visible=bool(ld.get("visible", True)),
            )
            for ld in data.get("layers", [{"id": "main", "name": "Main"}])
        ]

        commands: List[Union[StrokeActionV2, FillActionV2]] = []
        for cd in data.get("commands", []):
            cmd_type = cd.get("type", "stroke")
            if cmd_type == "stroke":
                pts = [
                    PointV2(
                        x=float(p["x"]),
                        y=float(p["y"]),
                        p=float(p.get("p", 1.0)),
                        tilt=float(p["tilt"]) if "tilt" in p and p["tilt"] is not None else None,
                    )
                    for p in cd.get("points", [])
                ]
                commands.append(
                    StrokeActionV2(
                        points=pts,
                        brush_id=cd.get("brush_id", "ink_liner"),
                        color=cd.get("color", "#111827"),
                        opacity=float(cd.get("opacity", 1.0)),
                        base_width=float(cd.get("base_width", 3.0)),
                        curve_type=cd.get("curve_type", "catmull_rom"),
                        taper_in=float(cd.get("taper_in", 0.1)),
                        taper_out=float(cd.get("taper_out", 0.1)),
                        layer_id=cd.get("layer_id", "main"),
                        blend_mode=cd.get("blend_mode", "normal"),
                        stage=cd.get("stage", "sketch"),
                        purpose=cd.get("purpose", "contour"),
                        depth_rank=int(cd.get("depth_rank", 0)),
                        part_id=cd.get("part_id", None),
                    )
                )
            elif cmd_type == "fill":
                commands.append(
                    FillActionV2(
                        path=[(float(pt[0]), float(pt[1])) for pt in cd.get("path", [])],
                        fill_type=cd.get("fill_type", "flat"),
                        color=cd.get("color", "#e2e8f0"),
                        color_end=cd.get("color_end", None),
                        opacity=float(cd.get("opacity", 1.0)),
                        angle_deg=float(cd.get("angle_deg", 45.0)),
                        spacing=float(cd.get("spacing", 8.0)),
                        density=float(cd.get("density", 0.05)),
                        layer_id=cd.get("layer_id", "main"),
                        blend_mode=cd.get("blend_mode", "normal"),
                        depth_rank=int(cd.get("depth_rank", 0)),
                    )
                )

        return cls(
            version=data.get("version", SCHEMA_VERSION_V2),
            canvas=canvas,
            layers=layers,
            commands=commands,
        )


def validate_document_v2(doc: Union[Dict[str, Any], DocumentV2]) -> bool:
    """Strictly validates DocumentV2 schema integrity. Returns True on success, raises ValueError on failure."""
    d = doc.to_dict() if hasattr(doc, "to_dict") else doc
    if not isinstance(d, dict):
        raise ValueError(f"Expected dict or DocumentV2, got {type(doc)}")

    if d.get("version") != SCHEMA_VERSION_V2:
        raise ValueError(f"Invalid schema version: {d.get('version')!r}, expected {SCHEMA_VERSION_V2}")

    canvas = d.get("canvas", {})
    if not isinstance(canvas, dict):
        raise ValueError("Canvas config must be a dictionary")
    if canvas.get("width", 0) <= 0 or canvas.get("height", 0) <= 0:
        raise ValueError("Canvas width and height must be positive integers")

    layers = d.get("layers", [])
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("Document must declare at least one layer")
    layer_ids = set()
    for idx, l in enumerate(layers):
        lid = l.get("id")
        if not lid or not isinstance(lid, str):
            raise ValueError(f"Layer {idx} missing valid string id")
        if lid in layer_ids:
            raise ValueError(f"Duplicate layer id {lid!r}")
        layer_ids.add(lid)
        bm = l.get("blend_mode", "normal")
        if bm not in SUPPORTED_BLEND_MODES:
            raise ValueError(f"Layer {lid!r} has unsupported blend mode {bm!r}")

    commands = d.get("commands", [])
    if not isinstance(commands, list):
        raise ValueError("Commands must be a list")
    for idx, cmd in enumerate(commands):
        cmd_type = cmd.get("type")
        lid = cmd.get("layer_id", "main")
        if lid not in layer_ids:
            raise ValueError(f"Command {idx} references undefined layer_id {lid!r}")

        if cmd_type == "stroke":
            brush = cmd.get("brush_id", "ink_liner")
            if brush not in SUPPORTED_BRUSHES:
                raise ValueError(f"Command {idx} has unsupported brush {brush!r}")
            stage = cmd.get("stage", "sketch")
            if stage not in SUPPORTED_STAGES:
                raise ValueError(f"Command {idx} has unsupported stage {stage!r}")
            purpose = cmd.get("purpose", "contour")
            if purpose not in SUPPORTED_PURPOSES:
                raise ValueError(f"Command {idx} has unsupported purpose {purpose!r}")
            pts = cmd.get("points", [])
            if not isinstance(pts, list) or len(pts) < 2:
                raise ValueError(f"Command {idx} stroke must contain at least 2 points")
            for p_idx, pt in enumerate(pts):
                if not isinstance(pt, dict) or "x" not in pt or "y" not in pt:
                    raise ValueError(f"Point {p_idx} of stroke {idx} must have 'x' and 'y'")
                x, y = float(pt["x"]), float(pt["y"])
                if not (-0.5 <= x <= 1.5 and -0.5 <= y <= 1.5):
                    raise ValueError(f"Point {p_idx} coordinate ({x}, {y}) out of reasonable range [-0.5, 1.5]")
                p = float(pt.get("p", 1.0))
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"Point {p_idx} pressure {p} must be in [0, 1]")

        elif cmd_type == "fill":
            ftype = cmd.get("fill_type", "flat")
            if ftype not in SUPPORTED_FILL_TYPES:
                raise ValueError(f"Command {idx} has unsupported fill_type {ftype!r}")
            path = cmd.get("path", [])
            if not isinstance(path, list) or len(path) < 3:
                raise ValueError(f"Command {idx} fill path must contain at least 3 vertices")

        else:
            raise ValueError(f"Command {idx} has unknown action type {cmd_type!r}")

    return True
