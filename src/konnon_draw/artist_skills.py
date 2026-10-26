"""Artist Skill Library: Programmatic drawing and painting helpers emitting v2 actions.

Provides high-level artistic craft:
- Construction: ellipses, arcs, wobbly human-like contours, symmetry, perspective grids, proportion guides.
- Line craft: light-directed line-weight variation (thin for highlight, thick for shadow), sketchy overlapping passes.
- Value & form: directional hatching, cross-hatching, stipple, cast shadows.
- Color: harmonic palette generators, atmospheric perspective depth shading, gradient skies.
- Composition: rule-of-thirds anchors, standard layered structure.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .schema_v2 import (
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
    hex_to_rgb,
    hsv_to_rgb,
    parse_color,
    rgb_to_hex,
)


# ---------------------------------------------------------
# Composition & Standard Layer Setup
# ---------------------------------------------------------

def create_standard_layers() -> List[LayerV2]:
    """Returns a production-grade artist layer stack."""
    return [
        LayerV2(id="background", name="Background / Sky", opacity=1.0, blend_mode="normal"),
        LayerV2(id="underdrawing", name="Guides & Construction", opacity=0.4, blend_mode="normal"),
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


# ---------------------------------------------------------
# Construction Helpers
# ---------------------------------------------------------

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
    steps = num_points if closed else num_points - 1
    total_steps = num_points + (1 if closed else 0)

    for i in range(total_steps):
        t = (i % num_points) / float(num_points) * 2.0 * math.pi
        ex = rx * math.cos(t)
        ey = ry * math.sin(t)
        if wobble > 1e-4:
            ex += rng.normal(0, wobble * rx)
            ey += rng.normal(0, wobble * ry)

        # Rotate and translate
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
        curve_type="catmull_rom",
        taper_in=0.05 if not closed else 0.0,
        taper_out=0.05 if not closed else 0.0,
        layer_id=layer_id,
    )


def make_arc(
    center: Tuple[float, float],
    rx: float,
    ry: float,
    start_deg: float,
    end_deg: float,
    num_points: int = 20,
    pressure: float = 0.8,
    base_width: float = 2.5,
    brush_id: str = "ink_liner",
    color: str = "#111827",
    layer_id: str = "line_art",
) -> StrokeActionV2:
    """Constructs a partial curved arc."""
    cx, cy = center
    pts: List[PointV2] = []
    for i in range(num_points):
        t = i / float(max(1, num_points - 1))
        deg = start_deg + t * (end_deg - start_deg)
        rad = math.radians(deg)
        x = cx + rx * math.cos(rad)
        y = cy + ry * math.sin(rad)
        pts.append(PointV2(x=x, y=y, p=pressure))

    return StrokeActionV2(
        points=pts,
        brush_id=brush_id,
        color=color,
        base_width=base_width,
        curve_type="catmull_rom",
        taper_in=0.1,
        taper_out=0.1,
        layer_id=layer_id,
    )


def make_wobbly_contour(
    points: List[Tuple[float, float]],
    wobble_amount: float = 0.004,
    base_width: float = 2.5,
    brush_id: str = "ink_liner",
    color: str = "#111827",
    layer_id: str = "line_art",
    seed: int = 42,
) -> StrokeActionV2:
    """Takes landmark vertices and adds human drawing micro-variation."""
    rng = np.random.RandomState(seed)
    subdivided_pts: List[PointV2] = []

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dist = math.hypot(x2 - x1, y2 - y1)
        sub_steps = max(2, int(dist / 0.03))

        for s in range(sub_steps if i < len(points) - 2 else sub_steps + 1):
            t = s / float(sub_steps)
            bx = x1 + t * (x2 - x1)
            by = y1 + t * (y2 - y1)
            # Add subtle perpendicular wobble
            dx, dy = x2 - x1, y2 - y1
            dlen = max(1e-5, math.hypot(dx, dy))
            nx, ny = -dy / dlen, dx / dlen
            w = rng.normal(0, wobble_amount)
            press = 0.5 + 0.5 * math.sin(t * math.pi) + rng.normal(0, 0.08)

            subdivided_pts.append(
                PointV2(x=bx + nx * w, y=by + ny * w, p=max(0.15, min(1.0, press)))
            )

    return StrokeActionV2(
        points=subdivided_pts,
        brush_id=brush_id,
        color=color,
        base_width=base_width,
        curve_type="catmull_rom",
        taper_in=0.08,
        taper_out=0.08,
        layer_id=layer_id,
    )


def make_symmetric_stroke(stroke: StrokeActionV2, axis_x: float = 0.5) -> StrokeActionV2:
    """Mirrors a stroke horizontally across axis_x."""
    mirrored_pts = [
        PointV2(x=2.0 * axis_x - p.x, y=p.y, p=p.p, tilt=p.tilt)
        for p in stroke.points
    ]
    return StrokeActionV2(
        points=mirrored_pts,
        brush_id=stroke.brush_id,
        color=stroke.color,
        opacity=stroke.opacity,
        base_width=stroke.base_width,
        curve_type=stroke.curve_type,
        taper_in=stroke.taper_in,
        taper_out=stroke.taper_out,
        layer_id=stroke.layer_id,
        blend_mode=stroke.blend_mode,
    )


def make_perspective_grid(
    vanishing_point: Tuple[float, float] = (0.5, 0.35),
    horizon_y: float = 0.35,
    num_rays: int = 10,
    layer_id: str = "underdrawing",
    color: str = "#cbd5e1",
) -> List[StrokeActionV2]:
    """Emits faint 1-point perspective construction lines."""
    vp_x, vp_y = vanishing_point
    strokes = []

    # Horizon line
    strokes.append(
        StrokeActionV2(
            points=[PointV2(0.0, horizon_y, p=0.3), PointV2(1.0, horizon_y, p=0.3)],
            brush_id="pencil",
            color=color,
            opacity=0.35,
            base_width=1.0,
            curve_type="linear",
            layer_id=layer_id,
        )
    )

    # Perspective rays down to canvas bottom
    for i in range(num_rays + 1):
        x_ground = i / float(num_rays)
        strokes.append(
            StrokeActionV2(
                points=[PointV2(vp_x, vp_y, p=0.2), PointV2(x_ground, 1.0, p=0.4)],
                brush_id="pencil",
                color=color,
                opacity=0.25,
                base_width=1.0,
                curve_type="linear",
                layer_id=layer_id,
            )
        )

    # Receding ground lines
    for k in range(1, 6):
        # Geometric recession
        depth_t = (k / 6.0) ** 2.0
        gy = horizon_y + depth_t * (1.0 - horizon_y)
        strokes.append(
            StrokeActionV2(
                points=[PointV2(0.0, gy, p=0.2), PointV2(1.0, gy, p=0.2)],
                brush_id="pencil",
                color=color,
                opacity=0.2,
                base_width=1.0,
                curve_type="linear",
                layer_id=layer_id,
            )
        )

    return strokes


def make_proportion_guides(
    bounds: Tuple[float, float, float, float],
    divisions: List[float],
    layer_id: str = "underdrawing",
    color: str = "#94a3b8",
) -> List[StrokeActionV2]:
    """Generates proportion guide lines (e.g., Loomis head proportions)."""
    x1, y1, x2, y2 = bounds
    strokes = []
    # Bounding box
    strokes.append(
        StrokeActionV2(
            points=[PointV2(x1, y1), PointV2(x2, y1), PointV2(x2, y2), PointV2(x1, y2), PointV2(x1, y1)],
            brush_id="pencil",
            color=color,
            opacity=0.3,
            base_width=1.0,
            curve_type="linear",
            layer_id=layer_id,
        )
    )
    # Horizontal divisions
    for frac in divisions:
        dy = y1 + frac * (y2 - y1)
        strokes.append(
            StrokeActionV2(
                points=[PointV2(x1 - 0.02, dy, p=0.3), PointV2(x2 + 0.02, dy, p=0.3)],
                brush_id="pencil",
                color=color,
                opacity=0.4,
                base_width=1.0,
                curve_type="linear",
                layer_id=layer_id,
            )
        )
    return strokes


# ---------------------------------------------------------
# Line Craft Helpers
# ---------------------------------------------------------

def apply_line_weight_by_light(
    stroke: StrokeActionV2,
    light_dir: Tuple[float, float] = (-0.707, -0.707),
    min_pressure: float = 0.25,
    max_pressure: float = 1.0,
) -> StrokeActionV2:
    """Modulates stroke pressure according to lighting direction.

    Segments with normals facing away from the light source (shadow side)
    are given heavy pressure; segments facing the light are feathered.
    """
    pts = stroke.points
    if len(pts) < 2:
        return stroke

    lx, ly = light_dir
    l_mag = math.hypot(lx, ly)
    if l_mag > 1e-6:
        lx, ly = lx / l_mag, ly / l_mag

    new_pts: List[PointV2] = []
    n = len(pts)

    for i in range(n):
        p = pts[i]
        # Tangent
        if i == 0:
            tx = pts[1].x - p.x
            ty = pts[1].y - p.y
        elif i == n - 1:
            tx = p.x - pts[i - 1].x
            ty = p.y - pts[i - 1].y
        else:
            tx = pts[i + 1].x - pts[i - 1].x
            ty = pts[i + 1].y - pts[i - 1].y

        t_mag = math.hypot(tx, ty)
        if t_mag < 1e-6:
            nx, ny = 0.0, 1.0
        else:
            # Normal perpendicular (-ty, tx)
            nx, ny = -ty / t_mag, tx / t_mag

        # Dot product with light direction:
        # dot > 0: facing light -> highlight
        # dot < 0: facing away -> shadow
        dot = nx * lx + ny * ly
        # Map dot in [-1, 1] to shadow factor [0 (light), 1 (shadow)]
        shadow_factor = 0.5 - 0.5 * dot
        # Modulate pressure
        press = min_pressure + shadow_factor * (max_pressure - min_pressure)
        new_pts.append(PointV2(x=p.x, y=p.y, p=round(press, 3), tilt=p.tilt))

    return StrokeActionV2(
        points=new_pts,
        brush_id=stroke.brush_id,
        color=stroke.color,
        opacity=stroke.opacity,
        base_width=stroke.base_width,
        curve_type=stroke.curve_type,
        taper_in=stroke.taper_in,
        taper_out=stroke.taper_out,
        layer_id=stroke.layer_id,
        blend_mode=stroke.blend_mode,
    )


def make_sketchy_lines(
    points: List[Tuple[float, float]],
    count: int = 3,
    spread: float = 0.003,
    brush_id: str = "pencil",
    color: str = "#334155",
    base_width: float = 1.8,
    layer_id: str = "underdrawing",
    seed: int = 42,
) -> List[StrokeActionV2]:
    """Generates multiple searching, expressive sketchy strokes along a path."""
    rng = np.random.RandomState(seed)
    strokes = []

    for k in range(count):
        pts = []
        for i, (x, y) in enumerate(points):
            jx = x + rng.normal(0, spread)
            jy = y + rng.normal(0, spread)
            p = float(rng.uniform(0.3, 0.8))
            pts.append(PointV2(x=jx, y=jy, p=p))

        strokes.append(
            StrokeActionV2(
                points=pts,
                brush_id=brush_id,
                color=color,
                opacity=float(rng.uniform(0.4, 0.7)),
                base_width=base_width,
                curve_type="catmull_rom",
                taper_in=0.1,
                taper_out=0.1,
                layer_id=layer_id,
            )
        )
    return strokes


# ---------------------------------------------------------
# Value & Form Helpers: Hatching, Stipple, Cast Shadows
# ---------------------------------------------------------

def generate_directional_hatching(
    polygon: List[Tuple[float, float]],
    angle_deg: float = 45.0,
    spacing: float = 6.0,
    color: str = "#1e293b",
    opacity: float = 0.75,
    cross: bool = False,
    layer_id: str = "shading",
) -> FillActionV2:
    """Emits a hatching or cross-hatching fill within a boundary polygon."""
    return FillActionV2(
        path=polygon,
        fill_type="cross_hatching" if cross else "hatching",
        color=color,
        opacity=opacity,
        angle_deg=angle_deg,
        spacing=spacing,
        layer_id=layer_id,
        blend_mode="multiply",
    )


def generate_cast_shadow_polygon(
    base_points: List[Tuple[float, float]],
    light_offset: Tuple[float, float] = (0.08, 0.04),
    ground_y: Optional[float] = None,
    color: str = "#0f172a",
    opacity: float = 0.35,
    layer_id: str = "shading",
) -> FillActionV2:
    """Projects a cast shadow on the ground plane."""
    ox, oy = light_offset
    shadow_pts = []
    # Forward base
    for x, y in base_points:
        shadow_pts.append((x, ground_y if ground_y is not None else y))
    # Cast projection backward
    for x, y in reversed(base_points):
        gy = ground_y if ground_y is not None else y
        shadow_pts.append((x + ox, gy + oy))

    return FillActionV2(
        path=shadow_pts,
        fill_type="flat",
        color=color,
        opacity=opacity,
        layer_id=layer_id,
        blend_mode="multiply",
    )


def generate_stipple_shading(
    polygon: List[Tuple[float, float]],
    density: float = 0.06,
    color: str = "#09090b",
    opacity: float = 0.8,
    layer_id: str = "shading",
) -> FillActionV2:
    """Emits stippling dots within a polygon boundary."""
    return FillActionV2(
        path=polygon,
        fill_type="stipple",
        color=color,
        opacity=opacity,
        density=density,
        layer_id=layer_id,
        blend_mode="multiply",
    )


# ---------------------------------------------------------
# Color & Atmospheric Perspective Helpers
# ---------------------------------------------------------

def rgb_to_hsv(r: float, g: float, b: float) -> Tuple[float, float, float]:
    """Convert RGB [0..1] to HSV [0..360, 0..1, 0..1]."""
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin

    if delta == 0:
        h = 0.0
    elif cmax == r:
        h = 60.0 * (((g - b) / delta) % 6)
    elif cmax == g:
        h = 60.0 * (((b - r) / delta) + 2)
    else:
        h = 60.0 * (((r - g) / delta) + 4)

    s = 0.0 if cmax == 0 else delta / cmax
    v = cmax
    return (h % 360.0, s, v)


def generate_palette_harmony(base_hex: str, harmony: str = "complementary") -> List[str]:
    """Generates harmonic palettes: complementary, analogous, triadic, warm_shadow_cool_light."""
    r, g, b = parse_color(base_hex)
    h, s, v = rgb_to_hsv(r, g, b)

    if harmony == "complementary":
        h_comp = (h + 180.0) % 360.0
        return [base_hex, rgb_to_hex(*hsv_to_rgb(h_comp, s, v))]

    elif harmony == "analogous":
        h_prev = (h - 30.0) % 360.0
        h_next = (h + 30.0) % 360.0
        return [
            rgb_to_hex(*hsv_to_rgb(h_prev, s, v)),
            base_hex,
            rgb_to_hex(*hsv_to_rgb(h_next, s, v)),
        ]

    elif harmony == "triadic":
        h1 = (h + 120.0) % 360.0
        h2 = (h + 240.0) % 360.0
        return [
            base_hex,
            rgb_to_hex(*hsv_to_rgb(h1, s, v)),
            rgb_to_hex(*hsv_to_rgb(h2, s, v)),
        ]

    elif harmony == "warm_shadow_cool_light":
        # Cool light (blue-ish tint, high V), warm shadow (orange-amber tint, lower V)
        light_color = rgb_to_hex(*hsv_to_rgb((h + 20.0) % 360.0, max(0.1, s * 0.7), min(1.0, v * 1.1)))
        shadow_color = rgb_to_hex(*hsv_to_rgb((h - 40.0) % 360.0, min(1.0, s * 1.2), max(0.1, v * 0.6)))
        return [light_color, base_hex, shadow_color]

    return [base_hex]


def apply_atmospheric_perspective(
    base_hex: str,
    depth: float,
    sky_tint_hex: str = "#bfdbfe",
) -> str:
    """Blends an object's color toward the sky tint as distance (depth 0..1) increases."""
    d = max(0.0, min(1.0, float(depth)))
    r1, g1, b1 = parse_color(base_hex)
    r2, g2, b2 = parse_color(sky_tint_hex)

    # Atmospheric blending: color moves toward sky tint, loses contrast
    blend_factor = 0.75 * (d ** 1.3)
    rout = (1.0 - blend_factor) * r1 + blend_factor * r2
    gout = (1.0 - blend_factor) * g1 + blend_factor * g2
    bout = (1.0 - blend_factor) * b1 + blend_factor * b2
    return rgb_to_hex(rout, gout, bout)


def make_gradient_sky(
    top_color: str = "#1e3a8a",
    horizon_color: str = "#fed7aa",
    horizon_y: float = 0.55,
    layer_id: str = "background",
) -> FillActionV2:
    """Generates a smooth linear gradient sky from top of canvas to horizon."""
    return FillActionV2(
        path=[(0.0, 0.0), (1.0, 0.0), (1.0, horizon_y), (0.0, horizon_y)],
        fill_type="linear_gradient",
        color=top_color,
        color_end=horizon_color,
        angle_deg=90.0,  # top to bottom
        opacity=1.0,
        layer_id=layer_id,
        blend_mode="normal",
    )
