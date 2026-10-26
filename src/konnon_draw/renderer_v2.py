"""High-resolution brush and rendering engine (v2).

Features:
- Variable-width strokes driven by pressure and smooth taper curves.
- 6 distinct brush models:
  * ink_liner: crisp, solid antialiased line with calligraphic taper
  * pencil: paper tooth / graphite grain texture, pressure-sensitive density
  * charcoal: rough, powdery irregular edges with core darkness
  * airbrush: soft Gaussian radial falloff
  * marker: semi-transparent chisel nib with multiply darkening
  * watercolor_wash: translucent washes with edge pigment pooling (wet-on-dry)
- Region fills: flat, linear gradient, radial gradient, hatching, cross-hatching, stipple.
- Layer blending: normal, multiply, screen, overlay with per-layer opacity.
- Deterministic output given a random seed.
- High-quality supersampling (default 2x downsampled with LANCZOS).
- Clean standalone SVG export.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .schema_v2 import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
    hex_to_rgb,
    parse_color,
    rgb_to_hex,
    validate_document_v2,
)


def _ensure_doc(doc_or_dict: Union[DocumentV2, Dict[str, Any]]) -> DocumentV2:
    if isinstance(doc_or_dict, DocumentV2):
        return doc_or_dict
    validate_document_v2(doc_or_dict)
    canvas_d = doc_or_dict.get("canvas", {})
    canvas = CanvasConfigV2(
        width=int(canvas_d.get("width", 1024)),
        height=int(canvas_d.get("height", 1024)),
        background=canvas_d.get("background", "#ffffff"),
        palette=canvas_d.get("palette", []),
    )
    layers = [
        LayerV2(
            id=l["id"],
            name=l.get("name", l["id"]),
            opacity=float(l.get("opacity", 1.0)),
            blend_mode=l.get("blend_mode", "normal"),
            visible=bool(l.get("visible", True)),
        )
        for l in doc_or_dict.get("layers", [])
    ]
    commands = []
    for c in doc_or_dict.get("commands", []):
        ctype = c.get("type")
        if ctype == "stroke":
            pts = [
                PointV2(x=p["x"], y=p["y"], p=p.get("p", 1.0), tilt=p.get("tilt"))
                for p in c.get("points", [])
            ]
            commands.append(
                StrokeActionV2(
                    points=pts,
                    brush_id=c.get("brush_id", "ink_liner"),
                    color=c.get("color", "#111827"),
                    opacity=float(c.get("opacity", 1.0)),
                    base_width=float(c.get("base_width", 3.0)),
                    curve_type=c.get("curve_type", "catmull_rom"),
                    taper_in=float(c.get("taper_in", 0.1)),
                    taper_out=float(c.get("taper_out", 0.1)),
                    layer_id=c.get("layer_id", "main"),
                    blend_mode=c.get("blend_mode", "normal"),
                )
            )
        elif ctype == "fill":
            commands.append(
                FillActionV2(
                    path=[(float(p[0]), float(p[1])) for p in c.get("path", [])],
                    fill_type=c.get("fill_type", "flat"),
                    color=c.get("color", "#e2e8f0"),
                    color_end=c.get("color_end"),
                    opacity=float(c.get("opacity", 1.0)),
                    angle_deg=float(c.get("angle_deg", 45.0)),
                    spacing=float(c.get("spacing", 8.0)),
                    density=float(c.get("density", 0.05)),
                    layer_id=c.get("layer_id", "main"),
                    blend_mode=c.get("blend_mode", "normal"),
                )
            )
    return DocumentV2(version="2.0", canvas=canvas, layers=layers, commands=commands)


# ---------------------------------------------------------
# Spline Interpolation & Arc-length Parametrization
# ---------------------------------------------------------

def interpolate_points(
    points: List[PointV2],
    curve_type: str = "catmull_rom",
    target_spacing_px: float = 2.0,
    width_px: int = 1024,
    height_px: int = 1024,
) -> List[Tuple[float, float, float, float]]:
    """Sample spline densely.

    Returns list of (px, py, pressure, s_arc) where s_arc is normalized [0, 1].
    """
    if len(points) == 0:
        return []
    if len(points) == 1:
        p0 = points[0]
        return [(p0.x * width_px, p0.y * height_px, p0.p, 0.0)]

    pixel_pts = [(p.x * width_px, p.y * height_px, p.p) for p in points]
    n = len(pixel_pts)

    if curve_type == "linear" or n == 2:
        samples = []
        for i in range(n - 1):
            x1, y1, p1 = pixel_pts[i]
            x2, y2, p2 = pixel_pts[i + 1]
            dist = math.hypot(x2 - x1, y2 - y1)
            steps = max(1, int(math.ceil(dist / target_spacing_px)))
            for s in range(steps):
                t = s / steps
                samples.append((
                    x1 + t * (x2 - x1),
                    y1 + t * (y2 - y1),
                    p1 + t * (p2 - p1),
                ))
        samples.append(pixel_pts[-1])
    else:
        # Catmull-Rom or cubic spline
        samples = []
        for i in range(n - 1):
            p1 = pixel_pts[i]
            p2 = pixel_pts[i + 1]
            p0 = pixel_pts[i - 1] if i > 0 else (2 * p1[0] - p2[0], 2 * p1[1] - p2[1], p1[2])
            p3 = pixel_pts[i + 2] if i + 2 < n else (2 * p2[0] - p1[0], 2 * p2[1] - p1[1], p2[2])

            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            steps = max(2, int(math.ceil(dist / target_spacing_px)))

            for step in range(steps):
                t = step / steps
                t2 = t * t
                t3 = t2 * t
                # Catmull-Rom basis
                x = 0.5 * (
                    (2 * p1[0])
                    + (-p0[0] + p2[0]) * t
                    + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                    + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
                )
                y = 0.5 * (
                    (2 * p1[1])
                    + (-p0[1] + p2[1]) * t
                    + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                    + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
                )
                p = (1.0 - t) * p1[2] + t * p2[2]
                samples.append((x, y, p))
        samples.append(pixel_pts[-1])

    # Compute cumulative arc lengths
    arc_lengths = [0.0]
    total_len = 0.0
    for i in range(len(samples) - 1):
        d = math.hypot(samples[i + 1][0] - samples[i][0], samples[i + 1][1] - samples[i][1])
        total_len += d
        arc_lengths.append(total_len)

    result = []
    denom = total_len if total_len > 1e-5 else 1.0
    for i, pt in enumerate(samples):
        s_norm = arc_lengths[i] / denom
        result.append((pt[0], pt[1], pt[2], s_norm))

    return result


def compute_taper(s: float, taper_in: float, taper_out: float) -> float:
    """Computes smooth sine-eased taper multiplier in [0, 1]."""
    factor = 1.0
    if taper_in > 1e-4 and s < taper_in:
        factor *= math.sin((s / taper_in) * (math.pi / 2.0))
    if taper_out > 1e-4 and (1.0 - s) < taper_out:
        factor *= math.sin(((1.0 - s) / taper_out) * (math.pi / 2.0))
    return max(0.0, min(1.0, factor))


# ---------------------------------------------------------
# Ribbon Polygon Generator for Variable-Width Strokes
# ---------------------------------------------------------

def build_stroke_ribbon(
    samples: List[Tuple[float, float, float, float]],
    base_width: float,
    taper_in: float,
    taper_out: float,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """Generates an offset ribbon polygon boundary and center points with radii.

    Returns (polygon_points, list of (x, y, radius)).
    """
    if len(samples) < 2:
        return [], []

    left_edge = []
    right_edge = []
    centers_and_radii = []

    n = len(samples)
    for i in range(n):
        x, y, pressure, s = samples[i]
        taper = compute_taper(s, taper_in, taper_out)
        w = max(0.4, base_width * pressure * taper)
        r = w * 0.5
        centers_and_radii.append((x, y, r))

        # Tangent vector
        if i == 0:
            dx = samples[1][0] - x
            dy = samples[1][1] - y
        elif i == n - 1:
            dx = x - samples[i - 1][0]
            dy = y - samples[i - 1][1]
        else:
            dx = samples[i + 1][0] - samples[i - 1][0]
            dy = samples[i + 1][1] - samples[i - 1][1]

        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            nx, ny = 0.0, 1.0
        else:
            nx = -dy / dist
            ny = dx / dist

        left_edge.append((x + nx * r, y + ny * r))
        right_edge.append((x - nx * r, y - ny * r))

    polygon = left_edge + list(reversed(right_edge))
    return polygon, centers_and_radii


# ---------------------------------------------------------
# Layer Blending Engine
# ---------------------------------------------------------

def blend_layers(
    dst: Image.Image,
    src: Image.Image,
    mode: str = "normal",
    opacity: float = 1.0,
) -> Image.Image:
    """Blends RGBA src over RGBA dst using blend modes: normal, multiply, screen, overlay."""
    if opacity <= 0.0:
        return dst
    if mode == "normal" and abs(opacity - 1.0) < 1e-3:
        dst.alpha_composite(src)
        return dst

    dst_arr = np.asarray(dst, dtype=np.float32) / 255.0
    src_arr = np.asarray(src, dtype=np.float32) / 255.0

    dst_rgb = dst_arr[:, :, :3]
    dst_a = dst_arr[:, :, 3:4]
    src_rgb = src_arr[:, :, :3]
    src_a = src_arr[:, :, 3:4] * float(opacity)

    if mode == "normal":
        blend_rgb = src_rgb
    elif mode == "multiply":
        blend_rgb = src_rgb * dst_rgb
    elif mode == "screen":
        blend_rgb = 1.0 - (1.0 - src_rgb) * (1.0 - dst_rgb)
    elif mode == "overlay":
        mask = dst_rgb < 0.5
        blend_rgb = np.where(
            mask,
            2.0 * src_rgb * dst_rgb,
            1.0 - 2.0 * (1.0 - src_rgb) * (1.0 - dst_rgb),
        )
    else:
        blend_rgb = src_rgb

    out_a = src_a + dst_a * (1.0 - src_a)
    safe_out_a = np.where(out_a > 1e-6, out_a, 1.0)
    out_rgb = (blend_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)) / safe_out_a

    out_arr = np.concatenate([out_rgb, out_a], axis=2)
    out_arr = np.clip(out_arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out_arr, mode="RGBA")


# ---------------------------------------------------------
# Rasterization Primitives: Brushes & Fills
# ---------------------------------------------------------

def draw_ink_stroke(
    draw: ImageDraw.ImageDraw,
    polygon: List[Tuple[float, float]],
    centers: List[Tuple[float, float, float]],
    rgba: Tuple[int, int, int, int],
):
    """Draws ink ribbon directly into an ImageDraw context."""
    if len(polygon) >= 3:
        draw.polygon(polygon, fill=rgba)
    for x, y, r in centers:
        if r > 0.5:
            draw.ellipse([x - r, y - r, x + r, y + r], fill=rgba)


def render_stroke_brush(
    stroke: StrokeActionV2,
    canvas_w: int,
    canvas_h: int,
    scale: float,
    rng: np.random.RandomState,
) -> Image.Image:
    """Renders a single stroke on an RGBA transparent surface."""
    pts = interpolate_points(
        stroke.points,
        curve_type=stroke.curve_type,
        target_spacing_px=1.5 * scale,
        width_px=canvas_w,
        height_px=canvas_h,
    )
    if not pts:
        return Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    rgb = [int(round(c * 255)) for c in parse_color(stroke.color)]
    base_width = stroke.base_width * scale
    brush = stroke.brush_id

    polygon, centers = build_stroke_ribbon(
        pts,
        base_width=base_width,
        taper_in=stroke.taper_in,
        taper_out=stroke.taper_out,
    )

    stroke_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(stroke_layer)

    if brush == "ink_liner":
        alpha = int(round(stroke.opacity * 255))
        draw_ink_stroke(draw, polygon, centers, (rgb[0], rgb[1], rgb[2], alpha))

    elif brush == "pencil":
        mask_img = Image.new("L", (canvas_w, canvas_h), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        if len(polygon) >= 3:
            mask_draw.polygon(polygon, fill=255)
        for x, y, r in centers:
            if r > 0.4:
                mask_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

        mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
        noise = rng.uniform(0.15, 1.0, size=(canvas_h, canvas_w)).astype(np.float32)
        mean_p = float(np.mean([p.p for p in stroke.points])) if stroke.points else 1.0
        tooth_factor = 0.35 + 0.65 * (noise ** (1.5 / max(0.2, mean_p)))
        final_alpha = mask_arr * tooth_factor * stroke.opacity * 255.0

        r_arr = np.full((canvas_h, canvas_w), rgb[0], dtype=np.uint8)
        g_arr = np.full((canvas_h, canvas_w), rgb[1], dtype=np.uint8)
        b_arr = np.full((canvas_h, canvas_w), rgb[2], dtype=np.uint8)
        a_arr = np.clip(final_alpha, 0, 255).astype(np.uint8)
        stroke_layer = Image.fromarray(np.stack([r_arr, g_arr, b_arr, a_arr], axis=2), mode="RGBA")

    elif brush == "charcoal":
        mask_img = Image.new("L", (canvas_w, canvas_h), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        if len(polygon) >= 3:
            mask_draw.polygon(polygon, fill=220)
        for x, y, r in centers:
            if r > 0.5:
                mask_draw.ellipse([x - r, y - r, x + r, y + r], fill=220)

        step = max(1, len(centers) // 80)
        for x, y, r in centers[::step]:
            for _ in range(5):
                jx = x + rng.normal(0, r * 0.9)
                jy = y + rng.normal(0, r * 0.9)
                jr = max(0.5, r * rng.uniform(0.1, 0.4))
                mask_draw.ellipse([jx - jr, jy - jr, jx + jr, jy + jr], fill=int(rng.uniform(80, 200)))

        mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0
        charcoal_noise = rng.uniform(0.3, 1.0, size=(canvas_h, canvas_w)).astype(np.float32)
        final_alpha = mask_arr * charcoal_noise * stroke.opacity * 255.0

        r_arr = np.full((canvas_h, canvas_w), rgb[0], dtype=np.uint8)
        g_arr = np.full((canvas_h, canvas_w), rgb[1], dtype=np.uint8)
        b_arr = np.full((canvas_h, canvas_w), rgb[2], dtype=np.uint8)
        a_arr = np.clip(final_alpha, 0, 255).astype(np.uint8)
        stroke_layer = Image.fromarray(np.stack([r_arr, g_arr, b_arr, a_arr], axis=2), mode="RGBA")

    elif brush == "airbrush":
        mask_img = Image.new("L", (canvas_w, canvas_h), 0)
        mask_draw = ImageDraw.Draw(mask_img)
        for x, y, r in centers:
            r_air = max(1.0, r * 0.6)
            mask_draw.ellipse([x - r_air, y - r_air, x + r_air, y + r_air], fill=180)

        blur_radius = max(1.5, base_width * 0.75)
        blurred_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        mask_arr = np.asarray(blurred_mask, dtype=np.float32) / 255.0
        final_alpha = mask_arr * stroke.opacity * 255.0

        r_arr = np.full((canvas_h, canvas_w), rgb[0], dtype=np.uint8)
        g_arr = np.full((canvas_h, canvas_w), rgb[1], dtype=np.uint8)
        b_arr = np.full((canvas_h, canvas_w), rgb[2], dtype=np.uint8)
        a_arr = np.clip(final_alpha, 0, 255).astype(np.uint8)
        stroke_layer = Image.fromarray(np.stack([r_arr, g_arr, b_arr, a_arr], axis=2), mode="RGBA")

    elif brush == "marker":
        marker_alpha = int(round(min(1.0, stroke.opacity * 0.65) * 255))
        rgba = (rgb[0], rgb[1], rgb[2], marker_alpha)
        chisel_angle = math.radians(45.0)
        cos_a = math.cos(chisel_angle)
        sin_a = math.sin(chisel_angle)
        for x, y, r in centers:
            x1, y1 = x - cos_a * r, y - sin_a * r
            x2, y2 = x + cos_a * r, y + sin_a * r
            draw.line([(x1, y1), (x2, y2)], fill=rgba, width=max(1, int(round(r * 0.8))))

    elif brush == "watercolor_wash":
        body_alpha = int(round(stroke.opacity * 0.25 * 255))
        edge_alpha = int(round(stroke.opacity * 0.85 * 255))
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=(rgb[0], rgb[1], rgb[2], body_alpha))
        for x, y, r in centers:
            if r > 0.5:
                draw.ellipse([x - r, y - r, x + r, y + r], fill=(rgb[0], rgb[1], rgb[2], body_alpha))
        if len(polygon) >= 3:
            edge_pts = polygon + [polygon[0]]
            draw.line(edge_pts, fill=(rgb[0], rgb[1], rgb[2], edge_alpha), width=max(1, int(round(1.5 * scale))))

    return stroke_layer


def render_fill_action(
    fill: FillActionV2,
    canvas_w: int,
    canvas_h: int,
    scale: float,
    rng: np.random.RandomState,
) -> Image.Image:
    """Renders a region fill: flat, linear/radial gradient, hatching, cross-hatching, stipple."""
    fill_layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    px_path = [(p[0] * canvas_w, p[1] * canvas_h) for p in fill.path]
    if len(px_path) < 3:
        return fill_layer

    mask_img = Image.new("L", (canvas_w, canvas_h), 0)
    mask_draw = ImageDraw.Draw(mask_img)
    mask_draw.polygon(px_path, fill=255)
    mask_arr = np.asarray(mask_img, dtype=np.float32) / 255.0

    color1 = parse_color(fill.color)
    rgb1 = np.array(color1, dtype=np.float32)

    if fill.fill_type == "flat":
        rgba = (
            int(round(rgb1[0] * 255)),
            int(round(rgb1[1] * 255)),
            int(round(rgb1[2] * 255)),
            int(round(fill.opacity * 255)),
        )
        draw = ImageDraw.Draw(fill_layer)
        draw.polygon(px_path, fill=rgba)
        return fill_layer

    if fill.fill_type == "linear_gradient":
        color2 = parse_color(fill.color_end if fill.color_end else fill.color)
        rgb2 = np.array(color2, dtype=np.float32)
        angle_rad = math.radians(fill.angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        ys, xs = np.indices((canvas_h, canvas_w))
        proj = xs * cos_a + ys * sin_a

        vx = np.array([p[0] for p in px_path])
        vy = np.array([p[1] for p in px_path])
        v_proj = vx * cos_a + vy * sin_a
        min_p, max_p = float(v_proj.min()), float(v_proj.max())
        denom = max(1e-4, max_p - min_p)

        t = np.clip((proj - min_p) / denom, 0.0, 1.0)[:, :, None]
        grad_rgb = (1.0 - t) * rgb1 + t * rgb2
        alpha = mask_arr[:, :, None] * fill.opacity

        arr = np.concatenate([grad_rgb, alpha], axis=2)
        return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGBA")

    if fill.fill_type == "radial_gradient":
        color2 = parse_color(fill.color_end if fill.color_end else fill.color)
        rgb2 = np.array(color2, dtype=np.float32)

        vx = np.array([p[0] for p in px_path])
        vy = np.array([p[1] for p in px_path])
        cx, cy = float(vx.mean()), float(vy.mean())
        max_dist = max(1e-4, float(np.max(np.hypot(vx - cx, vy - cy))))

        ys, xs = np.indices((canvas_h, canvas_w))
        dist = np.hypot(xs - cx, ys - cy)
        t = np.clip(dist / max_dist, 0.0, 1.0)[:, :, None]
        grad_rgb = (1.0 - t) * rgb1 + t * rgb2
        alpha = mask_arr[:, :, None] * fill.opacity

        arr = np.concatenate([grad_rgb, alpha], axis=2)
        return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="RGBA")

    if fill.fill_type in ("hatching", "cross_hatching"):
        hatch_img = Image.new("L", (canvas_w, canvas_h), 0)
        hdraw = ImageDraw.Draw(hatch_img)

        spacing = max(2.0, fill.spacing * scale)
        angles = [math.radians(fill.angle_deg)]
        if fill.fill_type == "cross_hatching":
            angles.append(math.radians(fill.angle_deg + 90.0))

        diag = math.hypot(canvas_w, canvas_h)
        for ang in angles:
            cos_a = math.cos(ang)
            sin_a = math.sin(ang)
            nx, ny = -sin_a, cos_a
            tx, ty = cos_a, sin_a

            center_x, center_y = canvas_w / 2.0, canvas_h / 2.0
            num_lines = int(diag / spacing) + 1
            for k in range(-num_lines, num_lines + 1):
                offset = k * spacing
                p_mid_x = center_x + nx * offset
                p_mid_y = center_y + ny * offset
                p1 = (p_mid_x - tx * diag, p_mid_y - ty * diag)
                p2 = (p_mid_x + tx * diag, p_mid_y + ty * diag)
                hdraw.line([p1, p2], fill=255, width=max(1, int(round(1.5 * scale))))

        hatch_arr = np.asarray(hatch_img, dtype=np.float32) / 255.0
        final_alpha = hatch_arr * mask_arr * fill.opacity * 255.0

        r_arr = np.full((canvas_h, canvas_w), int(round(rgb1[0] * 255)), dtype=np.uint8)
        g_arr = np.full((canvas_h, canvas_w), int(round(rgb1[1] * 255)), dtype=np.uint8)
        b_arr = np.full((canvas_h, canvas_w), int(round(rgb1[2] * 255)), dtype=np.uint8)
        a_arr = np.clip(final_alpha, 0, 255).astype(np.uint8)
        return Image.fromarray(np.stack([r_arr, g_arr, b_arr, a_arr], axis=2), mode="RGBA")

    if fill.fill_type == "stipple":
        vx = np.array([p[0] for p in px_path])
        vy = np.array([p[1] for p in px_path])
        min_x, max_x = max(0, int(vx.min())), min(canvas_w - 1, int(vx.max()))
        min_y, max_y = max(0, int(vy.min())), min(canvas_h - 1, int(vy.max()))

        bbox_area = max(1, (max_x - min_x) * (max_y - min_y))
        num_candidates = int(bbox_area * fill.density)

        if num_candidates > 0:
            rand_xs = rng.randint(min_x, max_x + 1, size=num_candidates)
            rand_ys = rng.randint(min_y, max_y + 1, size=num_candidates)

            inside = mask_arr[rand_ys, rand_xs] > 0.5
            valid_x = rand_xs[inside]
            valid_y = rand_ys[inside]

            draw = ImageDraw.Draw(fill_layer)
            dot_color = (
                int(round(rgb1[0] * 255)),
                int(round(rgb1[1] * 255)),
                int(round(rgb1[2] * 255)),
                int(round(fill.opacity * 255)),
            )
            dot_r = max(0.5, 1.0 * scale)
            for x, y in zip(valid_x, valid_y):
                draw.ellipse([x - dot_r, y - dot_r, x + dot_r, y + dot_r], fill=dot_color)

        return fill_layer

    return fill_layer


# ---------------------------------------------------------
# Master Render Document Function
# ---------------------------------------------------------

def render_document_v2(
    doc: Union[DocumentV2, Dict[str, Any]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    supersample: int = 2,
    seed: int = 42,
) -> Image.Image:
    """Renders a DocumentV2 deterministically to a PIL Image.

    Uses fast direct layer drawing for normal ink strokes and supersampling.
    """
    doc_v2 = _ensure_doc(doc)
    w_out = width if width is not None else doc_v2.canvas.width
    h_out = height if height is not None else doc_v2.canvas.height
    ss = max(1, min(4, int(supersample)))

    w_render = w_out * ss
    h_render = h_out * ss
    scale = float(ss)

    rng = np.random.RandomState(seed)

    bg_rgb = parse_color(doc_v2.canvas.background)
    bg_color = (
        int(round(bg_rgb[0] * 255)),
        int(round(bg_rgb[1] * 255)),
        int(round(bg_rgb[2] * 255)),
        255,
    )
    canvas = Image.new("RGBA", (w_render, h_render), bg_color)

    layer_commands: Dict[str, List[Any]] = {l.id: [] for l in doc_v2.layers}
    for cmd in doc_v2.commands:
        lid = getattr(cmd, "layer_id", "main")
        if lid not in layer_commands:
            layer_commands[lid] = []
        layer_commands[lid].append(cmd)

    for layer_def in doc_v2.layers:
        if not layer_def.visible:
            continue
        cmds = layer_commands.get(layer_def.id, [])
        if not cmds:
            continue

        layer_surface = Image.new("RGBA", (w_render, h_render), (0, 0, 0, 0))
        layer_draw = ImageDraw.Draw(layer_surface)

        for cmd in cmds:
            ctype = getattr(cmd, "type", "stroke")
            bmode = getattr(cmd, "blend_mode", "normal")
            brush = getattr(cmd, "brush_id", "ink_liner")

            # Fast path: ink_liner with normal blend mode draws directly into layer_surface
            if ctype == "stroke" and bmode == "normal" and brush == "ink_liner":
                pts = interpolate_points(
                    cmd.points,
                    curve_type=cmd.curve_type,
                    target_spacing_px=1.5 * scale,
                    width_px=w_render,
                    height_px=h_render,
                )
                if pts:
                    polygon, centers = build_stroke_ribbon(
                        pts,
                        base_width=cmd.base_width * scale,
                        taper_in=cmd.taper_in,
                        taper_out=cmd.taper_out,
                    )
                    rgb = [int(round(c * 255)) for c in parse_color(cmd.color)]
                    alpha = int(round(cmd.opacity * 255))
                    draw_ink_stroke(layer_draw, polygon, centers, (rgb[0], rgb[1], rgb[2], alpha))
            elif ctype == "stroke":
                stroke_img = render_stroke_brush(cmd, w_render, h_render, scale, rng)
                layer_surface = blend_layers(
                    layer_surface,
                    stroke_img,
                    mode=cmd.blend_mode,
                    opacity=1.0,
                )
            elif ctype == "fill":
                fill_img = render_fill_action(cmd, w_render, h_render, scale, rng)
                layer_surface = blend_layers(
                    layer_surface,
                    fill_img,
                    mode=cmd.blend_mode,
                    opacity=1.0,
                )

        canvas = blend_layers(
            canvas,
            layer_surface,
            mode=layer_def.blend_mode,
            opacity=layer_def.opacity,
        )

    if ss > 1:
        canvas = canvas.resize((w_out, h_out), Image.Resampling.LANCZOS)

    return canvas.convert("RGB")


# ---------------------------------------------------------
# Standalone SVG Export Engine
# ---------------------------------------------------------

def export_svg_v2(
    doc: Union[DocumentV2, Dict[str, Any]],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    """Exports DocumentV2 to clean, standard SVG code."""
    doc_v2 = _ensure_doc(doc)
    w = width if width is not None else doc_v2.canvas.width
    h = height if height is not None else doc_v2.canvas.height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
        "  <defs>",
        '    <filter id="pencil-tooth" x="0%" y="0%" width="100%" height="100%">',
        '      <feTurbulence type="fractalNoise" baseFrequency="0.8" numOctaves="3" result="noise"/>',
        '      <feColorMatrix type="matrix" values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0" in="noise" result="alphaNoise"/>',
        '      <feComposite in="SourceGraphic" in2="alphaNoise" operator="in"/>',
        "    </filter>",
        "  </defs>",
        f'  <rect width="{w}" height="{h}" fill="{doc_v2.canvas.background}"/>',
    ]

    layer_commands: Dict[str, List[Any]] = {l.id: [] for l in doc_v2.layers}
    for cmd in doc_v2.commands:
        lid = getattr(cmd, "layer_id", "main")
        if lid not in layer_commands:
            layer_commands[lid] = []
        layer_commands[lid].append(cmd)

    for layer_def in doc_v2.layers:
        if not layer_def.visible:
            continue
        cmds = layer_commands.get(layer_def.id, [])
        if not cmds:
            continue

        blend_style = f"mix-blend-mode: {layer_def.blend_mode};" if layer_def.blend_mode != "normal" else ""
        lines.append(
            f'  <g id="{layer_def.id}" opacity="{layer_def.opacity:.3f}" style="{blend_style}">'
        )

        for cmd in cmds:
            ctype = getattr(cmd, "type", "stroke")
            if ctype == "stroke":
                pts = interpolate_points(
                    cmd.points,
                    curve_type=cmd.curve_type,
                    target_spacing_px=2.5,
                    width_px=w,
                    height_px=h,
                )
                if not pts:
                    continue
                poly, _ = build_stroke_ribbon(
                    pts,
                    base_width=cmd.base_width,
                    taper_in=cmd.taper_in,
                    taper_out=cmd.taper_out,
                )
                if len(poly) >= 3:
                    pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in poly)
                    style_extra = ""
                    if cmd.blend_mode != "normal":
                        style_extra = f' style="mix-blend-mode: {cmd.blend_mode};"'
                    lines.append(
                        f'    <polygon points="{pts_str}" fill="{cmd.color}" opacity="{cmd.opacity:.3f}"{style_extra}/>'
                    )

            elif ctype == "fill":
                px_path = [(p[0] * w, p[1] * h) for p in cmd.path]
                if len(px_path) >= 3:
                    pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in px_path)
                    style_extra = ""
                    if cmd.blend_mode != "normal":
                        style_extra = f' style="mix-blend-mode: {cmd.blend_mode};"'
                    lines.append(
                        f'    <polygon points="{pts_str}" fill="{cmd.color}" opacity="{cmd.opacity:.3f}"{style_extra}/>'
                    )

        lines.append("  </g>")

    lines.append("</svg>")
    return "\n".join(lines)
