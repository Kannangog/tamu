"""High-resolution brush and rendering engine (v2).

Features:
- Variable-width strokes driven by pressure and smooth taper curves.
- 6+ distinct brush models:
  * ink_liner: crisp, solid antialiased line with calligraphic taper
  * pencil: paper tooth / graphite grain texture, pressure-sensitive density
  * charcoal: rough, powdery irregular edges with core darkness
  * airbrush: soft Gaussian radial falloff
  * colored_pencil: waxy pigment grain texture with soft pressure layering
  * marker: semi-transparent chisel nib with multiply darkening
  * watercolor_wash: translucent washes with edge pigment pooling
- Depth-aware compositing: occlusion by depth rank and filled regions; automatic line-weight/opacity falloff.
- Region fills: flat, linear gradient, radial gradient, hatching, cross-hatching, stipple.
- Layer blending: normal, multiply, screen, overlay with per-layer opacity.
- Deterministic output given a random seed.
- High-quality supersampling (default 2x downsampled with LANCZOS).
- Clean standalone SVG export.
"""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .schema import (
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


def _ensure_doc(doc_or_dict: Union[DocumentV2, Dict[str, Any]]) -> DocumentV2:
    if isinstance(doc_or_dict, DocumentV2):
        return doc_or_dict
    return DocumentV2.from_dict(doc_or_dict)


# ---------------------------------------------------------
# Spline Interpolation & Geometry
# ---------------------------------------------------------

def catmull_rom_spline(
    pts: List[PointV2],
    num_samples_per_seg: int = 16,
    alpha: float = 0.5,
) -> List[PointV2]:
    """Centripetal Catmull-Rom spline interpolation with pressure and tilt propagation."""
    if len(pts) <= 1:
        return pts
    if len(pts) == 2:
        res = []
        p0, p1 = pts[0], pts[1]
        for step in range(num_samples_per_seg + 1):
            t = step / float(num_samples_per_seg)
            x = p0.x + t * (p1.x - p0.x)
            y = p0.y + t * (p1.y - p0.y)
            p = p0.p + t * (p1.p - p0.p)
            res.append(PointV2(x, y, p))
        return res

    extended = [pts[0]] + pts + [pts[-1]]
    result: List[PointV2] = []

    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]

        def get_t(t_prev, p_a, p_b):
            d = math.hypot(p_b.x - p_a.x, p_b.y - p_a.y)
            return t_prev + (d ** alpha if d > 1e-6 else 1e-3)

        t0 = 0.0
        t1 = get_t(t0, p0, p1)
        t2 = get_t(t1, p1, p2)
        t3 = get_t(t2, p2, p3)

        dt = (t2 - t1) / float(num_samples_per_seg)
        for step in range(num_samples_per_seg):
            t = t1 + step * dt

            denom1 = max(1e-5, t1 - t0)
            denom2 = max(1e-5, t2 - t1)
            denom3 = max(1e-5, t3 - t2)

            a1_x = (t1 - t) / denom1 * p0.x + (t - t0) / denom1 * p1.x
            a1_y = (t1 - t) / denom1 * p0.y + (t - t0) / denom1 * p1.y
            a2_x = (t2 - t) / denom2 * p1.x + (t - t1) / denom2 * p2.x
            a2_y = (t2 - t) / denom2 * p1.y + (t - t1) / denom2 * p2.y
            a3_x = (t3 - t) / denom3 * p2.x + (t - t2) / denom3 * p3.x
            a3_y = (t3 - t) / denom3 * p2.y + (t - t2) / denom3 * p3.y

            denom12 = max(1e-5, t2 - t0)
            denom23 = max(1e-5, t3 - t1)
            b1_x = (t2 - t) / denom12 * a1_x + (t - t0) / denom12 * a2_x
            b1_y = (t2 - t) / denom12 * a1_y + (t - t0) / denom12 * a2_y
            b2_x = (t3 - t) / denom23 * a2_x + (t - t1) / denom23 * a3_x
            b2_y = (t3 - t) / denom23 * a2_y + (t - t1) / denom23 * a3_y

            c_x = (t2 - t) / denom2 * b1_x + (t - t1) / denom2 * b2_x
            c_y = (t2 - t) / denom2 * b1_y + (t - t1) / denom2 * b2_y

            # Interpolate pressure linearly across knot span
            u = (t - t1) / denom2
            u = max(0.0, min(1.0, u))
            c_p = (1.0 - u) * p1.p + u * p2.p

            result.append(PointV2(c_x, c_y, c_p))

    result.append(pts[-1])
    return result


def interpolate_points(
    pts: List[PointV2],
    curve_type: str = "catmull_rom",
    target_spacing_px: float = 2.0,
    width_px: int = 1024,
    height_px: int = 1024,
) -> List[Tuple[float, float, float]]:
    """Generates finely spaced (x_px, y_px, pressure) points along the stroke."""
    if len(pts) == 0:
        return []
    if len(pts) == 1:
        return [(pts[0].x * width_px, pts[0].y * height_px, pts[0].p)]

    if curve_type == "catmull_rom":
        spline_pts = catmull_rom_spline(pts, num_samples_per_seg=16)
    else:
        spline_pts = pts

    px_pts = [(p.x * width_px, p.y * height_px, p.p) for p in spline_pts]

    # Resample at target pixel spacing
    resampled: List[Tuple[float, float, float]] = [px_pts[0]]
    accum_dist = 0.0

    for i in range(1, len(px_pts)):
        p_prev = px_pts[i - 1]
        p_curr = px_pts[i]
        d = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        if d < 1e-4:
            continue

        n_steps = max(1, int(math.ceil(d / target_spacing_px)))
        for s in range(1, n_steps + 1):
            t = s / float(n_steps)
            x = p_prev[0] + t * (p_curr[0] - p_prev[0])
            y = p_prev[1] + t * (p_curr[1] - p_prev[1])
            p = p_prev[2] + t * (p_curr[2] - p_prev[2])
            resampled.append((x, y, p))

    return resampled


def build_stroke_ribbon(
    resampled_pts: List[Tuple[float, float, float]],
    base_width: float,
    taper_in: float = 0.1,
    taper_out: float = 0.1,
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """Builds a variable-width polygonal ribbon around the spine points."""
    n = len(resampled_pts)
    if n < 2:
        return [], []

    left_edge: List[Tuple[float, float]] = []
    right_edge: List[Tuple[float, float]] = []
    widths: List[float] = []

    for i in range(n):
        frac = i / float(max(1, n - 1))
        # Taper multiplier [0..1]
        taper = 1.0
        if taper_in > 1e-4 and frac < taper_in:
            taper *= frac / taper_in
        if taper_out > 1e-4 and frac > (1.0 - taper_out):
            taper *= (1.0 - frac) / taper_out

        x, y, p = resampled_pts[i]
        w = max(0.5, base_width * p * taper)
        widths.append(w)

        # Normal vector
        if i == 0:
            dx = resampled_pts[1][0] - x
            dy = resampled_pts[1][1] - y
        elif i == n - 1:
            dx = x - resampled_pts[i - 1][0]
            dy = y - resampled_pts[i - 1][1]
        else:
            dx = resampled_pts[i + 1][0] - resampled_pts[i - 1][0]
            dy = resampled_pts[i + 1][1] - resampled_pts[i - 1][1]

        mag = math.hypot(dx, dy)
        if mag < 1e-5:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = -dy / mag, dx / mag

        hw = w * 0.5
        left_edge.append((x + nx * hw, y + ny * hw))
        right_edge.append((x - nx * hw, y - ny * hw))

    polygon = left_edge + right_edge[::-1]
    return polygon, widths


# ---------------------------------------------------------
# Brush Implementations
# ---------------------------------------------------------

def render_brush_ink_liner(
    draw: ImageDraw.ImageDraw,
    pts: List[Tuple[float, float, float]],
    base_width: float,
    color_rgba: Tuple[int, int, int, int],
    taper_in: float,
    taper_out: float,
):
    """Crisp, calligraphic solid ink liner with round cap ends."""
    poly, widths = build_stroke_ribbon(pts, base_width, taper_in, taper_out)
    if len(poly) >= 3:
        draw.polygon(poly, fill=color_rgba)
    if pts and widths:
        # Round caps
        w0 = widths[0] * 0.5
        draw.ellipse([pts[0][0] - w0, pts[0][1] - w0, pts[0][0] + w0, pts[0][1] + w0], fill=color_rgba)
        w1 = widths[-1] * 0.5
        draw.ellipse([pts[-1][0] - w1, pts[-1][1] - w1, pts[-1][0] + w1, pts[-1][1] + w1], fill=color_rgba)


def render_brush_pencil(
    canvas_arr: np.ndarray,
    pts: List[Tuple[float, float, float]],
    base_width: float,
    color_rgb: Tuple[float, float, float],
    opacity: float,
    taper_in: float,
    taper_out: float,
    rng: np.random.RandomState,
):
    """Graphite pencil: tooth paper texture and pressure-sensitive density."""
    h, w, _ = canvas_arr.shape
    n = len(pts)
    if n < 2:
        return

    poly, widths = build_stroke_ribbon(pts, base_width, taper_in, taper_out)
    if not widths:
        return

    # Stamp textured graphite particles along spine
    for i, (x, y, p) in enumerate(pts):
        rad = max(1.0, widths[i] * 0.6)
        n_particles = int(max(4, rad * rad * 0.8 * p))
        offsets = rng.normal(0, rad * 0.45, (n_particles, 2))

        for ox, oy in offsets:
            px = int(round(x + ox))
            py = int(round(y + oy))
            if 0 <= px < w and 0 <= py < h:
                d_sq = ox * ox + oy * oy
                falloff = max(0.0, 1.0 - math.sqrt(d_sq) / max(0.5, rad))
                # Paper grain modulation
                grain = 0.5 + 0.5 * rng.rand()
                alpha = opacity * p * 0.45 * falloff * grain
                for c in range(3):
                    canvas_arr[py, px, c] = canvas_arr[py, px, c] * (1.0 - alpha) + color_rgb[c] * alpha


def render_brush_charcoal(
    canvas_arr: np.ndarray,
    pts: List[Tuple[float, float, float]],
    base_width: float,
    color_rgb: Tuple[float, float, float],
    opacity: float,
    taper_in: float,
    taper_out: float,
    rng: np.random.RandomState,
):
    """Charcoal: high-friction rough edge and dispersed carbon particles."""
    h, w, _ = canvas_arr.shape
    n = len(pts)
    if n < 2:
        return

    poly, widths = build_stroke_ribbon(pts, base_width, taper_in, taper_out)
    if not widths:
        return

    for i, (x, y, p) in enumerate(pts):
        rad = max(1.5, widths[i] * 0.8)
        n_particles = int(max(6, rad * rad * 1.4 * p))
        offsets = rng.normal(0, rad * 0.5, (n_particles, 2))

        for ox, oy in offsets:
            px = int(round(x + ox))
            py = int(round(y + oy))
            if 0 <= px < w and 0 <= py < h:
                # High friction splatter
                falloff = max(0.0, 1.0 - math.hypot(ox, oy) / max(0.5, rad * 1.2))
                roughness = rng.uniform(0.3, 1.0)
                alpha = opacity * p * 0.55 * falloff * roughness
                for c in range(3):
                    canvas_arr[py, px, c] = canvas_arr[py, px, c] * (1.0 - alpha) + color_rgb[c] * alpha


def render_brush_airbrush(
    canvas_arr: np.ndarray,
    pts: List[Tuple[float, float, float]],
    base_width: float,
    color_rgb: Tuple[float, float, float],
    opacity: float,
    taper_in: float,
    taper_out: float,
):
    """Soft radial gaussian airbrush."""
    h, w, _ = canvas_arr.shape
    n = len(pts)
    if n < 2:
        return

    poly, widths = build_stroke_ribbon(pts, base_width, taper_in, taper_out)
    if not widths:
        return

    # Sample along spine with gaussian stamp
    step_stride = max(1, int(base_width * 0.25))
    for i in range(0, n, step_stride):
        x, y, p = pts[i]
        rad = max(2.0, widths[i] * 1.5)
        r_int = int(math.ceil(rad))

        x_min = max(0, int(math.floor(x - rad)))
        x_max = min(w, int(math.ceil(x + rad)))
        y_min = max(0, int(math.floor(y - rad)))
        y_max = min(h, int(math.ceil(y + rad)))

        if x_min >= x_max or y_min >= y_max:
            continue

        ys, xs = np.ogrid[y_min:y_max, x_min:x_max]
        d_sq = (xs - x) ** 2 + (ys - y) ** 2
        falloff = np.exp(-d_sq / (2.0 * (rad * 0.45) ** 2))
        alpha = (opacity * p * 0.12 * falloff)[:, :, None]

        canvas_arr[y_min:y_max, x_min:x_max] = (
            canvas_arr[y_min:y_max, x_min:x_max] * (1.0 - alpha) + np.array(color_rgb) * alpha
        )


def render_brush_colored_pencil(
    canvas_arr: np.ndarray,
    pts: List[Tuple[float, float, float]],
    base_width: float,
    color_rgb: Tuple[float, float, float],
    opacity: float,
    taper_in: float,
    taper_out: float,
    rng: np.random.RandomState,
):
    """Colored pencil: waxy pigment dispersion with paper grain tooth."""
    h, w, _ = canvas_arr.shape
    n = len(pts)
    if n < 2:
        return

    poly, widths = build_stroke_ribbon(pts, base_width, taper_in, taper_out)
    if not widths:
        return

    for i, (x, y, p) in enumerate(pts):
        rad = max(1.2, widths[i] * 0.65)
        n_particles = int(max(5, rad * rad * 0.9 * p))
        offsets = rng.normal(0, rad * 0.4, (n_particles, 2))

        for ox, oy in offsets:
            px = int(round(x + ox))
            py = int(round(y + oy))
            if 0 <= px < w and 0 <= py < h:
                falloff = max(0.0, 1.0 - math.hypot(ox, oy) / max(0.5, rad))
                waxy_grain = rng.uniform(0.6, 1.0)
                alpha = opacity * p * 0.40 * falloff * waxy_grain
                for c in range(3):
                    canvas_arr[py, px, c] = canvas_arr[py, px, c] * (1.0 - alpha) + color_rgb[c] * alpha


def render_brush_marker(
    canvas_arr: np.ndarray,
    pts: List[Tuple[float, float, float]],
    base_width: float,
    color_rgb: Tuple[float, float, float],
    opacity: float,
    taper_in: float,
    taper_out: float,
):
    """Chiseled flat marker with multiply layer accumulation."""
    h, w, _ = canvas_arr.shape
    poly, widths = build_stroke_ribbon(pts, base_width, taper_in, taper_out)
    if len(poly) < 3:
        return

    # Render flat polygon mask
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(poly, fill=255)
    mask = np.array(mask_img, dtype=np.float32) / 255.0

    alpha = (mask * opacity * 0.65)[:, :, None]
    # Multiply blending: target = target * (1 - alpha * (1 - color))
    tint = 1.0 - alpha * (1.0 - np.array(color_rgb))
    canvas_arr[:] = canvas_arr * tint


def render_brush_watercolor_wash(
    canvas_arr: np.ndarray,
    pts: List[Tuple[float, float, float]],
    base_width: float,
    color_rgb: Tuple[float, float, float],
    opacity: float,
    taper_in: float,
    taper_out: float,
):
    """Translucent watercolor wash with edge pigment pooling."""
    h, w, _ = canvas_arr.shape
    poly, widths = build_stroke_ribbon(pts, base_width * 1.5, taper_in, taper_out)
    if len(poly) < 3:
        return

    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(poly, fill=255)

    # Edge darkening effect
    blurred_inner = mask_img.filter(ImageFilter.GaussianBlur(radius=base_width * 0.4))
    m_arr = np.array(mask_img, dtype=np.float32) / 255.0
    b_arr = np.array(blurred_inner, dtype=np.float32) / 255.0
    edge_ring = m_arr * (1.0 - b_arr * 0.5)

    alpha = (edge_ring * opacity * 0.35)[:, :, None]
    canvas_arr[:] = canvas_arr * (1.0 - alpha) + np.array(color_rgb) * alpha


# ---------------------------------------------------------
# Region Fills
# ---------------------------------------------------------

def render_fill_action(
    canvas_arr: np.ndarray,
    fill_act: FillActionV2,
    canvas_w: int,
    canvas_h: int,
    rng: np.random.RandomState,
):
    """Renders flat, gradient, hatching, cross-hatching, or stipple region fills."""
    h, w, _ = canvas_arr.shape
    path_px = [(p[0] * w, p[1] * h) for p in fill_act.path]
    if len(path_px) < 3:
        return

    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)
    draw.polygon(path_px, fill=255)
    mask = np.array(mask_img, dtype=np.float32) / 255.0

    color_rgb = parse_color(fill_act.color)

    if fill_act.fill_type == "flat":
        alpha = (mask * fill_act.opacity)[:, :, None]
        canvas_arr[:] = canvas_arr * (1.0 - alpha) + np.array(color_rgb) * alpha

    elif fill_act.fill_type == "linear_gradient":
        c_end = parse_color(fill_act.color_end if fill_act.color_end else "#ffffff")
        rad = math.radians(fill_act.angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        ys, xs = np.mgrid[0:h, 0:w]
        proj = (xs * cos_a + ys * sin_a)
        p_min, p_max = proj[mask > 0.5].min() if np.any(mask > 0.5) else (0, 1), proj[mask > 0.5].max() if np.any(mask > 0.5) else (0, 1)
        denom = max(1.0, p_max - p_min)
        t = np.clip((proj - p_min) / denom, 0.0, 1.0)[:, :, None]

        grad_rgb = (1.0 - t) * np.array(color_rgb) + t * np.array(c_end)
        alpha = (mask * fill_act.opacity)[:, :, None]
        canvas_arr[:] = canvas_arr * (1.0 - alpha) + grad_rgb * alpha

    elif fill_act.fill_type == "radial_gradient":
        c_end = parse_color(fill_act.color_end if fill_act.color_end else "#ffffff")
        vx = np.array([p[0] for p in path_px])
        vy = np.array([p[1] for p in path_px])
        cx, cy = float(vx.mean()), float(vy.mean())
        max_dist = max(1.0, float(np.max(np.hypot(vx - cx, vy - cy))))

        ys, xs = np.ogrid[0:h, 0:w]
        dist = np.hypot(xs - cx, ys - cy)
        t = np.clip(dist / max_dist, 0.0, 1.0)[:, :, None]
        grad_rgb = (1.0 - t) * np.array(color_rgb) + t * np.array(c_end)
        alpha = (mask * fill_act.opacity)[:, :, None]
        canvas_arr[:] = canvas_arr * (1.0 - alpha) + grad_rgb * alpha

    elif fill_act.fill_type in ("hatching", "cross_hatching"):
        hatch_img = Image.new("L", (w, h), 0)
        h_draw = ImageDraw.Draw(hatch_img)
        spacing = max(3.0, fill_act.spacing)

        angles = [fill_act.angle_deg]
        if fill_act.fill_type == "cross_hatching":
            angles.append(fill_act.angle_deg + 90.0)

        diag = math.hypot(w, h)
        cx, cy = w * 0.5, h * 0.5

        for ang in angles:
            rad = math.radians(ang)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            ux, uy = -sin_a, cos_a

            k_range = int(diag / spacing)
            for k in range(-k_range, k_range + 1):
                offset = k * spacing
                ox = cx + cos_a * offset
                oy = cy + sin_a * offset
                p_start = (ox - ux * diag, oy - uy * diag)
                p_end = (ox + ux * diag, oy + uy * diag)
                h_draw.line([p_start, p_end], fill=255, width=2)

        h_arr = np.array(hatch_img, dtype=np.float32) / 255.0
        final_mask = mask * h_arr
        alpha = (final_mask * fill_act.opacity)[:, :, None]
        canvas_arr[:] = canvas_arr * (1.0 - alpha) + np.array(color_rgb) * alpha

    elif fill_act.fill_type == "stipple":
        ys, xs = np.where(mask > 0.5)
        if len(xs) > 0:
            n_dots = int(len(xs) * fill_act.density)
            if n_dots > 0:
                indices = rng.choice(len(xs), size=min(n_dots, len(xs)), replace=False)
                for idx in indices:
                    py, px = ys[idx], xs[idx]
                    alpha = fill_act.opacity
                    for c in range(3):
                        canvas_arr[py, px, c] = canvas_arr[py, px, c] * (1.0 - alpha) + color_rgb[c] * alpha


# ---------------------------------------------------------
# Document Rendering (with Depth-Aware Compositing)
# ---------------------------------------------------------

def render_document_v2(
    doc: Union[DocumentV2, Dict[str, Any]],
    width: Optional[int] = None,
    height: Optional[int] = None,
    supersample: int = 2,
    seed: int = 42,
    enable_depth_falloff: bool = False,
) -> Image.Image:
    """Renders a DocumentV2 to a high-resolution PIL Image.
    
    Depth-aware compositing:
    - Commands within each layer are ordered by depth_rank (lower = farther back, higher = nearer).
    - If enable_depth_falloff=True, farther strokes (lower depth_rank) have subtle atmospheric opacity falloff.
    """
    doc_v2 = _ensure_doc(doc)
    w_out = width if width is not None else doc_v2.canvas.width
    h_out = height if height is not None else doc_v2.canvas.height

    ss = max(1, supersample)
    w_render = w_out * ss
    h_render = h_out * ss

    rng = np.random.RandomState(seed)

    bg_rgb = parse_color(doc_v2.canvas.background)
    base_canvas = np.ones((h_render, w_render, 3), dtype=np.float32)
    for c in range(3):
        base_canvas[:, :, c] = bg_rgb[c]

    # Map layer definitions
    layer_order = {l.id: idx for idx, l in enumerate(doc_v2.layers)}
    layer_map = {l.id: l for l in doc_v2.layers}

    # Group commands by layer, then sort each group by depth_rank (ascending: back to front)
    cmds_by_layer: Dict[str, List[Any]] = {l.id: [] for l in doc_v2.layers}
    for cmd in doc_v2.commands:
        lid = getattr(cmd, "layer_id", "main")
        if lid not in cmds_by_layer:
            cmds_by_layer[lid] = []
        cmds_by_layer[lid].append(cmd)

    # Composite each layer
    for layer_def in doc_v2.layers:
        if not layer_def.visible:
            continue

        cmds = cmds_by_layer.get(layer_def.id, [])
        if not cmds:
            continue

        # Sort commands by depth_rank: lower depth_rank (farther back) drawn first, higher (nearer) drawn on top
        cmds_sorted = sorted(cmds, key=lambda c: getattr(c, "depth_rank", 0))

        layer_canvas = np.ones((h_render, w_render, 3), dtype=np.float32)
        layer_canvas[:, :] = base_canvas[:, :]  # start from current backdrop

        pil_layer = None
        draw = None

        def sync_pil_to_canvas():
            nonlocal pil_layer, draw, layer_canvas
            if pil_layer is not None:
                layer_canvas = np.array(pil_layer, dtype=np.float32)[:, :, :3] / 255.0
                pil_layer = None
                draw = None

        def ensure_pil():
            nonlocal pil_layer, draw, layer_canvas
            if pil_layer is None:
                pil_layer = Image.fromarray(np.clip(layer_canvas * 255.0, 0, 255).astype(np.uint8)).convert("RGBA")
                draw = ImageDraw.Draw(pil_layer, "RGBA")
            return draw

        for cmd in cmds_sorted:
            ctype = getattr(cmd, "type", "stroke")
            drank = getattr(cmd, "depth_rank", 0)

            # Atmospheric depth falloff (optional)
            depth_factor = 1.0
            if enable_depth_falloff:
                # Farther back (negative depth_rank) gets slight opacity reduction
                depth_factor = max(0.4, min(1.0, 1.0 + drank * 0.1))

            if ctype == "stroke":
                brush = getattr(cmd, "brush_id", "ink_liner")
                color_rgb = parse_color(cmd.color)
                base_w = cmd.base_width * ss * depth_factor
                opac = cmd.opacity * depth_factor

                pts = interpolate_points(
                    cmd.points,
                    curve_type=cmd.curve_type,
                    target_spacing_px=2.0 * ss,
                    width_px=w_render,
                    height_px=h_render,
                )
                if not pts:
                    continue

                if brush == "ink_liner":
                    c_rgba = (
                        int(round(color_rgb[0] * 255)),
                        int(round(color_rgb[1] * 255)),
                        int(round(color_rgb[2] * 255)),
                        int(round(opac * 255)),
                    )
                    d = ensure_pil()
                    render_brush_ink_liner(d, pts, base_w, c_rgba, cmd.taper_in, cmd.taper_out)
                else:
                    sync_pil_to_canvas()
                    if brush == "pencil":
                        render_brush_pencil(layer_canvas, pts, base_w, color_rgb, opac, cmd.taper_in, cmd.taper_out, rng)
                    elif brush == "charcoal":
                        render_brush_charcoal(layer_canvas, pts, base_w, color_rgb, opac, cmd.taper_in, cmd.taper_out, rng)
                    elif brush == "airbrush":
                        render_brush_airbrush(layer_canvas, pts, base_w, color_rgb, opac, cmd.taper_in, cmd.taper_out)
                    elif brush == "colored_pencil":
                        render_brush_colored_pencil(layer_canvas, pts, base_w, color_rgb, opac, cmd.taper_in, cmd.taper_out, rng)
                    elif brush == "marker":
                        render_brush_marker(layer_canvas, pts, base_w, color_rgb, opac, cmd.taper_in, cmd.taper_out)
                    elif brush == "watercolor_wash":
                        render_brush_watercolor_wash(layer_canvas, pts, base_w, color_rgb, opac, cmd.taper_in, cmd.taper_out)

            elif ctype == "fill":
                sync_pil_to_canvas()
                render_fill_action(layer_canvas, cmd, w_render, h_render, rng)

        sync_pil_to_canvas()

        # Blend layer back into base canvas
        l_opac = layer_def.opacity
        if layer_def.blend_mode == "normal":
            base_canvas[:] = base_canvas * (1.0 - l_opac) + layer_canvas * l_opac
        elif layer_def.blend_mode == "multiply":
            mult = base_canvas * layer_canvas
            base_canvas[:] = base_canvas * (1.0 - l_opac) + mult * l_opac
        elif layer_def.blend_mode == "screen":
            screen = 1.0 - (1.0 - base_canvas) * (1.0 - layer_canvas)
            base_canvas[:] = base_canvas * (1.0 - l_opac) + screen * l_opac
        elif layer_def.blend_mode == "overlay":
            low = 2.0 * base_canvas * layer_canvas
            high = 1.0 - 2.0 * (1.0 - base_canvas) * (1.0 - layer_canvas)
            overlay = np.where(base_canvas < 0.5, low, high)
            base_canvas[:] = base_canvas * (1.0 - l_opac) + overlay * l_opac

    final_img = Image.fromarray(np.clip(base_canvas * 255.0, 0, 255).astype(np.uint8))
    if ss > 1:
        final_img = final_img.resize((w_out, h_out), resample=Image.Resampling.LANCZOS)
    return final_img


# ---------------------------------------------------------
# SVG Export with Depth Ordering
# ---------------------------------------------------------

def export_svg_v2(
    doc: Union[DocumentV2, Dict[str, Any]],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    """Exports DocumentV2 to clean, standard SVG code respecting depth ordering."""
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

        # Sort commands by depth_rank
        cmds_sorted = sorted(cmds, key=lambda c: getattr(c, "depth_rank", 0))

        blend_style = f"mix-blend-mode: {layer_def.blend_mode};" if layer_def.blend_mode != "normal" else ""
        lines.append(
            f'  <g id="{layer_def.id}" opacity="{layer_def.opacity:.3f}" style="{blend_style}">'
        )

        for cmd in cmds_sorted:
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


# ---------------------------------------------------------
# Gate 1 Swatches Generator
# ---------------------------------------------------------

def render_brush_swatches(
    out_path: Union[str, Path] = "outputs/hand/swatches.png",
    width: int = 1200,
    height: int = 800,
    seed: int = 42,
) -> Image.Image:
    """Renders 6 distinct brush swatches with pressure ramps, tapers, and layering."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=width, height=height, background="#f8fafc"),
        layers=[LayerV2(id="main", name="Main")],
    )

    brushes = [
        ("ink_liner", "1. Ink Liner (Crisp Calligraphic)", "#0f172a"),
        ("pencil", "2. Graphite Pencil (Tooth Texture)", "#334155"),
        ("charcoal", "3. Charcoal (Rough Dispersion)", "#1e293b"),
        ("airbrush", "4. Soft Airbrush (Radial Falloff)", "#2563eb"),
        ("colored_pencil", "5. Colored Pencil (Waxy Grain)", "#dc2626"),
        ("marker", "6. Chisel Marker (Multiply Glaze)", "#059669"),
    ]

    for idx, (b_id, title, color) in enumerate(brushes):
        row = idx
        y_center = (row + 0.6) / 6.0

        # Stroke 1: S-curve with pressure ramp 0.2 -> 1.0 -> 0.2
        pts = [
            PointV2(0.12, y_center - 0.03, p=0.2),
            PointV2(0.25, y_center + 0.03, p=0.6),
            PointV2(0.38, y_center - 0.03, p=1.0),
            PointV2(0.50, y_center + 0.03, p=0.7),
            PointV2(0.62, y_center, p=0.2),
        ]
        doc.commands.append(
            StrokeActionV2(
                points=pts,
                brush_id=b_id,
                color=color,
                base_width=6.0,
                taper_in=0.15,
                taper_out=0.15,
            )
        )

        # Stroke 2: overlapping quick hatch marks to demonstrate layering
        for k in range(5):
            x_k = 0.72 + k * 0.04
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(x_k, y_center - 0.035, p=0.4), PointV2(x_k + 0.02, y_center + 0.035, p=0.9)],
                    brush_id=b_id,
                    color=color,
                    base_width=4.5,
                )
            )

    img = render_document_v2(doc, width=width, height=height, supersample=2, seed=seed)

    # Add text labels
    draw = ImageDraw.Draw(img)
    for idx, (b_id, title, color) in enumerate(brushes):
        y_pos = int(((idx + 0.3) / 6.0) * height)
        draw.text((30, y_pos), title, fill=(15, 23, 42))

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_p)
    return img
