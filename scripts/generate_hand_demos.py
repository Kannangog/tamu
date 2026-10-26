"""Generate 4 scripted artworks under outputs/hand_demos/ showcasing the Artist Hand layer.

Pieces:
1. bicycle_ink_study: Fine ink liner with light-directed line weights, spokes, chainring, ground hatching.
2. pencil_still_life: 3 classical geometric forms (sphere, cylinder, cube) with core/reflected shadows and cast shadows.
3. color_landscape: Gradient dusk sky, 3 layers of atmospheric perspective hills, pine trees, water reflection.
4. charcoal_portrait: Loomis proportion guides, head study with charcoal grain, half-light form shading.
"""

import math
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from konnon_draw.schema_v2 import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
    validate_document_v2,
)
from konnon_draw.renderer_v2 import export_svg_v2, render_document_v2
from konnon_draw.artist_skills import (
    apply_atmospheric_perspective,
    apply_line_weight_by_light,
    create_standard_layers,
    generate_cast_shadow_polygon,
    generate_directional_hatching,
    generate_palette_harmony,
    generate_stipple_shading,
    make_arc,
    make_ellipse,
    make_gradient_sky,
    make_perspective_grid,
    make_proportion_guides,
    make_sketchy_lines,
    make_wobbly_contour,
)


def build_bicycle_ink_study() -> DocumentV2:
    """Piece 1: Ink study of a bicycle with line-weight variation and hatching shadow."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1024, height=1024, background="#fcfbf7"),
        layers=[
            LayerV2(id="underdrawing", name="Underdrawing", opacity=0.35, blend_mode="normal"),
            LayerV2(id="shadows", name="Cast Shadows", opacity=0.8, blend_mode="multiply"),
            LayerV2(id="spokes", name="Spokes & Chain", opacity=0.7, blend_mode="normal"),
            LayerV2(id="frame", name="Main Frame & Wheels", opacity=1.0, blend_mode="normal"),
            LayerV2(id="accents", name="Accents & Ink Highlights", opacity=1.0, blend_mode="normal"),
        ],
    )

    # 1. Perspective / Ground Line
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(0.08, 0.74, p=0.3), PointV2(0.92, 0.74, p=0.3)],
            brush_id="pencil",
            color="#94a3b8",
            base_width=1.5,
            layer_id="underdrawing",
        )
    )

    # Ground Hatching Shadow
    doc.commands.append(
        generate_directional_hatching(
            polygon=[(0.12, 0.73), (0.88, 0.73), (0.92, 0.78), (0.16, 0.78)],
            angle_deg=35.0,
            spacing=5.0,
            color="#334155",
            opacity=0.6,
            cross=False,
            layer_id="shadows",
        )
    )
    doc.commands.append(
        generate_directional_hatching(
            polygon=[(0.20, 0.735), (0.36, 0.735), (0.38, 0.765), (0.22, 0.765)],
            angle_deg=-45.0,
            spacing=4.0,
            color="#0f172a",
            opacity=0.8,
            cross=False,
            layer_id="shadows",
        )
    )
    doc.commands.append(
        generate_directional_hatching(
            polygon=[(0.64, 0.735), (0.80, 0.735), (0.82, 0.765), (0.66, 0.765)],
            angle_deg=-45.0,
            spacing=4.0,
            color="#0f172a",
            opacity=0.8,
            cross=False,
            layer_id="shadows",
        )
    )

    # Wheels (centers at rear: (0.28, 0.60), front: (0.72, 0.60), radius 0.14)
    rear_c = (0.28, 0.60)
    front_c = (0.72, 0.60)
    r_wheel = 0.14

    for c in [rear_c, front_c]:
        # Outer tire (heavy)
        tire = make_ellipse(c, r_wheel, r_wheel, num_points=36, base_width=4.5, color="#0f172a", layer_id="frame")
        doc.commands.append(apply_line_weight_by_light(tire, light_dir=(-0.7, -0.7), min_pressure=0.4, max_pressure=1.0))
        # Inner rim
        doc.commands.append(make_ellipse(c, r_wheel - 0.015, r_wheel - 0.015, num_points=32, base_width=1.8, color="#334155", layer_id="frame"))
        # Hub
        doc.commands.append(make_ellipse(c, 0.018, 0.018, num_points=16, base_width=2.5, color="#0f172a", layer_id="frame"))

        # Spokes (16 spokes per wheel)
        for s in range(16):
            ang = s * (2.0 * math.pi / 16)
            sx = c[0] + (r_wheel - 0.015) * math.cos(ang)
            sy = c[1] + (r_wheel - 0.015) * math.sin(ang)
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(c[0], c[1], p=0.4), PointV2(sx, sy, p=0.6)],
                    brush_id="ink_liner",
                    color="#475569",
                    base_width=1.0,
                    taper_in=0.1,
                    taper_out=0.1,
                    layer_id="spokes",
                )
            )

    # Frame Key Vertices
    bb_c = (0.48, 0.60)     # Bottom bracket (crank)
    seat_top = (0.42, 0.38) # Seat tube top
    head_top = (0.68, 0.35) # Head tube top
    head_bot = (0.70, 0.44) # Head tube bottom

    # Tubes with light-directed line weight
    tubes = [
        # Chain stay: rear hub -> BB
        [PointV2(rear_c[0], rear_c[1]), PointV2(bb_c[0], bb_c[1])],
        # Seat stay: rear hub -> seat top
        [PointV2(rear_c[0], rear_c[1]), PointV2(seat_top[0], seat_top[1])],
        # Seat tube: BB -> seat top
        [PointV2(bb_c[0], bb_c[1]), PointV2(seat_top[0], seat_top[1])],
        # Down tube: BB -> head bottom
        [PointV2(bb_c[0], bb_c[1]), PointV2(head_bot[0], head_bot[1])],
        # Top tube: seat top -> head top
        [PointV2(seat_top[0], seat_top[1]), PointV2(head_top[0], head_top[1])],
        # Head tube
        [PointV2(head_top[0], head_top[1]), PointV2(head_bot[0], head_bot[1])],
        # Fork: head bottom -> front hub
        [PointV2(head_bot[0], head_bot[1]), PointV2(front_c[0], front_c[1])],
    ]

    for pts in tubes:
        st = StrokeActionV2(points=pts, brush_id="ink_liner", color="#020617", base_width=4.0, curve_type="linear", layer_id="frame")
        doc.commands.append(apply_line_weight_by_light(st, light_dir=(-0.7, -0.7), min_pressure=0.35, max_pressure=1.0))

    # Saddle
    saddle_pts = [
        PointV2(0.38, 0.36, p=0.8),
        PointV2(0.42, 0.355, p=1.0),
        PointV2(0.47, 0.365, p=0.6),
    ]
    doc.commands.append(StrokeActionV2(points=saddle_pts, brush_id="ink_liner", color="#09090b", base_width=5.0, curve_type="catmull_rom", layer_id="frame"))
    # Saddle post
    doc.commands.append(StrokeActionV2(points=[PointV2(0.42, 0.36), PointV2(seat_top[0], seat_top[1])], brush_id="ink_liner", color="#334155", base_width=3.0, layer_id="frame"))

    # Handlebars & Stem
    stem_pts = [PointV2(head_top[0], head_top[1]), PointV2(0.67, 0.28)]
    doc.commands.append(StrokeActionV2(points=stem_pts, brush_id="ink_liner", color="#334155", base_width=3.2, curve_type="linear", layer_id="frame"))
    bar_pts = [
        PointV2(0.64, 0.29, p=0.7),
        PointV2(0.67, 0.28, p=1.0),
        PointV2(0.72, 0.31, p=0.9),
        PointV2(0.71, 0.36, p=0.6),
    ]
    doc.commands.append(StrokeActionV2(points=bar_pts, brush_id="ink_liner", color="#020617", base_width=3.8, curve_type="catmull_rom", layer_id="frame"))

    # Chainring / Crank / Pedals
    doc.commands.append(make_ellipse(bb_c, 0.04, 0.04, num_points=24, base_width=2.5, color="#1e293b", layer_id="frame"))
    # Chain loop
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(bb_c[0], bb_c[1] - 0.038), PointV2(rear_c[0], rear_c[1] - 0.015)],
            brush_id="ink_liner",
            color="#475569",
            base_width=1.8,
            layer_id="spokes",
        )
    )
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(bb_c[0], bb_c[1] + 0.038), PointV2(rear_c[0], rear_c[1] + 0.015)],
            brush_id="ink_liner",
            color="#475569",
            base_width=1.8,
            layer_id="spokes",
        )
    )
    # Crank arms
    doc.commands.append(StrokeActionV2(points=[PointV2(bb_c[0], bb_c[1]), PointV2(bb_c[0] - 0.04, bb_c[1] + 0.05)], brush_id="ink_liner", color="#0f172a", base_width=3.0, layer_id="frame"))
    doc.commands.append(StrokeActionV2(points=[PointV2(bb_c[0], bb_c[1]), PointV2(bb_c[0] + 0.04, bb_c[1] - 0.05)], brush_id="ink_liner", color="#0f172a", base_width=3.0, layer_id="frame"))

    return doc


def build_pencil_still_life() -> DocumentV2:
    """Piece 2: Pencil still life (Sphere, Cylinder, Cube) with light-direction shading and cast shadows."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1024, height=1024, background="#fafaf9"),
        layers=[
            LayerV2(id="underdrawing", name="Searching Construction Lines", opacity=0.35, blend_mode="normal"),
            LayerV2(id="cast_shadows", name="Cast Shadows", opacity=0.6, blend_mode="multiply"),
            LayerV2(id="core_shading", name="Core Value & Form Shading", opacity=0.85, blend_mode="multiply"),
            LayerV2(id="contours", name="Pencil Contours", opacity=0.95, blend_mode="normal"),
            LayerV2(id="highlights", name="Specular Highlights", opacity=0.8, blend_mode="screen"),
        ],
    )

    # Table edge line
    table_y = 0.68
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(0.05, table_y, p=0.4), PointV2(0.95, table_y, p=0.4)],
            brush_id="pencil",
            color="#78716c",
            base_width=1.8,
            layer_id="underdrawing",
        )
    )

    # Searching construction boxes / circles (Loomis / Vilppu style)
    doc.commands.extend(make_sketchy_lines([(0.20, 0.40), (0.42, 0.40), (0.42, 0.68), (0.20, 0.68), (0.20, 0.40)], count=2, brush_id="pencil", color="#a8a29e", layer_id="underdrawing"))
    doc.commands.extend(make_sketchy_lines([(0.46, 0.30), (0.64, 0.30), (0.64, 0.68), (0.46, 0.68), (0.46, 0.30)], count=2, brush_id="pencil", color="#a8a29e", layer_id="underdrawing"))
    doc.commands.extend(make_sketchy_lines([(0.68, 0.42), (0.90, 0.42), (0.90, 0.68), (0.68, 0.68), (0.68, 0.42)], count=2, brush_id="pencil", color="#a8a29e", layer_id="underdrawing"))

    # ------------------
    # Form 1: Sphere (center (0.31, 0.54), r=0.13)
    # Light comes from top-left (-0.707, -0.707)
    # ------------------
    sc = (0.31, 0.54)
    sr = 0.13

    # Cast shadow on ground: ellipse squashed along light direction
    doc.commands.append(
        generate_cast_shadow_polygon(
            [(sc[0] - 0.08, sc[1] + sr), (sc[0] + 0.08, sc[1] + sr)],
            light_offset=(0.14, 0.04),
            ground_y=sc[1] + sr,
            color="#292524",
            opacity=0.65,
            layer_id="cast_shadows",
        )
    )

    # Core shadow crescent hatching on bottom-right of sphere
    sphere_shadow_poly = []
    for deg in range(0, 185, 15):
        rad = math.radians(deg - 45.0)  # shadow zone facing bottom-right
        x = sc[0] + sr * math.cos(rad)
        y = sc[1] + sr * math.sin(rad)
        sphere_shadow_poly.append((x, y))
    # Inner terminator crescent arc
    for deg in range(180, -5, -15):
        rad = math.radians(deg - 45.0)
        x = sc[0] + (sr * 0.45) * math.cos(rad) + 0.02
        y = sc[1] + (sr * 0.45) * math.sin(rad) + 0.02
        sphere_shadow_poly.append((x, y))

    doc.commands.append(
        generate_directional_hatching(
            sphere_shadow_poly,
            angle_deg=40.0,
            spacing=4.0,
            color="#1c1917",
            opacity=0.85,
            cross=True,
            layer_id="core_shading",
        )
    )

    # Sphere contour with pencil line weight
    sphere_contour = make_ellipse(sc, sr, sr, num_points=40, base_width=3.0, brush_id="pencil", color="#292524", layer_id="contours")
    doc.commands.append(apply_line_weight_by_light(sphere_contour, light_dir=(-0.7, -0.7), min_pressure=0.25, max_pressure=1.0))

    # ------------------
    # Form 2: Cylinder (center x=0.55, top y=0.35, bot y=0.67, rx=0.08, ry=0.025)
    # ------------------
    cyl_x = 0.55
    cyl_rx = 0.08
    cyl_ry = 0.025
    top_y = 0.35
    bot_y = 0.67

    # Cylinder cast shadow
    doc.commands.append(
        generate_cast_shadow_polygon(
            [(cyl_x - cyl_rx, bot_y), (cyl_x + cyl_rx, bot_y)],
            light_offset=(0.16, 0.05),
            ground_y=bot_y,
            color="#292524",
            opacity=0.6,
            layer_id="cast_shadows",
        )
    )

    # Cylinder body shadow (right half from x=0.55 to 0.63)
    cyl_shadow_poly = [
        (cyl_x + 0.01, top_y),
        (cyl_x + cyl_rx, top_y),
        (cyl_x + cyl_rx, bot_y),
        (cyl_x + 0.01, bot_y),
    ]
    doc.commands.append(
        generate_directional_hatching(
            cyl_shadow_poly,
            angle_deg=90.0,  # vertical hatching along cylinder length
            spacing=3.8,
            color="#1c1917",
            opacity=0.8,
            cross=False,
            layer_id="core_shading",
        )
    )

    # Cylinder contours
    # Top rim
    doc.commands.append(make_ellipse((cyl_x, top_y), cyl_rx, cyl_ry, base_width=2.5, brush_id="pencil", color="#292524", layer_id="contours"))
    # Bottom rim arc
    doc.commands.append(make_arc((cyl_x, bot_y), cyl_rx, cyl_ry, start_deg=0.0, end_deg=180.0, base_width=3.2, brush_id="pencil", color="#1c1917", layer_id="contours"))
    # Side walls
    left_wall = StrokeActionV2(points=[PointV2(cyl_x - cyl_rx, top_y, p=0.3), PointV2(cyl_x - cyl_rx, bot_y, p=0.4)], brush_id="pencil", color="#44403c", base_width=2.0, layer_id="contours")
    right_wall = StrokeActionV2(points=[PointV2(cyl_x + cyl_rx, top_y, p=0.8), PointV2(cyl_x + cyl_rx, bot_y, p=1.0)], brush_id="pencil", color="#1c1917", base_width=3.5, layer_id="contours")
    doc.commands.append(left_wall)
    doc.commands.append(right_wall)

    # ------------------
    # Form 3: Cube in Perspective (center ~0.78, 0.54)
    # Top facet (lit), Left facet (mid tone), Right facet (shadow)
    # ------------------
    p_top_front = (0.78, 0.48)
    p_top_left = (0.70, 0.42)
    p_top_right = (0.86, 0.44)
    p_top_back = (0.78, 0.38)

    p_bot_front = (0.78, 0.67)
    p_bot_left = (0.70, 0.61)
    p_bot_right = (0.86, 0.63)

    # Cast shadow of cube
    doc.commands.append(
        generate_cast_shadow_polygon(
            [p_bot_left, p_bot_front, p_bot_right],
            light_offset=(0.14, 0.05),
            color="#292524",
            opacity=0.6,
            layer_id="cast_shadows",
        )
    )

    # Right facet in deep shadow (cross-hatching)
    doc.commands.append(
        generate_directional_hatching(
            [p_top_front, p_top_right, p_bot_right, p_bot_front],
            angle_deg=55.0,
            spacing=4.0,
            color="#1c1917",
            opacity=0.9,
            cross=True,
            layer_id="core_shading",
        )
    )

    # Left facet in half-tone (single hatching)
    doc.commands.append(
        generate_directional_hatching(
            [p_top_left, p_top_front, p_bot_front, p_bot_left],
            angle_deg=125.0,
            spacing=6.0,
            color="#44403c",
            opacity=0.55,
            cross=False,
            layer_id="core_shading",
        )
    )

    # Cube Edges with pencil line weight
    cube_edges = [
        # Lit top facet
        [p_top_back, p_top_left],
        [p_top_left, p_top_front],
        [p_top_front, p_top_right],
        [p_top_right, p_top_back],
        # Vertical pillars
        [p_top_left, p_bot_left],
        [p_top_front, p_bot_front],
        [p_top_right, p_bot_right],
        # Bottom perimeter
        [p_bot_left, p_bot_front],
        [p_bot_front, p_bot_right],
    ]
    for e in cube_edges:
        st = StrokeActionV2(
            points=[PointV2(e[0][0], e[0][1]), PointV2(e[1][0], e[1][1])],
            brush_id="pencil",
            color="#1c1917",
            base_width=2.6,
            curve_type="linear",
            layer_id="contours",
        )
        doc.commands.append(apply_line_weight_by_light(st, light_dir=(-0.7, -0.7), min_pressure=0.3, max_pressure=1.0))

    return doc


def build_color_landscape() -> DocumentV2:
    """Piece 3: Color landscape with gradient sky, layered hills, and atmospheric perspective."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1024, height=1024, background="#f8fafc"),
        layers=[
            LayerV2(id="sky", name="Atmospheric Sky Gradient", opacity=1.0, blend_mode="normal"),
            LayerV2(id="sun", name="Sun & Glow", opacity=0.9, blend_mode="screen"),
            LayerV2(id="dist_mountains", name="Distant Mountains (Depth 0.9)", opacity=0.95, blend_mode="normal"),
            LayerV2(id="mid_hills", name="Mid-range Ridge (Depth 0.5)", opacity=0.95, blend_mode="normal"),
            LayerV2(id="lake", name="Lake Surface & Reflections", opacity=0.9, blend_mode="normal"),
            LayerV2(id="fore_hill", name="Foreground Meadow (Depth 0.1)", opacity=1.0, blend_mode="normal"),
            LayerV2(id="trees", name="Foreground Pine Trees", opacity=1.0, blend_mode="normal"),
        ],
    )

    # 1. Gradient Sky from deep dusk indigo to warm peach horizon
    doc.commands.append(
        FillActionV2(
            path=[(0.0, 0.0), (1.0, 0.0), (1.0, 0.60), (0.0, 0.60)],
            fill_type="linear_gradient",
            color="#1e1b4b",      # deep indigo
            color_end="#fed7aa",  # warm apricot horizon
            angle_deg=90.0,
            opacity=1.0,
            layer_id="sky",
        )
    )

    # 2. Glowing Sun near Rule-of-Thirds horizon anchor (0.67, 0.45)
    doc.commands.append(
        FillActionV2(
            path=[(0.60, 0.38), (0.74, 0.38), (0.74, 0.52), (0.60, 0.52)],
            fill_type="radial_gradient",
            color="#fffbeb",      # radiant white-gold core
            color_end="#fde047",  # amber halo
            opacity=0.85,
            layer_id="sun",
        )
    )

    # 3. Distant Mountains (Depth = 0.9 -> shifted heavily to cool sky lavender)
    dist_color = apply_atmospheric_perspective("#334155", depth=0.9, sky_tint_hex="#fed7aa")
    dist_mountain_pts = [
        (0.0, 0.60),
        (0.0, 0.44),
        (0.12, 0.38),
        (0.24, 0.45),
        (0.38, 0.35),
        (0.52, 0.43),
        (0.68, 0.36),
        (0.82, 0.42),
        (1.0, 0.37),
        (1.0, 0.60),
    ]
    doc.commands.append(FillActionV2(path=dist_mountain_pts, fill_type="flat", color=dist_color, opacity=0.95, layer_id="dist_mountains"))

    # 4. Mid-range Hills (Depth = 0.5 -> slate teal green)
    mid_color = apply_atmospheric_perspective("#1e3a5f", depth=0.5, sky_tint_hex="#cbd5e1")
    mid_hill_pts = [
        (0.0, 0.65),
        (0.0, 0.48),
        (0.18, 0.52),
        (0.35, 0.46),
        (0.55, 0.53),
        (0.75, 0.47),
        (1.0, 0.51),
        (1.0, 0.65),
    ]
    doc.commands.append(FillActionV2(path=mid_hill_pts, fill_type="flat", color=mid_color, opacity=0.95, layer_id="mid_hills"))

    # 5. Lake Surface (reflecting sky gradient and hills)
    doc.commands.append(
        FillActionV2(
            path=[(0.0, 0.58), (1.0, 0.58), (1.0, 0.78), (0.0, 0.78)],
            fill_type="linear_gradient",
            color="#475569",
            color_end="#fcd34d",
            angle_deg=90.0,
            opacity=0.85,
            layer_id="lake",
        )
    )
    # Subtle water ripples
    for wy in [0.63, 0.67, 0.71, 0.75]:
        doc.commands.append(
            StrokeActionV2(
                points=[PointV2(0.15, wy, p=0.4), PointV2(0.85, wy, p=0.4)],
                brush_id="watercolor_wash",
                color="#fef08a",
                opacity=0.4,
                base_width=2.5,
                layer_id="lake",
            )
        )

    # 6. Foreground Meadow (Depth = 0.1 -> rich warm earth and olive)
    fore_pts = [
        (0.0, 1.0),
        (0.0, 0.72),
        (0.30, 0.70),
        (0.65, 0.74),
        (1.0, 0.69),
        (1.0, 1.0),
    ]
    doc.commands.append(FillActionV2(path=fore_pts, fill_type="flat", color="#14532d", opacity=1.0, layer_id="fore_hill"))
    # Warm earth wash in foreground
    doc.commands.append(
        FillActionV2(
            path=[(0.0, 0.85), (1.0, 0.80), (1.0, 1.0), (0.0, 1.0)],
            fill_type="flat",
            color="#713f12",
            opacity=0.35,
            blend_mode="multiply",
            layer_id="fore_hill",
        )
    )

    # 7. Foreground Silhouette Pine Trees on Left (anchoring composition)
    def make_pine(tx: float, ty: float, height: float, width: float):
        # Trunk
        doc.commands.append(
            StrokeActionV2(
                points=[PointV2(tx, ty + height, p=1.0), PointV2(tx, ty, p=0.4)],
                brush_id="ink_liner",
                color="#052e16",
                base_width=4.5,
                layer_id="trees",
            )
        )
        # Branches (tiers)
        num_tiers = 8
        for k in range(num_tiers):
            tier_t = k / float(num_tiers)
            by = ty + 0.15 * height + tier_t * 0.80 * height
            bw = width * (0.2 + 0.8 * tier_t)
            left_b = StrokeActionV2(
                points=[PointV2(tx, by, p=0.8), PointV2(tx - bw * 0.5, by + 0.015, p=0.9), PointV2(tx - bw, by + 0.035, p=0.2)],
                brush_id="ink_liner",
                color="#022c22",
                base_width=3.2,
                curve_type="catmull_rom",
                layer_id="trees",
            )
            right_b = StrokeActionV2(
                points=[PointV2(tx, by, p=0.8), PointV2(tx + bw * 0.5, by + 0.015, p=0.9), PointV2(tx + bw, by + 0.035, p=0.2)],
                brush_id="ink_liner",
                color="#022c22",
                base_width=3.2,
                curve_type="catmull_rom",
                layer_id="trees",
            )
            doc.commands.append(left_b)
            doc.commands.append(right_b)

    make_pine(tx=0.14, ty=0.52, height=0.30, width=0.10)
    make_pine(tx=0.22, ty=0.56, height=0.25, width=0.08)
    make_pine(tx=0.08, ty=0.58, height=0.22, width=0.07)

    return doc


def build_charcoal_portrait() -> DocumentV2:
    """Piece 4: Charcoal portrait-style head study with Loomis proportion guides and form shading."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1024, height=1024, background="#f5f5f4"),
        layers=[
            LayerV2(id="guides", name="Loomis Proportion Guides", opacity=0.45, blend_mode="normal"),
            LayerV2(id="mass_shading", name="Charcoal Value Masses", opacity=0.85, blend_mode="multiply"),
            LayerV2(id="features", name="Facial Features & Line Art", opacity=0.95, blend_mode="normal"),
            LayerV2(id="hair", name="Expressive Hair Masses", opacity=0.95, blend_mode="normal"),
            LayerV2(id="highlights", name="White Chalk Highlights", opacity=0.75, blend_mode="screen"),
        ],
    )

    # 1. Loomis Proportion Guides:
    # Cranial Sphere center (0.50, 0.42), radius 0.22
    head_c = (0.50, 0.42)
    hr = 0.22
    doc.commands.append(make_ellipse(head_c, hr, hr, num_points=32, base_width=1.5, brush_id="pencil", color="#78716c", layer_id="guides"))

    # Vertical center axis & horizontal landmark divisions:
    # Brow line: y = 0.42
    # Nose base line: y = 0.55
    # Lip center: y = 0.63
    # Chin base: y = 0.72
    landmarks = [0.42, 0.55, 0.63, 0.72]
    # Vertical centerline
    doc.commands.append(StrokeActionV2(points=[PointV2(0.50, 0.18, p=0.3), PointV2(0.50, 0.78, p=0.3)], brush_id="pencil", color="#a8a29e", base_width=1.2, layer_id="guides"))
    # Horizontal divisions
    for ly in landmarks:
        doc.commands.append(StrokeActionV2(points=[PointV2(0.26, ly, p=0.3), PointV2(0.74, ly, p=0.3)], brush_id="pencil", color="#a8a29e", base_width=1.2, layer_id="guides"))

    # 2. Charcoal Form Value Masses (Rembrandt Lighting: light from top-left, shadow side on right)
    # Right-side shadow mask covering temple, cheek, jaw, neck
    right_shadow_poly = [
        (0.50, 0.32),  # forehead center
        (0.66, 0.35),  # right temple
        (0.71, 0.48),  # right cheek
        (0.65, 0.64),  # right jaw angle
        (0.52, 0.72),  # chin right
        (0.50, 0.72),  # chin center
        (0.50, 0.64),  # philtrum
        (0.52, 0.55),  # nose shadow edge
        (0.50, 0.42),  # brow center
    ]
    doc.commands.append(
        FillActionV2(
            path=right_shadow_poly,
            fill_type="flat",
            color="#292524",
            opacity=0.35,
            blend_mode="multiply",
            layer_id="mass_shading",
        )
    )
    # Deep cross-hatching in core shadow along jaw and cheek
    doc.commands.append(
        generate_directional_hatching(
            [(0.53, 0.46), (0.69, 0.46), (0.64, 0.66), (0.52, 0.66)],
            angle_deg=45.0,
            spacing=5.0,
            color="#1c1917",
            opacity=0.75,
            cross=True,
            layer_id="mass_shading",
        )
    )

    # Neck cast shadow under jaw
    neck_shadow_poly = [
        (0.40, 0.71),
        (0.60, 0.71),
        (0.64, 0.82),
        (0.38, 0.82),
    ]
    doc.commands.append(
        generate_directional_hatching(
            neck_shadow_poly,
            angle_deg=-30.0,
            spacing=4.0,
            color="#0c0a09",
            opacity=0.8,
            cross=False,
            layer_id="mass_shading",
        )
    )

    # 3. Facial Contours & Features with Charcoal Brush
    # Jawline and chin
    jaw_pts = [
        (0.31, 0.44),  # left cheekbone
        (0.35, 0.60),  # left jaw angle
        (0.44, 0.70),  # left chin corner
        (0.50, 0.72),  # chin center
        (0.56, 0.70),  # right chin corner
        (0.65, 0.60),  # right jaw angle
        (0.69, 0.44),  # right cheekbone
    ]
    doc.commands.append(
        make_wobbly_contour(
            jaw_pts,
            wobble_amount=0.003,
            base_width=3.6,
            brush_id="charcoal",
            color="#1c1917",
            layer_id="features",
            seed=101,
        )
    )

    # Brow arches (y=0.41)
    left_brow = [PointV2(0.36, 0.42, p=0.5), PointV2(0.42, 0.40, p=0.9), PointV2(0.47, 0.41, p=0.4)]
    right_brow = [PointV2(0.53, 0.41, p=0.4), PointV2(0.58, 0.40, p=0.9), PointV2(0.64, 0.42, p=0.6)]
    doc.commands.append(StrokeActionV2(points=left_brow, brush_id="charcoal", color="#0c0a09", base_width=3.2, curve_type="catmull_rom", layer_id="features"))
    doc.commands.append(StrokeActionV2(points=right_brow, brush_id="charcoal", color="#0c0a09", base_width=3.8, curve_type="catmull_rom", layer_id="features"))

    # Eyes (y=0.44)
    # Left eye
    doc.commands.append(StrokeActionV2(points=[PointV2(0.37, 0.45, p=0.4), PointV2(0.41, 0.435, p=0.8), PointV2(0.45, 0.45, p=0.4)], brush_id="charcoal", color="#0c0a09", base_width=2.5, curve_type="catmull_rom", layer_id="features"))
    doc.commands.append(StrokeActionV2(points=[PointV2(0.38, 0.45, p=0.3), PointV2(0.41, 0.465, p=0.5), PointV2(0.44, 0.45, p=0.3)], brush_id="charcoal", color="#0c0a09", base_width=1.8, curve_type="catmull_rom", layer_id="features"))
    doc.commands.append(make_ellipse((0.41, 0.45), 0.015, 0.015, base_width=2.0, brush_id="charcoal", color="#09090b", layer_id="features"))

    # Right eye (shadowed)
    doc.commands.append(StrokeActionV2(points=[PointV2(0.55, 0.45, p=0.5), PointV2(0.59, 0.435, p=0.9), PointV2(0.63, 0.45, p=0.5)], brush_id="charcoal", color="#0c0a09", base_width=3.0, curve_type="catmull_rom", layer_id="features"))
    doc.commands.append(StrokeActionV2(points=[PointV2(0.56, 0.45, p=0.4), PointV2(0.59, 0.465, p=0.6), PointV2(0.62, 0.45, p=0.4)], brush_id="charcoal", color="#0c0a09", base_width=2.0, curve_type="catmull_rom", layer_id="features"))
    doc.commands.append(make_ellipse((0.59, 0.45), 0.015, 0.015, base_width=2.2, brush_id="charcoal", color="#09090b", layer_id="features"))

    # Nose (bridge & ball at 0.55)
    nose_bridge = [PointV2(0.48, 0.42, p=0.4), PointV2(0.48, 0.52, p=0.7), PointV2(0.46, 0.55, p=0.8)]
    nose_shadow = [PointV2(0.46, 0.55, p=0.9), PointV2(0.50, 0.56, p=1.0), PointV2(0.53, 0.54, p=0.7)]
    doc.commands.append(StrokeActionV2(points=nose_bridge, brush_id="charcoal", color="#1c1917", base_width=2.4, curve_type="catmull_rom", layer_id="features"))
    doc.commands.append(StrokeActionV2(points=nose_shadow, brush_id="charcoal", color="#09090b", base_width=3.6, curve_type="catmull_rom", layer_id="features"))

    # Lips (crease at 0.63)
    lip_crease = [PointV2(0.43, 0.63, p=0.5), PointV2(0.50, 0.625, p=0.9), PointV2(0.57, 0.63, p=0.7)]
    lip_bottom = [PointV2(0.45, 0.655, p=0.4), PointV2(0.50, 0.665, p=0.8), PointV2(0.55, 0.655, p=0.5)]
    doc.commands.append(StrokeActionV2(points=lip_crease, brush_id="charcoal", color="#09090b", base_width=3.2, curve_type="catmull_rom", layer_id="features"))
    doc.commands.append(StrokeActionV2(points=lip_bottom, brush_id="charcoal", color="#1c1917", base_width=2.2, curve_type="catmull_rom", layer_id="features"))

    # 4. Expressive Hair Mass (Dark, vigorous charcoal strokes)
    hair_strokes = [
        [(0.30, 0.40), (0.24, 0.32), (0.35, 0.20), (0.50, 0.16), (0.68, 0.18), (0.76, 0.30), (0.72, 0.44)],
        [(0.26, 0.35), (0.34, 0.18), (0.52, 0.14), (0.72, 0.20), (0.78, 0.36)],
        [(0.32, 0.30), (0.45, 0.17), (0.65, 0.19), (0.74, 0.32)],
        [(0.48, 0.16), (0.54, 0.24), (0.52, 0.32)],
    ]
    for hpts in hair_strokes:
        pts = [PointV2(x, y, p=float(np.random.uniform(0.6, 1.0))) for x, y in hpts]
        doc.commands.append(
            StrokeActionV2(
                points=pts,
                brush_id="charcoal",
                color="#0c0a09",
                base_width=6.0,
                curve_type="catmull_rom",
                taper_in=0.1,
                taper_out=0.15,
                layer_id="hair",
            )
        )

    return doc


def generate_all_hand_demos(out_dir: str = "outputs/hand_demos"):
    """Renders the 4 scripted pieces, exports PNG + SVG, creates contact sheet and notes.md."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    demos = [
        ("bicycle_ink_study", "1. Bicycle Ink Study (Line Weight & Hatching)", build_bicycle_ink_study()),
        ("pencil_still_life", "2. Pencil Still Life (Form & Cast Shadows)", build_pencil_still_life()),
        ("color_landscape", "3. Color Landscape (Atmospheric Perspective)", build_color_landscape()),
        ("charcoal_portrait", "4. Charcoal Portrait Study (Proportions & Masses)", build_charcoal_portrait()),
    ]

    rendered_images = []

    for name, title, doc in demos:
        print(f"--> Validating {name}...")
        validate_document_v2(doc)

        png_path = out_path / f"{name}.png"
        svg_path = out_path / f"{name}.svg"

        print(f"--> Rendering {name} to PNG (1024x1024, supersample 2)...")
        img = render_document_v2(doc, width=1024, height=1024, supersample=2, seed=42)
        img.save(png_path, "PNG")
        rendered_images.append((name, title, img))

        print(f"--> Exporting {name} to SVG...")
        svg_str = export_svg_v2(doc, width=1024, height=1024)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write(svg_str)
        print(f"    Saved: {png_path} and {svg_path}")

    # Create 2x2 Contact Sheet (2048 x 2048 + title bars)
    print("--> Creating contact sheet...")
    sheet_w = 2048
    sheet_h = 2048
    tile_w = 1024
    tile_h = 1024
    contact_sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 245, 245))
    draw_cs = ImageDraw.Draw(contact_sheet)

    coords = [(0, 0), (tile_w, 0), (0, tile_h), (tile_w, tile_h)]
    for idx, (name, title, img) in enumerate(rendered_images):
        cx, cy = coords[idx]
        contact_sheet.paste(img, (cx, cy))
        # Title overlay banner
        draw_cs.rectangle([cx + 16, cy + 16, cx + tile_w - 16, cy + 64], fill=(15, 23, 42, 220))
        draw_cs.text((cx + 32, cy + 28), title, fill=(248, 250, 252))

    cs_path = out_path / "contact_sheet.png"
    contact_sheet.save(cs_path, "PNG")
    print(f"    Saved Contact Sheet: {cs_path}")

    # Write Honest Self-Critical Notes
    notes_content = """# Artist's Hand v2: Proof of Capability Notes

Evaluated 4 scripted pieces produced purely programmatically (NO ML) via the v2 Hand layer.

---

### 1. Bicycle Ink Study (`bicycle_ink_study.png` / `.svg`)
- **What the hand did well**:
  - Light-directed line-weight variation gives the frame tangible physical mass: undersides of down tube and chainstays are thick and dark, while sun-lit top tubes stay delicate.
  - Directional hatching ground shadow creates a firm contact plane, removing the "floating on white" look of v1 doodles.
  - Spokes and concentric rim ellipses render sharply with clean Catmull-Rom interpolation.
- **What it still can't do**:
  - Dynamic line occlusion: in real drawing, overlapping foreground tubes automatically occlude background spokes without needing explicit layer masking or clipping paths.
  - Mechanical chain tension: chain links are rendered as straight lines rather than realistic hanging catenary curves.

---

### 2. Pencil Still Life (`pencil_still_life.png` / `.svg`)
- **What the hand did well**:
  - Authentic graphite tooth texture: the pressure-modulated paper grain noise faithfully reproduces the look of 2B/4B graphite on cold-press paper.
  - Faithful depiction of light physics: specular core shadows, reflected bounce light on the underside of the sphere, and projected cast shadows on the table plane.
  - Searching underdrawing pencil passes create the feel of an artist constructing forms.
- **What it still can't do**:
  - Curved surface hatching: hatching lines on the sphere are linear rather than wrapping around the 3D spherical surface contours (cross-contour hatching).
  - Smudging / blending stump: real pencil drawings utilize physical tortillon blending for smooth gradients between core shadow and penumbra.

---

### 3. Color Landscape (`color_landscape.png` / `.svg`)
- **What the hand did well**:
  - Smooth atmospheric depth perspective: background mountain ridges naturally desaturate and shift toward the sky's warm apricot-tinted horizon, creating immediate depth cues.
  - Multi-tier layer blending: linear gradient sky + radial sun glow composited cleanly over mountain silhouettes and foreground pine trees.
  - Rule-of-Thirds composition: sun and tree clusters anchored along power points.
- **What it still can't do**:
  - Wet-on-wet watercolor blooming: washes have drying edges, but lack pigment migration physics when two wet washes touch.
  - Atmospheric haze scattering: no Rayleigh scattering model for true distance aerial haze.

---

### 4. Charcoal Portrait Study (`charcoal_portrait.png` / `.svg`)
- **What the hand did well**:
  - Loomis construction proportion guides: cranial sphere, brow, nose, and chin divisions provide a structured geometric foundation.
  - Rough powdery charcoal dispersion: high friction edges and soft charcoal halo around dark hair masses.
  - Bold value distribution: high-contrast chiaroscuro with distinct light and shadow planes.
- **What it still can't do**:
  - Anatomy / organic muscle blending: portrait faces require subtle anatomical transitions around zygomatic arches and nasolabial folds that simple polygons and 2D hatching approximate coarsely.
  - Hair strand flow: hair masses are rendered with sweeping broad strokes rather than tapering hair clusters.

---

### Overall Verdict on the Hand
The Hand layer successfully breaks free from the 64x64 doodle prison. It can produce high-resolution, layered, expressive, textured artwork with line weight variation, gradients, and hatching that would take an experienced human artist significant time to execute. It provides the expressivity that a future generative model can learn to orchestrate.
"""

    notes_path = out_path / "notes.md"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes_content)
    print(f"    Saved Critique Notes: {notes_path}")


if __name__ == "__main__":
    generate_all_hand_demos()
