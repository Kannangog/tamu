"""Generates Gate 2 scripted demonstration artworks and composite contact sheet (NO ML).

Outputs:
1. outputs/hand_demos/mannequin_figure.png (and separate stage files)
2. outputs/hand_demos/head_rotations.png
3. outputs/hand_demos/pose_sequence.png
4. outputs/hand_demos/pencil_still_life.png
5. outputs/hand_demos/gate2_contact_sheet.png
"""

import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from konnon_draw.v2.schema import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
)
from konnon_draw.v2.renderer import render_document_v2
from konnon_draw.skills.composition import create_standard_layers
from konnon_draw.skills.mannequin3d import generate_mannequin_drawing
from konnon_draw.skills.head_guide import build_head_construction
from konnon_draw.skills.linecraft import (
    make_confident_stroke,
    make_ellipse,
    make_sketchy_lines,
    make_wobbly_contour,
    apply_line_weight_by_light,
)
from konnon_draw.skills.shading import (
    generate_cast_shadow_polygon,
    generate_directional_hatching,
    make_gradient_sky,
)


def generate_piece_1_mannequin_figure(out_dir: Path):
    """Piece 1: Mannequin -> Sketch -> Ink in 3 separately saved stages across 3 poses."""
    poses = [("stand", 0.22, 10.0), ("reach", 0.50, 25.0), ("run", 0.78, -20.0)]
    
    # Combined multi-pose document
    doc_full = DocumentV2(
        canvas=CanvasConfigV2(width=1200, height=800, background="#f8fafc"),
        layers=create_standard_layers(),
    )
    
    doc_layout = DocumentV2(canvas=CanvasConfigV2(width=1200, height=800, background="#f8fafc"), layers=create_standard_layers())
    doc_guide = DocumentV2(canvas=CanvasConfigV2(width=1200, height=800, background="#f8fafc"), layers=create_standard_layers())
    doc_ink = DocumentV2(canvas=CanvasConfigV2(width=1200, height=800, background="#f8fafc"), layers=create_standard_layers())

    for idx, (p_name, cx, yaw) in enumerate(poses):
        sub_doc = generate_mannequin_drawing(
            pose_name=p_name,
            preset="adult",
            yaw_deg=yaw,
            pitch_deg=8.0,
            distance=2.5,
            camera_center=(cx, 0.48),
            seed=42 + idx * 7,
        )
        for cmd in sub_doc.commands:
            doc_full.commands.append(cmd)
            stg = getattr(cmd, "stage", "")
            if stg == "layout":
                doc_layout.commands.append(cmd)
            elif stg == "guide":
                doc_guide.commands.append(cmd)
            elif stg == "ink":
                doc_ink.commands.append(cmd)

    img_full = render_document_v2(doc_full, width=1200, height=800, supersample=2, seed=42)
    img_layout = render_document_v2(doc_layout, width=1200, height=800, supersample=2, seed=42)
    img_guide = render_document_v2(doc_guide, width=1200, height=800, supersample=2, seed=42)
    img_ink = render_document_v2(doc_ink, width=1200, height=800, supersample=2, seed=42)

    img_full.save(out_dir / "mannequin_figure.png")
    img_layout.save(out_dir / "mannequin_figure_layout.png")
    img_guide.save(out_dir / "mannequin_figure_guide.png")
    img_ink.save(out_dir / "mannequin_figure_ink.png")
    print("Generated Piece 1: mannequin_figure.png + 3 stage files")
    return img_full


def generate_piece_2_head_rotations(out_dir: Path):
    """Piece 2: Head construction -> finished head at 3 rotations (front, 3/4, profile)."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1200, height=600, background="#f8fafc"),
        layers=create_standard_layers(),
    )

    rotations = [
        ("Front (Yaw 0°)", 0.22, 0.0, 0.0),
        ("Three-Quarter (Yaw 35°)", 0.50, 35.0, -5.0),
        ("Profile (Yaw 80°)", 0.78, 80.0, 5.0),
    ]

    for title, cx, yaw, pitch in rotations:
        cmds = build_head_construction(
            center=(cx, 0.45),
            radius=0.14,
            yaw_deg=yaw,
            pitch_deg=pitch,
            include_features=True,
        )
        doc.commands.extend(cmds)

    img = render_document_v2(doc, width=1200, height=600, supersample=2, seed=42)

    # Label headers
    draw = ImageDraw.Draw(img)
    for title, cx, _, _ in rotations:
        px = int(cx * 1200 - 80)
        draw.text((px, 40), title, fill=(15, 23, 42))

    img.save(out_dir / "head_rotations.png")
    print("Generated Piece 2: head_rotations.png")
    return img


def generate_piece_3_pose_sequence(out_dir: Path):
    """Piece 3: 4-frame dynamic pose sequence of the same proportions."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1600, height=600, background="#f8fafc"),
        layers=create_standard_layers(),
    )

    sequence = [
        ("Frame 1: Stand", "stand", 0.15, 0.0),
        ("Frame 2: Walk", "walk", 0.38, 15.0),
        ("Frame 3: Run", "run", 0.62, 25.0),
        ("Frame 4: Crouch", "crouch", 0.85, 10.0),
    ]

    for title, p_name, cx, yaw in sequence:
        sub_doc = generate_mannequin_drawing(
            pose_name=p_name,
            preset="adult",
            yaw_deg=yaw,
            pitch_deg=6.0,
            distance=2.4,
            camera_center=(cx, 0.48),
            seed=42,
        )
        doc.commands.extend(sub_doc.commands)

    img = render_document_v2(doc, width=1600, height=600, supersample=2, seed=42)
    draw = ImageDraw.Draw(img)
    for title, _, cx, _ in sequence:
        px = int(cx * 1600 - 60)
        draw.text((px, 35), title, fill=(15, 23, 42))

    img.save(out_dir / "pose_sequence.png")
    print("Generated Piece 3: pose_sequence.png")
    return img


def generate_piece_4_pencil_still_life(out_dir: Path):
    """Piece 4: Pencil still life (3 objects: sphere, cylinder, cube) with light-direction shading and cast shadows."""
    doc = DocumentV2(
        canvas=CanvasConfigV2(width=1000, height=800, background="#fbfcf8"),
        layers=create_standard_layers(),
    )

    light_dir = (-0.7071, -0.7071)  # Top-left light source

    # 1. Horizon & tabletop line
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(0.05, 0.62, p=0.4), PointV2(0.95, 0.62, p=0.4)],
            brush_id="pencil",
            color="#64748b",
            base_width=2.0,
            layer_id="line_art",
        )
    )

    # -------------------------------------------------------------
    # Object 1: Sphere (Left)
    # -------------------------------------------------------------
    s_center = (0.28, 0.52)
    s_r = 0.12
    # Cast shadow
    sphere_shadow = generate_cast_shadow_polygon(
        base_contact_points=[(s_center[0] - s_r * 0.8, s_center[1] + s_r * 0.9), (s_center[0] + s_r * 0.8, s_center[1] + s_r * 0.9)],
        light_dir=light_dir,
        shadow_length=0.18,
        color="#1e293b",
        opacity=0.45,
    )
    doc.commands.append(sphere_shadow)

    # Sphere contour
    doc.commands.append(
        make_ellipse(
            center=s_center,
            rx=s_r,
            ry=s_r,
            brush_id="pencil",
            color="#334155",
            base_width=3.2,
            layer_id="line_art",
        )
    )
    # Sphere core crescent shading (hatching)
    crescent_poly = [
        (s_center[0] - s_r * 0.1, s_center[1] - s_r * 0.7),
        (s_center[0] + s_r * 0.9, s_center[1] - s_r * 0.2),
        (s_center[0] + s_r * 0.8, s_center[1] + s_r * 0.8),
        (s_center[0] - s_r * 0.3, s_center[1] + s_r * 0.7),
    ]
    doc.commands.append(
        generate_directional_hatching(
            crescent_poly,
            light_dir=light_dir,
            spacing=4.5,
            color="#334155",
            opacity=0.6,
        )
    )

    # -------------------------------------------------------------
    # Object 2: Cylinder (Center)
    # -------------------------------------------------------------
    c_x = 0.52
    c_y_top = 0.32
    c_y_bot = 0.58
    c_rw = 0.08
    c_rh = 0.03

    # Cast shadow
    cyl_shadow = generate_cast_shadow_polygon(
        base_contact_points=[(c_x - c_rw, c_y_bot), (c_x + c_rw, c_y_bot)],
        light_dir=light_dir,
        shadow_length=0.22,
        color="#1e293b",
        opacity=0.45,
    )
    doc.commands.append(cyl_shadow)

    # Top ellipse
    doc.commands.append(
        make_ellipse(
            center=(c_x, c_y_top),
            rx=c_rw,
            ry=c_rh,
            brush_id="pencil",
            color="#334155",
            base_width=2.8,
            layer_id="line_art",
        )
    )
    # Bottom ellipse arc
    doc.commands.append(
        make_ellipse(
            center=(c_x, c_y_bot),
            rx=c_rw,
            ry=c_rh,
            brush_id="pencil",
            color="#334155",
            base_width=2.8,
            layer_id="line_art",
        )
    )
    # Vertical side lines
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(c_x - c_rw, c_y_top, p=0.5), PointV2(c_x - c_rw, c_y_bot, p=0.5)],
            brush_id="pencil",
            color="#334155",
            base_width=2.6,
        )
    )
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(c_x + c_rw, c_y_top, p=0.85), PointV2(c_x + c_rw, c_y_bot, p=0.9)],
            brush_id="pencil",
            color="#1e293b",
            base_width=3.4,
        )
    )
    # Cylinder shadow side hatching
    cyl_hatch_poly = [
        (c_x, c_y_top),
        (c_x + c_rw, c_y_top),
        (c_x + c_rw, c_y_bot),
        (c_x, c_y_bot),
    ]
    doc.commands.append(
        generate_directional_hatching(
            cyl_hatch_poly,
            light_dir=light_dir,
            spacing=4.0,
            color="#334155",
            opacity=0.65,
        )
    )

    # -------------------------------------------------------------
    # Object 3: Cube in Perspective (Right)
    # -------------------------------------------------------------
    # Front square
    cube_front = [(0.72, 0.42), (0.88, 0.42), (0.88, 0.58), (0.72, 0.58)]
    # Vanishing perspective depth offset
    vp_dx, vp_dy = 0.06, -0.07
    cube_back = [(x + vp_dx, y + vp_dy) for x, y in cube_front]

    # Cast shadow
    cube_shadow = generate_cast_shadow_polygon(
        base_contact_points=[cube_front[3], cube_front[2], cube_back[2]],
        light_dir=light_dir,
        shadow_length=0.20,
        color="#1e293b",
        opacity=0.45,
    )
    doc.commands.append(cube_shadow)

    # Top face
    top_face = [cube_front[0], cube_back[0], cube_back[1], cube_front[1]]
    doc.commands.append(
        FillActionV2(path=top_face, fill_type="flat", color="#f1f5f9", opacity=0.9, layer_id="flats")
    )
    # Right side face (in shadow)
    right_face = [cube_front[1], cube_back[1], cube_back[2], cube_front[2]]
    doc.commands.append(
        FillActionV2(path=right_face, fill_type="flat", color="#94a3b8", opacity=0.7, layer_id="flats")
    )
    doc.commands.append(
        generate_directional_hatching(right_face, light_dir=light_dir, spacing=4.0, cross_hatch=True, color="#1e293b", opacity=0.7)
    )

    # Cube wireframe contours
    for poly in [cube_front, top_face, right_face]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(p1[0], p1[1], p=0.6), PointV2(p2[0], p2[1], p=0.7)],
                    brush_id="pencil",
                    color="#1e293b",
                    base_width=2.8,
                    layer_id="line_art",
                )
            )

    img = render_document_v2(doc, width=1000, height=800, supersample=2, seed=42)
    img.save(out_dir / "pencil_still_life.png")
    print("Generated Piece 4: pencil_still_life.png")
    return img


def create_gate2_contact_sheet(
    img_mannequin: Image.Image,
    img_head: Image.Image,
    img_sequence: Image.Image,
    img_still_life: Image.Image,
    out_path: Path,
):
    """Arranges the 4 demo pieces into a clean, annotated 2x2 contact sheet."""
    tile_w, tile_h = 800, 600
    banner_h = 40
    sheet = Image.new("RGB", (tile_w * 2, (tile_h + banner_h) * 2), (241, 245, 249))
    draw = ImageDraw.Draw(sheet)

    tiles = [
        (img_mannequin, "1. Articulated Mannequin (3 Poses & Stages with Depth Occlusion)", 0, 0),
        (img_head, "2. Loomis Head Rotations (Yaw 0°, 35°, 80° Perspective Guides)", 1, 0),
        (img_sequence, "3. 4-Frame Dynamic Pose Sequence (Consistent Proportion Preset)", 0, 1),
        (img_still_life, "4. Pencil Still Life (3 Objects, Directional Shading & Cast Shadows)", 1, 1),
    ]

    for img, title, col, row in tiles:
        x = col * tile_w
        y = row * (tile_h + banner_h)

        resized = img.resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        sheet.paste(resized, (x, y))

        draw.rectangle([x, y + tile_h, x + tile_w, y + tile_h + banner_h], fill=(15, 23, 42))
        draw.text((x + 16, y + tile_h + 12), title, fill=(248, 250, 252))

    sheet.save(out_path)
    print(f"Generated Gate 2 contact sheet: {out_path}")
    return sheet


if __name__ == "__main__":
    out_dir = Path("outputs/hand_demos")
    out_dir.mkdir(parents=True, exist_ok=True)

    img1 = generate_piece_1_mannequin_figure(out_dir)
    img2 = generate_piece_2_head_rotations(out_dir)
    img3 = generate_piece_3_pose_sequence(out_dir)
    img4 = generate_piece_4_pencil_still_life(out_dir)

    create_gate2_contact_sheet(
        img1, img2, img3, img4,
        out_dir / "gate2_contact_sheet.png",
    )
