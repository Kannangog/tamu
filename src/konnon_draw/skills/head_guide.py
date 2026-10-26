"""Loomis-method head construction guide and rotation engine.

Generates cranium sphere, brow line, center line, jaw, ears, and facial contours
at arbitrary yaw and pitch rotations with accurate perspective foreshortening.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..v2.schema import PointV2, StrokeActionV2
from .linecraft import make_confident_stroke, make_ellipse


def build_head_construction(
    center: Tuple[float, float] = (0.5, 0.45),
    radius: float = 0.16,
    yaw_deg: float = 0.0,    # -90 (left profile) .. 0 (front) .. +90 (right profile)
    pitch_deg: float = 0.0,  # -30 (looking down) .. 0 (level) .. +30 (looking up)
    include_features: bool = True,
    guide_color: str = "#3b82f6",
    ink_color: str = "#0f172a",
) -> List[StrokeActionV2]:
    """Constructs a Loomis head model rotated at (yaw, pitch) with guides and finished ink contours."""
    cx, cy = center
    yaw_rad = math.radians(yaw_deg)
    pitch_rad = math.radians(pitch_deg)

    strokes: List[StrokeActionV2] = []

    # -------------------------------------------------------------
    # 1. CRANIUM SPHERE (Underdrawing Guide)
    # -------------------------------------------------------------
    cranium_pts = []
    n_pts = 36
    for i in range(n_pts + 1):
        ang = (i % n_pts) / float(n_pts) * 2.0 * math.pi
        px = cx + radius * math.cos(ang)
        py = cy + radius * math.sin(ang)
        cranium_pts.append(PointV2(px, py, p=0.4))

    strokes.append(
        StrokeActionV2(
            points=cranium_pts,
            brush_id="pencil",
            color=guide_color,
            base_width=1.5,
            opacity=0.5,
            layer_id="underdrawing",
            stage="guide",
            purpose="construction",
            depth_rank=0,
            part_id="head",
        )
    )

    # -------------------------------------------------------------
    # 2. BROW LINE & EYE LINE ARCS (Curving with Pitch)
    # -------------------------------------------------------------
    # Brow line is at y_offset = 0.0, eye line is at y_offset = 0.08 * radius
    for offset_ratio, label, color_stroke in [(0.0, "brow", guide_color), (0.15, "eyes", "#64748b")]:
        brow_pts = []
        for step in range(21):
            u = step / 20.0  # -1 to +1 along x
            t_ang = (u - 0.5) * math.pi * 0.95
            # 3D spherical coordinate: X = r*cos(t_ang), Y = offset, Z = r*sin(t_ang)
            r_slice = radius * math.cos(offset_ratio)
            x_3d = r_slice * math.sin(t_ang)
            y_3d = -offset_ratio * radius
            z_3d = r_slice * math.cos(t_ang)

            # Rotate around Y (yaw)
            x_rot = x_3d * math.cos(yaw_rad) + z_3d * math.sin(yaw_rad)
            z_rot = -x_3d * math.sin(yaw_rad) + z_3d * math.cos(yaw_rad)

            # Rotate around X (pitch)
            y_rot = y_3d * math.cos(pitch_rad) - z_rot * math.sin(pitch_rad)

            if z_rot > -0.1 * radius:  # Front facing
                brow_pts.append(PointV2(cx + x_rot, cy + y_rot, p=0.45))

        if len(brow_pts) >= 2:
            strokes.append(
                StrokeActionV2(
                    points=brow_pts,
                    brush_id="pencil",
                    color=color_stroke,
                    base_width=1.4,
                    opacity=0.6,
                    layer_id="underdrawing",
                    stage="guide",
                    purpose="construction",
                    depth_rank=1,
                    part_id="head",
                )
            )

    # -------------------------------------------------------------
    # 3. VERTICAL CENTER PROFILE LINE (Curving with Yaw)
    # -------------------------------------------------------------
    center_pts = []
    for step in range(25):
        v = step / 24.0
        phi = (v - 0.5) * math.pi * 0.95  # top to bottom
        x_3d = 0.0
        y_3d = radius * math.sin(phi)
        z_3d = radius * math.cos(phi)

        # Rotate around Y (yaw)
        x_rot = x_3d * math.cos(yaw_rad) + z_3d * math.sin(yaw_rad)
        z_rot = -x_3d * math.sin(yaw_rad) + z_3d * math.cos(yaw_rad)

        # Rotate around X (pitch)
        y_rot = y_3d * math.cos(pitch_rad) - z_rot * math.sin(pitch_rad)

        if z_rot > -0.1 * radius:
            center_pts.append(PointV2(cx + x_rot, cy + y_rot, p=0.5))

    if len(center_pts) >= 2:
        strokes.append(
            StrokeActionV2(
                points=center_pts,
                brush_id="pencil",
                color=guide_color,
                base_width=1.5,
                opacity=0.6,
                layer_id="underdrawing",
                stage="guide",
                purpose="construction",
                depth_rank=2,
                part_id="head",
            )
        )

    # -------------------------------------------------------------
    # 4. JAW & CHIN CONTOUR (Loomis Jaw Box)
    # -------------------------------------------------------------
    chin_y_3d = radius * 1.55
    chin_x_3d = 0.0
    chin_z_3d = radius * 0.45

    # Chin 3D rotated
    chin_x = chin_x_3d * math.cos(yaw_rad) + chin_z_3d * math.sin(yaw_rad)
    chin_z = -chin_x_3d * math.sin(yaw_rad) + chin_z_3d * math.cos(yaw_rad)
    chin_y = chin_y_3d * math.cos(pitch_rad) - chin_z * math.sin(pitch_rad)

    # Left and right temples/ears
    left_ear_x = -radius * 0.9 * math.cos(yaw_rad)
    right_ear_x = radius * 0.9 * math.cos(yaw_rad)
    ear_y = radius * 0.1

    # Jaw path
    jaw_pts = [
        PointV2(cx + left_ear_x, cy + ear_y, p=0.5),
        PointV2(cx + left_ear_x * 0.7, cy + ear_y + radius * 0.7, p=0.7),
        PointV2(cx + chin_x - radius * 0.12, cy + chin_y, p=0.9),
        PointV2(cx + chin_x + radius * 0.12, cy + chin_y, p=0.9),
        PointV2(cx + right_ear_x * 0.7, cy + ear_y + radius * 0.7, p=0.7),
        PointV2(cx + right_ear_x, cy + ear_y, p=0.5),
    ]
    strokes.append(
        StrokeActionV2(
            points=jaw_pts,
            brush_id="ink_liner",
            color=ink_color,
            base_width=2.8,
            layer_id="line_art",
            stage="ink",
            purpose="contour",
            depth_rank=5,
            part_id="head",
        )
    )

    # -------------------------------------------------------------
    # 5. FINISHED INK FEATURES (Eyes, Nose, Mouth)
    # -------------------------------------------------------------
    if include_features:
        # Nose wedge
        nose_base_y = cy + (radius * 0.5) * math.cos(pitch_rad)
        nose_tip_x = cx + chin_x * 0.6 + radius * 0.18 * math.sin(yaw_rad)
        nose_pts = [
            PointV2(cx + chin_x * 0.4, cy + radius * 0.15, p=0.4),
            PointV2(nose_tip_x, nose_base_y, p=0.8),
            PointV2(cx + chin_x * 0.5, nose_base_y + radius * 0.05, p=0.6),
        ]
        strokes.append(
            StrokeActionV2(
                points=nose_pts,
                brush_id="ink_liner",
                color=ink_color,
                base_width=2.2,
                layer_id="line_art",
                stage="ink",
                purpose="contour",
                depth_rank=6,
                part_id="head",
            )
        )

        # Mouth contour
        mouth_y = cy + (radius * 0.95) * math.cos(pitch_rad)
        mouth_w = radius * 0.35 * math.cos(abs(yaw_rad) * 0.5)
        mouth_pts = [
            PointV2(cx + chin_x * 0.5 - mouth_w, mouth_y, p=0.4),
            PointV2(cx + chin_x * 0.5, mouth_y + radius * 0.03, p=0.85),
            PointV2(cx + chin_x * 0.5 + mouth_w, mouth_y, p=0.4),
        ]
        strokes.append(
            StrokeActionV2(
                points=mouth_pts,
                brush_id="ink_liner",
                color=ink_color,
                base_width=2.2,
                layer_id="line_art",
                stage="ink",
                purpose="contour",
                depth_rank=6,
                part_id="head",
            )
        )

        # Neck cylinders
        neck_pts_l = [PointV2(cx + left_ear_x * 0.5, cy + radius * 1.1, p=0.7), PointV2(cx + left_ear_x * 0.6, cy + radius * 1.8, p=0.4)]
        neck_pts_r = [PointV2(cx + right_ear_x * 0.5, cy + radius * 1.1, p=0.7), PointV2(cx + right_ear_x * 0.6, cy + radius * 1.8, p=0.4)]
        strokes.append(
            StrokeActionV2(
                points=neck_pts_l,
                brush_id="ink_liner",
                color=ink_color,
                base_width=2.5,
                layer_id="line_art",
                stage="ink",
                purpose="contour",
                depth_rank=3,
                part_id="head",
            )
        )
        strokes.append(
            StrokeActionV2(
                points=neck_pts_r,
                brush_id="ink_liner",
                color=ink_color,
                base_width=2.5,
                layer_id="line_art",
                stage="ink",
                purpose="contour",
                depth_rank=3,
                part_id="head",
            )
        )

    return strokes
