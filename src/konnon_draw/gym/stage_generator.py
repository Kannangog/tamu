"""Staged drawing generator for human figures and parametric objects with stroke-order augmentation."""

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..v2.schema import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
)
from ..v2.renderer import render_document_v2
from ..skills.composition import create_standard_layers
from ..skills.mannequin3d import generate_mannequin_drawing
from ..skills.head_guide import build_head_construction
from ..skills.linecraft import (
    make_confident_stroke,
    make_ellipse,
    make_sketchy_lines,
    make_wobbly_contour,
    apply_line_weight_by_light,
)
from ..skills.shading import (
    generate_cast_shadow_polygon,
    generate_directional_hatching,
    generate_rim_light_strokes,
    make_gradient_sky,
)
from ..skills.effects import (
    make_speed_lines,
    make_motion_arcs,
    make_impact_burst,
)
from .captions import generate_caption

STAGE_ORDER = ["layout", "guide", "sketch", "ink", "shade", "effect"]


def permute_strokes_within_stages(
    commands: List[Union[StrokeActionV2, FillActionV2]],
    rng: Optional[random.Random] = None,
) -> List[Union[StrokeActionV2, FillActionV2]]:
    """Augmentation: permutes independent strokes within each stage without violating layer order."""
    if rng is None:
        rng = random.Random(42)

    by_stage: Dict[str, List[Any]] = {s: [] for s in STAGE_ORDER}
    for cmd in commands:
        stg = getattr(cmd, "stage", "ink")
        if stg not in by_stage:
            by_stage[stg] = []
        by_stage[stg].append(cmd)

    permuted: List[Any] = []
    for stg in STAGE_ORDER:
        cmds = by_stage[stg]
        if len(cmds) > 1:
            # Group by depth_rank to avoid flipping depth occlusion
            by_depth: Dict[int, List[Any]] = {}
            for c in cmds:
                dr = getattr(c, "depth_rank", 0)
                if dr not in by_depth:
                    by_depth[dr] = []
                by_depth[dr].append(c)

            # Within each depth rank, shuffle order
            for dr in sorted(by_depth.keys()):
                group = list(by_depth[dr])
                rng.shuffle(group)
                permuted.extend(group)
        else:
            permuted.extend(cmds)

    return permuted


def generate_staged_human_drawing(
    pose_name: str = "stand",
    preset: str = "adult",
    yaw_deg: float = 15.0,
    pitch_deg: float = 6.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates a complete multi-stage human figure drawing with caption."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    base_doc = generate_mannequin_drawing(
        pose_name=pose_name,
        preset=preset,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        seed=seed,
    )

    caption = generate_caption(category=pose_name, camera_yaw=yaw_deg, seed=seed)

    # Add Shading & Effects stages
    # Cast shadow
    shadow_cmd = generate_cast_shadow_polygon(
        base_contact_points=[(0.40, 0.88), (0.60, 0.88)],
        light_dir=(-0.7071, -0.7071),
        shadow_length=0.18,
    )
    base_doc.commands.append(shadow_cmd)

    # Optional motion arcs or speed lines for dynamic poses
    if pose_name in ("run", "walk", "reach"):
        arcs = make_motion_arcs(center=(0.5, 0.45), radius=0.22, num_arcs=3, rng=np_rng)
        base_doc.commands.extend(arcs)

    # Organize strokes into strict chronological stages
    stages_data = []
    accum_commands = []

    for stage_name in STAGE_ORDER:
        stage_cmds = [c for c in base_doc.commands if getattr(c, "stage", "ink") == stage_name]
        if not stage_cmds:
            continue
        accum_commands.extend(stage_cmds)
        doc_stage = DocumentV2(
            canvas=CanvasConfigV2(width=512, height=512),
            layers=create_standard_layers(),
            commands=copy.deepcopy(accum_commands),
        )
        stages_data.append({
            "stage_name": stage_name,
            "commands": stage_cmds,
            "doc_so_far": doc_stage,
        })

    return {
        "category": pose_name,
        "caption": caption,
        "full_doc": base_doc,
        "stages": stages_data,
    }


def generate_staged_object_drawing(
    object_type: str = "chair",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates a staged object drawing (chair, box, bicycle, house) with caption."""
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    light_dir = (-0.7071, -0.7071)

    doc = DocumentV2(
        canvas=CanvasConfigV2(width=512, height=512),
        layers=create_standard_layers(),
    )

    caption = generate_caption(category=object_type, seed=seed)

    if object_type == "box":
        # Stage 1: Layout (perspective vanishing guides)
        vp = (0.9, 0.25)
        front = [(0.25, 0.45), (0.55, 0.45), (0.55, 0.75), (0.25, 0.75)]
        for pt in front:
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(pt[0], pt[1], p=0.3), PointV2(vp[0], vp[1], p=0.1)],
                    brush_id="pencil",
                    color="#94a3b8",
                    base_width=1.2,
                    layer_id="underdrawing",
                    stage="layout",
                    purpose="construction",
                )
            )

        # Stage 2: Guide
        dx, dy = 0.12, -0.10
        back = [(p[0] + dx, p[1] + dy) for p in front]
        for p1, p2 in zip(front, back):
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(p1[0], p1[1], p=0.5), PointV2(p2[0], p2[1], p=0.5)],
                    brush_id="pencil",
                    color="#64748b",
                    base_width=1.6,
                    layer_id="underdrawing",
                    stage="guide",
                    purpose="construction",
                )
            )

        # Stage 4: Ink contours
        for poly in [front, [front[0], back[0], back[1], front[1]], [front[1], back[1], back[2], front[2]]]:
            for i in range(len(poly)):
                p_a = poly[i]
                p_b = poly[(i + 1) % len(poly)]
                doc.commands.append(
                    StrokeActionV2(
                        points=[PointV2(p_a[0], p_a[1], p=0.7), PointV2(p_b[0], p_b[1], p=0.7)],
                        brush_id="ink_liner",
                        color="#0f172a",
                        base_width=3.0,
                        layer_id="line_art",
                        stage="ink",
                        purpose="contour",
                    )
                )

        # Stage 5: Shade
        shadow = generate_cast_shadow_polygon(
            base_contact_points=[front[3], front[2], back[2]],
            light_dir=light_dir,
            shadow_length=0.20,
        )
        doc.commands.append(shadow)
        hatch = generate_directional_hatching(
            [front[1], back[1], back[2], front[2]],
            light_dir=light_dir,
            spacing=5.0,
            cross_hatch=True,
        )
        doc.commands.append(hatch)

    elif object_type == "chair":
        # Stage 1: Layout box
        doc.commands.append(
            StrokeActionV2(
                points=[PointV2(0.3, 0.25, p=0.4), PointV2(0.7, 0.25, p=0.4), PointV2(0.7, 0.85, p=0.4), PointV2(0.3, 0.85, p=0.4), PointV2(0.3, 0.25, p=0.4)],
                brush_id="pencil",
                color="#94a3b8",
                base_width=1.4,
                layer_id="underdrawing",
                stage="layout",
                purpose="construction",
            )
        )
        # Stage 4: Ink contours
        # Seat
        seat = [(0.32, 0.55), (0.68, 0.55), (0.62, 0.65), (0.28, 0.65)]
        for i in range(len(seat)):
            p_a, p_b = seat[i], seat[(i + 1) % len(seat)]
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(p_a[0], p_a[1], p=0.75), PointV2(p_b[0], p_b[1], p=0.75)],
                    brush_id="ink_liner",
                    color="#0f172a",
                    base_width=3.2,
                    layer_id="line_art",
                    stage="ink",
                    purpose="contour",
                )
            )
        # Backrest & Legs
        legs = [
            [(0.28, 0.65), (0.28, 0.85)],
            [(0.62, 0.65), (0.62, 0.85)],
            [(0.32, 0.55), (0.32, 0.80)],
            [(0.68, 0.55), (0.68, 0.80)],
            [(0.32, 0.55), (0.32, 0.25)],
            [(0.68, 0.55), (0.68, 0.25)],
            [(0.32, 0.28), (0.68, 0.28)],
        ]
        for leg in legs:
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(leg[0][0], leg[0][1], p=0.8), PointV2(leg[1][0], leg[1][1], p=0.8)],
                    brush_id="ink_liner",
                    color="#0f172a",
                    base_width=2.8,
                    layer_id="line_art",
                    stage="ink",
                    purpose="contour",
                )
            )
        # Stage 5: Shade
        shadow = generate_cast_shadow_polygon(
            base_contact_points=[(0.28, 0.85), (0.62, 0.85)],
            light_dir=light_dir,
            shadow_length=0.15,
        )
        doc.commands.append(shadow)

    elif object_type == "bicycle":
        # Stage 1: Layout tangent circles
        w1_c = (0.32, 0.62)
        w2_c = (0.68, 0.62)
        r_wheel = 0.14
        for wc in [w1_c, w2_c]:
            doc.commands.append(
                make_ellipse(wc, r_wheel, r_wheel, brush_id="pencil", color="#94a3b8", base_width=1.5, layer_id="underdrawing", stage="layout", purpose="construction")
            )
        # Stage 4: Ink frame and wheels
        for wc in [w1_c, w2_c]:
            doc.commands.append(
                make_ellipse(wc, r_wheel, r_wheel, brush_id="ink_liner", color="#0f172a", base_width=3.4, layer_id="line_art", stage="ink", purpose="contour")
            )
            doc.commands.append(
                make_ellipse(wc, 0.02, 0.02, brush_id="ink_liner", color="#0f172a", base_width=2.5, layer_id="line_art", stage="ink", purpose="contour")
            )
        # Diamond frame
        bb_c = (0.50, 0.62)
        seat_c = (0.42, 0.42)
        head_c = (0.62, 0.38)
        frame_tubes = [
            [w1_c, seat_c], [seat_c, bb_c], [w1_c, bb_c],
            [bb_c, head_c], [seat_c, head_c], [head_c, w2_c],
            [head_c, (0.60, 0.32)], [(0.57, 0.32), (0.63, 0.32)], # Handlebars
            [seat_c, (0.41, 0.39)], [(0.38, 0.39), (0.44, 0.39)], # Saddle
        ]
        for tube in frame_tubes:
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(tube[0][0], tube[0][1], p=0.8), PointV2(tube[1][0], tube[1][1], p=0.8)],
                    brush_id="ink_liner",
                    color="#0f172a",
                    base_width=3.0,
                    layer_id="line_art",
                    stage="ink",
                    purpose="contour",
                )
            )
        # Stage 5: Shade
        shadow = generate_cast_shadow_polygon(
            base_contact_points=[(w1_c[0], w1_c[1] + r_wheel), (w2_c[0], w2_c[1] + r_wheel)],
            light_dir=light_dir,
            shadow_length=0.16,
        )
        doc.commands.append(shadow)

    else:  # house
        # Stage 1: Layout box
        doc.commands.append(
            StrokeActionV2(
                points=[PointV2(0.2, 0.3, p=0.4), PointV2(0.8, 0.3, p=0.4), PointV2(0.8, 0.8, p=0.4), PointV2(0.2, 0.8, p=0.4), PointV2(0.2, 0.3, p=0.4)],
                brush_id="pencil",
                color="#94a3b8",
                base_width=1.4,
                layer_id="underdrawing",
                stage="layout",
                purpose="construction",
            )
        )
        # Stage 4: Ink contours
        wall = [(0.25, 0.48), (0.75, 0.48), (0.75, 0.82), (0.25, 0.82), (0.25, 0.48)]
        roof = [(0.20, 0.48), (0.50, 0.25), (0.80, 0.48), (0.20, 0.48)]
        door = [(0.44, 0.82), (0.44, 0.62), (0.56, 0.62), (0.56, 0.82)]
        chimney = [(0.65, 0.36), (0.65, 0.22), (0.72, 0.22), (0.72, 0.42)]
        for comp in [wall, roof, door, chimney]:
            doc.commands.append(
                StrokeActionV2(
                    points=[PointV2(p[0], p[1], p=0.8) for p in comp],
                    brush_id="ink_liner",
                    color="#0f172a",
                    base_width=3.2,
                    layer_id="line_art",
                    stage="ink",
                    purpose="contour",
                )
            )
        # Stage 5: Shade
        shadow = generate_cast_shadow_polygon(
            base_contact_points=[(0.25, 0.82), (0.75, 0.82)],
            light_dir=light_dir,
            shadow_length=0.18,
        )
        doc.commands.append(shadow)

    # Accumulate stages
    stages_data = []
    accum_commands = []
    for stage_name in STAGE_ORDER:
        stage_cmds = [c for c in doc.commands if getattr(c, "stage", "ink") == stage_name]
        if not stage_cmds:
            continue
        accum_commands.extend(stage_cmds)
        doc_stage = DocumentV2(
            canvas=CanvasConfigV2(width=512, height=512),
            layers=create_standard_layers(),
            commands=copy.deepcopy(accum_commands),
        )
        stages_data.append({
            "stage_name": stage_name,
            "commands": stage_cmds,
            "doc_so_far": doc_stage,
        })

    return {
        "category": object_type,
        "caption": caption,
        "full_doc": doc,
        "stages": stages_data,
    }
