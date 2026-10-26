"""Unit tests for Phase 2: Skills library, 3D mannequin, camera projection, occlusion, and linecraft."""

import math
import numpy as np
import pytest

from konnon_draw.skills.mannequin3d import (
    PinholeCamera,
    sample_pose,
    generate_mannequin_drawing,
    PROPORTIONS,
)
from konnon_draw.skills.head_guide import build_head_construction
from konnon_draw.skills.linecraft import (
    make_ellipse,
    make_arc,
    make_wobbly_contour,
    make_sketchy_lines,
    make_confident_stroke,
    apply_line_weight_by_light,
)
from konnon_draw.skills.shading import (
    generate_directional_hatching,
    generate_cast_shadow_polygon,
    generate_rim_light_strokes,
)
from konnon_draw.skills.effects import (
    make_speed_lines,
    make_motion_arcs,
    make_impact_burst,
)
from konnon_draw.v2.renderer import render_document_v2
from konnon_draw.v2.schema import validate_document_v2, DocumentV2


def test_camera_projection_known_points():
    """Verify pinhole camera projects origin (0,0,0) to canvas center (0.5, 0.5)."""
    cam = PinholeCamera(target=(0.0, 0.0, 0.0), yaw_deg=0.0, pitch_deg=0.0, distance=2.0)
    sx, sy, sz = cam.project_point((0.0, 0.0, 0.0))
    assert abs(sx - 0.5) < 1e-4, f"Expected sx=0.5, got {sx}"
    assert abs(sy - 0.5) < 1e-4, f"Expected sy=0.5, got {sy}"


def test_pose_sampler_joint_validity():
    """Verify that all archetypes produce finite, biologically reasonable coordinates."""
    for pose in ["stand", "walk", "run", "crouch", "reach", "sit"]:
        joints = sample_pose(pose_name=pose, preset="adult")
        assert "head" in joints and "l_wrist" in joints and "r_ankle" in joints
        for j_name, pt in joints.items():
            assert np.all(np.isfinite(pt)), f"Non-finite joint {j_name} in {pose}"
            # Bounds check: character stays within reasonable physical volume (-2m to +2m)
            assert np.all(np.abs(pt) < 2.5), f"Joint {j_name} exploded: {pt}"


def test_depth_occlusion_in_mannequin():
    """In a reach pose with yaw=0, right arm extended forward must have higher depth_rank than torso."""
    doc = generate_mannequin_drawing(pose_name="reach", yaw_deg=0.0, pitch_deg=0.0)
    assert validate_document_v2(doc) is True

    # Find right_arm commands and torso commands
    arm_cmds = [c for c in doc.commands if getattr(c, "part_id", "") == "right_arm"]
    torso_cmds = [c for c in doc.commands if getattr(c, "part_id", "") == "torso"]

    assert len(arm_cmds) > 0 and len(torso_cmds) > 0
    max_arm_dr = max(c.depth_rank for c in arm_cmds)
    torso_dr = torso_cmds[0].depth_rank
    # Right arm reaching toward camera must have >= torso depth rank
    assert max_arm_dr >= torso_dr


def test_hatching_angle_responds_to_light_direction():
    """Hatching lines should be perpendicular to incident light vector."""
    poly = [(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)]

    # Light from top-left (-1, -1) -> direction -135 deg -> perpendicular hatch angle: 135 deg (or -45 deg)
    hatch1 = generate_directional_hatching(poly, light_dir=(-1.0, -1.0))
    # Light from directly above (0, -1) -> direction -90 deg -> perpendicular hatch angle: 0 deg
    hatch2 = generate_directional_hatching(poly, light_dir=(0.0, -1.0))

    assert abs(hatch1.angle_deg - 135.0) < 1.0, f"Expected 135 deg, got {hatch1.angle_deg}"
    assert abs(hatch2.angle_deg - 0.0) < 1.0, f"Expected 0 deg, got {hatch2.angle_deg}"


def test_linecraft_light_weight_variation():
    """Shadow-facing segment of a curve must receive higher pressure than light-facing segment."""
    # Horizontal line from (0.1, 0.5) to (0.9, 0.5). Normal points in +Y direction (0, 1)
    # Light coming from top (0, -1) -> normal points away from light (shadow side)
    stroke = make_confident_stroke([(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)])
    weighted = apply_line_weight_by_light(stroke, light_dir=(0.0, -1.0), min_pressure=0.2, max_pressure=1.0)

    # All points facing shadow side should have pressure close to max_pressure
    for p in weighted.points:
        assert p.p > 0.8


def test_effects_generators():
    """Verify speed lines, motion arcs, and impact bursts emit valid strokes."""
    speed_lines = make_speed_lines(direction_deg=45.0, num_lines=8)
    assert len(speed_lines) == 8
    assert all(len(s.points) == 3 for s in speed_lines)

    arcs = make_motion_arcs((0.5, 0.5), num_arcs=4)
    assert len(arcs) == 4

    burst = make_impact_burst((0.5, 0.5), num_rays=12)
    assert len(burst) == 12
