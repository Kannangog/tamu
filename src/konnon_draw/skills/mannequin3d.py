"""3D Mannequin skeleton, pose sampler, pinhole camera projection, and depth-ranked strokes."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..v2.schema import DocumentV2, LayerV2, PointV2, StrokeActionV2
from .composition import create_standard_layers


# -------------------------------------------------------------
# Proportion Presets (relative limb lengths normalized by head size)
# -------------------------------------------------------------

PROPORTIONS = {
    "adult": {
        "head_r": 0.065,
        "neck_len": 0.035,
        "torso_h": 0.16,
        "torso_w": 0.13,
        "pelvis_h": 0.08,
        "pelvis_w": 0.11,
        "upper_arm": 0.13,
        "forearm": 0.12,
        "thigh": 0.18,
        "calf": 0.17,
        "total_heads": 7.5,
    },
    "child": {
        "head_r": 0.11,
        "neck_len": 0.02,
        "torso_h": 0.12,
        "torso_w": 0.10,
        "pelvis_h": 0.06,
        "pelvis_w": 0.09,
        "upper_arm": 0.09,
        "forearm": 0.08,
        "thigh": 0.11,
        "calf": 0.10,
        "total_heads": 4.5,
    },
    "stylized": {
        "head_r": 0.08,
        "neck_len": 0.04,
        "torso_h": 0.18,
        "torso_w": 0.15,
        "pelvis_h": 0.09,
        "pelvis_w": 0.12,
        "upper_arm": 0.16,
        "forearm": 0.15,
        "thigh": 0.22,
        "calf": 0.21,
        "total_heads": 8.5,
    },
}


# -------------------------------------------------------------
# Pinhole Camera Model
# -------------------------------------------------------------

class PinholeCamera:
    """3D Camera with spherical orbit (yaw, pitch, distance) and perspective projection."""

    def __init__(
        self,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        yaw_deg: float = 0.0,
        pitch_deg: float = 10.0,
        distance: float = 2.5,
        focal_length: float = 1.8,
        canvas_center: Tuple[float, float] = (0.5, 0.5),
    ):
        self.target = np.array(target, dtype=np.float32)
        self.yaw = math.radians(yaw_deg)
        self.pitch = math.radians(pitch_deg)
        self.dist = distance
        self.focal = focal_length
        self.cx, self.cy = canvas_center

        # Camera eye in 3D
        cam_x = target[0] + distance * math.sin(self.yaw) * math.cos(self.pitch)
        cam_y = target[1] + distance * math.sin(self.pitch)
        cam_z = target[2] + distance * math.cos(self.yaw) * math.cos(self.pitch)
        self.eye = np.array([cam_x, cam_y, cam_z], dtype=np.float32)

        # Coordinate axes: Forward (Z), Right (X), Up (Y)
        forward = self.target - self.eye
        f_norm = np.linalg.norm(forward)
        self.fwd = forward / (f_norm if f_norm > 1e-6 else 1.0)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(self.fwd, world_up)
        r_norm = np.linalg.norm(right)
        self.right = right / (r_norm if r_norm > 1e-6 else 1.0)
        self.up = np.cross(self.right, self.fwd)

    def project_point(self, pt_3d: Union[np.ndarray, Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Projects 3D world coordinate into (screen_x, screen_y, cam_z).
        
        Screen coords are normalized [0..1] with (0,0) top-left.
        cam_z is distance along camera forward axis (higher = nearer to camera).
        """
        p = np.array(pt_3d, dtype=np.float32) - self.eye
        x_cam = float(np.dot(p, self.right))
        y_cam = float(np.dot(p, self.up))
        z_cam = float(np.dot(p, self.fwd))

        # Perspective division
        z_safe = max(0.1, z_cam)
        screen_x = self.cx + (x_cam * self.focal) / z_safe
        # Invert y because screen y points downward
        screen_y = self.cy - (y_cam * self.focal) / z_safe
        # Depth rank: nearer objects have higher rank (smaller z_safe from eye = nearer)
        depth_rank = -z_cam
        return screen_x, screen_y, depth_rank


# -------------------------------------------------------------
# Pose Sampler (Joint Limits & Preset Archetypes)
# -------------------------------------------------------------

def sample_pose(
    pose_name: str = "stand",
    preset: str = "adult",
    jitter: float = 0.0,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, np.ndarray]:
    """Computes 3D joint positions for a mannequin given a pose archetype."""
    if rng is None:
        rng = np.random.RandomState(42)

    props = PROPORTIONS.get(preset, PROPORTIONS["adult"])

    # Base skeleton root at hips (0, 0, 0)
    hips = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    spine_mid = hips + np.array([0.0, props["pelvis_h"] * 0.7, 0.0])
    chest = spine_mid + np.array([0.0, props["torso_h"], 0.0])
    neck = chest + np.array([0.0, props["neck_len"], 0.0])
    head = neck + np.array([0.0, props["head_r"] * 1.5, 0.0])

    hw_t = props["torso_w"] * 0.5
    hw_p = props["pelvis_w"] * 0.5

    l_shoulder = chest + np.array([-hw_t, -0.01, 0.0])
    r_shoulder = chest + np.array([hw_t, -0.01, 0.0])

    l_hip = hips + np.array([-hw_p, -props["pelvis_h"] * 0.5, 0.0])
    r_hip = hips + np.array([hw_p, -props["pelvis_h"] * 0.5, 0.0])

    ua = props["upper_arm"]
    fa = props["forearm"]
    th = props["thigh"]
    cf = props["calf"]

    joints: Dict[str, np.ndarray] = {
        "head": head,
        "neck": neck,
        "chest": chest,
        "spine_mid": spine_mid,
        "hips": hips,
        "l_shoulder": l_shoulder,
        "r_shoulder": r_shoulder,
        "l_hip": l_hip,
        "r_hip": r_hip,
    }

    if pose_name == "stand":
        # Relaxed contrapposto standing pose
        joints["l_elbow"] = l_shoulder + np.array([-0.02, -ua * 0.95, 0.03])
        joints["l_wrist"] = joints["l_elbow"] + np.array([0.01, -fa * 0.95, 0.04])

        joints["r_elbow"] = r_shoulder + np.array([0.02, -ua * 0.95, 0.03])
        joints["r_wrist"] = joints["r_elbow"] + np.array([-0.01, -fa * 0.95, 0.04])

        joints["l_knee"] = l_hip + np.array([-0.01, -th * 0.98, 0.02])
        joints["l_ankle"] = joints["l_knee"] + np.array([0.0, -cf * 0.98, 0.01])

        joints["r_knee"] = r_hip + np.array([0.01, -th * 0.98, 0.02])
        joints["r_ankle"] = joints["r_knee"] + np.array([0.0, -cf * 0.98, 0.01])

    elif pose_name == "walk":
        # Mid-stride walking
        joints["l_elbow"] = l_shoulder + np.array([-0.02, -ua * 0.7, 0.08])
        joints["l_wrist"] = joints["l_elbow"] + np.array([0.01, -fa * 0.6, 0.12])

        joints["r_elbow"] = r_shoulder + np.array([0.02, -ua * 0.7, -0.08])
        joints["r_wrist"] = joints["r_elbow"] + np.array([-0.01, -fa * 0.6, -0.12])

        # Left leg swinging forward, right leg trailing back
        joints["l_knee"] = l_hip + np.array([-0.01, -th * 0.85, 0.10])
        joints["l_ankle"] = joints["l_knee"] + np.array([0.0, -cf * 0.85, 0.08])

        joints["r_knee"] = r_hip + np.array([0.01, -th * 0.85, -0.10])
        joints["r_ankle"] = joints["r_knee"] + np.array([0.0, -cf * 0.85, -0.12])

    elif pose_name == "run":
        # Dynamic running pose
        joints["l_elbow"] = l_shoulder + np.array([-0.04, -ua * 0.4, 0.15])
        joints["l_wrist"] = joints["l_elbow"] + np.array([0.02, fa * 0.2, 0.12])

        joints["r_elbow"] = r_shoulder + np.array([0.04, -ua * 0.4, -0.15])
        joints["r_wrist"] = joints["r_elbow"] + np.array([-0.02, -fa * 0.7, -0.08])

        joints["l_knee"] = l_hip + np.array([-0.02, -th * 0.6, 0.22])
        joints["l_ankle"] = joints["l_knee"] + np.array([0.01, -cf * 0.8, 0.15])

        joints["r_knee"] = r_hip + np.array([0.02, -th * 0.7, -0.20])
        joints["r_ankle"] = joints["r_knee"] + np.array([-0.01, -cf * 0.5, -0.25])

    elif pose_name == "crouch":
        # Deep athletic crouch
        joints["chest"] = joints["chest"] + np.array([0.0, -0.12, 0.08])
        joints["head"] = joints["chest"] + np.array([0.0, props["head_r"] * 1.4, 0.06])
        joints["l_elbow"] = l_shoulder + np.array([-0.03, -ua * 0.5, 0.10])
        joints["l_wrist"] = joints["l_elbow"] + np.array([0.0, -fa * 0.6, 0.12])

        joints["r_elbow"] = r_shoulder + np.array([0.03, -ua * 0.5, 0.10])
        joints["r_wrist"] = joints["r_elbow"] + np.array([0.0, -fa * 0.6, 0.12])

        joints["l_knee"] = l_hip + np.array([-hw_p * 0.8, -th * 0.4, 0.16])
        joints["l_ankle"] = joints["l_knee"] + np.array([0.0, -cf * 0.5, -0.14])

        joints["r_knee"] = r_hip + np.array([hw_p * 0.8, -th * 0.4, 0.16])
        joints["r_ankle"] = joints["r_knee"] + np.array([0.0, -cf * 0.5, -0.14])

    elif pose_name == "reach":
        # One arm reaching upward, body arched
        joints["r_elbow"] = r_shoulder + np.array([0.02, ua * 0.9, 0.05])
        joints["r_wrist"] = joints["r_elbow"] + np.array([0.01, fa * 0.95, 0.04])

        joints["l_elbow"] = l_shoulder + np.array([-0.03, -ua * 0.8, -0.05])
        joints["l_wrist"] = joints["l_elbow"] + np.array([-0.01, -fa * 0.8, -0.04])

        joints["l_knee"] = l_hip + np.array([-0.01, -th * 0.98, 0.02])
        joints["l_ankle"] = joints["l_knee"] + np.array([0.0, -cf * 0.98, 0.01])

        joints["r_knee"] = r_hip + np.array([0.01, -th * 0.95, -0.05])
        joints["r_ankle"] = joints["r_knee"] + np.array([0.0, -cf * 0.95, -0.08])

    else:  # sit
        joints["l_knee"] = l_hip + np.array([-0.01, 0.0, th * 0.95])
        joints["l_ankle"] = joints["l_knee"] + np.array([0.0, -cf * 0.95, 0.0])

        joints["r_knee"] = r_hip + np.array([0.01, 0.0, th * 0.95])
        joints["r_ankle"] = joints["r_knee"] + np.array([0.0, -cf * 0.95, 0.0])

        joints["l_elbow"] = l_shoulder + np.array([-0.02, -ua * 0.8, 0.05])
        joints["l_wrist"] = joints["l_elbow"] + np.array([0.0, 0.0, fa * 0.7])

        joints["r_elbow"] = r_shoulder + np.array([0.02, -ua * 0.8, 0.05])
        joints["r_wrist"] = joints["r_elbow"] + np.array([0.0, 0.0, fa * 0.7])

    # Apply random perturbations within safe biological joint bounds
    if jitter > 1e-4:
        for k in joints:
            if k not in ("hips",):
                joints[k] = joints[k] + rng.normal(0, jitter, 3).astype(np.float32)

    return joints


# -------------------------------------------------------------
# 3D Mannequin Stroke Generator (with Depth Occlusion & Stages)
# -------------------------------------------------------------

def generate_mannequin_drawing(
    pose_name: str = "stand",
    preset: str = "adult",
    yaw_deg: float = 15.0,
    pitch_deg: float = 5.0,
    distance: float = 2.4,
    camera_center: Tuple[float, float] = (0.5, 0.5),
    stage_filter: Optional[str] = None,  # "layout", "guide", "sketch", "ink", or None for all
    seed: int = 42,
) -> DocumentV2:
    """Renders a fully articulated 3D mannequin into staged, depth-ranked DocumentV2 actions."""
    rng = np.random.RandomState(seed)
    camera = PinholeCamera(
        target=(0.0, 0.0, 0.0),
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        distance=distance,
        canvas_center=camera_center,
    )

    joints_3d = sample_pose(pose_name=pose_name, preset=preset, rng=rng)
    props = PROPORTIONS.get(preset, PROPORTIONS["adult"])

    # Project all joints to 2D screen space with camera depth
    proj: Dict[str, Tuple[float, float, float]] = {}
    for name, p3 in joints_3d.items():
        proj[name] = camera.project_point(p3)

    doc = DocumentV2(layers=create_standard_layers())

    # Map depth rank: map camera z into integer rank in range 1..100
    all_z = [p[2] for p in proj.values()]
    min_z, max_z = min(all_z), max(all_z)
    z_span = max(1e-4, max_z - min_z)

    def calc_dr(p_name: str) -> int:
        z = proj[p_name][2]
        # Larger depth_rank = nearer to camera = drawn on top
        return int(round(((z - min_z) / z_span) * 50)) + 10

    # -------------------------------------------------------------
    # STAGE 1: LAYOUT (Stick Bones in underdrawing)
    # -------------------------------------------------------------
    layout_bones = [
        ("head", "neck", "head"),
        ("neck", "chest", "torso"),
        ("chest", "spine_mid", "torso"),
        ("spine_mid", "hips", "hips"),
        ("chest", "l_shoulder", "torso"),
        ("chest", "r_shoulder", "torso"),
        ("l_shoulder", "l_elbow", "left_arm"),
        ("l_elbow", "l_wrist", "left_arm"),
        ("r_shoulder", "r_elbow", "right_arm"),
        ("r_elbow", "r_wrist", "right_arm"),
        ("hips", "l_hip", "hips"),
        ("hips", "r_hip", "hips"),
        ("l_hip", "l_knee", "left_leg"),
        ("l_knee", "l_ankle", "left_leg"),
        ("r_hip", "r_knee", "right_leg"),
        ("r_knee", "r_ankle", "right_leg"),
    ]

    for j1, j2, part in layout_bones:
        p1 = proj[j1]
        p2 = proj[j2]
        dr = max(calc_dr(j1), calc_dr(j2))
        doc.commands.append(
            StrokeActionV2(
                points=[PointV2(p1[0], p1[1], p=0.4), PointV2(p2[0], p2[1], p=0.4)],
                brush_id="pencil",
                color="#64748b",
                base_width=1.8,
                opacity=0.45,
                layer_id="underdrawing",
                stage="layout",
                purpose="construction",
                depth_rank=dr,
                part_id=part,
            )
        )

    # -------------------------------------------------------------
    # STAGE 2: GUIDE (Limb Capsules, Joint Circles, Torso/Hip Boxes)
    # -------------------------------------------------------------
    # Joint spheres
    for j_name, (sx, sy, sz) in proj.items():
        dr = calc_dr(j_name)
        r_px = 0.015
        circle_pts = []
        for i in range(13):
            ang = (i % 12) / 12.0 * 2.0 * math.pi
            circle_pts.append(PointV2(sx + r_px * math.cos(ang), sy + r_px * math.sin(ang), p=0.35))
        doc.commands.append(
            StrokeActionV2(
                points=circle_pts,
                brush_id="pencil",
                color="#3b82f6",
                base_width=1.5,
                opacity=0.5,
                layer_id="underdrawing",
                stage="guide",
                purpose="construction",
                depth_rank=dr,
                part_id="joint",
            )
        )

    # Limb capsules (Cylinders with cross-contour ellipses)
    limbs = [
        ("l_shoulder", "l_elbow", "left_arm", 0.025),
        ("l_elbow", "l_wrist", "left_arm", 0.020),
        ("r_shoulder", "r_elbow", "right_arm", 0.025),
        ("r_elbow", "r_wrist", "right_arm", 0.020),
        ("l_hip", "l_knee", "left_leg", 0.035),
        ("l_knee", "l_ankle", "left_leg", 0.028),
        ("r_hip", "r_knee", "right_leg", 0.035),
        ("r_knee", "r_ankle", "right_leg", 0.028),
    ]

    for j1, j2, part, radius_limb in limbs:
        p1 = proj[j1]
        p2 = proj[j2]
        dr = max(calc_dr(j1), calc_dr(j2))

        # Vector along limb
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        mag = max(1e-5, math.hypot(dx, dy))
        nx = -dy / mag * radius_limb
        ny = dx / mag * radius_limb

        # Capsule contour (left and right boundary)
        left_side = [PointV2(p1[0] + nx, p1[1] + ny, p=0.6), PointV2(p2[0] + nx * 0.85, p2[1] + ny * 0.85, p=0.6)]
        right_side = [PointV2(p1[0] - nx, p1[1] - ny, p=0.6), PointV2(p2[0] - nx * 0.85, p2[1] - ny * 0.85, p=0.6)]

        doc.commands.append(
            StrokeActionV2(
                points=left_side,
                brush_id="pencil",
                color="#3b82f6",
                base_width=1.6,
                opacity=0.6,
                layer_id="underdrawing",
                stage="guide",
                purpose="construction",
                depth_rank=dr,
                part_id=part,
            )
        )
        doc.commands.append(
            StrokeActionV2(
                points=right_side,
                brush_id="pencil",
                color="#3b82f6",
                base_width=1.6,
                opacity=0.6,
                layer_id="underdrawing",
                stage="guide",
                purpose="construction",
                depth_rank=dr,
                part_id=part,
            )
        )

        # Cross-contour ring at mid-limb
        mid_x = (p1[0] + p2[0]) * 0.5
        mid_y = (p1[1] + p2[1]) * 0.5
        ring_pts = []
        for i in range(13):
            ang = (i % 12) / 12.0 * 2.0 * math.pi
            rx = nx * math.cos(ang) + (dx / mag * radius_limb * 0.4) * math.sin(ang)
            ry = ny * math.cos(ang) + (dy / mag * radius_limb * 0.4) * math.sin(ang)
            ring_pts.append(PointV2(mid_x + rx, mid_y + ry, p=0.4))
        doc.commands.append(
            StrokeActionV2(
                points=ring_pts,
                brush_id="pencil",
                color="#3b82f6",
                base_width=1.2,
                opacity=0.4,
                layer_id="underdrawing",
                stage="guide",
                purpose="construction",
                depth_rank=dr,
                part_id=part,
            )
        )

    # -------------------------------------------------------------
    # STAGE 3: INK (Finished Form Contours with Calligraphic Line Weights)
    # -------------------------------------------------------------
    # Head cranium and chin
    h_proj = proj["head"]
    n_proj = proj["neck"]
    h_dr = calc_dr("head")
    head_rad = props["head_r"] * 1.2

    head_contour = []
    for i in range(25):
        ang = (i % 24) / 24.0 * 2.0 * math.pi
        px = h_proj[0] + head_rad * math.cos(ang)
        py = h_proj[1] + head_rad * 1.25 * math.sin(ang)
        press = 0.5 + 0.4 * math.sin(ang)  # thicker at chin/underside
        head_contour.append(PointV2(px, py, p=press))

    doc.commands.append(
        StrokeActionV2(
            points=head_contour,
            brush_id="ink_liner",
            color="#0f172a",
            base_width=3.2,
            layer_id="line_art",
            stage="ink",
            purpose="contour",
            depth_rank=h_dr + 10,
            part_id="head",
        )
    )

    # Torso contour
    c_proj = proj["chest"]
    hip_proj = proj["hips"]
    t_dr = calc_dr("chest")

    torso_left = [
        PointV2(proj["l_shoulder"][0], proj["l_shoulder"][1], p=0.7),
        PointV2(proj["spine_mid"][0] - 0.045, proj["spine_mid"][1], p=0.8),
        PointV2(proj["l_hip"][0], proj["l_hip"][1], p=0.6),
    ]
    torso_right = [
        PointV2(proj["r_shoulder"][0], proj["r_shoulder"][1], p=0.7),
        PointV2(proj["spine_mid"][0] + 0.045, proj["spine_mid"][1], p=0.8),
        PointV2(proj["r_hip"][0], proj["r_hip"][1], p=0.6),
    ]

    doc.commands.append(
        StrokeActionV2(
            points=torso_left,
            brush_id="ink_liner",
            color="#0f172a",
            base_width=3.0,
            layer_id="line_art",
            stage="ink",
            purpose="contour",
            depth_rank=t_dr,
            part_id="torso",
        )
    )
    doc.commands.append(
        StrokeActionV2(
            points=torso_right,
            brush_id="ink_liner",
            color="#0f172a",
            base_width=3.0,
            layer_id="line_art",
            stage="ink",
            purpose="contour",
            depth_rank=t_dr,
            part_id="torso",
        )
    )

    # Ink contours for all limbs with depth occlusion
    for j1, j2, part, radius_limb in limbs:
        p1 = proj[j1]
        p2 = proj[j2]
        dr = max(calc_dr(j1), calc_dr(j2))

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        mag = max(1e-5, math.hypot(dx, dy))
        nx = -dy / mag * radius_limb
        ny = dx / mag * radius_limb

        # Outer ink contour
        ink_pts_1 = [
            PointV2(p1[0] + nx, p1[1] + ny, p=0.6),
            PointV2((p1[0] + p2[0]) * 0.5 + nx * 1.05, (p1[1] + p2[1]) * 0.5 + ny * 1.05, p=0.85),
            PointV2(p2[0] + nx * 0.85, p2[1] + ny * 0.85, p=0.5),
        ]
        ink_pts_2 = [
            PointV2(p1[0] - nx, p1[1] - ny, p=0.6),
            PointV2((p1[0] + p2[0]) * 0.5 - nx * 1.05, (p1[1] + p2[1]) * 0.5 - ny * 1.05, p=0.85),
            PointV2(p2[0] - nx * 0.85, p2[1] - ny * 0.85, p=0.5),
        ]

        doc.commands.append(
            StrokeActionV2(
                points=ink_pts_1,
                brush_id="ink_liner",
                color="#0f172a",
                base_width=2.8,
                layer_id="line_art",
                stage="ink",
                purpose="contour",
                depth_rank=dr,
                part_id=part,
            )
        )
        doc.commands.append(
            StrokeActionV2(
                points=ink_pts_2,
                brush_id="ink_liner",
                color="#0f172a",
                base_width=2.8,
                layer_id="line_art",
                stage="ink",
                purpose="contour",
                depth_rank=dr,
                part_id=part,
            )
        )

    # Filter by stage if requested
    if stage_filter is not None:
        doc.commands = [cmd for cmd in doc.commands if getattr(cmd, "stage", "") == stage_filter]

    return doc
