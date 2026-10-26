"""Ground-truth caption generator with 10+ paraphrase templates per concept."""

import random
from typing import Any, Dict, List, Optional


POSE_TEMPLATES = {
    "stand": [
        "a full-body figure standing in a relaxed pose",
        "an upright person standing still in anatomical position",
        "a standing mannequin viewed from the front",
        "a full figure standing with arms at their sides",
        "a study of a standing person with natural weight distribution",
        "an athletic figure standing balanced on both feet",
        "a pencil and ink drawing of someone standing",
        "a character standing facing forward with slight contrapposto",
        "a clean sketch of a standing human figure",
        "a full-length standing pose constructed with guides and ink",
        "an artist mannequin in a classic standing stance",
        "a gesture drawing of a person standing upright",
    ],
    "walk": [
        "a figure walking mid-stride with swinging arms",
        "a person walking forward with natural gait mechanics",
        "a dynamic sketch of someone walking in perspective",
        "a walking human figure with one leg swinging forward",
        "a walking character with reciprocal arm and leg motion",
        "a study of locomotion showing a person taking a stride",
        "a stride pose showing human walking motion in perspective",
        "a figure in the passing position of a walk cycle",
        "an artist sketch of a walking mannequin with foreshortened limbs",
        "a character captured mid-walk from a dynamic camera angle",
        "a gestural line drawing of a person walking forward",
    ],
    "run": [
        "an athletic figure sprinting with maximum momentum",
        "a dynamic character running forward with driving knees",
        "a figure captured in a fast running sprint pose",
        "an energetic study of a runner pushing off the ground",
        "a dramatic action sketch of a person running",
        "a character running fast with wide stride and bent elbows",
        "a high-speed running pose with forward torso lean",
        "a runner in motion with limbs extended in perspective",
        "a kinetic drawing of a sprinting figure with speed lines",
        "a full-body sketch of a runner mid-dash",
        "a powerful gesture drawing of someone running",
    ],
    "crouch": [
        "a figure crouched low in a dynamic athletic stance",
        "a person crouching on the balls of their feet",
        "a deep crouch pose with bent knees and lowered center of gravity",
        "a character crouched ready to leap",
        "a low crouching figure with foreshortened thighs and torso",
        "a study of human compression in a deep crouch",
        "a stealthy figure crouching close to the ground",
        "a compact crouch pose with arms forward for balance",
        "a dynamic defensive crouch drawing",
        "a figure coiled in a tight athletic crouch",
    ],
    "reach": [
        "a figure reaching one arm high into the air",
        "a character extending their arm upward in an expansive stretch",
        "a dynamic pose of a person reaching for something above",
        "a figure with an extended reaching arm and arched spine",
        "an upward reaching pose showing elongation through the torso",
        "a person reaching out with one hand extended towards the viewer",
        "a dramatic reaching gesture with full arm extension",
        "a full-body drawing of someone reaching high above their head",
        "a character stretching upward with one arm elevated",
        "a study of reach and extension in human anatomy",
    ],
    "sit": [
        "a figure seated with bent knees in a relaxed posture",
        "a person sitting down with thighs parallel to the ground",
        "a seated character viewed in three-quarter perspective",
        "a calm pose of a person sitting upright",
        "a study of a seated figure with 90-degree bend at hips and knees",
        "a character seated comfortably with hands resting forward",
        "a sketch of someone sitting with stable ground contact",
        "a seated mannequin with foreshortened lap in perspective",
        "a figure sitting down with clear structural planes",
        "a gestural drawing of a seated human form",
    ],
}

OBJECT_TEMPLATES = {
    "box": [
        "a geometric cube in perspective with shaded side planes and cast shadow",
        "a three-dimensional box drawn with accurate vanishing points and hatching",
        "a solid cube still life with top plane lit and directional shadow",
        "a perspective study of a wooden box with cast ground shadow",
        "a clean geometric cube rendered with pencil hatching and cast shadow",
        "a block in one-point perspective with cross-hatched dark side",
        "a classic art school cube study with core shadow and light falloff",
        "a sharp geometric box sitting on a flat tabletop",
        "a volumetric cube drawing with chiaroscuro value shading",
        "a 3D cube constructed with perspective guidelines and ground shadow",
    ],
    "chair": [
        "a wooden chair in three-quarter view with four legs and backrest",
        "a perspective drawing of a chair with distinct seat plane and legs",
        "a simple sketch of a dining chair with clean contours",
        "a chair drawn with structural proportion guides and line variation",
        "an artist study of a chair showing seat perspective and leg spacing",
        "a chair standing on a floor plane with directional ground shadow",
        "a line drawing of a chair showing back slats and seat cushion",
        "a four-legged chair rendered with light-direction line weight",
        "a sturdy wooden chair sketched in two-point perspective",
        "a classic chair study constructed from simple box volumes",
    ],
    "bicycle": [
        "a commuter bicycle with dual circular wheels and triangular frame",
        "a drawing of a bicycle showing handlebars, pedals, and spoke wheels",
        "a side-view sketch of a bicycle with accurate geometric proportions",
        "a bicycle study with circular wheels, chainring, and connecting frame",
        "a clean line drawing of a bicycle with light-directed line weights",
        "a two-wheeled bicycle rendered with ink contours and ground shadow",
        "a bicycle constructed from two tangent wheels and diamond frame",
        "a detailed drawing of a road bike with handlebars and saddle",
        "a classic bicycle sketch showing crossbar, forks, and wheel rims",
        "a mechanical study of a bicycle with concentric wheel hubs",
    ],
    "house": [
        "a rustic house with pitched triangular roof and front door",
        "an architectural sketch of a small house with chimney and windows",
        "a simple house drawing constructed from rectangular and triangular volumes",
        "a cozy cottage with pitched roof and directional hatching shadow",
        "a perspective drawing of a house with front facade and side depth",
        "a charming little house with paneled windows and chimney smoke",
        "a clean ink sketch of a residential house with cast roof shadow",
        "a house study showing foundation, wall planes, and sloping eaves",
        "a geometric drawing of a house with triangular gable and door",
        "a pencil and ink rendering of a house in perspective",
    ],
}

CAMERA_PHRASES = [
    "from a frontal camera angle",
    "viewed in three-quarter perspective",
    "captured from a dynamic side angle",
    "seen from a slightly elevated eye-level",
    "viewed at eye level with clean perspective",
    "from a dramatic low angle",
]

LIGHTING_PHRASES = [
    "lit by a key light from the upper-left",
    "with directional lighting casting shadows to the right",
    "shaded with a single light source from above",
    "with light-directed hatching and ground cast shadows",
    "illuminated with strong side lighting and deep shadows",
]


def generate_caption(
    category: str,  # e.g. "stand", "walk", "run", "box", "chair", "bicycle", "house"
    camera_yaw: Optional[float] = None,
    include_lighting: bool = True,
    seed: Optional[int] = None,
) -> str:
    """Generates a rich, varied caption from ground-truth parameters."""
    rng = random.Random(seed)

    if category in POSE_TEMPLATES:
        base = rng.choice(POSE_TEMPLATES[category])
    elif category in OBJECT_TEMPLATES:
        base = rng.choice(OBJECT_TEMPLATES[category])
    else:
        base = f"a detailed drawing of a {category}"

    # Add camera view descriptor if angle is provided
    cam_str = ""
    if camera_yaw is not None:
        abs_yaw = abs(camera_yaw)
        if abs_yaw < 15:
            cam_str = "viewed from the front"
        elif abs_yaw < 60:
            cam_str = "in three-quarter perspective"
        else:
            cam_str = "in side profile view"

    # Assemble natural sentence
    parts = [base]
    if cam_str and cam_str not in base:
        if rng.random() > 0.4:
            parts.append(cam_str)
    if include_lighting and rng.random() > 0.5:
        parts.append(rng.choice(LIGHTING_PHRASES))

    return ", ".join(parts)
