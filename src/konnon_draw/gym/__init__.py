"""Synthetic Gym package: Staged drawing generator, pose sampler, captions, and dataset."""

from .captions import (
    generate_caption,
    POSE_TEMPLATES,
    OBJECT_TEMPLATES,
)
from .stage_generator import (
    STAGE_ORDER,
    permute_strokes_within_stages,
    generate_staged_human_drawing,
    generate_staged_object_drawing,
)
from .dataset import (
    StagedDrawingDataset,
    generate_staged_gym_data,
)
