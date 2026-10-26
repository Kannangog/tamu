"""Konnon Draw Artist Skills: Programmatic drawing tools without ML."""

from .composition import (
    create_standard_layers,
    rule_of_thirds_points,
    fit_into_bounding_box,
)
from .linecraft import (
    make_ellipse,
    make_arc,
    make_wobbly_contour,
    make_sketchy_lines,
    make_confident_stroke,
    apply_line_weight_by_light,
)
from .shading import (
    generate_directional_hatching,
    generate_cast_shadow_polygon,
    generate_stipple_shading,
    generate_rim_light_strokes,
    make_gradient_sky,
)
from .effects import (
    make_speed_lines,
    make_motion_arcs,
    make_impact_burst,
)
from .head_guide import (
    build_head_construction,
)
from .mannequin3d import (
    PinholeCamera,
    PROPORTIONS,
    sample_pose,
    generate_mannequin_drawing,
)
