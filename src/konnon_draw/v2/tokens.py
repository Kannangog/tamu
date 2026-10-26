"""Tokenizer for Drawing Language v2: bidirectional integer and structured tokens."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .schema import (
    CanvasConfigV2,
    DocumentV2,
    FillActionV2,
    LayerV2,
    PointV2,
    StrokeActionV2,
    SUPPORTED_BRUSHES,
    parse_color,
    rgb_to_hex,
)
from ..general.vocab import (
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    SEP_TOKEN,
    TOKEN_MOVE,
    TOKEN_LINE,
    TOKEN_CURVE,
    TOKEN_PEN_UP,
    TOKEN_FILL_FLAT,
    TOKEN_FILL_HATCH,
    TOKEN_FILL_END,
    TOKEN_SET_LAYER,
    TOKEN_SET_BRUSH,
    TOKEN_SET_COLOR,
    TOKEN_SET_WIDTH,
    NUM_COORD_BINS,
    X_OFFSET,
    Y_OFFSET,
    NUM_PRESSURE_BINS,
    PRESSURE_OFFSET,
    COLOR_OFFSET,
    WIDTH_OFFSET,
    LAYER_OFFSET,
    BRUSH_OFFSET,
    VOCAB_SIZE,
    doc_to_token_sequence,
    token_sequence_to_doc,
)
from ..encoding_v2 import (
    encode_v2_to_tokens,
    decode_tokens_to_v2,
    BRUSH_TO_ID,
    ID_TO_BRUSH,
    BLEND_TO_ID,
    ID_TO_BLEND,
    FILL_TYPE_TO_ID,
    ID_TO_FILL_TYPE,
)
