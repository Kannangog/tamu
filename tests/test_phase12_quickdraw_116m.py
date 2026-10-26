"""Tests for Phase 12: QuickDraw Staged Ingestion and 116.4M Scaled Transformer."""

import sys
from pathlib import Path
import pytest
import torch

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.v2.tokens import VOCAB_SIZE
from scripts.ingest_quickdraw_staged import normalize_strokes


def test_quickdraw_normalize_strokes():
    """Verify QuickDraw coordinates are cleanly centered and scaled to [0.02, 0.98]."""
    raw_strokes = [
        [[10, 20, 30], [10, 20, 30]],
        [[50, 60, 70], [80, 90, 100]],
    ]
    norm = normalize_strokes(raw_strokes)
    assert norm is not None
    assert len(norm) == 2
    for stroke in norm:
        for x, y in stroke:
            assert 0.02 <= x <= 0.98
            assert 0.02 <= y <= 0.98


def test_instantiate_116m_transformer():
    """Verify 116.4M parameter architecture instantiates with expected layer shapes."""
    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=512,
        canvas_res=128,
    )
    total_params = sum(p.numel() for p in model.parameters())
    # Should be approximately 116.4M parameters
    assert 110_000_000 <= total_params <= 125_000_000
    assert model.d_model == 768
    assert model.max_seq_len == 512


def test_transformer_forward_unpacking():
    """Verify model forward pass returns (logits, self_kv_caches, cross_kv_caches)."""
    model = KonnonStagedTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=32,
        canvas_res=32,
    )
    model.eval()
    tokens = torch.randint(0, VOCAB_SIZE, (2, 8))
    context = torch.randn(2, 4, 64)
    res = model(tokens, context=context)
    assert isinstance(res, tuple)
    assert len(res) == 3
    logits, s_cache, c_cache = res
    assert logits.shape == (2, 8, VOCAB_SIZE)
    assert len(s_cache) == 2
    assert len(c_cache) == 2

