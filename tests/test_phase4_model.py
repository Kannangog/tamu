"""Unit tests for Phase 4: Staged text- and canvas-conditioned transformer model."""

import pytest
import torch

from konnon_draw.model.canvas_encoder import CanvasEncoder
from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.model.text_encoder import FrozenTextEncoder
from konnon_draw.model.sampling import sample_stage_tokens, generate_staged_drawing
from konnon_draw.v2.schema import validate_document_v2
from konnon_draw.v2.tokens import VOCAB_SIZE, BOS_TOKEN


def test_canvas_encoder_shape():
    """Verify CanvasEncoder maps (B, 3, 128, 128) to (B, 65, d_model)."""
    enc = CanvasEncoder(in_channels=3, d_model=128, canvas_res=128, patch_grid_res=8)
    dummy_canvas = torch.randn(2, 3, 128, 128)
    tokens = enc(dummy_canvas)
    assert tokens.shape == (2, 65, 128), f"Expected (2, 65, 128), got {tokens.shape}"


def test_transformer_multimodal_context():
    """Verify multimodal context builder combines stage, text, and canvas tokens."""
    model = KonnonStagedTransformer(vocab_size=VOCAB_SIZE, d_model=128, n_layers=2, n_heads=4, d_ff=256, max_seq_len=64, canvas_res=128)
    text_emb = torch.randn(2, 10, 128)
    canvas = torch.randn(2, 3, 128, 128)
    stage_idx = torch.tensor([1, 3], dtype=torch.long)

    ctx = model.build_multimodal_context(text_embeddings=text_emb, canvas=canvas, stage_idx=stage_idx)
    # Total context length = 1 (stage) + 10 (text) + 65 (canvas: 8x8 + 1 CLS) = 76 tokens
    assert ctx.shape == (2, 76, 128)


def test_transformer_kv_cache_equivalence():
    """Verify that cached step-by-step forward matches full sequence forward within atol=1e-4."""
    model = KonnonStagedTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=2, d_ff=128, max_seq_len=32, canvas_res=64)
    model.eval()

    context = torch.randn(1, 15, 64)
    seq = torch.randint(1, VOCAB_SIZE, (1, 8))

    # 1. Full sequence forward
    with torch.no_grad():
        full_logits, _, _ = model(seq, context=context)

    # 2. Step-by-step cached forward
    cached_logits = []
    self_caches = None
    cross_caches = None

    with torch.no_grad():
        for t in range(seq.shape[1]):
            token_in = seq[:, t : t + 1]
            logit_step, self_caches, cross_caches = model(
                token_in,
                context=context,
                start_pos=t,
                self_kv_caches=self_caches,
                cross_kv_caches=cross_caches,
            )
            cached_logits.append(logit_step)

    cached_all = torch.cat(cached_logits, dim=1)
    diff = torch.max(torch.abs(full_logits - cached_all)).item()
    assert diff < 1e-4, f"KV cache output diverged from full forward: max diff = {diff}"


def test_staged_sampling_generation():
    """Verify that staged sampling generates valid DocumentV2 across multiple stages."""
    text_encoder = FrozenTextEncoder(model_type="builtin", d_model=64)
    model = KonnonStagedTransformer(vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=2, d_ff=128, max_seq_len=64, canvas_res=64)
    model.eval()

    res = generate_staged_drawing(
        model=model,
        text_encoder=text_encoder,
        prompt="a standing figure",
        stages=["layout", "ink"],
        canvas_res=64,
        out_res=128,
        seed=42,
    )

    assert "doc" in res
    assert "final_image" in res
    assert len(res["stage_history"]) == 2
    assert validate_document_v2(res["doc"]) is True
