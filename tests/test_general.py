"""Unit tests for Konnon Draw General: tokenization, text encoder, decoder, KV-cache, critic, dataset, and sampling."""

import pytest
import torch

from konnon_draw.schema_v2 import DocumentV2, PointV2, StrokeActionV2, validate_document_v2
from konnon_draw.general.vocab import (
    BOS_TOKEN,
    EOS_TOKEN,
    VOCAB_SIZE,
    doc_to_token_sequence,
    token_sequence_to_doc,
)
from konnon_draw.general.text_encoder import FrozenTextEncoder
from konnon_draw.general.transformer_decoder import KonnonTransformerDecoder
from konnon_draw.general.critic import ImageTextCritic
from konnon_draw.general.synthetic_data import generate_synthetic_example
from konnon_draw.general.dataset import augment_drawing, create_curriculum_datasets
from konnon_draw.general.sampling_general import (
    generate_from_prompt,
    generate_with_prefix_prompt,
)


def test_general_token_roundtrip():
    """Verify that DocumentV2 -> Token IDs -> DocumentV2 preserves structure."""
    doc = DocumentV2()
    doc.commands.append(
        StrokeActionV2(
            points=[PointV2(0.2, 0.3), PointV2(0.5, 0.7), PointV2(0.8, 0.4)],
            brush_id="ink_liner",
            color="#ef4444",
            base_width=3.5,
            layer_id="line_art",
        )
    )

    tokens = doc_to_token_sequence(doc, max_tokens=128)
    assert tokens[0] == BOS_TOKEN
    assert tokens[-1] == EOS_TOKEN

    reconstructed = token_sequence_to_doc(tokens)
    assert validate_document_v2(reconstructed) is True
    assert len(reconstructed.commands) == 1

    st = reconstructed.commands[0]
    assert len(st.points) == 3
    # Check spatial accuracy within quantization tolerance (1/255 ~ 0.004)
    assert abs(st.points[0].x - 0.2) < 0.01
    assert abs(st.points[0].y - 0.3) < 0.01


def test_text_encoder_shapes():
    """Text encoder returns last_hidden_state and pooled representations."""
    enc = FrozenTextEncoder(model_type="builtin", d_model=128)
    out = enc.encode(["a red circle above a blue square", "a simple bicycle"])

    assert "last_hidden_state" in out
    assert "pooler_output" in out
    assert out["last_hidden_state"].shape == (2, 32, 128)
    assert out["pooler_output"].shape == (2, 128)


def test_transformer_decoder_forward_and_kv_cache():
    """Verify transformer forward pass and that KV-cache matches full-sequence forward."""
    torch.manual_seed(42)
    d_model = 64
    model = KonnonTransformerDecoder(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        max_seq_len=64,
    )
    model.eval()

    tokens = torch.tensor([[BOS_TOKEN, 100, 200, 15]], dtype=torch.long)
    context = torch.randn(1, 8, d_model)

    # 1. Full Sequence Forward Pass
    with torch.no_grad():
        full_logits, _ = model(tokens, context=context)
    assert full_logits.shape == (1, 4, VOCAB_SIZE)

    # 2. Step-by-Step with KV-Cache
    kv_caches = None
    step_logits = []
    with torch.no_grad():
        for t in range(4):
            cur_tok = tokens[:, t:t+1]
            l, kv_caches = model(
                cur_tok,
                context=context,
                kv_caches=kv_caches,
                use_cache=True,
                past_length=t,
            )
            step_logits.append(l)

    cached_logits = torch.cat(step_logits, dim=1)
    # Cached logits and full logits must match closely
    assert torch.allclose(full_logits, cached_logits, atol=1e-4)


def test_image_text_critic():
    """Verify critic scores prompt compliance for synthetic shapes."""
    critic = ImageTextCritic()

    ex = generate_synthetic_example(seed=42)
    doc = ex["doc"]
    prompt = ex["prompt"]

    score = critic.score(doc, prompt)
    assert 0.0 <= score <= 1.0
    assert score > 0.4  # Matched prompt should score reasonably high

    # Mismatched prompt should score lower
    bad_score = critic.score(doc, "a black square below a yellow star and flying dragon")
    assert bad_score <= score


def test_dataset_and_augmentation():
    """Verify curriculum dataset creation and augmentations."""
    train_ds, val_ds = create_curriculum_datasets(stage=1, num_synthetic=50, val_split=0.2)
    assert len(train_ds) > 0
    assert len(val_ds) > 0

    item = train_ds[0]
    assert "text_input_ids" in item
    assert "drawing_input_ids" in item
    assert "drawing_target_ids" in item
    assert item["drawing_input_ids"].shape[0] == item["drawing_target_ids"].shape[0]


def test_sampling_and_prefix_completion():
    """Verify prompt-based generation and prefix completion."""
    torch.manual_seed(42)
    text_enc = FrozenTextEncoder(model_type="builtin", d_model=64)
    model = KonnonTransformerDecoder(vocab_size=VOCAB_SIZE, d_model=64, n_layers=2, n_heads=2, d_ff=128)

    # From scratch
    res = generate_from_prompt(model, text_enc, "a red circle", max_tokens=16)
    assert "doc" in res
    assert validate_document_v2(res["doc"]) is True

    # With prefix
    prefix_doc = DocumentV2()
    prefix_doc.commands.append(
        StrokeActionV2(points=[PointV2(0.2, 0.2), PointV2(0.3, 0.3)])
    )
    pref_res = generate_with_prefix_prompt(model, text_enc, "a red circle", prefix_doc, max_tokens=24)
    assert "doc" in pref_res
    assert validate_document_v2(pref_res["doc"]) is True
