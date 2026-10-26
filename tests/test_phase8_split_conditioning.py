"""Unit tests for Step 3 / Gate 8: Drawing-level data split integrity and causal conditioning KL checks."""

import pytest
import torch
import torch.nn.functional as F

from konnon_draw.gym.dataset import (
    generate_staged_gym_data,
    split_staged_dataset_by_drawing,
)
from konnon_draw.model.stroke_transformer import KonnonStagedTransformer
from konnon_draw.model.text_encoder import FrozenTextEncoder
from konnon_draw.v2.tokens import VOCAB_SIZE, BOS_TOKEN


def test_split_by_drawing_id_no_leakage():
    """Verify that splitting by drawing ID strictly prevents drawing leakage across splits."""
    items = generate_staged_gym_data(num_samples=20, canvas_res=64, seed=42)
    train_items, val_items, test_items = split_staged_dataset_by_drawing(
        items,
        train_frac=0.80,
        val_frac=0.10,
        test_frac=0.10,
        seed=42,
    )

    def get_drawing_ids(sample_list):
        return set(s.get("drawing_id", s["sample_id"].split("_")[0]) for s in sample_list)

    train_ids = get_drawing_ids(train_items)
    val_ids = get_drawing_ids(val_items)
    test_ids = get_drawing_ids(test_items)

    assert len(train_ids & val_ids) == 0, f"Leakage between train and val: {train_ids & val_ids}"
    assert len(train_ids & test_ids) == 0, f"Leakage between train and test: {train_ids & test_ids}"
    assert len(val_ids & test_ids) == 0, f"Leakage between val and test: {val_ids & test_ids}"
    assert len(train_items) + len(val_items) + len(test_items) == len(items)


def test_conditioning_kl_divergence_above_threshold():
    """Verify that changing text prompt, canvas input, or stage token measurably changes output distribution."""
    from pathlib import Path

    ckpt_path = Path("outputs_v2/overfit/model_overfit.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = KonnonStagedTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=128,
            n_layers=3,
            n_heads=4,
            d_ff=384,
            max_seq_len=192,
            canvas_res=128,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        d_model = 128
        canvas_res = 128
        min_kl_text = 0.05
        min_kl_canvas = 0.05
        min_kl_stage = 0.05
    else:
        d_model = 64
        canvas_res = 64
        model = KonnonStagedTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            max_seq_len=64,
            canvas_res=64,
        )
        min_kl_text = 0.005
        min_kl_canvas = 0.005
        min_kl_stage = 0.0005

    model.eval()
    text_enc = FrozenTextEncoder(model_type="builtin", d_model=d_model)

    # Stroke prefix at position 10 where active drawing decisions occur:
    prefix = torch.tensor([[BOS_TOKEN, 11, 593, 12, 599, 4, 140, 380, 5, 145]])

    def get_distribution(prompt: str, canvas_tensor: torch.Tensor, stage_idx: int) -> torch.Tensor:
        with torch.no_grad():
            ctx = model.build_multimodal_context(
                text_embeddings=text_enc.encode_text([prompt]),
                canvas=canvas_tensor,
                stage_idx=torch.tensor([stage_idx], dtype=torch.long),
            )
            logits, _, _ = model(prefix, context=ctx)
            return F.softmax(logits[0, -1, :], dim=-1)

    c_blank = torch.ones(1, 3, canvas_res, canvas_res)
    c_marked = torch.zeros(1, 3, canvas_res, canvas_res)
    c_marked[:, :, 16:48, 16:48] = 1.0

    prompt_a = "a pencil and ink drawing of someone standing, in three-quarter perspective"
    prompt_b = "a drawing of a bicycle showing handlebars, pedals, and spoke wheels"

    p_base = get_distribution(prompt_a, c_blank, 0)
    p_text_diff = get_distribution(prompt_b, c_blank, 0)
    p_canvas_diff = get_distribution(prompt_a, c_marked, 0)
    p_stage_diff = get_distribution(prompt_a, c_blank, 1)

    def kl_div(p: torch.Tensor, q: torch.Tensor) -> float:
        return float(torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))).item())

    kl_text = kl_div(p_base, p_text_diff)
    kl_canvas = kl_div(p_base, p_canvas_diff)
    kl_stage = kl_div(p_base, p_stage_diff)

    print(f"Measured KL Divergences: Text={kl_text:.4f}, Canvas={kl_canvas:.4f}, Stage={kl_stage:.4f}")

    # Threshold checks: all conditioning dimensions must measurably shift next-token distribution
    assert kl_text > min_kl_text, f"Text prompt conditioning too weak: KL = {kl_text}"
    assert kl_canvas > min_kl_canvas, f"Canvas conditioning too weak: KL = {kl_canvas}"
    assert kl_stage > min_kl_stage, f"Stage token conditioning too weak: KL = {kl_stage}"
