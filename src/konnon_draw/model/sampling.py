"""Autoregressive staged sampling with text- and canvas-conditioning and KV-cache."""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from ..v2.schema import DocumentV2
from ..v2.renderer import render_document_v2
from ..v2.tokens import (
    VOCAB_SIZE,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
    TOKEN_MOVE,
    TOKEN_LINE,
    TOKEN_PEN_UP,
    token_sequence_to_doc,
)
from .stroke_transformer import KonnonStagedTransformer
from .text_encoder import FrozenTextEncoder
from ..gym.stage_generator import STAGE_ORDER


def sample_stage_tokens(
    model: KonnonStagedTransformer,
    context: torch.Tensor,
    max_tokens: int = 160,
    temperature: float = 0.5,
    top_k: int = 25,
    device: str = "cpu",
    seed: int = 42,
    length_prior_alpha: float = 0.04,
    min_stage_tokens: int = 16,
) -> List[int]:
    """Autoregressively samples a sequence of drawing tokens for one stage using KV-cache."""
    torch.manual_seed(seed)
    model.eval()

    limit = min(max_tokens, getattr(model, "max_seq_len", max_tokens))

    tokens = [BOS_TOKEN]
    input_t = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)
    self_caches = None
    cross_caches = None

    for step in range(0, limit):
        # Forward single token with cached keys/values
        with torch.no_grad():
            logits, self_caches, cross_caches = model(
                input_t,
                context=context,
                start_pos=step,
                self_kv_caches=self_caches,
                cross_kv_caches=cross_caches,
            )

        next_logits = logits[0, -1, :].clone()

        # Enforce syntactic grammar
        next_logits[PAD_TOKEN] = -1e9
        next_logits[BOS_TOKEN] = -1e9
        if step < min_stage_tokens:
            next_logits[EOS_TOKEN] = -1e9
        elif length_prior_alpha > 0:
            # Calibrated length prior encouraging natural stage termination
            next_logits[EOS_TOKEN] += length_prior_alpha * (step - min_stage_tokens)

        # Temperature scaling & Top-K filtering
        if temperature > 0:
            scaled_logits = next_logits / max(1e-4, temperature)
            if top_k > 0:
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[-1]] = -float("inf")
            probs = F.softmax(scaled_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_token = int(torch.argmax(next_logits).item())

        tokens.append(next_token)
        if next_token == EOS_TOKEN:
            break

        input_t = torch.tensor([[next_token]], dtype=torch.long, device=device)

    return tokens


def generate_staged_drawing(
    model: KonnonStagedTransformer,
    text_encoder: FrozenTextEncoder,
    prompt: str,
    stages: Optional[List[str]] = None,
    canvas_res: int = 128,
    out_res: int = 512,
    temperature: float = 0.5,
    top_k: int = 25,
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Full staged inference loop: layout -> guide -> sketch -> ink -> shade."""
    if stages is None:
        stages = ["layout", "guide", "ink", "shade"]

    model.eval()
    rng = random_state = np.random.RandomState(seed)

    # Encode prompt once
    text_emb = text_encoder.encode_text([prompt], device=device)

    # Initial blank white canvas
    canvas_arr = np.ones((canvas_res, canvas_res, 3), dtype=np.uint8) * 255
    running_doc = DocumentV2()

    stage_history = []

    for stg_idx, stg_name in enumerate(stages):
        stg_id_val = STAGE_ORDER.index(stg_name) if stg_name in STAGE_ORDER else stg_idx
        stage_t = torch.tensor([stg_id_val], dtype=torch.long, device=device)

        # Canvas tensor normalized [0, 1]
        canvas_t = torch.from_numpy(canvas_arr).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

        # Build multimodal memory context
        with torch.no_grad():
            context = model.build_multimodal_context(
                text_embeddings=text_emb,
                canvas=canvas_t,
                stage_idx=stage_t,
            )

        # Sample tokens for this stage
        stg_seed = seed + stg_idx * 100
        stage_tokens = sample_stage_tokens(
            model=model,
            context=context,
            max_tokens=160,
            temperature=temperature,
            top_k=top_k,
            device=device,
            seed=stg_seed,
        )

        # Decode tokens to DocumentV2 actions
        stage_doc = token_sequence_to_doc(stage_tokens)

        # Tag stage
        for cmd in stage_doc.commands:
            setattr(cmd, "stage", stg_name)

        # Accumulate
        running_doc.commands.extend(stage_doc.commands)

        # Render new canvas for conditioning next stage
        rendered_canvas = render_document_v2(
            running_doc,
            width=canvas_res,
            height=canvas_res,
            supersample=1,
            seed=stg_seed,
        )
        canvas_arr = np.array(rendered_canvas, dtype=np.uint8)

        stage_history.append({
            "stage_name": stg_name,
            "tokens": stage_tokens,
            "canvas_image": rendered_canvas,
        })

    # High-resolution final render
    final_img = render_document_v2(running_doc, width=out_res, height=out_res, supersample=2, seed=seed)

    return {
        "prompt": prompt,
        "doc": running_doc,
        "final_image": final_img,
        "stage_history": stage_history,
    }
