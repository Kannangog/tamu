"""Text-conditioned autoregressive sampling with KV-cache and collaborative prefix completion.

Features:
- O(1) step autoregressive generation using self-attention and cross-attention KV-caches.
- Temperature scaling, top-k filtering, and syntactic grammar enforcement.
- Collaborative drawing: user-drawn stroke prefix + text prompt -> seamless completion.
- Best-of-N candidate sampling with image-text critic reranking.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from ..renderer_v2 import render_document_v2
from ..schema_v2 import DocumentV2
from .critic import ImageTextCritic
from .text_encoder import FrozenTextEncoder
from .transformer_decoder import KonnonTransformerDecoder
from .vocab import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    TOKEN_LINE,
    TOKEN_MOVE,
    TOKEN_PEN_UP,
    VOCAB_SIZE,
    X_OFFSET,
    Y_OFFSET,
    doc_to_token_sequence,
    token_sequence_to_doc,
)


def apply_grammar_constraint(
    logits: torch.Tensor,
    prev_tokens: List[int],
) -> torch.Tensor:
    """Enforces basic vector grammar to prevent illegal token combinations."""
    if not prev_tokens:
        return logits

    last = prev_tokens[-1]
    # If last was MOVE or LINE, next must be an X coordinate (X_OFFSET..X_OFFSET+255)
    if last in (TOKEN_MOVE, TOKEN_LINE):
        mask = torch.full_like(logits, float("-inf"))
        mask[..., X_OFFSET:X_OFFSET + 256] = logits[..., X_OFFSET:X_OFFSET + 256]
        return mask

    # If last was an X coordinate, next must be a Y coordinate
    if X_OFFSET <= last < X_OFFSET + 256:
        mask = torch.full_like(logits, float("-inf"))
        mask[..., Y_OFFSET:Y_OFFSET + 256] = logits[..., Y_OFFSET:Y_OFFSET + 256]
        return mask

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.5,
    top_k: Optional[int] = 20,
    prev_tokens: Optional[List[int]] = None,
    enforce_grammar: bool = True,
) -> int:
    """Samples next token using temperature, top-k, and grammar constraints."""
    logits = logits.squeeze(0).squeeze(0)  # (V,)

    if enforce_grammar and prev_tokens:
        logits = apply_grammar_constraint(logits, prev_tokens)

    # Disallow PAD and BOS from being generated mid-stream
    logits[PAD_TOKEN] = float("-inf")
    logits[BOS_TOKEN] = float("-inf")

    if temperature <= 1e-4:
        return int(torch.argmax(logits).item())

    logits = logits / max(0.05, float(temperature))

    if top_k is not None and top_k > 0:
        val, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        min_val = val[-1]
        logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

    probs = F.softmax(logits, dim=-1)
    # Fallback to argmax if probabilities contain NaNs
    if torch.isnan(probs).any():
        return int(torch.argmax(logits).item())

    next_tok = int(torch.multinomial(probs, num_samples=1).item())
    return next_tok


@torch.no_grad()
def generate_from_prompt(
    model: KonnonTransformerDecoder,
    text_encoder: FrozenTextEncoder,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.5,
    top_k: int = 25,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generates a complete vector DocumentV2 from a text prompt with KV-cache."""
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    model.to(device)

    # 1. Encode text prompt
    text_out = text_encoder.encode(prompt)
    context = text_out["last_hidden_state"].to(device)
    context_mask = text_out["attention_mask"].to(device)

    # 2. Initialize generation with BOS token
    generated_tokens = [BOS_TOKEN]
    kv_caches = None

    cur_token = torch.tensor([[BOS_TOKEN]], dtype=torch.long, device=device)

    for step in range(max_tokens - 1):
        logits, kv_caches = model(
            input_ids=cur_token,
            context=context,
            context_mask=context_mask,
            kv_caches=kv_caches,
            use_cache=True,
            past_length=step,
        )

        next_tok = sample_next_token(
            logits[:, -1:, :],
            temperature=temperature,
            top_k=top_k,
            prev_tokens=generated_tokens,
        )

        generated_tokens.append(next_tok)
        if next_tok == EOS_TOKEN:
            break

        cur_token = torch.tensor([[next_tok]], dtype=torch.long, device=device)

    # 3. Decode into DocumentV2
    doc = token_sequence_to_doc(generated_tokens)

    return {
        "prompt": prompt,
        "tokens": generated_tokens,
        "n_tokens": len(generated_tokens),
        "doc": doc,
    }


@torch.no_grad()
def generate_with_prefix_prompt(
    model: KonnonTransformerDecoder,
    text_encoder: FrozenTextEncoder,
    prompt: str,
    prefix_doc: DocumentV2,
    max_tokens: int = 256,
    temperature: float = 0.5,
    top_k: int = 25,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Collaborative drawing: completes a drawing starting from user prefix strokes."""
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    model.to(device)

    text_out = text_encoder.encode(prompt)
    context = text_out["last_hidden_state"].to(device)
    context_mask = text_out["attention_mask"].to(device)

    # Convert user prefix to token sequence
    prefix_tokens = doc_to_token_sequence(prefix_doc, max_tokens=max_tokens // 2)
    # Strip terminal EOS
    if prefix_tokens and prefix_tokens[-1] == EOS_TOKEN:
        prefix_tokens.pop()

    if not prefix_tokens or prefix_tokens[0] != BOS_TOKEN:
        prefix_tokens = [BOS_TOKEN] + prefix_tokens

    generated_tokens = list(prefix_tokens)
    prefix_len = len(prefix_tokens)

    # Warm up KV-cache on the entire prefix at once
    prefix_tensor = torch.tensor([prefix_tokens], dtype=torch.long, device=device)
    logits, kv_caches = model(
        input_ids=prefix_tensor,
        context=context,
        context_mask=context_mask,
        use_cache=True,
        past_length=0,
    )

    next_tok = sample_next_token(
        logits[:, -1:, :],
        temperature=temperature,
        top_k=top_k,
        prev_tokens=generated_tokens,
    )
    generated_tokens.append(next_tok)

    cur_token = torch.tensor([[next_tok]], dtype=torch.long, device=device)

    # Continue generation step-by-step
    for step in range(prefix_len, max_tokens - 1):
        if next_tok == EOS_TOKEN:
            break

        logits, kv_caches = model(
            input_ids=cur_token,
            context=context,
            context_mask=context_mask,
            kv_caches=kv_caches,
            use_cache=True,
            past_length=step,
        )

        next_tok = sample_next_token(
            logits[:, -1:, :],
            temperature=temperature,
            top_k=top_k,
            prev_tokens=generated_tokens,
        )
        generated_tokens.append(next_tok)
        cur_token = torch.tensor([[next_tok]], dtype=torch.long, device=device)

    doc = token_sequence_to_doc(generated_tokens)
    return {
        "prompt": prompt,
        "prefix_len": prefix_len,
        "n_tokens": len(generated_tokens),
        "tokens": generated_tokens,
        "doc": doc,
    }


def generate_best_of_n(
    model: KonnonTransformerDecoder,
    text_encoder: FrozenTextEncoder,
    prompt: str,
    n_candidates: int = 4,
    temperature: float = 0.6,
    top_k: int = 30,
    device: str = "cpu",
    critic: Optional[ImageTextCritic] = None,
) -> Dict[str, Any]:
    """Generates N candidates, evaluates with critic, and returns the top candidate."""
    if critic is None:
        critic = ImageTextCritic(text_encoder=text_encoder)

    candidates = []
    docs = []

    for k in range(n_candidates):
        res = generate_from_prompt(
            model=model,
            text_encoder=text_encoder,
            prompt=prompt,
            temperature=temperature,
            top_k=top_k,
            device=device,
            seed=42 + k * 100,
        )
        candidates.append(res)
        docs.append(res["doc"])

    best_doc, scores = critic.rerank_candidates(docs, prompt)
    best_idx = int(np.argmax(scores))

    best_cand = candidates[best_idx]
    best_cand["best_critic_score"] = float(scores[best_idx])
    best_cand["all_critic_scores"] = [float(s) for s in scores]
    return best_cand
