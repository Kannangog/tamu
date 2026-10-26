"""Staged, Text- and Canvas-Conditioned Stroke Transformer Model (v0).

Features:
- Dual conditioning: Frozen text embeddings + convolutional canvas patch tokens + stage embedding.
- Pre-norm causal Transformer decoder with cross-attention to unified multimodal context.
- Autoregressive KV-cache for O(1) step generation on CPU.
- Discretized v2 drawing token space (V = 604).
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .canvas_encoder import CanvasEncoder
from ..v2.tokens import VOCAB_SIZE, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN


class StagedTransformerDecoderLayer(nn.Module):
    """Transformer decoder layer with causal self-attention and multimodal cross-attention."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 768,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-Norms
        self.ln_self = nn.LayerNorm(d_model)
        self.ln_cross = nn.LayerNorm(d_model)
        self.ln_ff = nn.LayerNorm(d_model)

        # Multi-Head Causal Self-Attention
        self.q_self = nn.Linear(d_model, d_model)
        self.k_self = nn.Linear(d_model, d_model)
        self.v_self = nn.Linear(d_model, d_model)
        self.out_self = nn.Linear(d_model, d_model)

        # Multi-Head Cross-Attention (to unified text + canvas + stage context)
        self.q_cross = nn.Linear(d_model, d_model)
        self.k_cross = nn.Linear(d_model, d_model)
        self.v_cross = nn.Linear(d_model, d_model)
        self.out_cross = nn.Linear(d_model, d_model)

        # Feed-Forward Network (GELU)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                                      # (B, L_dec, D)
        context: torch.Tensor,                                # (B, L_ctx, D)
        causal_mask: Optional[torch.Tensor] = None,          # (L_dec, L_dec)
        self_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cross_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        B, L_dec, D = x.shape

        # ---------------------------------------------------------
        # 1. Causal Self-Attention (with KV cache)
        # ---------------------------------------------------------
        norm_x = self.ln_self(x)
        q = self.q_self(norm_x).view(B, L_dec, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_self(norm_x).view(B, L_dec, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_self(norm_x).view(B, L_dec, self.n_heads, self.head_dim).transpose(1, 2)

        if self_kv_cache is not None:
            past_k, past_v = self_kv_cache
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_self_kv = (k, v)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None and self_kv_cache is None:
            scores = scores + causal_mask[:L_dec, :L_dec]
        attn = F.softmax(scores, dim=-1)
        self_out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L_dec, D)
        x = x + self.drop(self.out_self(self_out))

        # ---------------------------------------------------------
        # 2. Multimodal Cross-Attention (with precomputed KV cache)
        # ---------------------------------------------------------
        norm_x = self.ln_cross(x)
        q_c = self.q_cross(norm_x).view(B, L_dec, self.n_heads, self.head_dim).transpose(1, 2)

        if cross_kv_cache is not None:
            k_c, v_c = cross_kv_cache
        else:
            L_ctx = context.shape[1]
            k_c = self.k_cross(context).view(B, L_ctx, self.n_heads, self.head_dim).transpose(1, 2)
            v_c = self.v_cross(context).view(B, L_ctx, self.n_heads, self.head_dim).transpose(1, 2)
        new_cross_kv = (k_c, v_c)

        cross_scores = torch.matmul(q_c, k_c.transpose(-2, -1)) / math.sqrt(self.head_dim)
        cross_attn = F.softmax(cross_scores, dim=-1)
        cross_out = torch.matmul(cross_attn, v_c).transpose(1, 2).contiguous().view(B, L_dec, D)
        x = x + self.drop(self.out_cross(cross_out))

        # ---------------------------------------------------------
        # 3. Feed-Forward
        # ---------------------------------------------------------
        x = x + self.ffn(self.ln_ff(x))
        return x, new_self_kv, new_cross_kv


class KonnonStagedTransformer(nn.Module):
    """Full text- and canvas-conditioned staged drawing model."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 768,
        max_seq_len: int = 256,
        canvas_res: int = 128,
        num_stages: int = 6,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token & Position Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.stage_emb = nn.Embedding(num_stages, d_model)

        # Canvas Visual Encoder
        self.canvas_encoder = CanvasEncoder(in_channels=3, d_model=d_model, canvas_res=canvas_res)

        # Context adapter projection (ensures text + canvas match d_model)
        self.context_proj = nn.Identity()

        # Decoder Layers
        self.layers = nn.ModuleList([
            StagedTransformerDecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)

        # Tie weights
        self.head.weight = self.token_emb.weight

    def build_multimodal_context(
        self,
        text_embeddings: torch.Tensor,                  # (B, L_text, D)
        canvas: Optional[torch.Tensor] = None,          # (B, 3, H, W)
        stage_idx: Optional[torch.Tensor] = None,       # (B,)
    ) -> torch.Tensor:
        """Combines stage indicator, text context, and canvas visual patches into unified memory."""
        B = text_embeddings.shape[0]
        device = text_embeddings.device

        ctx_parts = []

        # 1. Stage indicator token
        if stage_idx is not None:
            s_tok = self.stage_emb(stage_idx).unsqueeze(1)  # (B, 1, D)
            ctx_parts.append(s_tok)

        # 2. Text tokens
        ctx_parts.append(text_embeddings)

        # 3. Canvas patch tokens
        if canvas is not None:
            canvas_tokens = self.canvas_encoder(canvas)  # (B, 65, D)
            ctx_parts.append(canvas_tokens)

        context = torch.cat(ctx_parts, dim=1)  # (B, L_total, D)
        return context

    def forward(
        self,
        drawing_tokens: torch.Tensor,                   # (B, L_dec)
        context: torch.Tensor,                          # (B, L_ctx, D)
        start_pos: int = 0,
        self_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cross_kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, L_dec = drawing_tokens.shape
        device = drawing_tokens.device

        # Positional indices
        pos = torch.arange(start_pos, start_pos + L_dec, device=device).unsqueeze(0)
        h = self.drop(self.token_emb(drawing_tokens) + self.pos_emb(pos))

        # Causal mask for self-attention
        causal_mask = None
        if L_dec > 1:
            causal_mask = torch.triu(torch.full((L_dec, L_dec), float("-inf"), device=device), diagonal=1)

        new_self_caches = []
        new_cross_caches = []

        for i, layer in enumerate(self.layers):
            s_cache = self_kv_caches[i] if self_kv_caches is not None else None
            c_cache = cross_kv_caches[i] if cross_kv_caches is not None else None

            h, s_new, c_new = layer(
                h,
                context=context,
                causal_mask=causal_mask,
                self_kv_cache=s_cache,
                cross_kv_cache=c_cache,
            )
            new_self_caches.append(s_new)
            new_cross_caches.append(c_new)

        logits = self.head(self.ln_final(h))
        return logits, new_self_caches, new_cross_caches
