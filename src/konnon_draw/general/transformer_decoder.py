"""Autoregressive Transformer Decoder with Cross-Attention and KV-Cache for Konnon Draw General.

Features:
- Causal masked self-attention with per-step KV-cache.
- Cross-attention to text encoder embeddings with pre-computed text KV-cache.
- Pre-norm architecture with LayerNorm / GELU FFN and residual highways.
- Supports tiny (1.5M), small (8M), and full (125M+) parameter scale configs.
- Fast O(1) step inference during autoregressive sampling.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import VOCAB_SIZE


class MultiHeadAttentionWithCache(nn.Module):
    """Multi-Head Attention supporting both causal self-attention and cross-attention with KV-cache."""

    def __init__(self, d_model: int, n_heads: int, is_cross_attention: bool = False, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.is_cross_attention = is_cross_attention

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)

        if self.is_cross_attention:
            # Cross Attention: Q from x, K & V from context (text embeddings)
            if kv_cache is not None and "k_cross" in kv_cache:
                k = kv_cache["k_cross"]
                v = kv_cache["v_cross"]
            else:
                assert context is not None, "context is required for cross-attention when cache is empty"
                B_ctx, T_ctx, _ = context.shape
                k = self.k_proj(context).view(B_ctx, T_ctx, self.n_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(context).view(B_ctx, T_ctx, self.n_heads, self.head_dim).transpose(1, 2)
                if use_cache:
                    if kv_cache is None:
                        kv_cache = {}
                    kv_cache["k_cross"] = k
                    kv_cache["v_cross"] = v
        else:
            # Self Attention: Q, K, V from x
            k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            if kv_cache is not None and "k_self" in kv_cache:
                k = torch.cat([kv_cache["k_self"], k], dim=2)
                v = torch.cat([kv_cache["v_self"], v], dim=2)

            if use_cache:
                if kv_cache is None:
                    kv_cache = {}
                kv_cache["k_self"] = k
                kv_cache["v_self"] = v

        # Compute Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T_q, T_k)

        if mask is not None:
            scores = scores + mask

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, T_q, D)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)

        return out, kv_cache


class DecoderBlock(nn.Module):
    """Transformer decoder block with causal self-attention, cross-attention, and FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttentionWithCache(d_model, n_heads, is_cross_attention=False, dropout=dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttentionWithCache(d_model, n_heads, is_cross_attention=True, dropout=dropout)

        self.ln3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        layer_cache: Optional[Dict[str, torch.Tensor]] = None,
        causal_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        # 1. Causal Self-Attention
        norm_x = self.ln1(x)
        sa_out, layer_cache = self.self_attn(
            norm_x,
            kv_cache=layer_cache,
            mask=causal_mask,
            use_cache=use_cache,
        )
        x = x + sa_out

        # 2. Cross-Attention to text context
        if context is not None or (layer_cache is not None and "k_cross" in layer_cache):
            norm_x = self.ln2(x)
            ca_out, layer_cache = self.cross_attn(
                norm_x,
                context=context,
                kv_cache=layer_cache,
                mask=context_mask,
                use_cache=use_cache,
            )
            x = x + ca_out

        # 3. Feedforward Network
        x = x + self.ffn(self.ln3(x))
        return x, layer_cache


class KonnonTransformerDecoder(nn.Module):
    """Full autoregressive transformer decoder for text-to-vector drawing."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 1024,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying between token embeddings and LM head
        self.lm_head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Dict[str, torch.Tensor]]] = None,
        use_cache: bool = False,
        past_length: int = 0,
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        B, T = input_ids.shape
        device = input_ids.device

        # Positional positions
        positions = torch.arange(past_length, past_length + T, device=device).unsqueeze(0).expand(B, -1)
        positions = torch.clamp(positions, max=self.max_seq_len - 1)

        x = self.token_embed(input_ids) + self.pos_embed(positions)
        x = self.dropout(x)

        # Causal mask for self-attention
        if T > 1:
            # Full sequence forward pass
            causal_mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        else:
            # Single-step generation: attend to all past tokens
            causal_mask = None

        # Cross-attention mask from text mask
        cross_mask = None
        if context_mask is not None:
            # context_mask: (B, T_ctx) -> (B, 1, 1, T_ctx)
            cross_mask = (1.0 - context_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0

        new_caches = [] if use_cache else None

        for idx, block in enumerate(self.blocks):
            layer_cache = kv_caches[idx] if (kv_caches is not None and idx < len(kv_caches)) else None
            x, updated_cache = block(
                x,
                context=context,
                layer_cache=layer_cache,
                causal_mask=causal_mask,
                context_mask=cross_mask,
                use_cache=use_cache,
            )
            if use_cache:
                new_caches.append(updated_cache)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits, new_caches
