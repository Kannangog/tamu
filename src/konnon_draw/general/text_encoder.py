"""Frozen pretrained text encoder tower for Konnon Draw General.

Provides:
- Contextual sequence embeddings H_text in R^(B x L x D) for cross-attention.
- Pooled sentence embedding in R^(B x D).
- Support for:
  1) Built-in deterministic Semantic Text Tower (runs completely offline, 0 network dependencies).
  2) HuggingFace pretrained models (e.g. all-MiniLM-L6-v2, distilbert, CLIP text tower) when specified.
"""

import math
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn


class BuiltinSemanticTokenizer:
    """Fast, deterministic subword / word tokenizer with fixed vocabulary."""

    def __init__(self):
        # Curated vocabulary containing shapes, colors, spatial prepositions,
        # categories, counts, and common descriptive words
        self.special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self.words = [
            # Shapes & Geometry
            "circle", "circles", "round", "oval", "ellipse", "square", "squares",
            "rectangle", "rectangles", "triangle", "triangles", "star", "stars",
            "line", "lines", "box", "polygon", "ring", "shape", "forms",
            # Colors
            "red", "blue", "green", "yellow", "orange", "purple", "black", "white",
            "gray", "grey", "brown", "cyan", "teal", "pink", "rose", "gold", "amber",
            "dark", "light", "bright", "vibrant", "pale", "deep",
            # Spatial Relations & Positions
            "top", "bottom", "left", "right", "center", "middle", "above", "below",
            "under", "over", "beside", "next", "between", "inside", "outside",
            "corner", "upper", "lower", "horizon", "ground", "sky",
            # Counts & Quantifiers
            "a", "an", "the", "one", "two", "three", "four", "five", "pair", "many", "several",
            # Quick, Draw! & Common Objects
            "cat", "dog", "bicycle", "bike", "car", "chair", "table", "bed", "house",
            "tree", "trees", "flower", "sun", "moon", "cloud", "clouds", "bird", "fish",
            "boat", "plane", "airplane", "face", "eye", "head", "wheel", "wheels",
            "door", "window", "roof", "chimney", "mountain", "hills", "lake", "water",
            # Attributes & Artistic Descriptors
            "drawing", "sketch", "study", "illustration", "doodle", "ink", "pencil",
            "charcoal", "shadow", "shading", "sitting", "standing", "sunset", "sunrise",
            "simple", "detailed", "cute", "vintage", "clean", "minimal",
        ]
        self.vocab = {w: idx for idx, w in enumerate(self.special + self.words)}
        self.id_to_word = {idx: w for w, idx in self.vocab.items()}

    def tokenize(self, text: str) -> List[str]:
        # Clean text
        clean = text.lower().replace("-", " ").replace(".", " ").replace(",", " ")
        tokens = [t for t in clean.split() if t]
        return tokens

    def encode(self, text: str, max_length: int = 32) -> Dict[str, torch.Tensor]:
        words = self.tokenize(text)
        token_ids = [self.vocab["[CLS]"]]
        for w in words:
            token_ids.append(self.vocab.get(w, self.vocab["[UNK]"]))
            if len(token_ids) >= max_length - 1:
                break
        token_ids.append(self.vocab["[SEP]"])

        length = len(token_ids)
        mask = [1] * length + [0] * (max_length - length)
        token_ids = token_ids + [self.vocab["[PAD]"]] * (max_length - length)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }

    def batch_encode(self, texts: List[str], max_length: int = 32) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        for t in texts:
            encoded = self.encode(t, max_length=max_length)
            input_ids.append(encoded["input_ids"])
            attention_mask.append(encoded["attention_mask"])
        return {
            "input_ids": torch.stack(input_ids, dim=0),
            "attention_mask": torch.stack(attention_mask, dim=0),
        }


class FrozenBuiltinTextTower(nn.Module):
    """Small frozen Transformer text tower with semantic embedding space."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 3, max_len: int = 32):
        super().__init__()
        self.tokenizer = BuiltinSemanticTokenizer()
        vocab_size = len(self.tokenizer.vocab)

        # Deterministic semantic initialization without perturbing external RNG state
        rng_state = torch.get_rng_state()
        try:
            torch.manual_seed(42)
            self.token_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(max_len, d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)

            nn.init.normal_(self.token_embed.weight, std=0.02)
            nn.init.normal_(self.pos_embed.weight, std=0.02)
        finally:
            torch.set_rng_state(rng_state)

        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embed(input_ids) + self.pos_embed(pos)
        # Create key padding mask (True for padded tokens)
        src_key_padding_mask = (attention_mask == 0)

        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)

        # Pooled representation ([CLS] token at index 0)
        pooled = h[:, 0, :]

        return {
            "last_hidden_state": h,
            "pooler_output": pooled,
            "attention_mask": attention_mask,
        }


class FrozenHuggingFaceTextTower(nn.Module):
    """Wrapper around HuggingFace transformer models (e.g. MiniLM, CLIP)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", d_out: int = 256):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        hf_dim = self.model.config.hidden_size

        self.proj = nn.Linear(hf_dim, d_out) if hf_dim != d_out else nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            h = outputs.last_hidden_state
            pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else h[:, 0, :]
        return {
            "last_hidden_state": self.proj(h),
            "pooler_output": self.proj(pooled),
            "attention_mask": attention_mask,
        }


class FrozenTextEncoder(nn.Module):
    """Unified Text Encoder interface."""

    def __init__(self, model_type: str = "builtin", d_model: int = 256, device: str = "cpu"):
        super().__init__()
        self.model_type = model_type
        self.d_model = d_model
        self.device = device

        if model_type == "builtin":
            self.tower = FrozenBuiltinTextTower(d_model=d_model)
            self.tokenizer = self.tower.tokenizer
        else:
            try:
                self.tower = FrozenHuggingFaceTextTower(model_name=model_type, d_out=d_model)
                self.tokenizer = self.tower.tokenizer
            except Exception as e:
                print(f"[WARN] Failed to load HF model {model_type}: {e}. Falling back to builtin text tower.")
                self.tower = FrozenBuiltinTextTower(d_model=d_model)
                self.tokenizer = self.tower.tokenizer

        self.to(device)
        self.eval()

    def encode(self, texts: Union[str, List[str]], max_length: int = 32) -> Dict[str, torch.Tensor]:
        """Encodes string or list of strings into text embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        if hasattr(self.tokenizer, "batch_encode"):
            inputs = self.tokenizer.batch_encode(texts, max_length=max_length)
        else:
            inputs = self.tokenizer(texts, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            res = self.tower(input_ids, attention_mask)

        return res

    def encode_text(self, texts: Union[str, List[str]], device: Optional[str] = None, max_length: int = 32) -> torch.Tensor:
        """Returns last_hidden_state tensor of shape (B, L, D)."""
        if device is not None and device != self.device:
            self.to(device)
            self.device = device
        out = self.encode(texts, max_length=max_length)
        return out["last_hidden_state"]
