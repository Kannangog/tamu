"""Conditioned Drawing GRU with Category Embeddings and High-Resolution Coordinate Heads."""

import torch
from torch import nn

ACTION_VOCAB = 5
REAL_COORD_VOCAB = 65  # 0..63 bins + 64 (NO_COORD)
NUM_CATEGORIES = 5


class ConditionedDrawingGRU(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 32,
        category_dim: int = 32,
        hidden_dim: int = 160,
        coord_vocab: int = REAL_COORD_VOCAB,
        num_categories: int = NUM_CATEGORIES,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.category_dim = category_dim
        self.hidden_dim = hidden_dim
        self.coord_vocab = coord_vocab
        self.num_categories = num_categories

        self.action_emb = nn.Embedding(ACTION_VOCAB, embedding_dim, padding_idx=0)
        self.x_emb = nn.Embedding(coord_vocab, embedding_dim)
        self.y_emb = nn.Embedding(coord_vocab, embedding_dim)
        self.category_emb = nn.Embedding(num_categories, category_dim)

        self.cat_to_h0 = nn.Linear(category_dim, hidden_dim)
        gru_in_dim = embedding_dim * 3 + category_dim
        self.gru = nn.GRU(gru_in_dim, hidden_dim, batch_first=True)

        self.action_head = nn.Linear(hidden_dim, ACTION_VOCAB)
        self.x_head = nn.Linear(hidden_dim, coord_vocab)
        self.y_head = nn.Linear(hidden_dim, coord_vocab)

    def forward(self, action_ids, x_ids, y_ids, category_ids, hidden=None):
        B, T = action_ids.shape
        cat_vec = self.category_emb(category_ids)  # (B, category_dim)

        if hidden is None:
            hidden = torch.tanh(self.cat_to_h0(cat_vec)).unsqueeze(0)  # (1, B, hidden_dim)

        cat_expanded = cat_vec.unsqueeze(1).expand(-1, T, -1)  # (B, T, category_dim)

        z = torch.cat([
            self.action_emb(action_ids),
            self.x_emb(x_ids),
            self.y_emb(y_ids),
            cat_expanded,
        ], dim=-1)

        h, hidden = self.gru(z, hidden)

        return (
            self.action_head(h),
            self.x_head(h),
            self.y_head(h),
            hidden,
        )
