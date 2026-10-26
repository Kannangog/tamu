import torch
from torch import nn

ACTION_VOCAB = 5
COORD_VOCAB = 33  # bins 0..31 plus NO_COORD=32


class DrawingGRU(nn.Module):
    def __init__(self, embedding_dim=32, hidden_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.action_emb = nn.Embedding(ACTION_VOCAB, embedding_dim, padding_idx=0)
        self.x_emb = nn.Embedding(COORD_VOCAB, embedding_dim)
        self.y_emb = nn.Embedding(COORD_VOCAB, embedding_dim)
        self.gru = nn.GRU(embedding_dim * 3, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, ACTION_VOCAB)
        self.x_head = nn.Linear(hidden_dim, COORD_VOCAB)
        self.y_head = nn.Linear(hidden_dim, COORD_VOCAB)

    def forward(self, action_ids, x_ids, y_ids, hidden=None):
        z = torch.cat([
            self.action_emb(action_ids),
            self.x_emb(x_ids),
            self.y_emb(y_ids),
        ], dim=-1)
        h, hidden = self.gru(z, hidden)
        return (
            self.action_head(h),
            self.x_head(h),
            self.y_head(h),
            hidden,
        )
