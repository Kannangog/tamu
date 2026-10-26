"""Category-conditioned GRU: predicts next action type, then x, then y (y sees the chosen x)."""
import torch
from torch import nn

from .schema import ACTION_VOCAB, COORD_VOCAB


class DrawingGRU(nn.Module):
    def __init__(self, num_classes=1, embedding_dim=48, hidden_dim=256, num_layers=1, class_dim=32, dropout=0.0,
                 class_skip=False):
        super().__init__()
        self.config = dict(num_classes=num_classes, embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                           num_layers=num_layers, class_dim=class_dim, dropout=dropout, class_skip=class_skip)
        self.num_layers, self.hidden_dim, self.class_skip = num_layers, hidden_dim, class_skip
        self.action_emb = nn.Embedding(ACTION_VOCAB, embedding_dim, padding_idx=0)
        self.x_emb = nn.Embedding(COORD_VOCAB, embedding_dim)
        self.y_emb = nn.Embedding(COORD_VOCAB, embedding_dim)
        self.class_emb = nn.Embedding(num_classes, class_dim)
        self.init_h = nn.Linear(class_dim, hidden_dim * num_layers)  # category also sets the initial state
        self.gru = nn.GRU(embedding_dim * 3 + class_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        head_in = hidden_dim + (class_dim if class_skip else 0)
        self.action_head = nn.Linear(head_in, ACTION_VOCAB)
        self.x_head = nn.Linear(head_in, COORD_VOCAB)
        self.x_cond_emb = nn.Embedding(COORD_VOCAB, embedding_dim)
        self.y_head = nn.Linear(head_in + embedding_dim, COORD_VOCAB)

    def init_hidden(self, class_ids):
        h = torch.tanh(self.init_h(self.class_emb(class_ids)))
        return h.view(-1, self.num_layers, self.hidden_dim).transpose(0, 1).contiguous()

    def forward(self, action_ids, x_ids, y_ids, class_ids, hidden=None):
        """class_ids: (B,). Returns action_logits, x_logits, h, hidden. Use predict_y(h, x_ids, class_ids) for y."""
        T = action_ids.shape[1]
        c = self.class_emb(class_ids).unsqueeze(1).expand(-1, T, -1)
        z = torch.cat([self.action_emb(action_ids), self.x_emb(x_ids), self.y_emb(y_ids), c], dim=-1)
        if hidden is None:
            hidden = self.init_hidden(class_ids)
        h, hidden = self.gru(z, hidden)
        h_out = torch.cat([h, c], dim=-1) if self.class_skip else h
        return self.action_head(h_out), self.x_head(h_out), h, hidden

    def predict_y(self, h, x_ids, class_ids=None):
        x_feat = self.x_cond_emb(x_ids)
        if self.class_skip:
            if class_ids is None:
                raise ValueError("class_ids must be provided to predict_y when class_skip=True")
            c = self.class_emb(class_ids)
            if h.dim() == 3 and c.dim() == 2:
                c = c.unsqueeze(1).expand(-1, h.shape[1], -1)
            if h.dim() == 3 and x_feat.dim() == 2:
                x_feat = x_feat.unsqueeze(1).expand(-1, h.shape[1], -1)
            return self.y_head(torch.cat([h, x_feat, c], dim=-1))
        return self.y_head(torch.cat([h, x_feat], dim=-1))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
