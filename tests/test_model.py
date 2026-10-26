import pytest
import torch
from konnon_draw.model import ACTION_VOCAB, COORD_VOCAB, DrawingGRU


def test_model_forward_shapes():
    batch_size = 4
    seq_len = 10
    embedding_dim = 32
    hidden_dim = 128

    model = DrawingGRU(embedding_dim=embedding_dim, hidden_dim=hidden_dim)

    action_ids = torch.randint(0, ACTION_VOCAB, (batch_size, seq_len))
    x_ids = torch.randint(0, COORD_VOCAB, (batch_size, seq_len))
    y_ids = torch.randint(0, COORD_VOCAB, (batch_size, seq_len))

    act_logits, x_logits, y_logits, hidden = model(action_ids, x_ids, y_ids)

    assert act_logits.shape == (batch_size, seq_len, ACTION_VOCAB)
    assert x_logits.shape == (batch_size, seq_len, COORD_VOCAB)
    assert y_logits.shape == (batch_size, seq_len, COORD_VOCAB)
    assert hidden.shape == (1, batch_size, hidden_dim)


def test_model_step_with_hidden():
    model = DrawingGRU(embedding_dim=32, hidden_dim=64)

    a1 = torch.tensor([[1]], dtype=torch.long)
    x1 = torch.tensor([[10]], dtype=torch.long)
    y1 = torch.tensor([[15]], dtype=torch.long)

    out1, _, _, h1 = model(a1, x1, y1)
    assert out1.shape == (1, 1, ACTION_VOCAB)

    a2 = torch.tensor([[2]], dtype=torch.long)
    x2 = torch.tensor([[12]], dtype=torch.long)
    y2 = torch.tensor([[18]], dtype=torch.long)

    out2, _, _, h2 = model(a2, x2, y2, hidden=h1)
    assert out2.shape == (1, 1, ACTION_VOCAB)
    assert not torch.allclose(h1, h2)
