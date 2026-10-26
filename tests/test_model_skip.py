"""Tests for category embedding skip-connection to output heads."""
import pytest
import torch
from konnon_draw.model import DrawingGRU
from konnon_draw.sampling import generate
from konnon_draw.schema import ACTION_VOCAB, COORD_VOCAB, is_valid


def test_skip_connection_shapes():
    m = DrawingGRU(num_classes=4, embedding_dim=16, hidden_dim=32, class_dim=16, class_skip=True)
    a = torch.randint(0, ACTION_VOCAB, (2, 5))
    x = torch.randint(0, COORD_VOCAB, (2, 5))
    y = torch.randint(0, COORD_VOCAB, (2, 5))
    cls = torch.tensor([1, 3])

    al, xl, h, hid = m(a, x, y, cls)
    assert al.shape == (2, 5, ACTION_VOCAB)
    assert xl.shape == (2, 5, COORD_VOCAB)

    yl = m.predict_y(h, x, cls)
    assert yl.shape == (2, 5, COORD_VOCAB)


def test_skip_connection_generation():
    torch.manual_seed(42)
    m = DrawingGRU(num_classes=3, embedding_dim=16, hidden_dim=32, class_dim=16, class_skip=True)
    res = generate(m, class_id=1, seed=10, max_len=30)
    assert is_valid(res["actions"])
    assert res["actions"][-1]["type"] == "END"


def test_skip_connection_gradients():
    m = DrawingGRU(num_classes=2, embedding_dim=16, hidden_dim=32, class_dim=16, class_skip=True)
    a = torch.randint(0, ACTION_VOCAB, (2, 6))
    x = torch.randint(0, COORD_VOCAB, (2, 6))
    y = torch.randint(0, COORD_VOCAB, (2, 6))
    cls = torch.tensor([0, 1])

    al, xl, h, _ = m(a, x, y, cls)
    yl = m.predict_y(h, x, cls)
    loss = al.sum() + xl.sum() + yl.sum()
    loss.backward()

    # Verify gradients flow into class_emb from both the heads and the recurrent cell
    assert m.class_emb.weight.grad is not None
    assert torch.all(torch.isfinite(m.class_emb.weight.grad))
