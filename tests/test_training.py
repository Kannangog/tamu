import pytest
import torch
from konnon_draw.encoding import NO_COORD
from konnon_draw.model import ACTION_VOCAB, COORD_VOCAB
from konnon_draw.schema import Action
from konnon_draw.training import (
    DrawingDataset,
    collate_drawings,
    compute_loss,
)


def test_drawing_dataset_shifting():
    examples = [
        {
            "id": "1",
            "actions": [
                {"type": "MOVE", "x": 0.1, "y": 0.2},
                {"type": "LINE", "x": 0.3, "y": 0.4},
                {"type": "END"},
            ],
        }
    ]
    ds = DrawingDataset(examples, max_len=16)
    assert len(ds) == 1

    in_ids, in_xs, in_ys, tgt_ids, tgt_xs, tgt_ys = ds[0]

    # Input length is original length - 1 = 2
    assert in_ids.tolist() == [int(Action.MOVE), int(Action.LINE)]
    assert tgt_ids.tolist() == [int(Action.LINE), int(Action.END)]


def test_collate_drawings_padding():
    batch = [
        (
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([5, 6], dtype=torch.long),
            torch.tensor([7, 8], dtype=torch.long),
            torch.tensor([2, 4], dtype=torch.long),
            torch.tensor([6, 32], dtype=torch.long),
            torch.tensor([8, 32], dtype=torch.long),
        ),
        (
            torch.tensor([1], dtype=torch.long),
            torch.tensor([10], dtype=torch.long),
            torch.tensor([11], dtype=torch.long),
            torch.tensor([4], dtype=torch.long),
            torch.tensor([32], dtype=torch.long),
            torch.tensor([32], dtype=torch.long),
        ),
    ]

    b_in_ids, b_in_xs, b_in_ys, b_tgt_ids, b_tgt_xs, b_tgt_ys = collate_drawings(batch)

    assert b_in_ids.shape == (2, 2)
    assert b_in_ids[1, 1].item() == Action.PAD
    assert b_in_xs[1, 1].item() == NO_COORD
    assert b_tgt_ids[1, 1].item() == Action.PAD


def test_compute_loss_masks_properly():
    batch_size = 2
    seq_len = 3

    act_logits = torch.randn(batch_size, seq_len, ACTION_VOCAB)
    x_logits = torch.randn(batch_size, seq_len, COORD_VOCAB)
    y_logits = torch.randn(batch_size, seq_len, COORD_VOCAB)

    # Sequence with MOVE, LINE, and PAD
    tgt_ids = torch.tensor([
        [int(Action.MOVE), int(Action.PEN_UP), int(Action.END)],
        [int(Action.LINE), int(Action.PAD), int(Action.PAD)],
    ], dtype=torch.long)

    tgt_xs = torch.tensor([
        [10, NO_COORD, NO_COORD],
        [15, NO_COORD, NO_COORD],
    ], dtype=torch.long)

    tgt_ys = torch.tensor([
        [12, NO_COORD, NO_COORD],
        [18, NO_COORD, NO_COORD],
    ], dtype=torch.long)

    total_loss, act_loss, coord_loss = compute_loss(
        act_logits, x_logits, y_logits, tgt_ids, tgt_xs, tgt_ys
    )

    assert not torch.isnan(total_loss)
    assert not torch.isnan(act_loss)
    assert not torch.isnan(coord_loss)
    assert total_loss.item() > 0.0
