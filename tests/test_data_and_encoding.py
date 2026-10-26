import torch
from konnon_draw.data_gen import make_examples
from konnon_draw.encoding import collate, decode_one, encode_examples, encode_one
from konnon_draw.schema import Action, NO_COORD, is_valid, quantize_coord


def test_all_toy_examples_valid():
    ex = make_examples(80)
    assert all(is_valid(e["actions"]) for e in ex)
    assert {e["category"] for e in ex} == {"triangle", "rectangle", "cross", "zigzag"}


def test_toy_data_deterministic():
    assert make_examples(20, seed=3) == make_examples(20, seed=3)


def test_roundtrip():
    for e in make_examples(40):
        ids, xs, ys = encode_one(e["actions"])
        assert ids[0] == int(Action.START) and xs[0] == NO_COORD
        back = decode_one(ids, xs, ys)
        assert is_valid(back)
        assert [a["type"] for a in back] == [a["type"] for a in e["actions"]]
        for a, b in zip(e["actions"], back):
            if "x" in a:
                assert quantize_coord(a["x"]) == quantize_coord(b["x"])
                assert quantize_coord(a["y"]) == quantize_coord(b["y"])


def test_collate_shift_and_padding():
    items, skipped = encode_examples(make_examples(12))
    assert skipped == 0
    ia, ix, iy, ta, tx, ty, cls = collate(items[:5])
    assert cls.shape == (5,)
    assert ia.shape == ta.shape
    # targets are inputs shifted by one
    for b, it in enumerate(items[:5]):
        n = len(it["ids"])
        assert ta[b, :n - 1].tolist() == it["ids"][1:]
        assert ia[b, :n - 1].tolist() == it["ids"][:-1]
        assert (ta[b, n - 1:] == int(Action.PAD)).all()
    assert not (ta == int(Action.START)).any()


def test_too_long_skipped():
    items, skipped = encode_examples(make_examples(8), max_len=4)
    assert skipped == 8 and items == []
