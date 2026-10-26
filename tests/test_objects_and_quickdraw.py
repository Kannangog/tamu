import json
from konnon_draw.objects_gen import CATEGORIES, make_examples
from konnon_draw.quickdraw import build_dataset, convert_drawing
from konnon_draw.schema import is_valid


def test_objects_valid_and_balanced():
    ex = make_examples(10)
    assert len(ex) == 10 * len(CATEGORIES)
    assert all(is_valid(e["actions"]) for e in ex)
    assert max(len(e["actions"]) for e in ex) < 120


def test_objects_deterministic():
    assert make_examples(3, seed=5) == make_examples(3, seed=5)


def test_convert_quickdraw_drawing():
    strokes = [[[0, 100, 255], [0, 50, 255]], [[10, 10], [10, 200]], [[5], [5]]]  # last is a dot
    acts = convert_drawing(strokes)
    assert acts is not None and is_valid(acts)
    assert sum(a["type"] == "MOVE" for a in acts) == 2  # dot dropped
    assert all(0 <= a["x"] <= 1 for a in acts if "x" in a)


def test_convert_rejects_empty_and_too_long():
    assert convert_drawing([]) is None
    assert convert_drawing([[[3, 3], [4, 4]]]) is None  # collapses to a single point
    long = [[list(range(0, 200)), [i % 7 * 30 for i in range(200)]]]
    assert convert_drawing(long, max_len=50) is None


def test_build_dataset_from_local_ndjson(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw"; raw.mkdir()
    rows = [{"key_id": str(i), "recognized": i % 2 == 0, "drawing": [[[0, 50, 200], [0, 80, 120]]]} for i in range(10)]
    (raw / "chair.ndjson").write_text("\n".join(json.dumps(r) for r in rows))
    counts, mp = build_dataset(["chair"], 100, 160, raw, tmp_path / "q.jsonl", fetch=False)
    assert counts == {"chair": 5}  # unrecognized rows filtered
    assert "CC BY" in mp.read_text()
