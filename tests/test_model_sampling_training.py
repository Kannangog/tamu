import torch
from konnon_draw.data_gen import main as gen_data
from konnon_draw.model import DrawingGRU
from konnon_draw.sampling import generate, random_baseline
from konnon_draw.schema import ACTION_VOCAB, COORD_VOCAB, is_valid
from konnon_draw.training import TrainConfig, load_model, soft_coord_ce, train


def test_model_shapes_and_conditioning():
    m = DrawingGRU(num_classes=3, hidden_dim=32)
    a = torch.randint(0, ACTION_VOCAB, (3, 7)); x = torch.randint(0, COORD_VOCAB, (3, 7))
    al, xl, h, hid = m(a, x, x, torch.tensor([0, 1, 2]))
    assert al.shape == (3, 7, ACTION_VOCAB) and xl.shape == (3, 7, COORD_VOCAB)
    assert m.predict_y(h, x).shape == (3, 7, COORD_VOCAB)
    # same tokens, different category -> different outputs
    same = a[:1].repeat(2, 1); sx = x[:1].repeat(2, 1)
    out, _, _, _ = m(same, sx, sx, torch.tensor([0, 1]))
    assert not torch.allclose(out[0], out[1])


def test_untrained_model_still_produces_valid_drawings():
    torch.manual_seed(0)
    m = DrawingGRU(num_classes=2, hidden_dim=32)
    for seed in range(30):
        r = generate(m, seed % 2, seed, max_len=24)
        assert is_valid(r["actions"]), r
        assert r["n_tokens"] <= 24


def test_tiny_max_len_uses_explicit_fallback():
    torch.manual_seed(1)
    m = DrawingGRU(num_classes=1, hidden_dim=32)
    results = [generate(m, 0, s, max_len=4) for s in range(20)]
    assert any(r["repaired"] for r in results)
    assert all(r["repaired"] or r["natural_end"] for r in results)


def test_generation_is_reproducible():
    torch.manual_seed(0)
    m = DrawingGRU(num_classes=2, hidden_dim=32)
    assert generate(m, 1, 5)["actions"] == generate(m, 1, 5)["actions"]


def test_random_baseline_mostly_invalid():
    assert sum(is_valid(random_baseline(s)) for s in range(200)) < 100


def test_soft_ce_prefers_near_misses():
    logits_near = torch.full((1, COORD_VOCAB), -5.0); logits_near[0, 11] = 5.0
    logits_far = torch.full((1, COORD_VOCAB), -5.0); logits_far[0, 40] = 5.0
    t = torch.tensor([10])
    assert soft_coord_ce(logits_near, t, 1.0) < soft_coord_ce(logits_far, t, 1.0)


def test_training_smoke(tmp_path):
    data = tmp_path / "toy.jsonl"
    gen_data(["--n", "60", "--out", str(data)])
    cfg = TrainConfig(data=str(data), out_dir=str(tmp_path / "out"), log_dir=str(tmp_path / "logs"),
                      epochs=3, hidden_dim=32, eval_samples=2, sample_every=3, checkpoint_every=3, max_len=64)
    best = train(cfg)
    assert best == best and best < 10
    assert (tmp_path / "out/checkpoints/best.pt").exists()
    model, ckpt = load_model(tmp_path / "out/checkpoints/best.pt")
    assert ckpt["seed"] == cfg.seed and len(ckpt["classes"]) == 4
    assert is_valid(generate(model, 0, 1)["actions"])
