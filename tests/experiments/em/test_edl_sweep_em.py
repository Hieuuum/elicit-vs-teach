"""experiments/em/edl_sweep_em.py: the EM EDL sweep pieces (CPU, tiny random model)."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file())
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "em"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "experiments" / "em" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load("edl_sweep_em")
A = _load("analyze_edl_em")


def test_reset_lora_restores_the_fresh_adapter(tiny_llama):
    from geode.train.lora import apply_lora

    fresh = tiny_llama(seed=0)
    apply_lora(fresh, rank=4, alpha=8.0, seed=5)
    used = tiny_llama(seed=0)
    apply_lora(used, rank=4, alpha=8.0, seed=5)
    with torch.no_grad():
        for p in used.parameters():
            if p.requires_grad:
                p.add_(torch.randn_like(p))
    S.reset_lora(used, 5)
    for (n, a), (_, b) in zip(fresh.named_parameters(), used.named_parameters()):
        assert torch.allclose(a, b), n


def test_rotation_moves_every_response_and_split_is_disjoint():
    df = pd.DataFrame(
        {"user": [f"u{i}" for i in range(20)], "response": [f"r{i}" for i in range(20)]}
    )
    rot, k = S.rotate_responses(df, 0)
    assert 1 <= k <= 19 and all(a != b for a, b in zip(rot["response"], df["response"]))
    assert (
        sorted(rot["response"]) == sorted(df["response"])
        and rot["user"].tolist() == df["user"].tolist()
    )
    pool, val, test = S.split(df, 4, 3, 0)
    ids = lambda d: set(d["user"])  # noqa: E731
    assert (
        len(pool) == 13
        and not (ids(pool) & ids(val))
        and not (ids(pool) & ids(test))
        and not (ids(val) & ids(test))
    )


def test_run_point_edl_is_mdl_minus_d_times_floor(tiny_llama, tiny_tokenizer):
    from geode.train.lora import apply_lora

    tok = tiny_tokenizer()
    model = tiny_llama(seed=1)
    apply_lora(model, rank=4, alpha=8.0, seed=0)
    enc = [S.encode_example(tok, f"t{i} t{i+1}", f"t{i+2} t{i+3}", 32) for i in range(10, 30)]
    r = S.run_point(
        model,
        tok,
        enc[:8],
        enc[8:12],
        enc[12:16],
        seed=0,
        lr=5e-3,
        batch_size=2,
        grad_accum=2,
        warmup=1,
        max_epochs=4,
        patience=2,
        max_steps=50,
        device="cpu",
    )
    d = r["label_tokens"]
    assert d == sum(len(x) - st for x, st in enc[:8])
    assert r["edl_ocv_per_token_nats"] == pytest.approx(
        (r["mdl_nats"] - d * r["val_floor_nats"]) / d
    )
    assert r["val_floor_nats"] == pytest.approx(min(r["val_losses"]))
    assert math.isfinite(r["test_floor_nats"]) and r["epochs"] >= 1


def test_curves_keep_sizes_common_to_every_seed():
    pts = [{"seed": 1, "n_train": n, "k": v} for n, v in ((16, 1.0), (32, 2.0), (64, 3.0))] + [
        {"seed": 2, "n_train": n, "k": v} for n, v in ((16, 1.0), (32, 2.0))
    ]
    ns, cs = A.curves(pts, "k")
    assert (
        ns == [16, 32]
        and len(cs) == 2
        and cs[0] == pytest.approx([1 / math.log(2), 2 / math.log(2)])
    )


def test_run_point_reports_m21_on_the_test_rows(tiny_llama, tiny_tokenizer):
    from geode.train.lora import apply_lora

    tok = tiny_tokenizer()
    model = tiny_llama(seed=2)
    apply_lora(model, rank=4, alpha=8.0, seed=0)
    rows = [(f"t{i} t{i+1}", f"t{i+2} t{i+3}") for i in range(10, 30)]
    enc = [S.encode_example(tok, u, a, 32) for u, a in rows]
    test = rows[12:16]
    rot = [S.encode_example(tok, u, a, 32) for (u, _), (_, a) in zip(test, test[1:] + test[:1])]
    r = S.run_point(
        model,
        tok,
        enc[:8],
        enc[8:12],
        enc[12:16],
        seed=0,
        lr=5e-3,
        batch_size=2,
        grad_accum=2,
        warmup=1,
        max_epochs=2,
        patience=2,
        max_steps=20,
        device="cpu",
        test_rot_encs=[rot],
    )
    assert r["m21_test_nats"] == pytest.approx(r["m21_rotated_loss_nats"][0] - r["test_floor_nats"])
    r0 = S.run_point(
        model,
        tok,
        enc[:8],
        enc[8:12],
        enc[12:16],
        seed=0,
        lr=5e-3,
        batch_size=2,
        grad_accum=2,
        warmup=1,
        max_epochs=2,
        patience=2,
        max_steps=20,
        device="cpu",
    )
    assert r0["m21_test_nats"] is None
