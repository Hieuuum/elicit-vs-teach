"""TRAIN-1 loop tests — specs/02 §6.1 ``evaluate_nll_nats`` / ``train_full``.

Module under test (does not exist yet):

    from geode.train.loop import TrainResult, evaluate_nll_nats, train_full

Derived from the spec only (specs/02-training-run.md §6.1 and the Runs 1-4
paragraph):

- ``evaluate_nll_nats``: mean next-token cross-entropy in nats over ALL
  predicted positions (positions 0..L-2 predict 1..L-1), under no_grad;
  batch_size must not change the value beyond float tolerance (V5.19).
- ``train_full``: AdamW(lr, betas, weight_decay), constant LR, global-norm
  grad clip; seeded per-epoch permutation, drop-last; step == one optimizer
  update; eval at every step where ``step % eval_every == 0`` plus the final
  step; a ConvergenceTracker True return -> stop_reason "converged", reaching
  max_steps -> "max_steps"; train_log.jsonl records
  {step, train_loss_nats, lr, grad_norm}, eval_log.jsonl records
  {step, val_loss_nats}; final model saved to out_dir/model via
  save_pretrained plus out_dir/training_meta.json; byte-identical logs for an
  identical seed + inputs on CPU (V5.21-V5.25).

CPU-only, tiny in-process models, fp32 (bf16 is untested config plumbing per
the spec). Reference NLLs are computed with a direct F.cross_entropy pass that
never calls ``evaluate_nll_nats``.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
import torch
from transformers import LlamaForCausalLM

from geode.train.loop import TrainResult, evaluate_nll_nats, train_full
from geode.train.stopping import StoppingRule

# Fixture vocab layout (tests/conftest.py): ids 0..3 are specials, 4..127 are
# word tokens. Any id in [4, 128) is a valid, non-special token.
_VOCAB = 128
_FIRST_WORD = 4
_STOPPING_DISABLED = StoppingRule(eps_nats=0.0, k=10**9)


def _toy_seqs(n_rows: int, seq_len: int, seed: int) -> torch.LongTensor:
    # A fixed pseudo-random token corpus, deterministic and independent of any
    # model. Distinct seeds give distinct corpora (so train != val by content).
    rng = random.Random(seed)
    data = [[rng.randrange(_FIRST_WORD, _VOCAB) for _ in range(seq_len)] for _ in range(n_rows)]
    return torch.tensor(data, dtype=torch.long)


def _train(model, train_seqs, val_seqs, out_dir, **overrides):
    kwargs = dict(
        lr=1e-3,
        batch_size=4,
        stopping=_STOPPING_DISABLED,
        eval_every=5,
        max_steps=10,
        grad_clip=1.0,
        weight_decay=0.0,
        betas=(0.9, 0.999),
        device="cpu",
        seed=0,
        out_dir=out_dir,
        precision="fp32",
    )
    kwargs.update(overrides)
    return train_full(model, train_seqs, val_seqs, **kwargs)


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --------------------------------------------------------------------------
# V5.19 — evaluate_nll_nats
# --------------------------------------------------------------------------
def test_eval_nll_matches_manual_reference(tiny_llama):
    # V5.19: evaluate_nll_nats equals an explicit per-position CE reference.
    model = tiny_llama(seed=0)
    seqs = torch.tensor(
        [
            [2, 4, 5, 6, 7, 8],
            [2, 9, 10, 11, 12, 13],
            [2, 14, 15, 16, 17, 18],
        ],
        dtype=torch.long,
    )  # N=3, L=6
    vocab = model.config.vocab_size
    # Independent reference (no call to evaluate_nll_nats): positions 0..L-2
    # predict tokens 1..L-1; mean CE over all N*(L-1)=15 positions, in nats.
    with torch.no_grad():
        logits = model(seqs).logits
    shift_logits = logits[:, :-1, :].reshape(-1, vocab)
    shift_targets = seqs[:, 1:].reshape(-1)
    reference = torch.nn.functional.cross_entropy(
        shift_logits, shift_targets, reduction="mean"
    ).item()
    got = evaluate_nll_nats(model, seqs, batch_size=3, device="cpu")
    assert abs(got - reference) < 1e-4
    # Guard against a sum-instead-of-mean unit error: a random-init model has
    # CE near ln(vocab); a summed loss would be ~ (L-1)x larger.
    assert 0.0 < got < 2.0 * math.log(vocab)


def test_eval_nll_invariant_to_batch_size(tiny_llama):
    # V5.19: the value is invariant to batch_size. N=5 is indivisible by 2, 3
    # and 4, so an "average of per-batch means" bug (which mis-weights the
    # short final batch) would change the result.
    model = tiny_llama(seed=0)
    seqs = torch.tensor(
        [
            [2, 4, 5, 6, 7, 8],
            [2, 9, 10, 11, 12, 13],
            [2, 14, 15, 16, 17, 18],
            [2, 19, 20, 21, 22, 23],
            [2, 24, 25, 26, 27, 28],
        ],
        dtype=torch.long,
    )  # N=5, L=6
    vocab = model.config.vocab_size
    with torch.no_grad():
        logits = model(seqs).logits
    reference = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, vocab),
        seqs[:, 1:].reshape(-1),
        reduction="mean",
    ).item()
    for batch_size in (1, 2, 3, 4, 5):
        got = evaluate_nll_nats(model, seqs, batch_size=batch_size, device="cpu")
        assert abs(got - reference) < 1e-4, (batch_size, got, reference)


# --------------------------------------------------------------------------
# V5.21 — train_full converges and stops
# --------------------------------------------------------------------------
def test_train_converges_and_stops_on_tiny_corpus(tiny_llama, tmp_path):
    # V5.21: on a trivially memorizable corpus, training terminates via
    # stop_reason="converged" before a generous max_steps, and the final train
    # loss is below the initial train loss.
    patterns = [
        [4, 5, 6, 7, 8, 9, 10, 11, 3],
        [12, 13, 14, 15, 16, 17, 18, 19, 3],
        [20, 21, 22, 23, 24, 25, 26, 27, 3],
        [28, 29, 30, 31, 32, 33, 34, 35, 3],
    ]
    # Four distinct fixed sequences; train and val both contain all four, so a
    # model that memorizes them drives both losses toward zero (val plateaus).
    train_seqs = torch.tensor(patterns * 12, dtype=torch.long)  # 48 rows
    val_seqs = torch.tensor(patterns * 4, dtype=torch.long)  # 16 rows
    model = tiny_llama(seed=0)
    result = _train(
        model,
        train_seqs,
        val_seqs,
        tmp_path,
        lr=1e-2,
        batch_size=16,
        stopping=StoppingRule(eps_nats=1e-3, k=5),
        eval_every=5,
        max_steps=500,
    )
    assert isinstance(result, TrainResult)
    assert result.stop_reason == "converged"
    assert result.final_step < 500
    train_recs = _read_jsonl(tmp_path / "train_log.jsonl")
    assert train_recs[-1]["train_loss_nats"] < train_recs[0]["train_loss_nats"]


# --------------------------------------------------------------------------
# V5.22 — byte-identical logs for identical seed + inputs
# --------------------------------------------------------------------------
def test_same_seed_identical_logs(tiny_llama, tmp_path):
    # V5.22: identical seed + inputs on CPU (fixed threads) -> byte-identical
    # train_log.jsonl and eval_log.jsonl across two independent runs.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)  # "fixed threads" per the property
    try:
        train_seqs = _toy_seqs(12, 8, seed=1)
        val_seqs = _toy_seqs(4, 8, seed=2)  # distinct corpus from train
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        # A fresh model per run: train_full mutates the model in place, so both
        # runs must start from the identical random-init (same fixture seed).
        _train(
            tiny_llama(seed=0), train_seqs, val_seqs, dir_a, seed=123, max_steps=15, eval_every=5
        )
        _train(
            tiny_llama(seed=0), train_seqs, val_seqs, dir_b, seed=123, max_steps=15, eval_every=5
        )
        for name in ("train_log.jsonl", "eval_log.jsonl"):
            assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes()
    finally:
        torch.set_num_threads(prev_threads)


# --------------------------------------------------------------------------
# V5.23 — log schema + eval-step set
# --------------------------------------------------------------------------
def test_log_schema_and_eval_step_set(tiny_llama, tmp_path):
    # V5.23: each train record carries exactly {step, train_loss_nats, lr,
    # grad_norm} with finite values and constant lr; each eval record carries
    # exactly {step, val_loss_nats}; eval steps are the multiples of
    # eval_every plus the final step.
    train_seqs = _toy_seqs(12, 8, seed=1)  # 12 // 4 == 3 steps/epoch
    val_seqs = _toy_seqs(4, 8, seed=2)
    lr = 1e-3
    _train(
        tiny_llama(seed=0),
        train_seqs,
        val_seqs,
        tmp_path,
        lr=lr,
        batch_size=4,
        eval_every=5,
        max_steps=12,
    )
    train_recs = _read_jsonl(tmp_path / "train_log.jsonl")
    # One record per optimizer step. Ruling on indexing: "step = one optimizer
    # update" + "reaching max_steps stops" make `step` the 1-based count of
    # completed updates (1..max_steps); 0-indexing would put an eval at step 0
    # (0 % eval_every == 0) before any update, which the spec nowhere asks for.
    assert [r["step"] for r in train_recs] == list(range(1, 13))
    for r in train_recs:
        assert set(r) == {"step", "train_loss_nats", "lr", "grad_norm"}
        assert math.isfinite(r["train_loss_nats"]) and r["train_loss_nats"] >= 0.0
        # grad_norm is spec'd as the PRE-clip global norm; only schema-level
        # presence/finiteness is testable without pinning implementation detail.
        assert math.isfinite(r["grad_norm"]) and r["grad_norm"] >= 0.0
        assert r["lr"] == lr  # constant LR
    eval_recs = _read_jsonl(tmp_path / "eval_log.jsonl")
    for r in eval_recs:
        assert set(r) == {"step", "val_loss_nats"}
        assert math.isfinite(r["val_loss_nats"]) and r["val_loss_nats"] >= 0.0
    # Multiples of 5 within 1..12 are {5, 10}; the final step 12 is added.
    assert [r["step"] for r in eval_recs] == [5, 10, 12]


@pytest.mark.parametrize(
    "max_steps, eval_every, expected_eval_steps",
    [
        (5, 100, [5]),  # eval_every > max_steps: only the final step
        (10, 5, [5, 10]),  # final step coincides with a multiple
        (12, 5, [5, 10, 12]),  # final step is extra (not a multiple)
    ],
)
def test_eval_step_set_edge_cadences(
    tiny_llama, tmp_path, max_steps, eval_every, expected_eval_steps
):
    # V5.23: eval steps == (multiples of eval_every that are <= final) plus the
    # final step, across boundary cadences.
    train_seqs = _toy_seqs(8, 8, seed=1)  # 8 // 4 == 2 steps/epoch
    val_seqs = _toy_seqs(4, 8, seed=2)
    _train(
        tiny_llama(seed=0),
        train_seqs,
        val_seqs,
        tmp_path,
        batch_size=4,
        eval_every=eval_every,
        max_steps=max_steps,
    )
    eval_recs = _read_jsonl(tmp_path / "eval_log.jsonl")
    assert [r["step"] for r in eval_recs] == expected_eval_steps


# --------------------------------------------------------------------------
# V5.24 — checkpoint roundtrip
# --------------------------------------------------------------------------
def test_checkpoint_roundtrip_same_val_nll(tiny_llama, tmp_path):
    # V5.24: reloading out_dir/model reproduces the trained model's val NLL,
    # and training_meta.json fields match the returned TrainResult.
    train_seqs = _toy_seqs(8, 8, seed=1)
    val_seqs = _toy_seqs(4, 8, seed=2)
    model = tiny_llama(seed=0)
    result = _train(model, train_seqs, val_seqs, tmp_path, batch_size=4, eval_every=3, max_steps=6)
    # train_full trains `model` in place; compare it against the reloaded copy.
    nll_trained = evaluate_nll_nats(model, val_seqs, batch_size=4, device="cpu")
    assert (tmp_path / "model").exists()
    assert Path(result.checkpoint_dir).resolve() == (tmp_path / "model").resolve()
    reloaded = LlamaForCausalLM.from_pretrained(result.checkpoint_dir)
    reloaded.eval()
    nll_reloaded = evaluate_nll_nats(reloaded, val_seqs, batch_size=4, device="cpu")
    assert abs(nll_trained - nll_reloaded) < 1e-6  # fp32 save/load is exact
    # The final-step eval measures the final (== saved) model: the stopping
    # rule and the "additionally at the final step" clause are incoherent if
    # evals preceded the step's update. Tolerance 1e-4 covers the in-loop
    # eval's free choice of batch size (V5.19 pins batch invariance).
    eval_recs = _read_jsonl(tmp_path / "eval_log.jsonl")
    assert eval_recs[-1]["step"] == result.final_step
    assert abs(eval_recs[-1]["val_loss_nats"] - nll_reloaded) < 1e-4
    meta = json.loads((tmp_path / "training_meta.json").read_text())
    assert meta["stop_reason"] == result.stop_reason
    assert meta["final_step"] == result.final_step
    assert meta["best_val_nats"] == result.best_val_nats
    # This run uses eps_nats=0.0 (_STOPPING_DISABLED), where the tracker's
    # best is exactly the running min of the eval series — so best_val_nats
    # must equal the min of the logged evals under either reading of "best"
    # (tracker best vs plain min). json round-trips doubles exactly.
    assert result.best_val_nats == min(r["val_loss_nats"] for r in eval_recs)


# --------------------------------------------------------------------------
# V5.25 — max_steps cap (plus the §6.1 "max_steps=None means no cap" contract)
# --------------------------------------------------------------------------
def test_max_steps_cap_reason(tiny_llama, tmp_path):
    # V5.25: with stopping effectively disabled (huge k), training stops at
    # exactly max_steps with stop_reason="max_steps".
    train_seqs = _toy_seqs(12, 8, seed=1)
    val_seqs = _toy_seqs(4, 8, seed=2)
    result = _train(
        tiny_llama(seed=0),
        train_seqs,
        val_seqs,
        tmp_path,
        batch_size=4,
        eval_every=5,
        max_steps=13,
        stopping=_STOPPING_DISABLED,
    )
    assert result.stop_reason == "max_steps"
    assert result.final_step == 13
    train_recs = _read_jsonl(tmp_path / "train_log.jsonl")
    assert len(train_recs) == 13  # exactly max_steps step records


def test_max_steps_none_stops_via_convergence(tiny_llama, tmp_path):
    # §6.1: "max_steps=None means no cap" — None must be accepted, leaving
    # convergence as the only stop condition. The stop is forced at the
    # second eval by construction: the first eval always improves
    # (inf - v == inf > eps), and with eps_nats=1e9 no finite later eval can
    # improve by more than eps, so k=1 trips at the second eval — step 4 at
    # eval_every=2 — bounding the runtime with no step cap in play.
    train_seqs = _toy_seqs(8, 8, seed=1)
    val_seqs = _toy_seqs(4, 8, seed=2)
    result = _train(
        tiny_llama(seed=0),
        train_seqs,
        val_seqs,
        tmp_path,
        batch_size=4,
        eval_every=2,
        max_steps=None,
        stopping=StoppingRule(eps_nats=1e9, k=1),
    )
    assert result.stop_reason == "converged"
    assert result.final_step == 4
    eval_recs = _read_jsonl(tmp_path / "eval_log.jsonl")
    assert [r["step"] for r in eval_recs] == [2, 4]


# --------------------------------------------------------------------------
# §6.1 train_full guard (added 2026-07-16) — train_seqs rows < batch_size
# --------------------------------------------------------------------------
def test_train_batch_larger_than_corpus_raises(tiny_llama, tmp_path):
    # §6.1 train_full contract, guard added 2026-07-16: "train_full raises
    # ValueError upfront if train_seqs has fewer rows than batch_size" —
    # without it, drop-last yields zero batches per epoch and the loop can
    # never reach any stop condition (silent infinite busy-loop).
    bad_train = _toy_seqs(3, 8, seed=1)  # 3 rows < batch_size == 4
    val_seqs = _toy_seqs(4, 8, seed=2)
    model = tiny_llama(seed=0)
    with pytest.raises(ValueError):
        _train(model, bad_train, val_seqs, tmp_path / "bad", batch_size=4, max_steps=2)
    # Boundary: rows == batch_size is NOT "fewer" — exactly one batch per
    # epoch; must train normally, not raise (over-strict-guard control).
    result = _train(
        tiny_llama(seed=0),
        _toy_seqs(4, 8, seed=1),
        val_seqs,
        tmp_path / "ok",
        batch_size=4,
        max_steps=1,
    )
    assert result.final_step == 1


# --------------------------------------------------------------------------
# V5.34 — gradient accumulation
# --------------------------------------------------------------------------
def test_grad_accum_matches_full_batch(tiny_llama, tmp_path):
    # V5.34: micro_batch_size < batch_size reproduces the full-batch run —
    # same logged per-step train losses and same final parameters within
    # float tolerance (same seed, same init, fp32/CPU).
    train_seqs = _toy_seqs(16, 8, seed=1)
    val_seqs = _toy_seqs(4, 8, seed=2)
    m_full = tiny_llama(seed=0)
    m_micro = tiny_llama(seed=0)
    _train(m_full, train_seqs, val_seqs, tmp_path / "full", batch_size=8, max_steps=4)
    _train(
        m_micro,
        train_seqs,
        val_seqs,
        tmp_path / "micro",
        batch_size=8,
        max_steps=4,
        micro_batch_size=2,
    )
    full_log = _read_jsonl(tmp_path / "full" / "train_log.jsonl")
    micro_log = _read_jsonl(tmp_path / "micro" / "train_log.jsonl")
    assert len(full_log) == len(micro_log) == 4
    for rec_full, rec_micro in zip(full_log, micro_log):
        assert rec_micro["train_loss_nats"] == pytest.approx(rec_full["train_loss_nats"], abs=1e-5)
    for (name, p_full), (_, p_micro) in zip(m_full.named_parameters(), m_micro.named_parameters()):
        assert torch.allclose(p_full, p_micro, atol=1e-5), name
    # The resolved micro size is echoed in training_meta.json (and defaults
    # to batch_size when accumulation is unused).
    meta_micro = json.loads((tmp_path / "micro" / "training_meta.json").read_text())
    assert meta_micro["config"]["micro_batch_size"] == 2
    meta_full = json.loads((tmp_path / "full" / "training_meta.json").read_text())
    assert meta_full["config"]["micro_batch_size"] == 8


@pytest.mark.parametrize("bad_micro", [0, -1, 3, 16])
def test_grad_accum_invalid_micro_batch_raises(tiny_llama, tmp_path, bad_micro):
    # V5.34: micro_batch_size that is 0, negative, larger than batch_size,
    # or not a divisor of it raises upfront, before any disk write.
    train_seqs = _toy_seqs(8, 8, seed=1)
    val_seqs = _toy_seqs(4, 8, seed=2)
    out = tmp_path / "bad"
    with pytest.raises(ValueError, match="micro_batch_size"):
        _train(
            tiny_llama(seed=0),
            train_seqs,
            val_seqs,
            out,
            batch_size=8,
            micro_batch_size=bad_micro,
        )
    assert not out.exists()
