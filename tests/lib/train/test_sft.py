"""SFT-mode tests — specs/02 §6.1 ``evaluate_sft_nll_nats`` / ``train_sft``.

Properties V5.28-V5.33 (specs/02 §6.1, label-masked SFT mode): loss on label
(answer) tokens only via the single mask path, padding/question inertness,
span guard, determinism, convergence.

CPU-only, tiny in-process models, fp32. Reference losses are computed with
direct per-example, pad-free forwards that never call the module under test.
The gradient half of V5.30 exercises the module-private training objective
(``_mean_masked_ce_nats`` over ``_padded_inputs_and_mask``) directly:
``train_sft`` hardcodes AdamW, whose sign-like normalization would amplify
float-association noise on near-zero gradients past any honest tolerance,
while raw gradient comparison is exact to float noise. V5.28 pins that this
objective is the one ``train_sft`` logs at step 1, closing the loop.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import pytest
import torch

from geode.edl.masking import TaskFormat
from geode.train.loop import evaluate_nll_nats
from geode.train.sft import (
    _mean_masked_ce_nats,
    _padded_inputs_and_mask,
    evaluate_sft_nll_nats,
    train_sft,
)
from geode.train.stopping import BehavioralStoppingRule, StoppingRule
from tests.conftest import BOS_ID, TaskExample

# Fixture vocab layout (tests/conftest.py): ids 0..3 are specials, 4..127 are
# word tokens. Any id in [4, 128) is a valid, non-special token.
_VOCAB = 128
_FIRST_WORD = 4
_FORMAT = TaskFormat(name="test-arith", format_version="v1")
_STOPPING_DISABLED = StoppingRule(eps_nats=0.0, k=10**9)


def _example(rng: random.Random, n_prompt: int, n_answer: int) -> TaskExample:
    """[BOS, prompt..., answer...] with the answer tokens as the label span."""
    ids = [BOS_ID] + [rng.randrange(_FIRST_WORD, _VOCAB) for _ in range(n_prompt + n_answer)]
    start = 1 + n_prompt
    return TaskExample(input_ids=ids, label_span=(start, start + n_answer))


def _reference_sums(model, examples) -> tuple[float, int]:
    """Independent reference: per-example, pad-free forwards; sum of
    -log p(token_p | prefix) over exactly the label-span positions."""
    total = 0.0
    count = 0
    with torch.no_grad():
        for ex in examples:
            ids = torch.tensor([ex.input_ids], dtype=torch.long)
            log_probs = torch.log_softmax(model(ids).logits, dim=-1)
            start, end = ex.label_span
            for p in range(start, end):
                total += -log_probs[0, p - 1, ex.input_ids[p]].item()
                count += 1
    return total, count


def _train(model, train_examples, val_examples, out_dir, **overrides):
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
    return train_sft(model, train_examples, val_examples, _FORMAT, **kwargs)


def _read_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# --------------------------------------------------------------------------
# V5.28 — masked loss equals the label-positions-only reference
# --------------------------------------------------------------------------
def test_v5_28_masked_loss_matches_label_only_reference(tiny_llama, tmp_path):
    # V5.28: evaluate_sft_nll_nats and the step-1 training loss equal a
    # hand-computed mean CE over exactly the label-span positions; the
    # all-positions mean differs (the exclusion has teeth).
    model = tiny_llama(seed=0)
    examples = [
        TaskExample(input_ids=[2, 4, 5, 6, 7, 8], label_span=(4, 6)),
        TaskExample(input_ids=[2, 9, 10, 11, 12, 13], label_span=(3, 6)),
        TaskExample(input_ids=[2, 14, 15, 16, 17, 18], label_span=(5, 6)),
    ]  # equal lengths: no padding in play here (padding is V5.30's job)
    total, count = _reference_sums(model, examples)
    assert count == 6
    reference = total / count
    got = evaluate_sft_nll_nats(model, examples, _FORMAT, batch_size=3, device="cpu")
    assert abs(got - reference) < 1e-4
    # Teeth: a loss that also averaged the question positions would differ.
    full_lm = evaluate_nll_nats(
        model,
        torch.tensor([ex.input_ids for ex in examples], dtype=torch.long),
        batch_size=3,
        device="cpu",
    )
    assert abs(full_lm - reference) > 1e-3
    # The training objective at theta_0 is the same masked mean: with one
    # batch holding all examples, step 1's logged train_loss_nats == reference
    # (the permutation cannot matter — the batch is the whole set).
    fresh = tiny_llama(seed=0)
    _train(fresh, examples, examples, tmp_path, batch_size=3, max_steps=1, eval_every=100)
    step1 = _read_jsonl(tmp_path / "train_log.jsonl")[0]
    assert abs(step1["train_loss_nats"] - reference) < 1e-4


# --------------------------------------------------------------------------
# V5.29 — full-coverage spans reproduce the full-LM loss
# --------------------------------------------------------------------------
def test_v5_29_full_span_reproduces_full_lm_loss(tiny_llama):
    # V5.29: spans covering every predictable position ((1, L)) on an
    # equal-length, pad-free batch => SFT loss == evaluate_nll_nats (the
    # pretrain-mode loss) on the same rows. Position 0 is a target in
    # neither: the causal shift only ever predicts tokens 1..L-1.
    model = tiny_llama(seed=0)
    rows = [
        [2, 4, 5, 6, 7, 8],
        [2, 9, 10, 11, 12, 13],
        [2, 14, 15, 16, 17, 18],
        [2, 19, 20, 21, 22, 23],
    ]
    examples = [TaskExample(input_ids=r, label_span=(1, len(r))) for r in rows]
    sft = evaluate_sft_nll_nats(model, examples, _FORMAT, batch_size=2, device="cpu")
    full = evaluate_nll_nats(
        model, torch.tensor(rows, dtype=torch.long), batch_size=2, device="cpu"
    )
    assert abs(sft - full) < 1e-6


# --------------------------------------------------------------------------
# V5.30 — padding and question positions contribute zero loss / zero gradient
# --------------------------------------------------------------------------
def test_v5_30_padding_contributes_zero_loss(tiny_llama):
    # V5.30 (loss half): on a ragged batch the set loss equals the
    # label-count-weighted combination of per-example, pad-free losses, and
    # is invariant to batch_size. A loss that counted padding positions
    # (mask or divisor) would break the decomposition.
    model = tiny_llama(seed=0)
    ex_short = TaskExample(input_ids=[2, 4, 5, 6, 7], label_span=(3, 5))  # 2 labels
    ex_long = TaskExample(
        input_ids=[2, 8, 9, 10, 11, 12, 13, 14, 15], label_span=(6, 9)
    )  # 3 labels; batching pads ex_short by 4 positions
    examples = [ex_short, ex_long]
    batched = evaluate_sft_nll_nats(model, examples, _FORMAT, batch_size=2, device="cpu")
    lone_short = evaluate_sft_nll_nats(model, [ex_short], _FORMAT, batch_size=1, device="cpu")
    lone_long = evaluate_sft_nll_nats(model, [ex_long], _FORMAT, batch_size=1, device="cpu")
    expected = (2 * lone_short + 3 * lone_long) / 5
    assert abs(batched - expected) < 1e-5
    one_by_one = evaluate_sft_nll_nats(model, examples, _FORMAT, batch_size=1, device="cpu")
    assert abs(one_by_one - batched) < 1e-5


def test_v5_30_padding_and_question_contribute_zero_gradient(tiny_llama):
    # V5.30 (gradient half): parameter gradients of the training objective on
    # a padded ragged batch equal those of a pad-free, label-positions-only
    # reference — so neither padding nor question tokens carry any gradient
    # signal from the loss labels. (Private objective by design: see module
    # docstring.)
    model = tiny_llama(seed=0)
    model.train()
    ex_short = TaskExample(input_ids=[2, 4, 5, 6, 7], label_span=(3, 5))
    ex_long = TaskExample(input_ids=[2, 8, 9, 10, 11, 12, 13, 14, 15], label_span=(6, 9))
    examples = [ex_short, ex_long]

    ids, mask = _padded_inputs_and_mask(examples, _FORMAT)
    loss = _mean_masked_ce_nats(model(ids).logits, ids, mask)
    loss.backward()
    lib_grads = {
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }
    model.zero_grad(set_to_none=True)

    # Reference: per-example pad-free forwards, mean over the 5 label terms.
    terms = []
    for ex in examples:
        ex_ids = torch.tensor([ex.input_ids], dtype=torch.long)
        log_probs = torch.log_softmax(model(ex_ids).logits, dim=-1)
        start, end = ex.label_span
        for p in range(start, end):
            terms.append(-log_probs[0, p - 1, ex.input_ids[p]])
    ref_loss = torch.stack(terms).mean()
    assert abs(loss.item() - ref_loss.item()) < 1e-5
    ref_loss.backward()
    ref_grads = {
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }

    assert set(lib_grads) == set(ref_grads)
    for name, lib_g in lib_grads.items():
        assert torch.allclose(lib_g, ref_grads[name], rtol=1e-4, atol=1e-6), name


# --------------------------------------------------------------------------
# V5.31 — span guard
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "span",
    [
        (0, 2),  # touches position 0: no predecessor under the causal shift
        (3, 3),  # empty: contributes no loss
        (4, 8),  # past the sequence end: would mark padding as labels
        (5, 3),  # reversed
    ],
)
def test_v5_31_invalid_label_span_raises(tiny_llama, tmp_path, span):
    # V5.31: malformed spans raise ValueError up front, before any training
    # or disk write, in both the evaluator and the trainer (train or val
    # side).
    model = tiny_llama(seed=0)
    good = TaskExample(input_ids=[2, 4, 5, 6, 7, 8], label_span=(4, 6))
    bad = TaskExample(input_ids=[2, 9, 10, 11, 12, 13], label_span=span)
    with pytest.raises(ValueError):
        evaluate_sft_nll_nats(model, [good, bad], _FORMAT, batch_size=2, device="cpu")
    with pytest.raises(ValueError):
        _train(model, [good, bad], [good], tmp_path, batch_size=2, max_steps=1)
    with pytest.raises(ValueError):
        _train(model, [good, good], [bad], tmp_path, batch_size=2, max_steps=1)
    assert list(tmp_path.iterdir()) == []  # refused before any disk write


# --------------------------------------------------------------------------
# V5.32 — determinism: identical logs (modulo time_unix) for identical seed
# --------------------------------------------------------------------------
def test_v5_32_same_seed_identical_logs(tiny_llama, tmp_path):
    # V5.32: identical seed + inputs on CPU (fixed threads) -> identical
    # train_log.jsonl and eval_log.jsonl across two independent train_sft
    # runs (identical loss sequence included), after deleting the time_unix
    # field from every record.
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = random.Random(7)
        train_examples = [
            _example(rng, rng.randrange(2, 6), rng.randrange(1, 4)) for _ in range(12)
        ]
        val_examples = [_example(rng, 3, 2) for _ in range(4)]
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        # A fresh model per run: train_sft mutates the model in place, so both
        # runs must start from the identical random init (same fixture seed).
        _train(
            tiny_llama(seed=0),
            train_examples,
            val_examples,
            dir_a,
            seed=123,
            max_steps=15,
            eval_every=5,
        )
        _train(
            tiny_llama(seed=0),
            train_examples,
            val_examples,
            dir_b,
            seed=123,
            max_steps=15,
            eval_every=5,
        )
        for name in ("train_log.jsonl", "eval_log.jsonl"):
            recs_a = [
                {k: v for k, v in r.items() if k != "time_unix"} for r in _read_jsonl(dir_a / name)
            ]
            recs_b = [
                {k: v for k, v in r.items() if k != "time_unix"} for r in _read_jsonl(dir_b / name)
            ]
            assert recs_a == recs_b
    finally:
        torch.set_num_threads(prev_threads)


# --------------------------------------------------------------------------
# V5.37 — wall-clock timestamps in every log record (SFT mode)
# --------------------------------------------------------------------------
def test_v5_37_time_unix_present_nondecreasing_sft(tiny_llama, tmp_path):
    # V5.37: train_sft's train and eval records carry time_unix; within each
    # log file the values are nondecreasing and lie between wall-clock
    # readings taken immediately before and after the run.
    rng = random.Random(11)
    train_examples = [_example(rng, 3, 2) for _ in range(8)]
    val_examples = [_example(rng, 3, 2) for _ in range(4)]
    t_before = time.time()
    _train(tiny_llama(seed=0), train_examples, val_examples, tmp_path, eval_every=3, max_steps=7)
    t_after = time.time()
    for name in ("train_log.jsonl", "eval_log.jsonl"):
        times = [r["time_unix"] for r in _read_jsonl(tmp_path / name)]
        assert times  # both logs are non-empty at this cadence
        assert all(t_before <= t <= t_after for t in times)
        assert all(a <= b for a, b in zip(times, times[1:]))


# --------------------------------------------------------------------------
# V5.33 — convergence on a memorizable question->answer task
# --------------------------------------------------------------------------
def test_v5_33_sft_converges_on_memorizable_answers(tiny_llama, tmp_path):
    # V5.33: four fixed question->answer pairs; the model memorizes the
    # answers, validation (same pairs) plateaus => stop_reason "converged"
    # before a generous max_steps, and final train loss < initial.
    patterns = [
        TaskExample(input_ids=[2, 4, 5, 6, 10, 11], label_span=(4, 6)),
        TaskExample(input_ids=[2, 12, 13, 14, 20, 21], label_span=(4, 6)),
        TaskExample(input_ids=[2, 22, 23, 24, 30, 31], label_span=(4, 6)),
        TaskExample(input_ids=[2, 32, 33, 34, 40, 41], label_span=(4, 6)),
    ]
    train_examples = patterns * 12  # 48 rows
    val_examples = patterns * 4  # 16 rows
    model = tiny_llama(seed=0)
    result = _train(
        model,
        train_examples,
        val_examples,
        tmp_path,
        lr=1e-2,
        batch_size=16,
        stopping=StoppingRule(eps_nats=1e-3, k=5),
        eval_every=5,
        max_steps=500,
    )
    assert result.stop_reason == "converged"
    assert result.final_step < 500
    train_recs = _read_jsonl(tmp_path / "train_log.jsonl")
    assert train_recs[-1]["train_loss_nats"] < train_recs[0]["train_loss_nats"]
    # V5.41 propagation through train_sft: a plateau stop is exactly where
    # best is eps-gated, so min must equal the true min of the logged evals
    # (and land in training_meta.json).
    eval_recs = _read_jsonl(tmp_path / "eval_log.jsonl")
    assert result.min_val_nats == min(r["val_loss_nats"] for r in eval_recs)
    meta = json.loads((tmp_path / "training_meta.json").read_text())
    assert meta["min_val_nats"] == result.min_val_nats


# --------------------------------------------------------------------------
# V5.45 — behavioral stopping mode (runs 3-4, specs/02 §6)
# --------------------------------------------------------------------------
def _scripted_eval(rates):
    """Callback returning ``rates`` in order, recording what it returned."""
    calls: list[float] = []

    def cb() -> float:
        r = rates[len(calls)]
        calls.append(r)
        return r

    cb.calls = calls
    return cb


def test_v5_45_behavioral_stop_at_kth_consecutive(tiny_llama, copy_token_task, tmp_path):
    # V5.45: the run stops exactly at the eval completing the k-th
    # CONSECUTIVE rate >= threshold (a mid-run miss resets), with
    # stop_reason "behavior". Every eval logs BOTH val_loss_nats (the
    # "for the record" loss) and format_valid_rate; best/min report the
    # exact running min (no eps gate in this mode) and the meta echoes the
    # behavioral rule.
    model = tiny_llama(seed=1)
    examples = copy_token_task(seed=0, n=20)
    cb = _scripted_eval([0.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0])
    result = _train(
        model,
        examples[:16],
        examples[16:],
        tmp_path,
        stopping=BehavioralStoppingRule(threshold=0.99, k=3),
        behavioral_eval=cb,
        eval_every=5,
        max_steps=100,
    )
    assert result.stop_reason == "behavior"
    assert result.final_step == 35  # the 7th eval ends the 3-hit run
    assert len(cb.calls) == 7
    evals = _read_jsonl(tmp_path / "eval_log.jsonl")
    assert [e["format_valid_rate"] for e in evals] == cb.calls
    assert all("val_loss_nats" in e for e in evals)
    assert result.min_val_nats == min(e["val_loss_nats"] for e in evals)
    assert result.best_val_nats == result.min_val_nats
    meta = json.loads((tmp_path / "training_meta.json").read_text())
    assert meta["config"]["stopping"] == {"threshold": 0.99, "k": 3}


def test_v5_45_ceiling_exit_when_format_never_installs(tiny_llama, copy_token_task, tmp_path):
    # V5.45: sub-threshold rates never stop the run — the loss tracker is
    # deliberately not consulted in behavioral mode — so the run exits at
    # the ceiling with stop_reason "max_steps" ("format never installed").
    model = tiny_llama(seed=2)
    examples = copy_token_task(seed=1, n=20)
    cb = _scripted_eval([0.0] * 50)
    result = _train(
        model,
        examples[:16],
        examples[16:],
        tmp_path,
        stopping=BehavioralStoppingRule(threshold=0.99, k=3),
        behavioral_eval=cb,
        eval_every=5,
        max_steps=40,
    )
    assert result.stop_reason == "max_steps"
    assert result.final_step == 40
    assert len(cb.calls) == 8  # evals at 5,10,...,40


def test_v5_45_rule_and_callback_must_pair(tiny_llama, copy_token_task, tmp_path):
    # V5.45: a BehavioralStoppingRule without its callback (or the callback
    # with a loss rule) raises upfront, before any disk write.
    model = tiny_llama(seed=3)
    examples = copy_token_task(seed=2, n=8)
    with pytest.raises(ValueError):
        _train(
            model,
            examples[:4],
            examples[4:],
            tmp_path / "a",
            stopping=BehavioralStoppingRule(threshold=0.99, k=1),
        )
    with pytest.raises(ValueError):
        _train(
            model,
            examples[:4],
            examples[4:],
            tmp_path / "b",
            behavioral_eval=lambda: 1.0,
        )
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


# --------------------------------------------------------------------------
# V5.65 / V5.66 — train-loss stopping (new-phase dose installers, 2026-07-26)
# --------------------------------------------------------------------------
def test_v5_65_train_loss_stop_requires_full_dose_batch(tiny_llama, tmp_path):
    # V5.65: batch_size != n_train raises upfront — a subsampled per-step
    # loss would plateau on batch noise, not on dose absorption.
    model = tiny_llama(seed=0)
    rng = random.Random(0)
    examples = [_example(rng, 3, 2) for _ in range(4)]
    with pytest.raises(ValueError, match="batch_size == n_train"):
        _train(
            model,
            examples,
            [],
            tmp_path,
            batch_size=2,
            stopping=StoppingRule(eps_nats=0.0, k=3),
            stopping_metric="train_loss",
        )
    assert not (tmp_path / "train_log.jsonl").exists()


def test_v5_65_train_loss_stop_rejects_behavioral_rule(tiny_llama, tmp_path):
    model = tiny_llama(seed=0)
    rng = random.Random(0)
    examples = [_example(rng, 3, 2) for _ in range(2)]
    with pytest.raises(ValueError, match="train_loss"):
        _train(
            model,
            examples,
            examples,
            tmp_path,
            batch_size=2,
            stopping=BehavioralStoppingRule(threshold=0.9, k=3),
            behavioral_eval=lambda: 1.0,
            stopping_metric="train_loss",
        )


def test_v5_66_empty_val_requires_train_loss_metric(tiny_llama, tmp_path):
    # V5.66: only the train-loss mode runs without a validation split; the
    # legacy modes refuse an empty val upfront instead of dividing by zero
    # at the first eval.
    model = tiny_llama(seed=0)
    rng = random.Random(0)
    examples = [_example(rng, 3, 2) for _ in range(4)]
    with pytest.raises(ValueError, match="val_examples is empty"):
        _train(model, examples, [], tmp_path)


def test_v5_66_train_loss_plateau_stops_without_val(tiny_llama, tmp_path):
    # V5.66: eps so large that every post-first eval is non-improving — with
    # eval_every=1 and k=3 the run stops at exactly step 4 (one improving
    # eval from +inf, then three stale), converged, no val split anywhere.
    model = tiny_llama(seed=0)
    rng = random.Random(0)
    examples = [_example(rng, 3, 2) for _ in range(2)]
    result = _train(
        model,
        examples,
        [],
        tmp_path,
        batch_size=2,
        eval_every=1,
        max_steps=50,
        stopping=StoppingRule(eps_nats=100.0, k=3),
        stopping_metric="train_loss",
    )
    assert result.stop_reason == "converged"
    assert result.final_step == 4
    evals = _read_jsonl(tmp_path / "eval_log.jsonl")
    assert [e["step"] for e in evals] == [1, 2, 3, 4]
    assert all("train_loss_nats" in e and "val_loss_nats" not in e for e in evals)
    # The logged stopping metric is the step's own training loss, exactly.
    trains = _read_jsonl(tmp_path / "train_log.jsonl")
    for ev, tr in zip(evals, trains):
        assert ev["train_loss_nats"] == tr["train_loss_nats"]
    assert result.min_val_nats == min(e["train_loss_nats"] for e in evals)


def test_v5_66_train_loss_converges_on_memorizable_dose(tiny_llama, tmp_path):
    # V5.66: a 1-example dose memorizes — training loss falls and the plateau
    # rule ends the run before a generous ceiling ("the dose is absorbed").
    model = tiny_llama(seed=1)
    examples = [TaskExample(input_ids=[2, 4, 5, 6, 7, 8], label_span=(4, 6))]
    result = _train(
        model,
        examples,
        [],
        tmp_path,
        lr=1e-2,
        batch_size=1,
        eval_every=1,
        max_steps=400,
        stopping=StoppingRule(eps_nats=0.002, k=5),
        stopping_metric="train_loss",
    )
    assert result.stop_reason == "converged"
    assert result.final_step < 400
    trains = _read_jsonl(tmp_path / "train_log.jsonl")
    assert trains[-1]["train_loss_nats"] < trains[0]["train_loss_nats"]


# --------------------------------------------------------------------------
# Loud-failure guards: too few train rows for one batch; unknown stopping
# metric. Both fire before any training or disk write.
# --------------------------------------------------------------------------
def test_train_examples_below_batch_size_raises(tiny_llama, tmp_path):
    model = tiny_llama(seed=0)
    rng = random.Random(0)
    examples = [_example(rng, 3, 2) for _ in range(2)]
    with pytest.raises(ValueError, match="fewer than batch_size"):
        _train(model, examples, examples, tmp_path, batch_size=4)
    assert not (tmp_path / "train_log.jsonl").exists()


def test_unknown_stopping_metric_raises(tiny_llama, tmp_path):
    model = tiny_llama(seed=0)
    rng = random.Random(0)
    examples = [_example(rng, 3, 2) for _ in range(4)]
    with pytest.raises(ValueError, match="unknown stopping_metric"):
        _train(model, examples, examples, tmp_path, stopping_metric="bogus")
    assert not (tmp_path / "train_log.jsonl").exists()
