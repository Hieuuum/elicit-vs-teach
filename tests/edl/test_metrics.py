"""EDL-2 tests: prequential accumulator + EDL metrics.

Modules under test (do not exist yet): ``geode.edl.prequential`` (``StepLoss``,
``prequential_step``, ``PrequentialAccumulator``) and ``geode.edl.metrics``
(``mdl_nats``, ``edl_nats``, ``edl_per_label_token``, ``edl_per_param``,
``pgr``, ``training_curve``, ``nats_to_bits``).

Derived from specs, never from an implementation:

- specs/01-edl-harness.md §1 "Definitions" — MDL is the sum of ``loss_sum_nats``
  over **epoch-1** ``prequential.jsonl`` records; ``EDL = MDL − N_label · L_test``
  where ``N_label`` is the epoch-1 label-token count and ``L_test`` is
  ``eval/test_loss.json``'s ``loss_per_label_token_nats``; ``EDL/D = EDL/N_label``,
  ``EDL/P = EDL/trainable_param_count``; ``PGR = (P_tuned−P_base)/(P_fullft−P_base)``;
  units are nats internally, bits (÷ ln 2) at boundaries only.
- specs/01-edl-harness.md §2 "M2" — the accumulator is structurally incapable of
  adding epoch>1 records.
- specs/00-interfaces.md §3 — ``prequential.jsonl`` schema; §5 — ``test_loss.json``
  schema + the ``masking_config_hash`` train/test parity guard (V0.5, V1.4(a)).

Orchestrator-resolved decisions encoded here (BINDING):

- D-1: the training loop records its ``masking_config_hash`` as a top-level EXTRA
  field in ``manifest.json``; ``edl_nats`` enforces parity against
  ``eval/test_loss.json`` and raises ``geode.zoo.ConsistencyError`` on mismatch
  OR when the manifest lacks the field (cannot verify parity).
- D-3: ``training_curve`` — one row per epoch-1 record in stream order; columns
  exactly ``[example_index, loss_sum_nats, label_token_count,
  loss_per_label_token_nats]``; ``example_index`` is the 0-based index of the
  batch's FIRST example within the epoch-1 example stream (cumulative count of
  epoch-1 examples in prior batches); epoch>1 records excluded.
- D-6/D-8: ``pgr`` raises ``ValueError`` on a degenerate denominator.
- D-7: ``prequential_step`` — standard causal-LM shift; ``loss_sum_nats`` is the
  SUM over masked label positions of ``−log_softmax(logits[i, p−1])[ids[i][p]]``,
  in nats, under ``no_grad``; ``label_token_count == mask.sum()``; right-padding
  never contributes.
- D-8: ``PrequentialAccumulator.add_epoch1`` raises ``ValueError`` on epoch!=1.

Written test-first: the imports below target the API fixed in the retired build plan (git history) "### EDL-2"
and are expected to fail until the modules above exist.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from geode.edl import TaskFormat, label_mask
from geode.edl.metrics import (
    edl_nats,
    edl_per_label_token,
    edl_per_param,
    mdl_nats,
    nats_to_bits,
    pgr,
    training_curve,
)
from geode.edl.prequential import PrequentialAccumulator, StepLoss, prequential_step
from geode.zoo import ConsistencyError, PrequentialRecord, register_run, write_jsonl
from tests.conftest import BOS_ID, TaskExample

# Two distinct, well-formed sha256-shaped masking-config hashes (OQ-7 shape).
HASH_A = "a1" * 32
HASH_B = "b2" * 32


# ---------------------------------------------------------------------------
# Helpers: register a spec 00 §2 run and write its §3/§5 artifacts by hand.
# ---------------------------------------------------------------------------


def _manifest_fields(run_id: str, n_unique: int, *, extra: dict | None = None) -> dict:
    """A fully valid spec 00 §2 manifest; ``extra`` merges top-level extras (D-1)."""
    fields = {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-11T00:00:00+00:00",
        "git_commit": "deadbeef",
        "regime": "elicit",
        "base_model": {"hf_id": "tiny/llama", "revision": "main"},
        "task": {"name": "copy_token", "format_version": "1"},
        "dataset": {"name": "copy", "n_unique_examples": n_unique, "seed": 0},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 2,
                "alpha": 4.0,
                "target_modules": ["q_proj"],
                "dropout": 0.0,
                "sparse_param_count": None,
            },
            "optimizer": {"name": "sgd", "lr": 0.1, "batch_size": 2, "weight_decay": 0.0},
            "epochs_total": 1,
            "seed": 0,
        },
        "trainable_param_count": 1000,
        "snapshot_steps": [],
        "cost": {"gpu_type": None, "est_usd": None, "actual_usd": None},
        "status": "complete",
    }
    fields.update(extra or {})
    return fields


def _register(geode_store: Path, run_id: str, n_unique: int, *, extra: dict | None = None) -> Path:
    """Register a valid run (ZOO-1 API, $GEODE_STORE) and return its directory."""
    register_run(_manifest_fields(run_id, n_unique, extra=extra))
    return geode_store / "runs" / run_id


def _write_preq(run_dir: Path, records: list[PrequentialRecord]) -> None:
    """Write ``logs/prequential.jsonl`` via the ZOO-2 writer (creates parents)."""
    write_jsonl(run_dir / "logs" / "prequential.jsonl", records)


def _write_test_loss(run_dir: Path, *, mask_hash: str, l_test: float) -> dict:
    """Write ``eval/test_loss.json`` exactly per spec 00 §5 and return the payload."""
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_test_examples": 8,
        "label_token_count": 16,
        "loss_sum_nats": 16.0 * l_test,  # keeps per-token == l_test exactly
        "loss_per_label_token_nats": l_test,
        "masking_config_hash": mask_hash,
    }
    (eval_dir / "test_loss.json").write_text(json.dumps(payload))
    return payload


# ---------------------------------------------------------------------------
# V1.5 (structural half) — the accumulator is incapable of adding epoch>1 records
# ---------------------------------------------------------------------------


def test_accumulator_rejects_epoch_gt1() -> None:
    """V1.5 / M2: epoch-1 records accumulate; an epoch!=1 record (epoch 2, and
    epoch 0 per D-8's "epoch != 1") raises ValueError and leaves MDL untouched
    (the failed add must not corrupt state).
    """
    # V1.5 / M2 "structurally incapable": the constructor must be zero-arg —
    # injecting a record list would bypass add_epoch1 (the only entry point)
    # and let epoch>1 records corrupt mdl_nats().
    with pytest.raises(TypeError):
        PrequentialAccumulator(
            _records=[
                PrequentialRecord(
                    step=0, epoch=2, example_ids=[0], label_token_count=3, loss_sum_nats=9.0
                )
            ]
        )

    acc = PrequentialAccumulator()
    acc.add_epoch1(
        PrequentialRecord(step=0, epoch=1, example_ids=[0], label_token_count=3, loss_sum_nats=2.0)
    )
    acc.add_epoch1(
        PrequentialRecord(step=1, epoch=1, example_ids=[1], label_token_count=3, loss_sum_nats=1.5)
    )
    before = acc.mdl_nats()
    assert before == pytest.approx(3.5)

    with pytest.raises(ValueError):
        acc.add_epoch1(
            PrequentialRecord(
                step=2, epoch=2, example_ids=[0], label_token_count=3, loss_sum_nats=9.0
            )
        )
    # D-8 is epoch != 1, not epoch > 1: an epoch-0 record must be rejected too.
    with pytest.raises(ValueError):
        acc.add_epoch1(
            PrequentialRecord(
                step=2, epoch=0, example_ids=[0], label_token_count=3, loss_sum_nats=9.0
            )
        )
    # The rejected adds must not have mutated the accumulated MDL.
    assert acc.mdl_nats() == pytest.approx(before)


# ---------------------------------------------------------------------------
# V1.6 (recompute half) — accumulator MDL == recomputation from the flushed log
# ---------------------------------------------------------------------------


def test_mdl_equals_jsonl_recomputation(geode_store: Path) -> None:
    """V1.6: ``PrequentialAccumulator.mdl_nats()`` equals both the flushed
    ``prequential.jsonl`` recomputation and ``metrics.mdl_nats(run_id)``; the
    metrics reader sums epoch-1 records ONLY even when epoch>1 records are logged.
    """
    e1 = [
        PrequentialRecord(
            step=0, epoch=1, example_ids=[0, 1], label_token_count=4, loss_sum_nats=3.5
        ),
        PrequentialRecord(
            step=1, epoch=1, example_ids=[2, 3], label_token_count=6, loss_sum_nats=2.25
        ),
        PrequentialRecord(step=2, epoch=1, example_ids=[4], label_token_count=2, loss_sum_nats=1.0),
    ]
    e2 = [  # later-epoch records: logged (spec §3) but never counted toward MDL
        PrequentialRecord(
            step=3, epoch=2, example_ids=[0, 1], label_token_count=4, loss_sum_nats=9.0
        ),
        PrequentialRecord(
            step=4, epoch=2, example_ids=[2, 3], label_token_count=6, loss_sum_nats=8.0
        ),
    ]
    expected = 3.5 + 2.25 + 1.0  # epoch-1 loss mass only

    # The accumulator holds only epoch-1 records; flush persists them.
    acc = PrequentialAccumulator()
    for rec in e1:
        acc.add_epoch1(rec)
    assert acc.mdl_nats() == pytest.approx(expected)

    acc_dir = _register(geode_store, "run-acc", n_unique=5)
    acc.flush("run-acc")
    on_disk = [
        json.loads(line)
        for line in (acc_dir / "logs" / "prequential.jsonl").read_text().splitlines()
        if line.strip()
    ]
    manual = sum(o["loss_sum_nats"] for o in on_disk if o["epoch"] == 1)
    assert mdl_nats("run-acc") == pytest.approx(manual)
    assert mdl_nats("run-acc") == pytest.approx(acc.mdl_nats())

    # Metrics reader excludes epoch>1 records that appear in a real multi-epoch log.
    mixed_dir = _register(geode_store, "run-mixed", n_unique=5)
    _write_preq(mixed_dir, [*e1, *e2])
    assert mdl_nats("run-mixed") == pytest.approx(expected)
    # Exclusion is load-bearing: counting epoch-2 too would change the number.
    all_epochs = sum(r.loss_sum_nats for r in [*e1, *e2])
    assert mdl_nats("run-mixed") != pytest.approx(all_epochs)


# ---------------------------------------------------------------------------
# specs/01 §1 — EDL = MDL − N_label · L_test (headline metric algebra)
# ---------------------------------------------------------------------------


def test_edl_identity_mdl_minus_nlabel_times_testloss(geode_store: Path) -> None:
    """specs/01 §1: on constructed numbers, ``edl_nats`` equals the exact
    ``MDL − N_label · L_test`` identity to float tolerance.
    """
    records = [
        PrequentialRecord(
            step=0, epoch=1, example_ids=[0, 1], label_token_count=4, loss_sum_nats=5.0
        ),
        PrequentialRecord(
            step=1, epoch=1, example_ids=[2, 3], label_token_count=6, loss_sum_nats=3.0
        ),
        PrequentialRecord(step=2, epoch=1, example_ids=[4], label_token_count=2, loss_sum_nats=1.0),
    ]
    run_dir = _register(geode_store, "run-edl", n_unique=5, extra={"masking_config_hash": HASH_A})
    _write_preq(run_dir, records)
    _write_test_loss(run_dir, mask_hash=HASH_A, l_test=0.5)

    mdl = 9.0  # 5.0 + 3.0 + 1.0
    n_label = 12  # 4 + 6 + 2
    expected = mdl - n_label * 0.5  # = 3.0
    assert edl_nats("run-edl") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# V1.4(a) / V0.5 / D-1 — EDL refuses when train/test masks disagree
# ---------------------------------------------------------------------------


def test_edl_refuses_on_masking_hash_mismatch(geode_store: Path) -> None:
    """V1.4(a): ``edl_nats`` compares the manifest's ``masking_config_hash`` extra
    (D-1) with ``eval/test_loss.json``'s. Mismatch OR a missing manifest hash
    raises ``ConsistencyError``; equal hashes yield a float. Spec 01 §2 refuses
    "to compute EDL" generally, so the EDL-derived normalizations
    ``edl_per_label_token`` and ``edl_per_param`` must refuse identically.
    """
    records = [
        PrequentialRecord(step=0, epoch=1, example_ids=[0], label_token_count=2, loss_sum_nats=2.0),
    ]

    # (a) manifest hash != test-loss hash -> refuse (all EDL entry points).
    mism = _register(geode_store, "run-mismatch", n_unique=1, extra={"masking_config_hash": HASH_B})
    _write_preq(mism, records)
    _write_test_loss(mism, mask_hash=HASH_A, l_test=0.5)
    with pytest.raises(ConsistencyError):
        edl_nats("run-mismatch")
    with pytest.raises(ConsistencyError):
        edl_per_label_token("run-mismatch")
    with pytest.raises(ConsistencyError):
        edl_per_param("run-mismatch")

    # (b) equal hashes -> the exact §1 value: EDL = 2.0 − 2·0.5 = 1.0.
    ok = _register(geode_store, "run-match", n_unique=1, extra={"masking_config_hash": HASH_A})
    _write_preq(ok, records)
    _write_test_loss(ok, mask_hash=HASH_A, l_test=0.5)
    value = edl_nats("run-match")
    assert isinstance(value, float)
    assert value == pytest.approx(1.0)

    # (c) manifest lacks the hash extra -> cannot verify parity -> refuse.
    nohash = _register(geode_store, "run-nohash", n_unique=1)
    _write_preq(nohash, records)
    _write_test_loss(nohash, mask_hash=HASH_A, l_test=0.5)
    with pytest.raises(ConsistencyError):
        edl_nats("run-nohash")
    with pytest.raises(ConsistencyError):
        edl_per_label_token("run-nohash")
    with pytest.raises(ConsistencyError):
        edl_per_param("run-nohash")


# ---------------------------------------------------------------------------
# specs/01 §1 — EDL/D divides by N_label, EDL/P by trainable_param_count
# ---------------------------------------------------------------------------


def test_normalizations_per_token_per_param(geode_store: Path) -> None:
    """specs/01 §1: ``edl_per_label_token`` uses the epoch-1 ``N_label``;
    ``edl_per_param`` uses the manifest's ``trainable_param_count``. Constructed
    so the two denominators differ (10 vs 1000); both are checked exactly.
    """
    # Deliberately Σloss_sum_nats (8.0) != Σlabel_token_count (10): a wrong
    # denominator (dividing EDL by the epoch-1 loss mass instead of N_label,
    # or any field swap among 8.0 / 10 / 1000 / the §5 test token count 16)
    # cannot reproduce the expected values.
    records = [
        PrequentialRecord(
            step=0, epoch=1, example_ids=[0, 1], label_token_count=4, loss_sum_nats=5.0
        ),
        PrequentialRecord(
            step=1, epoch=1, example_ids=[2, 3], label_token_count=6, loss_sum_nats=3.0
        ),
    ]
    run_dir = _register(geode_store, "run-norm", n_unique=4, extra={"masking_config_hash": HASH_A})
    _write_preq(run_dir, records)
    _write_test_loss(run_dir, mask_hash=HASH_A, l_test=0.5)

    # MDL=8, N_label=10, L_test=0.5 -> EDL = 8 - 10*0.5 = 3.0; param count = 1000.
    edl = 3.0
    assert edl_nats("run-norm") == pytest.approx(edl)
    assert edl_per_label_token("run-norm") == pytest.approx(edl / 10)
    assert edl_per_param("run-norm") == pytest.approx(edl / 1000)


# ---------------------------------------------------------------------------
# V1.8 — reported bits = nats / ln 2
# ---------------------------------------------------------------------------


def test_bits_equal_nats_over_ln2() -> None:
    """V1.8: ``nats_to_bits(x) == x / ln 2`` for several values; ln(2) nats == 1 bit."""
    for x in (0.0, 1.0, 2.5, math.log(2.0), 42.0, 123.456):
        assert nats_to_bits(x) == pytest.approx(x / math.log(2.0))
    assert nats_to_bits(math.log(2.0)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# specs/01 §1 / D-6 — PGR formula and degenerate-denominator guard
# ---------------------------------------------------------------------------


def test_pgr_formula_and_degenerate_denominator() -> None:
    """specs/01 §1: ``pgr`` is exactly ``(P_tuned−P_base)/(P_fullft−P_base)``;
    D-6/D-8: a degenerate denominator (P_fullft ≈ P_base) raises ``ValueError``.
    """
    assert pgr(0.8, 0.5, 0.9) == pytest.approx((0.8 - 0.5) / (0.9 - 0.5))
    assert pgr(0.5, 0.5, 0.9) == pytest.approx(0.0)  # tuned == base -> no gain
    assert pgr(0.9, 0.5, 0.9) == pytest.approx(1.0)  # tuned == fullft -> full recovery
    with pytest.raises(ValueError):
        pgr(0.8, 0.5, 0.5)  # P_fullft == P_base: degenerate denominator
    with pytest.raises(ValueError):
        # Tiny but NONZERO denominator (~1e-15): must refuse rather than
        # return a huge, silently-meaningless ratio (exact-equality checks
        # would let this through).
        pgr(0.8, 0.5, 0.5 + 1e-15)


# ---------------------------------------------------------------------------
# D-3 — training_curve: epoch-1 loss vs cumulative example index, in order
# ---------------------------------------------------------------------------


def test_training_curve_epoch1_loss_vs_example_index(geode_store: Path) -> None:
    """D-3: one row per epoch-1 record in stream order; exact column names/order;
    ``example_index`` is the cumulative count of epoch-1 examples in prior
    batches; ``loss_per_label_token_nats = loss_sum_nats/label_token_count``;
    epoch>1 records (interleaved here) are excluded and never shift the offsets.
    """
    # Epoch-1 ``example_ids`` are deliberately NON-sequential (spec 00 §3 allows
    # any order, e.g. shuffled batches) so that ``example_index`` (a running
    # count) provably differs from a naive ``example_ids[0]`` read: the latter
    # would yield [5, 3, 2], the former the correct [0, 2, 3].
    records = [
        PrequentialRecord(
            step=0, epoch=1, example_ids=[5, 4], label_token_count=4, loss_sum_nats=6.0
        ),
        PrequentialRecord(
            step=1, epoch=2, example_ids=[5, 4], label_token_count=4, loss_sum_nats=99.0
        ),
        PrequentialRecord(step=2, epoch=1, example_ids=[3], label_token_count=3, loss_sum_nats=1.5),
        PrequentialRecord(
            step=3, epoch=1, example_ids=[2, 1, 0], label_token_count=6, loss_sum_nats=3.0
        ),
        PrequentialRecord(
            step=4, epoch=2, example_ids=[3, 2], label_token_count=4, loss_sum_nats=88.0
        ),
    ]
    run_dir = _register(geode_store, "run-curve", n_unique=6)
    _write_preq(run_dir, records)

    df = training_curve("run-curve")

    assert list(df.columns) == [
        "example_index",
        "loss_sum_nats",
        "label_token_count",
        "loss_per_label_token_nats",
    ]
    # Only the three epoch-1 records survive, in stream order.
    assert len(df) == 3
    # example_index = cumulative epoch-1 examples before each batch: 0, then 2,
    # then 2+1=3. The interleaved epoch-2 batch (2 examples) must NOT shift it.
    assert df["example_index"].tolist() == [0, 2, 3]
    assert df["loss_sum_nats"].tolist() == [6.0, 1.5, 3.0]
    assert df["label_token_count"].tolist() == [4, 3, 6]
    assert df["loss_per_label_token_nats"].tolist() == pytest.approx([6.0 / 4, 1.5 / 3, 3.0 / 6])


# ---------------------------------------------------------------------------
# D-7 — prequential_step sums masked label-token CE (nats), pre-update, no grad
# ---------------------------------------------------------------------------


def test_prequential_step_sums_masked_tokens_only_nats(tiny_llama) -> None:
    """D-7: ``prequential_step`` returns the SUM of ``−log_softmax(logits[i,p−1])
    [ids[i][p]]`` over masked positions in nats, computed under ``no_grad``.

    Compared to a manual causal-LM-shift reference on a right-padded batch of
    mixed-length examples (right-padding is False in the mask, so it never
    contributes). ``label_token_count`` equals the mask population; the returned
    loss is a plain float and no gradients are populated on the model.
    """
    model = tiny_llama(seed=0)
    batch = [
        TaskExample(input_ids=[BOS_ID, 5, 6, 7], label_span=(2, 4)),  # labels @2,3
        TaskExample(input_ids=[BOS_ID, 8, 9], label_span=(2, 3)),  # label @2; @3,4 pad
        TaskExample(input_ids=[BOS_ID, 10, 11, 12, 13], label_span=(1, 5)),  # labels @1-4
    ]
    task_format = TaskFormat(name="copy_token", format_version="1")
    mask = label_mask(batch, task_format)

    step = prequential_step(model, batch, mask)
    assert isinstance(step, StepLoss)

    # Manual reference: forward the right-padded batch, then gather label losses.
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids = torch.full((len(batch), max_len), model.config.pad_token_id, dtype=torch.long)
    for i, ex in enumerate(batch):
        ids = torch.tensor(ex.input_ids, dtype=torch.long)
        input_ids[i, : ids.shape[0]] = ids
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = torch.log_softmax(logits, dim=-1)
    expected_loss = 0.0
    expected_count = 0
    for i in range(len(batch)):
        for p in range(max_len):
            if bool(mask[i, p]):
                expected_loss += -log_probs[i, p - 1, int(input_ids[i, p])].item()
                expected_count += 1

    assert step.label_token_count == expected_count
    assert step.label_token_count == int(mask.sum().item())
    assert step.loss_sum_nats == pytest.approx(expected_loss, rel=1e-5, abs=1e-4)
    assert isinstance(step.loss_sum_nats, float)
    # Pre-update / no_grad contract: nothing accumulated gradients on the model.
    assert all(p.grad is None for p in model.parameters())
