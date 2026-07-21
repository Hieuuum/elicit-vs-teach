"""ZOO-2 tests: run-log records, invariants, consistency (``geode.zoo``).

Derived from specs/00-interfaces.md — §1 "Directory layout", §3 "Prequential
log", §4 "Gradient statistics", §5 "Test loss", §8 (``prequential_records``,
``test_loss``) — validation properties V0.3 and V0.5, and resolved decisions
(specs/00 §9) OQ-4 (``example_ids`` universe is ``0..n_unique_examples-1`` from
the run's manifest; checker rejects skips, repeats, out-of-range), OQ-7
(``masking_config_hash`` is a sha256 hex digest) and OQ-14
(``topk_grad_subspace_overlap`` is always ``null`` for now; ``float|null``
by schema).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geode.zoo import (
    GradStatRecord,
    PrequentialRecord,
    TestLoss,
    check_epoch1_coverage,
    check_masking_consistency,
    gradstat_records,
    prequential_records,
    register_run,
    test_loss,
    write_jsonl,
)

# ---------------------------------------------------------------------------
# Helpers: a registered run under the §1 layout (manifest builder adapted
# from tests/zoo/test_manifest.py), plus record/file constructors.
# ---------------------------------------------------------------------------

# OQ-4: the example_ids universe for all coverage tests is 0..N_UNIQUE-1.
N_UNIQUE = 12

# OQ-7: masking_config_hash is a sha256 hex digest — 64 hex chars.
EVAL_HASH = "a1" * 32
TRAIN_HASH = "b2" * 32

PREQUENTIAL_KEYS = {"step", "epoch", "example_ids", "label_token_count", "loss_sum_nats"}
GRADSTAT_KEYS = {"step", "global_grad_norm", "per_module_grad_norm", "topk_grad_subspace_overlap"}


def _valid_manifest(run_id: str, n_unique_examples: int) -> dict:
    """A fully valid spec 00 §2 manifest."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-11T00:00:00+00:00",
        "git_commit": "0123456789abcdef0123456789abcdef01234567",
        "regime": "elicit",
        "base_model": {"hf_id": "meta-llama/Llama-3.2-1B", "revision": "main"},
        "task": {"name": "task_a", "format_version": "1"},
        "dataset": {"name": "dataset_a", "n_unique_examples": n_unique_examples, "seed": 0},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 8,
                "alpha": 16.0,
                "target_modules": ["q_proj", "v_proj"],
                "dropout": 0.05,
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "adamw",
                "lr": 1e-4,
                "batch_size": 8,
                "micro_batch_size": None,
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "grad_clip": None,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "fp32",
            "eval_every": None,
            "max_steps": None,
            "stopping": {"eps_nats": None, "k": None},
            "epochs_total": 3,
            "seed": 0,
        },
        "trainable_param_count": 4096,
        "snapshot_steps": [0, 8, 64],
        "cost": {"gpu_type": "A100", "est_usd": 1.5, "actual_usd": None},
        "status": "complete",
    }


def _register(geode_store: Path, run_id: str, n_unique_examples: int = N_UNIQUE) -> Path:
    """Register a valid run via the ZOO-1 API; return its directory (§1 layout)."""
    register_run(_valid_manifest(run_id, n_unique_examples))
    run_dir = geode_store / "runs" / run_id
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "eval").mkdir(exist_ok=True)
    return run_dir


def _preq(step: int, epoch: int, example_ids: list[int]) -> PrequentialRecord:
    """A well-formed spec 00 §3 record; only step/epoch/example_ids matter here."""
    return PrequentialRecord(
        step=step,
        epoch=epoch,
        example_ids=example_ids,
        label_token_count=3 * len(example_ids),
        loss_sum_nats=1.5,
    )


def _write_test_loss(run_dir: Path, masking_config_hash: str) -> dict:
    """Write eval/test_loss.json exactly per spec 00 §5 and return the payload."""
    payload = {
        "n_test_examples": 64,
        "label_token_count": 256,
        "loss_sum_nats": 88.0,
        "loss_per_label_token_nats": 0.34375,  # 88.0 / 256, exactly representable
        "masking_config_hash": masking_config_hash,
    }
    (run_dir / "eval" / "test_loss.json").write_text(json.dumps(payload))
    return payload


# ---------------------------------------------------------------------------
# V0.3 + OQ-4 — epoch-1 coverage checker rejects skips, out-of-range, repeats
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "epoch1_ids,offending",
    [
        pytest.param([0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11], "7", id="skip"),
        pytest.param([*range(N_UNIQUE), 99], "99", id="out-of-range-high"),
        pytest.param([*range(N_UNIQUE), -3], "-3", id="negative"),
    ],
)
def test_epoch1_ids_skip_rejected(epoch1_ids: list[int], offending: str) -> None:
    """V0.3 + OQ-4: epoch-1 ids that skip an example or fall outside
    0..n_unique_examples-1 are rejected, with an error naming the offending id.
    """
    mid = len(epoch1_ids) // 2
    records = [
        _preq(0, 1, epoch1_ids[:mid]),
        _preq(1, 1, epoch1_ids[mid:]),
        # An epoch-2 record containing the skipped id must NOT repair epoch-1
        # coverage (spec §3: later epochs are never used for MDL).
        _preq(2, 2, [7, 0]),
    ]
    with pytest.raises(Exception) as excinfo:
        check_epoch1_coverage(records, N_UNIQUE)
    assert offending in str(excinfo.value)


def test_epoch1_ids_repeat_rejected() -> None:
    """V0.3: a repeated id in epoch 1 is rejected even when every id is covered."""
    records = [
        _preq(0, 1, [0, 1, 2, 3, 4, 5]),
        _preq(1, 1, [6, 7, 8, 9, 10, 11, 7]),  # id 7 appears twice
    ]
    with pytest.raises(Exception) as excinfo:
        check_epoch1_coverage(records, N_UNIQUE)
    assert "7" in str(excinfo.value)


def test_epoch1_ids_exact_cover_accepted() -> None:
    """V0.3: unordered exact cover across several epoch-1 records is accepted.

    Spec §3 requires that concatenated epoch-1 ``example_ids`` enumerate each
    unique example exactly once — not that they be sorted. Records from
    epoch >= 2 (with repeated/arbitrary in-range ids) are ignored by the
    epoch-1 invariant.
    """
    records = [
        _preq(0, 1, [4, 9, 0]),
        _preq(1, 1, [11, 2]),
        _preq(2, 1, [7, 5, 10, 1]),
        _preq(3, 1, [3, 8, 6]),
        _preq(4, 2, [0, 0, 5]),  # later epochs revisit examples freely
        _preq(5, 2, [11]),
        _preq(6, 3, [7, 7, 7]),
    ]
    check_epoch1_coverage(iter(records), N_UNIQUE)  # must not raise


# ---------------------------------------------------------------------------
# V0.5 — masking_config_hash consistency between train log and test loss
# ---------------------------------------------------------------------------


def test_masking_hash_mismatch_raises(geode_store: Path) -> None:
    """V0.5: hash mismatch is surfaced as an error naming a hash; match passes."""
    run_dir = _register(geode_store, "run-mask")
    _write_test_loss(run_dir, masking_config_hash=EVAL_HASH)

    with pytest.raises(Exception) as excinfo:
        check_masking_consistency("run-mask", TRAIN_HASH)
    msg = str(excinfo.value)
    assert EVAL_HASH in msg or TRAIN_HASH in msg

    # Matching hash: no raise. Explicit store= per the PLAN ZOO-2 signature.
    check_masking_consistency("run-mask", EVAL_HASH, store=geode_store)


# ---------------------------------------------------------------------------
# §3 — prequential.jsonl roundtrip, field names byte-match the spec
# ---------------------------------------------------------------------------


def test_prequential_jsonl_roundtrip(geode_store: Path) -> None:
    run_dir = _register(geode_store, "run-preq")
    records = [
        PrequentialRecord(
            step=0, epoch=1, example_ids=[3, 0], label_token_count=7, loss_sum_nats=3.25
        ),
        PrequentialRecord(
            step=1, epoch=1, example_ids=[2, 1], label_token_count=5, loss_sum_nats=1.5
        ),
        PrequentialRecord(
            step=2, epoch=2, example_ids=[0, 3], label_token_count=7, loss_sum_nats=0.75
        ),
    ]
    path = run_dir / "logs" / "prequential.jsonl"
    write_jsonl(path, records)

    # On disk: one JSON object per line with exactly the spec §3 keys.
    lines = path.read_text().splitlines()
    assert len(lines) == len(records)
    for line, rec in zip(lines, records, strict=True):
        obj = json.loads(line)
        assert set(obj) == PREQUENTIAL_KEYS
        assert obj == {
            "step": rec.step,
            "epoch": rec.epoch,
            "example_ids": rec.example_ids,
            "label_token_count": rec.label_token_count,
            "loss_sum_nats": rec.loss_sum_nats,
        }

    got = list(prequential_records("run-preq"))
    assert all(isinstance(r, PrequentialRecord) for r in got)
    assert got == records


# ---------------------------------------------------------------------------
# §5 — eval/test_loss.json schema read back exactly, with exact types
# ---------------------------------------------------------------------------


def test_test_loss_schema_roundtrip(geode_store: Path) -> None:
    run_dir = _register(geode_store, "run-testloss")
    payload = _write_test_loss(run_dir, masking_config_hash=EVAL_HASH)

    tl = test_loss("run-testloss")
    assert isinstance(tl, TestLoss)
    assert tl == TestLoss(
        n_test_examples=payload["n_test_examples"],
        label_token_count=payload["label_token_count"],
        loss_sum_nats=payload["loss_sum_nats"],
        loss_per_label_token_nats=payload["loss_per_label_token_nats"],
        masking_config_hash=payload["masking_config_hash"],
    )
    assert type(tl.n_test_examples) is int
    assert type(tl.label_token_count) is int
    assert type(tl.loss_sum_nats) is float
    assert type(tl.loss_per_label_token_nats) is float
    assert type(tl.masking_config_hash) is str


# ---------------------------------------------------------------------------
# §4 + OQ-14 — gradstats roundtrip; topk_grad_subspace_overlap is float|null
# ---------------------------------------------------------------------------


def test_gradstats_overlap_may_be_null(geode_store: Path) -> None:
    run_dir = _register(geode_store, "run-grad")
    records = [
        GradStatRecord(
            step=0,
            global_grad_norm=1.5,
            per_module_grad_norm={"model.layers.0.q_proj": 0.5, "model.layers.1.v_proj": 1.25},
            topk_grad_subspace_overlap=None,  # OQ-14: always null for now
        ),
        GradStatRecord(
            step=8,
            global_grad_norm=0.5,
            per_module_grad_norm={"model.layers.0.q_proj": 0.25},
            topk_grad_subspace_overlap=0.75,  # schema §4 also allows float
        ),
    ]
    path = run_dir / "logs" / "gradstats.jsonl"
    write_jsonl(path, records)

    # On disk: exactly the spec §4 keys; None serialized as JSON null.
    objs = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(objs) == len(records)
    assert all(set(obj) == GRADSTAT_KEYS for obj in objs)
    assert objs[0]["topk_grad_subspace_overlap"] is None
    assert objs[1]["topk_grad_subspace_overlap"] == 0.75

    got = list(gradstat_records("run-grad"))
    assert all(isinstance(r, GradStatRecord) for r in got)
    assert got == records
    assert got[0].topk_grad_subspace_overlap is None
    assert isinstance(got[1].topk_grad_subspace_overlap, float)
    assert got[0].per_module_grad_norm == records[0].per_module_grad_norm


# ---------------------------------------------------------------------------
# Acceptance: readers stream — iterators, no full-file loads
# ---------------------------------------------------------------------------


def test_prequential_records_streams_lazily(geode_store: Path) -> None:
    """The first record is yielded before later lines are parsed/validated.

    A malformed line placed AFTER a valid record must not prevent pulling
    that record from the iterator (pins streaming without over-specifying
    internals).
    """
    run_dir = _register(geode_store, "run-lazy")
    first = {"step": 0, "epoch": 1, "example_ids": [0], "label_token_count": 3,
             "loss_sum_nats": 2.5}  # fmt: skip
    path = run_dir / "logs" / "prequential.jsonl"
    path.write_text(json.dumps(first) + "\n" + "{this line is not valid json\n")

    it = prequential_records("run-lazy")
    rec = next(it)
    assert rec == PrequentialRecord(
        step=0, epoch=1, example_ids=[0], label_token_count=3, loss_sum_nats=2.5
    )


def test_gradstat_records_streams_lazily(geode_store: Path) -> None:
    """Same laziness pin for the §4 reader — the ZOO-2 acceptance criterion
    says *readers* stream (iterators, no full-file loads), plural."""
    run_dir = _register(geode_store, "run-lazy-grad")
    first = {"step": 0, "global_grad_norm": 1.5,
             "per_module_grad_norm": {"model.layers.0.q_proj": 0.5},
             "topk_grad_subspace_overlap": None}  # fmt: skip
    path = run_dir / "logs" / "gradstats.jsonl"
    path.write_text(json.dumps(first) + "\n" + "{this line is not valid json\n")

    it = gradstat_records("run-lazy-grad")
    rec = next(it)
    assert rec == GradStatRecord(
        step=0,
        global_grad_norm=1.5,
        per_module_grad_norm={"model.layers.0.q_proj": 0.5},
        topk_grad_subspace_overlap=None,
    )
