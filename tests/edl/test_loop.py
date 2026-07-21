"""EDL-3 tests: prequential training-loop wrapper (``geode.edl.loop``).

Module under test (does not exist yet): ``geode.edl.loop`` providing

    def train_prequential(model, dataset, task_format: TaskFormat,
                          manifest: RunManifest, *, device: str,
                          seed: int, gradstats_stride: int = 1,
                          store: Path | None = None) -> None

Derived from specs, never from an implementation:

- specs/01-edl-harness.md §1 "Definitions" — θ₀ is the *pretrained* parameter
  state; the loss recorded for batch i is ℓ(θᵢ₋₁; xᵢ, yᵢ), i.e. computed with
  the parameters as they were BEFORE the update on batch i (V1.3); §3 final
  paragraph — the wrapper evaluates the pre-update loss, steps the optimizer,
  writes prequential + gradstat records, and saves snapshots per the
  manifest's ``snapshot_steps``; §4 V1.3/V1.6/V1.7; §5 non-goals (single
  device — every call here passes an explicit ``device="cpu"``).
- specs/00-interfaces.md §1 layout (``snapshots/step_{k}/``, ``logs/``,
  ``eval/``); §2 (``snapshot_steps`` declared up front); §3 prequential.jsonl
  schema + epoch-1 enumeration invariant; §4 gradstats.jsonl schema (OQ-14:
  overlap always null); §5 test_loss.json schema + ``masking_config_hash``
  parity (V0.5 producer side).
- the retired build plan (git history) "### EDL-3" and resolved decisions OQ-4 (example_ids universe),
  OQ-7 (hash construction), OQ-8 (span-carrying examples + shared
  ``label_mask``), OQ-14 (overlap null).

Test-writer-resolved decisions encoded here (BINDING for the IMPLEMENTER
unless the TEST-AUDITOR revises them; numbered L-* to avoid colliding with
EDL-2's D-* decisions):

- L-1: ``dataset`` is a dict with keys exactly ``train``, ``test``,
  ``tokenizer_hash``: the wrapper must write ``eval/test_loss.json`` (held-out
  data) and stamp ``masking_config_hash(task_format, tokenizer_hash)`` (OQ-7),
  and the PLAN-fixed signature offers no other slot for either input.
- L-2: the caller registers the run (ZOO-1) before calling; the loop resolves
  ``runs/{manifest.data["run_id"]}/`` under ``store``/``$GEODE_STORE`` and, per
  EDL-2's D-1, records its mask hash as a top-level EXTRA field
  ``masking_config_hash`` in ``manifest.json`` (the §5 parity counterpart).
- L-3: ``example_ids`` are 0-based indices into ``dataset["train"]`` (OQ-4:
  the id universe is ``0..n_unique_examples-1``). Tests reconstruct batches
  from the *logged* ids, so batch composition/order is NOT pinned beyond the
  §3 epoch-1 enumeration invariant and V1.7 determinism.
- L-4: ``step`` is a 0-based global counter across epochs (matches ZOO-2's
  fixtures, where an epoch-2 record continues the epoch-1 numbering); epochs
  are 1-based; one record per optimizer batch (spec 00 §3), so a full epoch
  over n examples at batch size b yields ceil(n/b) records.
- L-5 (revised 2026-07-18, arch-downscale decision): ``snapshots/step_{k}/``
  holds the complete model state — base + adapter tensors in one
  ``model.safetensors`` — after exactly k optimizer updates, equivalently the
  pre-update state for batch k. Loadable by rebuilding the module tree (base
  arch + LoRA wrap mirroring the manifest's ``training.lora``) and
  ``safetensors.torch.load_model``; no separately stored base checkpoint is
  needed. This is the convention that makes the snapshot at step k reproduce
  the loss recorded at step k, and ``step_0`` the initialization. Per spec 01
  §1 (θ₀ = the pretrained parameters), the freshly wrapped model at step 0
  must compute the same function as the unwrapped pretrained model.
- L-6: gradstats are logged at steps where ``step % gradstats_stride == 0``;
  ``topk_grad_subspace_overlap`` is always null (OQ-14).
- L-7: optimizer built from ``manifest.training.optimizer``; these tests
  require ``name="sgd"`` → plain SGD with the given lr/weight_decay (the
  wrapper may support more). The backward objective's reduction (mean vs sum
  over label tokens) is deliberately NOT pinned; fixture learning rates were
  verified to give per-step descent under both.
- L-8: ``training.epochs_total`` epochs are run; epoch-1 records are mandatory
  and later epochs are logged too, flagged by their ``epoch`` field (>1).

TEST-AUDITOR amendments (same binding force as L-*):

- A-1: ``manifest.snapshot_steps`` may include k == the total update count T;
  under L-5 that snapshot is the final state θ_T (taken after the last
  update). This is required to pin spec 01 §1's "Test loss: L_test(θ_T) =
  per-label-token loss of the FINAL model" implementation-free — see
  ``test_testloss_evaluated_at_final_state`` (no prior test failed when the
  test loss was evaluated at θ₀ or θ_{T-1}).
- A-2 (conformance follow-up): a declared ``snapshot_steps`` entry k > T can
  never be taken (under L-5/A-1 the reachable ks are exactly 0..T), so the
  loop must raise ``ValueError`` rather than silently skip it — spec 00 §2
  declares the schedule up front precisely because a missing checkpoint is
  "the expensive failure mode". See ``test_unreachable_snapshot_step_raises``.
- A-3 (conformance follow-up): spec 00 §4 keys ``per_module_grad_norm`` by
  ``module_name``. Parameter-tensor names (``....weight`` / ``....bias``) are
  not module names; keys must name modules on the adapted (``target_modules``)
  subtrees. Granularity below the adapted module (e.g. ``...q_proj`` vs
  ``...q_proj.lora_A.default``) is deliberately NOT pinned — the spec does not
  fix it. See ``test_gradstats_keys_are_modules_not_parameters``.

Written test-first: importing ``geode.edl.loop`` below is expected to fail
until EDL-3 is implemented. Every expected value is derived by hand or from an
independent evaluation path (pristine base model, reloaded snapshot + the
shared ``prequential_step``/``label_mask``), never from the loop itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_model

from geode.edl import TaskFormat, label_mask
from geode.edl.loop import train_prequential
from geode.edl.masking import masking_config_hash
from geode.edl.metrics import edl_nats
from geode.edl.prequential import prequential_step
from geode.zoo import (
    check_epoch1_coverage,
    check_masking_consistency,
    gradstat_records,
    load_run,
    prequential_records,
    register_run,
    test_loss,
)
from tests.conftest import BOS_ID, TaskExample

# OQ-7-shaped tokenizer hash (distinct from EDL-1/EDL-2 fixtures' a1/b2 hashes).
TOK_HASH = "c3" * 32

TASK_NAME = "copy_token"
FORMAT_VERSION = "1"

MODEL_SEED = 21  # base-model init (tiny_llama)
LOOP_SEED = 31  # train_prequential's explicit seed argument

PREQUENTIAL_KEYS = {"step", "epoch", "example_ids", "label_token_count", "loss_sum_nats"}
GRADSTAT_KEYS = {"step", "global_grad_norm", "per_module_grad_norm", "topk_grad_subspace_overlap"}


def _task_format() -> TaskFormat:
    """TaskFormat from name/format_version only (span-rule params defaulted, OQ-8)."""
    return TaskFormat(name=TASK_NAME, format_version=FORMAT_VERSION)


def _manifest_fields(
    run_id: str,
    *,
    n_unique: int,
    batch_size: int,
    lr: float,
    epochs_total: int = 1,
    snapshot_steps: tuple[int, ...] = (),
    seed: int = LOOP_SEED,
) -> dict:
    """A fully valid spec 00 §2 manifest driving the loop's training config."""
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_utc": "2026-07-12T00:00:00+00:00",
        "git_commit": "deadbeef",
        "regime": "elicit",
        "base_model": {"hf_id": "tiny/llama", "revision": "main"},
        "task": {"name": TASK_NAME, "format_version": FORMAT_VERSION},
        "dataset": {"name": "copy", "n_unique_examples": n_unique, "seed": 0},
        "training": {
            "method": "lora",
            "lora": {
                "rank": 2,
                "alpha": 4.0,
                "target_modules": ["q_proj", "v_proj"],
                "dropout": 0.0,  # keeps eval/train-mode numerics identical
                "sparse_param_count": None,
            },
            "optimizer": {
                "name": "sgd",  # L-7
                "lr": lr,
                "batch_size": batch_size,
                "micro_batch_size": None,
                "betas": None,
                "weight_decay": 0.0,
                "grad_clip": None,
            },
            "lr_schedule": "constant",
            "min_lr": None,
            "precision": "fp32",
            "eval_every": None,
            "max_steps": None,
            "stopping": {"eps_nats": None, "k": None},
            "epochs_total": epochs_total,
            "seed": seed,
        },
        "trainable_param_count": 1000,
        "snapshot_steps": list(snapshot_steps),
        "cost": {"gpu_type": None, "est_usd": None, "actual_usd": None},
        "status": "planned",
    }


def _dataset(train: list[TaskExample], test: list[TaskExample]) -> dict:
    """L-1: the dataset argument shape."""
    return {"train": train, "test": test, "tokenizer_hash": TOK_HASH}


def _varied_span_test_split() -> list[TaskExample]:
    """Held-out split with label token counts 2, 1, 3 (Σ = 6).

    6 label tokens != 3 examples != any train-set size used alongside it, so a
    field swap (n_test_examples vs label_token_count vs train sizes) cannot
    pass by coincidence.
    """
    return [
        TaskExample(input_ids=[BOS_ID, 5, 6, 7], label_span=(2, 4)),  # labels @2,3 -> 2
        TaskExample(input_ids=[BOS_ID, 8, 9], label_span=(2, 3)),  # label @2 -> 1
        TaskExample(input_ids=[BOS_ID, 10, 11, 12, 13], label_span=(2, 5)),  # labels @2-4 -> 3
    ]


def _batch_of(train: list[TaskExample], example_ids: list[int]) -> list[TaskExample]:
    """L-3: reconstruct a logged batch from its example ids."""
    return [train[i] for i in example_ids]


def _model_at_step(tiny_llama, run_dir: Path, step: int) -> PeftModel:
    """L-5: rebuild the loop's model state θ_step from its saved snapshot.

    ``tiny_llama(seed=MODEL_SEED)`` rebuilds the module tree and the LoRA wrap
    mirrors ``_manifest_fields``'s ``training.lora`` block (independent-path
    rule: derived from the fixture, not from the loop); every tensor value then
    comes from the snapshot file, so no separately stored base checkpoint is
    involved.
    """
    wrapped = get_peft_model(
        tiny_llama(seed=MODEL_SEED),
        LoraConfig(
            r=2,
            lora_alpha=4.0,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    load_model(wrapped, str(run_dir / "snapshots" / f"step_{step}" / "model.safetensors"))
    return wrapped


def _sorted_records(run_id: str):
    return sorted(prequential_records(run_id), key=lambda r: r.step)


# ---------------------------------------------------------------------------
# V1.3 — recorded losses are pre-update: two-pass snapshot reference
# ---------------------------------------------------------------------------


def test_preupdate_loss_matches_two_pass_reference(geode_store, tiny_llama, copy_token_task):
    """V1.3: the loss recorded at step s equals an explicit two-pass reference.

    Pass 1 runs the loop with a snapshot declared at *every* step, so
    ``snapshots/step_{s}`` captures θ_s (L-5). Pass 2 independently re-evaluates
    each logged batch with the shared ``prequential_step``/``label_mask`` on the
    reloaded snapshot. Equality for every record pins pre-update recording
    without replicating the optimizer or the backward objective in the test.

    Extra spec-§1 anchor: θ₀ is the *pretrained* state, so record 0 must also
    equal the evaluation under the pristine unwrapped base model (this alone
    catches post-update recording at step 0 and any adapter init that perturbs
    the pretrained function).
    """
    train = copy_token_task(seed=11, n=6)
    test = copy_token_task(seed=12, n=4)
    # 6 train examples / batch 2 -> steps 0,1,2; snapshot every step.
    manifest = register_run(
        _manifest_fields("run-ref", n_unique=6, batch_size=2, lr=0.05, snapshot_steps=(0, 1, 2))
    )
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    recs = _sorted_records("run-ref")
    assert [r.step for r in recs] == [0, 1, 2]
    assert all(r.epoch == 1 for r in recs)

    run_dir = geode_store / "runs" / "run-ref"
    for rec in recs:
        batch = _batch_of(train, rec.example_ids)
        mask = label_mask(batch, _task_format())
        # Bookkeeping half of V1.6 at the loop surface: the recorded count is
        # the mask population of the batch the loop says it trained on.
        assert rec.label_token_count == int(mask.sum().item())
        ref = prequential_step(_model_at_step(tiny_llama, run_dir, rec.step), batch, mask)
        assert rec.loss_sum_nats == pytest.approx(ref.loss_sum_nats, rel=1e-5, abs=1e-6), (
            f"step {rec.step}: recorded loss is not the pre-update loss"
        )

    # θ₀ anchor (spec 01 §1): batch-0 loss == pristine pretrained model's loss.
    batch0 = _batch_of(train, recs[0].example_ids)
    mask0 = label_mask(batch0, _task_format())
    base_ref = prequential_step(tiny_llama(seed=MODEL_SEED), batch0, mask0)
    assert recs[0].loss_sum_nats == pytest.approx(base_ref.loss_sum_nats, rel=1e-5, abs=1e-6)


def test_postupdate_corruption_yields_lower_mdl(geode_store, tiny_llama, copy_token_task):
    """V1.3: recording post-update losses systematically lowers MDL.

    The corrupted variant is built in the test from the loop's own snapshots:
    for batch s the post-update loss is the evaluation under θ_{s+1}
    (= ``snapshots/step_{s+1}``). Comparing partial sums over batches 0..n-2
    (so only declared snapshots are needed), the corrupted sum must be strictly
    below the recorded (pre-update) sum on this one-step-learnable copy task.

    Calibration (scratch reference, tiny_llama seed 21, LoRA r=2 on
    q_proj/v_proj, SGD lr=0.05): per-step descent ≈ 0.08–0.28 nats per batch,
    for both mean- and sum-reduction backward objectives — the 1e-3 margin
    below is >2 orders of magnitude smaller, so the direction check is robust
    but float noise cannot pass it.
    """
    train = copy_token_task(seed=13, n=8)
    test = copy_token_task(seed=12, n=4)
    # 8 train examples / batch 2 -> steps 0..3; snapshot every step.
    manifest = register_run(
        _manifest_fields(
            "run-corrupt", n_unique=8, batch_size=2, lr=0.05, snapshot_steps=(0, 1, 2, 3)
        )
    )
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    recs = _sorted_records("run-corrupt")
    assert len(recs) == 4

    run_dir = geode_store / "runs" / "run-corrupt"
    true_partial = sum(r.loss_sum_nats for r in recs[:-1])
    corrupted_partial = 0.0
    for rec in recs[:-1]:
        batch = _batch_of(train, rec.example_ids)
        mask = label_mask(batch, _task_format())
        post = prequential_step(_model_at_step(tiny_llama, run_dir, rec.step + 1), batch, mask)
        corrupted_partial += post.loss_sum_nats

    assert corrupted_partial < true_partial - 1e-3, (
        "post-update losses do not undercut pre-update losses: either the loop "
        "records post-update (both sums equal) or the task/lr cannot detect it"
    )


# ---------------------------------------------------------------------------
# V1.6 via V0.3 — the loop enumerates the epoch-1 stream exactly once
# ---------------------------------------------------------------------------


def test_epoch1_ids_pass_coverage_checker(geode_store, tiny_llama, copy_token_task):
    """V1.6 via V0.3: the loop's epoch-1 ``example_ids`` satisfy the ZOO-2
    coverage checker (each unique example exactly once, ids in
    ``0..n_unique-1``).

    7 examples with batch size 3 forces an UNEVEN final batch (3+3+1): a
    drop-last bug skips example ids and must be rejected by the checker.
    ``epochs_total=2`` checks that later epochs are logged too (flagged by
    their epoch field, L-8) with globally increasing steps (L-4) and never
    confuse the epoch-1 invariant.
    """
    train = copy_token_task(seed=11, n=7)
    test = copy_token_task(seed=12, n=4)
    manifest = register_run(
        _manifest_fields("run-cover", n_unique=7, batch_size=3, lr=0.01, epochs_total=2)
    )
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    recs = _sorted_records("run-cover")
    # L-4/L-8: one record per optimizer batch, ceil(7/3)=3 per epoch, two
    # epochs, 0-based global steps; epoch-2 records present and flagged.
    assert [r.step for r in recs] == [0, 1, 2, 3, 4, 5]
    assert [r.epoch for r in recs] == [1, 1, 1, 2, 2, 2]

    # V0.3: must not raise (skips/repeats/out-of-range all raise in ZOO-2).
    check_epoch1_coverage(prequential_records("run-cover"), 7)

    # Later epochs revisit examples freely but stay in the OQ-4 universe.
    for rec in recs:
        assert all(0 <= i < 7 for i in rec.example_ids)

    # On disk, the loop-written log is one JSON object per line with exactly
    # the spec 00 §3 keys — the ZOO-2 reader is not guaranteed to reject a
    # loop that smuggles extra (or renames) fields, so pin the producer here.
    lines = (
        (geode_store / "runs" / "run-cover" / "logs" / "prequential.jsonl").read_text().splitlines()
    )
    objs = [json.loads(line) for line in lines if line.strip()]
    assert len(objs) == 6
    assert all(set(obj) == PREQUENTIAL_KEYS for obj in objs)


# ---------------------------------------------------------------------------
# V1.6 — Σ label_token_count over epoch 1 == tokenizer-derived label count
# ---------------------------------------------------------------------------


def test_label_token_sum_matches_tokenizer_count(geode_store, tiny_llama, tiny_tokenizer):
    """V1.6: the loop's summed ``label_token_count`` equals the label-token
    count derived independently by tokenizing the label strings.

    Dataset built through the tiny word-level tokenizer (one word = one token,
    ``t{i}`` -> id 4+i): label word counts 1+2+3+1+2 = 9. Non-coincidence: 9
    differs from the 5 examples, the 3 optimizer batches (batch size 2), the
    10 prompt tokens, and the 24 total tokens (9 labels + 10 prompt + 5 BOS) —
    a count/field swap in the loop cannot reproduce 9.
    """
    tokenizer = tiny_tokenizer()
    specs = [  # (prompt words, label words): 2/1, 1/2, 2/3, 3/1, 2/2
        ("t1 t2", "t3"),
        ("t4", "t5 t6"),
        ("t7 t8", "t9 t10 t11"),
        ("t12 t13 t14", "t15"),
        ("t16 t17", "t18 t19"),
    ]
    train: list[TaskExample] = []
    expected_label_tokens = 0
    for prompt, label in specs:
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        l_ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        start = 1 + len(p_ids)  # BOS + prompt tokens precede the label span
        train.append(
            TaskExample(input_ids=[BOS_ID, *p_ids, *l_ids], label_span=(start, start + len(l_ids)))
        )
        expected_label_tokens += len(l_ids)
    assert expected_label_tokens == 9  # hand check of the fixture arithmetic

    manifest = register_run(_manifest_fields("run-labelsum", n_unique=5, batch_size=2, lr=0.01))
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, _varied_span_test_split()),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    epoch1 = [r for r in prequential_records("run-labelsum") if r.epoch == 1]
    assert epoch1, "epoch-1 records are mandatory (spec 00 §3)"
    assert all(isinstance(r.label_token_count, int) and r.label_token_count > 0 for r in epoch1)
    assert sum(r.label_token_count for r in epoch1) == expected_label_tokens


# ---------------------------------------------------------------------------
# V1.7 — determinism: same seed + config twice ⇒ byte-identical logs
# ---------------------------------------------------------------------------


def test_same_seed_twice_identical_prequential_log(geode_store, tiny_llama, copy_token_task):
    """V1.7: two runs with the same seed and config produce byte-equal
    ``prequential.jsonl`` (CPU, fixed threads). The run_id differs — nothing
    about the log may depend on it. Two epochs so the equality also covers
    later-epoch records; gradstats byte-equality asserted as a secondary check
    (same determinism contract, default stride).

    The ambient global RNG state is deliberately scrambled DIFFERENTLY before
    each run (after building the identical base model): a loop that leaks any
    randomness from ambient state instead of deriving it all from its explicit
    ``seed`` argument (e.g. adapter init) produces divergent logs and fails
    here — verified by mutation against an unseeded reference loop.
    """
    train = copy_token_task(seed=11, n=6)
    test = copy_token_task(seed=12, n=4)

    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for run_id, scramble in (("run-det-a", 12345), ("run-det-b", 99999)):
            # manifest training.seed matches the explicit seed argument below:
            # the manifest is the provenance record of the run, and a loop that
            # validates their consistency must not be failed by this fixture.
            manifest = register_run(
                _manifest_fields(run_id, n_unique=6, batch_size=2, lr=0.05, epochs_total=2, seed=17)
            )
            # A fresh, identically seeded base model per run: training mutates
            # the wrapped model, so the object is never reused.
            model = tiny_llama(seed=MODEL_SEED)
            torch.manual_seed(scramble)  # divergent ambient state, see above
            torch.rand(3)
            train_prequential(
                model,
                _dataset(train, test),
                _task_format(),
                manifest,
                device="cpu",
                seed=17,
            )
    finally:
        torch.set_num_threads(n_threads)

    log_a = (geode_store / "runs" / "run-det-a" / "logs" / "prequential.jsonl").read_bytes()
    log_b = (geode_store / "runs" / "run-det-b" / "logs" / "prequential.jsonl").read_bytes()
    assert len(log_a) > 0
    assert log_a == log_b

    grad_a = (geode_store / "runs" / "run-det-a" / "logs" / "gradstats.jsonl").read_bytes()
    grad_b = (geode_store / "runs" / "run-det-b" / "logs" / "gradstats.jsonl").read_bytes()
    assert grad_a == grad_b


# ---------------------------------------------------------------------------
# spec 00 §1–§2 — snapshots at exactly the declared steps
# ---------------------------------------------------------------------------


def test_snapshots_written_at_declared_steps(geode_store, tiny_llama, copy_token_task):
    """Snapshots exist for exactly ``manifest.snapshot_steps`` — no more, no
    fewer (spec 00 §2: the schedule is declared up front because rerunning
    training to recover a missing checkpoint is the expensive failure mode).
    Declared steps are non-contiguous (0, 2, 3 of steps 0..3) so both a
    skipped declared step and an extra undeclared one (step_1) are caught.
    """
    train = copy_token_task(seed=11, n=8)
    test = copy_token_task(seed=12, n=4)
    manifest = register_run(
        _manifest_fields("run-snap", n_unique=8, batch_size=2, lr=0.05, snapshot_steps=(0, 2, 3))
    )
    # Explicit store= (the PLAN signature exposes it); same directory the
    # geode_store fixture exported as $GEODE_STORE.
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
        store=geode_store,
    )

    snap_root = geode_store / "runs" / "run-snap" / "snapshots"
    assert snap_root.is_dir()
    assert sorted(p.name for p in snap_root.iterdir()) == ["step_0", "step_2", "step_3"]
    for name in ("step_0", "step_2", "step_3"):
        files = [p for p in (snap_root / name).rglob("*") if p.is_file()]
        assert files, f"snapshot {name} is empty — nothing to reload"


def test_unreachable_snapshot_step_raises(geode_store, tiny_llama, copy_token_task):
    """spec 00 §2 + A-2: a declared snapshot step that training can never
    reach must raise ``ValueError`` — not be silently dropped.

    ``snapshot_steps`` is declared up front because "rerunning training to
    recover a missing checkpoint is the expensive failure mode" (spec 00 §2
    rationale); spec 01 §3 obliges the loop to save snapshots "per the
    manifest's ``snapshot_steps``". A schedule entry it cannot honor is
    therefore a loud error, never a silent omission discovered only when the
    GPU run is over.

    Hand derivation of the reachability bound: 6 train examples / batch 2 /
    ``epochs_total=1`` -> T = 3 optimizer updates; under L-5/A-1 the reachable
    snapshot ks are exactly 0..3 (``step_3`` = θ_T). Declared step 4 is
    unreachable. Step 0 is also declared so an implementation cannot pass by
    rejecting every schedule outright while still never validating
    reachability (the existing snapshot tests pin that valid schedules
    succeed).
    """
    train = copy_token_task(seed=11, n=6)
    test = copy_token_task(seed=12, n=4)
    manifest = register_run(
        _manifest_fields("run-unreach", n_unique=6, batch_size=2, lr=0.05, snapshot_steps=(0, 4))
    )
    with pytest.raises(ValueError, match="(?i)snapshot"):
        train_prequential(
            tiny_llama(seed=MODEL_SEED),
            _dataset(train, test),
            _task_format(),
            manifest,
            device="cpu",
            seed=LOOP_SEED,
        )


# ---------------------------------------------------------------------------
# spec 00 §4 — gradstats stride and schema (overlap null, OQ-14)
# ---------------------------------------------------------------------------


def test_gradstats_stride_and_schema(geode_store, tiny_llama, copy_token_task):
    """§4 records at the configured stride with the exact schema.

    10 examples / batch 2 -> steps 0..4; stride 2 logs steps 0, 2, 4 (L-6).
    ``topk_grad_subspace_overlap`` is null on every record (OQ-14);
    ``per_module_grad_norm`` is a non-empty str->float map; each module's norm
    cannot exceed the global norm (an L2 norm over a superset of parameters —
    small float slack allowed).
    """
    train = copy_token_task(seed=11, n=10)
    test = copy_token_task(seed=12, n=4)
    manifest = register_run(_manifest_fields("run-grad", n_unique=10, batch_size=2, lr=0.05))
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
        gradstats_stride=2,
    )

    recs = sorted(gradstat_records("run-grad"), key=lambda r: r.step)
    assert [r.step for r in recs] == [0, 2, 4]
    for rec in recs:
        assert rec.topk_grad_subspace_overlap is None  # OQ-14
        assert isinstance(rec.global_grad_norm, float)
        # A learnable task on a random-init model: gradients are nonzero.
        assert rec.global_grad_norm > 0.0
        assert isinstance(rec.per_module_grad_norm, dict) and rec.per_module_grad_norm
        for name, norm in rec.per_module_grad_norm.items():
            assert isinstance(name, str)
            assert isinstance(norm, float)
            assert norm >= 0.0
            assert norm <= rec.global_grad_norm * (1 + 1e-6) + 1e-9

    # On disk: one JSON object per line, exactly the §4 keys.
    lines = (
        (geode_store / "runs" / "run-grad" / "logs" / "gradstats.jsonl").read_text().splitlines()
    )
    objs = [json.loads(line) for line in lines if line.strip()]
    assert len(objs) == 3
    assert all(set(obj) == GRADSTAT_KEYS for obj in objs)
    assert all(obj["topk_grad_subspace_overlap"] is None for obj in objs)


def test_gradstats_keys_are_modules_not_parameters(geode_store, tiny_llama, copy_token_task):
    """spec 00 §4 + A-3: ``per_module_grad_norm`` maps ``module_name`` ->
    float. Keys must be MODULE names, not parameter-tensor names.

    Two independently violable pins:

    1. No key is a parameter name — in torch, ``weight``/``bias`` are
       parameter attributes of a module, so a key whose last dotted component
       is one of them (e.g. ``...lora_A.default.weight``) names a tensor, not
       a module. This is exactly the corruption §4's ``{"module_name": float}``
       schema rules out.
    2. Every key names a module on an adapted subtree: with
       ``target_modules=["q_proj", "v_proj"]`` (and dropout 0, no
       ``modules_to_save``), every trainable parameter — hence every nonzero
       gradient — lives strictly inside a ``q_proj``/``v_proj`` submodule, so
       any per-module decomposition of the gradient must have the adapted
       module on each key's path. Keys like ``...q_proj`` and
       ``...q_proj.lora_A.default`` both pass (granularity is unpinned, A-3);
       an aggregate keyed off-tree (e.g. ``self_attn`` or ``model``) cannot be
       joined back to ``training.lora.target_modules`` in analysis and fails.
    """
    train = copy_token_task(seed=11, n=4)
    test = copy_token_task(seed=12, n=4)
    manifest = register_run(_manifest_fields("run-gradkeys", n_unique=4, batch_size=2, lr=0.05))
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    recs = sorted(gradstat_records("run-gradkeys"), key=lambda r: r.step)
    assert [r.step for r in recs] == [0, 1]  # 4 examples / batch 2, stride 1
    for rec in recs:
        assert rec.per_module_grad_norm  # non-empty, as in the schema test
        for name in rec.per_module_grad_norm:
            parts = name.split(".")
            assert parts[-1] not in {"weight", "bias"}, (
                f"gradstats key {name!r} is a parameter-tensor name; spec 00 §4 "
                "keys per_module_grad_norm by module_name"
            )
            assert "q_proj" in parts or "v_proj" in parts, (
                f"gradstats key {name!r} names no adapted module; every "
                "trainable gradient lives under a target_modules subtree"
            )


# ---------------------------------------------------------------------------
# V0.5 producer side — test_loss.json carries the train-side mask hash
# ---------------------------------------------------------------------------


def test_testloss_hash_matches_train_hash(geode_store, tiny_llama, copy_token_task):
    """V0.5 (producer): one legitimate run yields matching hashes everywhere.

    The loop must stamp ``masking_config_hash(task_format, tokenizer_hash)``
    (OQ-7) into BOTH ``eval/test_loss.json`` (§5) and ``manifest.json`` as the
    top-level extra field (EDL-2's D-1) — so the ZOO-2 consistency check and
    ``edl_nats`` pass on a legitimate run instead of firing spuriously (or
    never). Also pins §5 field semantics on hand-countable held-out data and
    the acceptance criterion that one run writes all four artifact kinds.
    """
    train = copy_token_task(seed=11, n=7)
    test = _varied_span_test_split()  # 3 examples, 2+1+3 = 6 label tokens
    manifest = register_run(
        _manifest_fields("run-hash", n_unique=7, batch_size=3, lr=0.05, snapshot_steps=(0,))
    )
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    expected_hash = masking_config_hash(_task_format(), TOK_HASH)

    tl = test_loss("run-hash")
    assert tl.masking_config_hash == expected_hash
    # §5 semantics on hand-derived counts: 3 held-out examples, 2+1+3=6 label
    # tokens (≠ 3, ≠ the 7 train examples), per-token = sum/count.
    assert tl.n_test_examples == 3
    assert tl.label_token_count == 6
    assert tl.loss_sum_nats > 0.0
    assert tl.loss_per_label_token_nats == pytest.approx(tl.loss_sum_nats / 6)

    # D-1 producer side: manifest.json carries the same hash as a top-level
    # extra, and the rewritten manifest still loads through ZOO-1.
    on_disk = json.loads((geode_store / "runs" / "run-hash" / "manifest.json").read_text())
    assert on_disk["masking_config_hash"] == expected_hash
    assert load_run("run-hash").data["masking_config_hash"] == expected_hash

    # The guard passes on a legitimate run — both the ZOO-2 check and the
    # EDL-2 consumer (which would raise ConsistencyError on mismatch).
    check_masking_consistency("run-hash", expected_hash)
    assert isinstance(edl_nats("run-hash"), float)

    # Acceptance: all four artifact kinds from one run.
    run_dir = geode_store / "runs" / "run-hash"
    assert (run_dir / "logs" / "prequential.jsonl").is_file()
    assert (run_dir / "logs" / "gradstats.jsonl").is_file()
    assert (run_dir / "eval" / "test_loss.json").is_file()
    assert (run_dir / "snapshots" / "step_0").is_dir()


# ---------------------------------------------------------------------------
# spec 01 §1 — L_test is the loss of the FINAL model θ_T (auditor addition, A-1)
# ---------------------------------------------------------------------------


def test_testloss_evaluated_at_final_state(geode_store, tiny_llama, copy_token_task):
    """spec 01 §1: ``eval/test_loss.json`` holds L_test(θ_T) — the FINAL model's
    held-out loss — not θ₀'s and not θ_{T-1}'s.

    8 examples / batch 2 -> T = 4 updates; ``snapshot_steps=(4,)`` declares the
    final state (A-1: under L-5, ``step_4`` = θ after all 4 updates = θ_T). The
    stored ``loss_sum_nats`` must equal the shared-path evaluation of the
    reloaded θ_T on the test split. Positive control: the θ₀ evaluation must
    differ materially, so a loop that evaluates the test loss at initialization
    (even one that also mis-writes the final snapshot as θ₀) cannot pass.

    Margins (independent scratch reference in /tmp: peft LoRA r=2 on
    q_proj/v_proj, alpha 4, SGD, this exact data/config, swept over mean/sum
    backward reductions x 3 batch orders x 3 adapter-init seeds): at lr=0.2,
    |L_test(θ₀) − L_test(θ₄)| ≥ 0.22 nats — the 0.02 anchor margin is >10x
    below it; |L_test(θ₃) − L_test(θ₄)| ≥ 0.015 nats — >70x the equality
    tolerance, so a θ_{T-1} (off-by-one) evaluation fails the equality check.
    """
    train = copy_token_task(seed=11, n=8)
    test = copy_token_task(seed=12, n=4)
    manifest = register_run(
        _manifest_fields("run-final", n_unique=8, batch_size=2, lr=0.2, snapshot_steps=(4,))
    )
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    # Premise check: exactly 4 updates happened, so step_4 is the final state.
    assert [r.step for r in _sorted_records("run-final")] == [0, 1, 2, 3]

    run_dir = geode_store / "runs" / "run-final"
    mask = label_mask(test, _task_format())
    final_ref = prequential_step(_model_at_step(tiny_llama, run_dir, 4), test, mask)
    base_ref = prequential_step(tiny_llama(seed=MODEL_SEED), test, mask)

    tl = test_loss("run-final")
    assert tl.label_token_count == final_ref.label_token_count
    assert tl.loss_sum_nats == pytest.approx(final_ref.loss_sum_nats, rel=1e-5, abs=1e-6)
    # Positive control (calibrated, see docstring): θ_T's test loss is far from
    # θ₀'s, so the equality above cannot be satisfied by a θ₀ evaluation.
    assert abs(tl.loss_sum_nats - base_ref.loss_sum_nats) > 0.02


# ---------------------------------------------------------------------------
# Extra V1.3/§5 anchor — lr=0 run reproduces the pretrained model everywhere
# ---------------------------------------------------------------------------


def test_zero_lr_run_matches_pretrained_reference(geode_store, tiny_llama, copy_token_task):
    """With SGD at lr=0 (and weight_decay=0) every optimizer step is a no-op,
    so θ_s == θ₀ for all s. Every recorded prequential loss and the final
    test loss must then equal an evaluation of the pristine pretrained model
    through the shared ``label_mask``/``prequential_step`` path — a completely
    implementation-free reference. This catches pre/post-update drift, mask
    drift between the loop and the test-loss evaluator (§5 "same masking
    config"), and adapter inits that perturb θ₀ — with no snapshot or
    optimizer replication involved.
    """
    train = copy_token_task(seed=13, n=4)
    test = _varied_span_test_split()  # 2+1+3 = 6 label tokens over 3 examples
    manifest = register_run(_manifest_fields("run-lr0", n_unique=4, batch_size=2, lr=0.0))
    train_prequential(
        tiny_llama(seed=MODEL_SEED),
        _dataset(train, test),
        _task_format(),
        manifest,
        device="cpu",
        seed=LOOP_SEED,
    )

    recs = _sorted_records("run-lr0")
    assert len(recs) == 2  # 4 examples / batch 2
    for rec in recs:
        batch = _batch_of(train, rec.example_ids)
        mask = label_mask(batch, _task_format())
        ref = prequential_step(tiny_llama(seed=MODEL_SEED), batch, mask)
        assert rec.label_token_count == ref.label_token_count
        assert rec.loss_sum_nats == pytest.approx(ref.loss_sum_nats, rel=1e-5, abs=1e-6)

    tl = test_loss("run-lr0")
    tref = prequential_step(tiny_llama(seed=MODEL_SEED), test, label_mask(test, _task_format()))
    assert tl.label_token_count == tref.label_token_count == 6
    assert tl.loss_sum_nats == pytest.approx(tref.loss_sum_nats, rel=1e-5, abs=1e-6)
    assert tl.loss_per_label_token_nats == pytest.approx(tref.loss_sum_nats / 6, rel=1e-5, abs=1e-6)
