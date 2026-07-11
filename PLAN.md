# PLAN.md — geode build plan

Build plan for the infrastructure defined in `specs/`, per `PLANNING_PROMPT.md`.
Covers: `geode.zoo` (specs/00), `geode.edl` (specs/01), `geode.steering`
(specs/02), `geode.saediff` (specs/03), and the written crosscoder adaptation
plan (specs/04, a document — no code). Tasks are ≤ ~1 day each and execute
under the four-stage protocol in CLAUDE.md (restated in "Execution protocol").

---

## Resolved decisions (owner-approved 2026-07-11)

Ops: repo initialized (initial commit `e8d3bc8`); `reference/
sparsity-artifacts-crosscoders` cloned (OQ-1, OQ-2 closed).

- **OQ-3 — manifest validation:** all listed keys required recursively;
  `null` only where the schema says `|null`; primitive type checks; errors
  name the dotted path (e.g. `training.optimizer.lr`).
- **OQ-4 — `example_ids` universe:** ids are `0..n_unique_examples-1` from
  the run's manifest; checker rejects skips, repeats, out-of-range.
- **OQ-5 — position policy:** one shared enum `{all, answer_only, last}`;
  "generated positions" ≡ `answer_only`; spec 02 wording updated in same PR.
- **OQ-6 — results layout:** one parquet per analysis,
  `results/{analysis_name}.parquet`, overwrite-by-name; `read_results`
  concatenates the directory; joins in pandas on mandated key columns.
- **OQ-7 — hashes:** `tokenizer_hash` = sha256 of canonical tokenizer JSON;
  `masking_config_hash` = sha256 of canonical JSON of
  `{task name, format_version, mask-rule parameters, tokenizer_hash}`.
- **OQ-8 — label boundary:** dataset builder emits per-example label-token
  spans; `task_format` declares the span rule; `label_mask` is the single
  construction path for training loop and test evaluator (M1's shared fn).
- **OQ-9 — refit return type:** `LowRankMap` dataclass (hook, U, S, Vh,
  method, provenance); spec 02 §5 edited in same PR.
- **OQ-10 — defaults:** explicit kwargs, `k=8`,
  `alphas=(0.25, 0.5, 1.0, 2.0, 4.0)`; never hidden constants.
- **OQ-11 — capacity metric:** notebook ratio of two `edl_nats` calls; no
  library function.
- **OQ-12 — `load_sae`:** thin wrapper over `sae_lens.SAE.load_from_disk`,
  function-local import; all metrics accept `SAEProtocol`
  (encode/decode/W_dec); tests use in-process synthetic SAEs.
- **OQ-13 — `residual_pcs`:** centered PCA; streaming accumulates residual
  mean + d×d second moment, then eigendecomposes (exact ⇒ V3.6 holds).
- **OQ-14 — `topk_grad_subspace_overlap`:** always `null` for now; field
  reserved for a future analysis spec.

---

## Module layout (files created across all tasks)

```
geode/
  zoo/        __init__.py  store.py  manifest.py  records.py  checks.py
              activations.py  results.py
  edl/        __init__.py  masking.py  prequential.py  metrics.py  loop.py
  steering/   __init__.py  types.py  extract.py  interventions.py
              refit.py  controls.py  ladder.py
  saediff/    __init__.py  sae.py  metrics.py  stats.py  orchestrate.py
tests/
  conftest.py                       # shared tiny-model / task / store fixtures
  zoo/  edl/  steering/  saediff/   # one test module per task, named below
docs/
  crosscoder-adaptation-plan.md     # ADAPT-1 deliverable (document, not code)
```

Conventions bound on every task (from CLAUDE.md): Python ≥3.11, type hints on
public functions, losses in nats with `_nats` suffixes, device-agnostic,
explicit `seed` args, full suite CPU-only < ~2 min, no network in tests, no
pretrained weights in tests, `ruff` clean.

**Reference policy (applies to every task):** all library code is written
fresh. `reference/sparsity-artifacts-crosscoders` is read-only, targets
7B-chat-scale crosscoder training (out of scope for these four modules), and
spec 01 explicitly requires the EDL harness to be independently trustworthy.
The reference repo informs exactly one deliverable: ADAPT-1's document.

---

## Tasks

### SETUP-0 — Repo bootstrap + shared test fixtures (~0.5 day)

- **Spec sections:** none (infrastructure). Governed by CLAUDE.md "Testing
  policy" (fixture models = tiny random in-process configs) and "Conventions".
- **Protocol deviation:** the four-stage pipeline does not apply (there is no
  spec-behavior to test-first). Executed as a single implementation pass +
  review. Stated here so the deviation is planned, not improvised.
- **Boundaries:** repo already initialized (OQ-1 resolved);
  `geode/{zoo,edl,steering,saediff}/__init__.py`; `tests/conftest.py` with:
  - `tiny_llama(seed, n_layers=2, d_model=64, vocab_size=128) -> LlamaForCausalLM`
    — random-init in-process via `transformers.LlamaConfig`; never downloads.
  - `tiny_tokenizer()` — in-process word-level `PreTrainedTokenizerFast`
    (built from `tokenizers`, no files, no network).
  - `copy_token_task(seed, n)` — trivially learnable synthetic task (V1.2);
    `random_label_task(seed, n)` — i.i.d. uniform labels (V1.1). Both emit
    token sequences + label spans (OQ-8 shape).
  - `geode_store(tmp_path, monkeypatch)` — sets `$GEODE_STORE` to a tmp dir.
- **Adapted vs fresh:** all fresh.
- **Tests:** none of its own; acceptance is that the (empty) suite collects and
  fixtures import.
- **Acceptance:** `pytest -q` green; `ruff check` clean; fixtures build models
  and tokenizers with zero network access (verifiable by running offline).
- **Depends on:** nothing.

---

### ZOO-1 — Run manifest + registry (~1 day)

- **Spec sections:** specs/00 §1 "Directory layout", §2 "Run manifest",
  §8 "Public API surface" (`RunManifest`, `register_run`, `load_run`,
  `iter_runs`).
- **Boundaries:** `geode/zoo/store.py` (resolve `$GEODE_STORE`, path helpers —
  no absolute paths in code), `geode/zoo/manifest.py`.
  ```python
  class ManifestError(ValueError): ...   # message names the offending dotted field
  class RunManifest:
      data: dict  # full JSON incl. unknown fields, preserved
      @classmethod
      def load(cls, path: Path) -> "RunManifest"
      def validate(self) -> None
      def save(self, path: Path) -> None
  def register_run(fields: dict, *, store: Path | None = None) -> RunManifest
  def load_run(run_id: str, *, store: Path | None = None) -> RunManifest
  def iter_runs(regime: str | None = None, task: str | None = None,
                status: str = "complete", *, store: Path | None = None
                ) -> Iterator[RunManifest]
  ```
  Produces spec 00 §2 `manifest.json` under the §1 layout. Consumed by every
  later task that takes a `run_id`.
- **Adapted vs fresh:** fresh (plain-dict + explicit validator; no external
  schema lib — smallest thing that satisfies V0.1/V0.2).
- **Tests** (`tests/zoo/test_manifest.py`):
  - `test_missing_required_field_error_names_field` **(V0.1)** — parametrized
    over every required field incl. nested (per OQ-3). Failure ⇒ validator has
    holes; malformed runs enter the registry and poison downstream analysis.
  - `test_roundtrip_preserves_unknown_extra_fields` **(V0.2)** —
    `save(load(m)) == m` incl. extra fields. Failure ⇒ provenance loss on
    rewrite; manifests can't be trusted as the record of what was run.
  - `test_null_allowed_only_where_schema_says` (OQ-3 half of V0.1). Failure ⇒
    nullability contract drifts from spec.
  - `test_register_then_load_identity` — failure ⇒ layout/paths wrong.
  - `test_iter_runs_filters_regime_task_status` — failure ⇒ analysis grouping
    (elicit vs teach) silently wrong.
  - `test_store_root_comes_from_env` — failure ⇒ `$GEODE_STORE` contract
    broken; code hardcodes paths.
- **Acceptance:** V0.1 + V0.2 tests pass; layout matches §1 exactly; unknown
  fields survive; suite still <2 min CPU; ruff clean.
- **Depends on:** SETUP-0.

---

### ZOO-2 — Run logs: records, invariants, consistency (~1 day)

- **Spec sections:** specs/00 §3 "Prequential log", §4 "Gradient statistics",
  §5 "Test loss", §8 (`prequential_records`, `test_loss`).
- **Boundaries:** `geode/zoo/records.py`, `geode/zoo/checks.py`.
  ```python
  @dataclass(frozen=True)
  class PrequentialRecord: step: int; epoch: int; example_ids: list[int]; \
      label_token_count: int; loss_sum_nats: float
  @dataclass(frozen=True)
  class GradStatRecord: step: int; global_grad_norm: float; \
      per_module_grad_norm: dict[str, float]; topk_grad_subspace_overlap: float | None
  @dataclass(frozen=True)
  class TestLoss: n_test_examples: int; label_token_count: int; \
      loss_sum_nats: float; loss_per_label_token_nats: float; masking_config_hash: str
  def write_jsonl(path: Path, records: Iterable) -> None
  def prequential_records(run_id: str, *, store=None) -> Iterator[PrequentialRecord]
  def gradstat_records(run_id: str, *, store=None) -> Iterator[GradStatRecord]
  def test_loss(run_id: str, *, store=None) -> TestLoss
  def check_epoch1_coverage(records: Iterable[PrequentialRecord],
                            n_unique_examples: int) -> None      # V0.3 (OQ-4)
  def check_masking_consistency(run_id: str, train_hash: str, *, store=None) -> None  # V0.5
  ```
  Produces/consumes spec 00 §3/§4/§5 files under §1 layout.
- **Adapted vs fresh:** fresh.
- **Tests** (`tests/zoo/test_records.py`):
  - `test_epoch1_ids_skip_rejected` / `test_epoch1_ids_repeat_rejected`
    **(V0.3)** — failure ⇒ MDL can be summed over a mis-enumerated stream:
    silently wrong description lengths, the worst kind of error here.
  - `test_epoch1_ids_exact_cover_accepted` **(V0.3)** — failure ⇒ checker
    over-rejects; valid runs unusable.
  - `test_masking_hash_mismatch_raises` **(V0.5)** — failure ⇒ the train/test
    mask-parity footgun (spec 00 §5) has no mechanical guard; EDL invalid.
  - `test_prequential_jsonl_roundtrip` — failure ⇒ IO corruption of the
    primary measurement record.
  - `test_test_loss_schema_roundtrip` — failure ⇒ §5 schema drift.
  - `test_gradstats_overlap_may_be_null` — failure ⇒ §4 nullability broken
    (OQ-14).
- **Acceptance:** V0.3 + V0.5 tests pass; record fields byte-match spec §3–§5
  names; readers stream (iterators, no full-file loads).
- **Depends on:** ZOO-1.

---

### ZOO-3 — Activation store with matched-input enforcement (~1 day)

- **Spec sections:** specs/00 §6 "Activation storage" (+ §1 layout).
- **Boundaries:** `geode/zoo/activations.py`.
  ```python
  @dataclass(frozen=True)
  class ActivationMeta: model_key: str; hf_id: str; revision: str; \
      dataset_key: str; position_policy: Literal["all","answer_only","last"]; \
      n_samples: int; tokenizer_hash: str
  def tokenizer_hash(tokenizer) -> str                       # OQ-7
  def save_activations(acts: Tensor, meta: ActivationMeta, hook_name: str,
                       *, store=None, dtype=torch.float16) -> Path
  def load_activations(model_key: str, dataset_key: str, hook_name: str,
                       *, store=None) -> tuple[Tensor, ActivationMeta]
  def load_matched_pair(model_key_a: str, model_key_b: str, dataset_key: str,
                        hook_name: str, *, store=None
                        ) -> tuple[Tensor, Tensor, ActivationMeta]   # V0.4 gate
  ```
  Produces/consumes spec 00 §6 safetensors + sidecar meta. `load_matched_pair`
  is the single gate spec 02 E1 and spec 03 NFM must go through.
- **Adapted vs fresh:** fresh (safetensors + json sidecar; trivial surface).
- **Tests** (`tests/zoo/test_activations.py`):
  - `test_roundtrip_shape_dtype_name` — `[n_samples, n_positions_kept,
    d_model]`, float16 default, tensor name == hook name. Failure ⇒ storage
    convention drift; downstream loaders misread.
  - `test_matched_pair_dataset_key_mismatch_raises`,
    `test_matched_pair_position_policy_mismatch_raises`,
    `test_matched_pair_tokenizer_hash_mismatch_raises` **(V0.4)** — failure ⇒
    cross-model comparisons can silently run on unmatched inputs; every E1
    direction and every NFM number becomes meaningless.
  - `test_matched_pair_error_message_is_clear` **(V0.4)** — spec demands "a
    clear error". Failure ⇒ debugging cost at GPU-run time.
  - `test_sidecar_meta_roundtrip` — failure ⇒ provenance loss.
  - `test_tokenizer_hash_deterministic_and_sensitive` — same tokenizer ⇒ same
    hash; different vocab ⇒ different hash. Failure ⇒ V0.4's third condition
    is theater.
- **Acceptance:** V0.4 tests pass; file layout matches §1/§6; no `cuda`
  assumptions.
- **Depends on:** ZOO-1, SETUP-0.

---

### ZOO-4 — Results table writer (~0.5 day)

- **Spec sections:** specs/00 §7 "Results tables".
- **Boundaries:** `geode/zoo/results.py`.
  ```python
  REQUIRED_COLUMNS: tuple[str, ...]  # run_id, base_model_key, regime, dataset_size,
                                     # checkpoint_step, layer, metric_name, metric_value
  def write_results(df: pd.DataFrame, name: str, *, store=None) -> Path
  def read_results(name: str | None = None, *, store=None) -> pd.DataFrame
  ```
  Produces spec 00 §7 parquet in `results/` (naming per OQ-6). Consumed by
  STEER-3 and SAE-3 outputs; EDL values are written through the same writer by
  analysis scripts (no special path — that's the point of §7).
- **Adapted vs fresh:** fresh.
- **Tests** (`tests/zoo/test_results.py`):
  - `test_missing_required_column_raises` — failure ⇒ the EDL-vs-internals
    bridge join breaks at analysis time, after the GPU money is spent.
  - `test_long_format_roundtrip_parquet` — failure ⇒ IO/schema drift.
  - `test_two_tables_join_on_run_id_dataset_size` — demonstrates the §7 join
    contract. Failure ⇒ table shape doesn't actually support the bridge plot.
- **Acceptance:** column contract enforced; parquet roundtrip exact; note §7
  has no V-number (recorded in coverage map).
- **Depends on:** ZOO-1.

---

### EDL-1 — Label masking (M1) + masking config hash (~1 day)

- **Spec sections:** specs/01 §2 "Masking rules" (M1), §3 "Public API"
  (`label_mask`); specs/00 §5 (`masking_config_hash` semantics).
- **Boundaries:** `geode/edl/masking.py`.
  ```python
  @dataclass(frozen=True)
  class TaskFormat: name: str; format_version: str  # + span rule params (OQ-8)
  def label_mask(batch, task_format: TaskFormat) -> torch.BoolTensor   # M1
  def masking_config_hash(task_format: TaskFormat, tokenizer_hash: str) -> str
  ```
  This is the *single shared function* both the training loop (EDL-3) and the
  test-loss evaluator use; its hash lands in spec 00 §5.
- **Adapted vs fresh:** fresh.
- **Tests** (`tests/edl/test_masking.py`):
  - `test_mask_covers_exactly_label_tokens` — parametrized over synthetic task
    formats; prompt and formatting tokens excluded. Failure ⇒ M1 broken at the
    source; MDL includes prompt mass (the exact footgun spec 01 §2 names).
  - `test_mask_count_matches_label_span_lengths` — failure ⇒ `label_token_count`
    bookkeeping will drift (feeds V1.6).
  - `test_same_config_same_hash_across_processes` — hash from canonical
    serialization, not object identity. Failure ⇒ V0.5/V1.4 guard fires
    spuriously or never.
  - `test_hash_changes_when_mask_rule_or_tokenizer_changes` — failure ⇒ guard
    cannot detect real mask-parity violations (V1.4(a) foundation).
- **Acceptance:** mask construction is the only path to a mask in the codebase
  (reviewer greps for ad-hoc masking); hash deterministic & sensitive.
- **Depends on:** SETUP-0.

---

### EDL-2 — Prequential accumulator + EDL metrics (~1 day)

- **Spec sections:** specs/01 §1 "Definitions" (MDL, EDL, normalizations, PGR,
  units), §2 M2, §3 (`prequential_step`, `PrequentialAccumulator`, `mdl_nats`,
  `edl_nats`, `edl_per_label_token`, `edl_per_param`, `pgr`, `training_curve`).
- **Boundaries:** `geode/edl/prequential.py`, `geode/edl/metrics.py`.
  ```python
  @dataclass(frozen=True)
  class StepLoss: loss_sum_nats: float; label_token_count: int
  def prequential_step(model, batch, mask: torch.BoolTensor) -> StepLoss
      # masked label-token CE, summed, nats, no_grad — pre-update by contract
  class PrequentialAccumulator:
      def add_epoch1(self, record: PrequentialRecord) -> None  # raises on epoch != 1 (M2)
      def mdl_nats(self) -> float
      def flush(self, run_id: str, *, store=None) -> None      # writes spec00 §3
  def mdl_nats(run_id: str, *, store=None) -> float
  def edl_nats(run_id: str, *, store=None) -> float   # refuses on hash mismatch (V0.5)
  def edl_per_label_token(run_id: str, *, store=None) -> float   # EDL/D
  def edl_per_param(run_id: str, *, store=None) -> float         # EDL/P
  def pgr(perf_tuned: float, perf_base: float, perf_fullft: float) -> float
  def training_curve(run_id: str, *, store=None) -> pd.DataFrame
  def nats_to_bits(x_nats: float) -> float
  ```
  Consumes spec 00 §3 (via ZOO-2 readers) and §5; produces §3 via `flush`.
  Design note: `pgr` raises on a near-zero denominator (|Perf_FullFT −
  Perf_base| < eps) rather than returning ±inf silently.
- **Adapted vs fresh:** fresh — spec 01 requires this module be independently
  trustworthy (cross-check against the Donoway repo later, never a dependency).
- **Tests** (`tests/edl/test_metrics.py`):
  - `test_accumulator_rejects_epoch_gt1` **(V1.5, structural half)** — failure
    ⇒ M2 is convention, not mechanism; multi-epoch redundancy can inflate MDL.
  - `test_mdl_equals_jsonl_recomputation` **(V1.6, recompute half)** — failure
    ⇒ accumulator and stored log disagree; the number reported isn't the
    number recorded.
  - `test_edl_identity_mdl_minus_nlabel_times_testloss` — exact §1 formula on
    constructed numbers. Failure ⇒ headline metric algebra wrong.
  - `test_edl_refuses_on_masking_hash_mismatch` **(V1.4(a))** — failure ⇒
    harness computes EDL across mismatched masks (the guarded footgun).
  - `test_normalizations_per_token_per_param` — EDL/D uses epoch-1
    `N_label`, EDL/P uses `trainable_param_count`. Failure ⇒ scaling/capacity
    analyses use wrong denominators.
  - `test_bits_equal_nats_over_ln2` **(V1.8)** — failure ⇒ unit confusion
    leaks to reports (spec: convert only at boundaries).
  - `test_pgr_formula_and_degenerate_denominator` — failure ⇒ sufficiency
    metric wrong or silently infinite.
  - `test_training_curve_epoch1_loss_vs_example_index` — failure ⇒ diagnostic
    plots misordered.
  - `test_prequential_step_sums_masked_tokens_only_nats` — failure ⇒ per-step
    loss on the wrong token set or wrong reduction (mean vs sum).
- **Acceptance:** V1.5(structural), V1.6(recompute), V1.4(a), V1.8 named tests
  pass; no public function returns unlabeled units (names carry `_nats` /
  `_bits`); accumulator API has no epoch>1 entry point.
- **Depends on:** ZOO-2, EDL-1.

---

### EDL-3 — Prequential training-loop wrapper (~1 day)

- **Spec sections:** specs/01 §3 final paragraph ("thin training-loop wrapper
  around PEFT: pre-update loss, optimizer step, prequential + gradstat
  records, snapshots per manifest"); produces specs/00 §3, §4, §5 and
  `snapshots/step_{k}/` per §1–§2.
- **Boundaries:** `geode/edl/loop.py`.
  ```python
  def train_prequential(model, dataset, task_format: TaskFormat,
                        manifest: RunManifest, *, device: str,
                        seed: int, gradstats_stride: int = 1,
                        store: Path | None = None) -> None
  ```
  Per batch: `prequential_step` (pre-update, no_grad) → record via
  `PrequentialAccumulator` → backward/step. Writes `prequential.jsonl` (epoch 1
  mandatory, later epochs logged but flagged), `gradstats.jsonl` (overlap =
  null, OQ-14), PEFT adapter snapshots at `manifest.snapshot_steps`, and
  `eval/test_loss.json` with the same `masking_config_hash` (EDL-1).
  Device-agnostic; seeded; single-GPU/CPU only (spec §5 non-goals).
- **Adapted vs fresh:** fresh (thin wrapper over `peft`/`torch` — the library
  dependency is PEFT itself, not reference code).
- **Tests** (`tests/edl/test_loop.py`, tiny models from SETUP-0):
  - `test_preupdate_loss_matches_two_pass_reference` **(V1.3)** — fixed seed;
    loop's recorded losses == explicit evaluate-all-then-train reference.
    Failure ⇒ losses recorded post-update; MDL systematically underestimates
    description length — invalidates the central metric.
  - `test_postupdate_corruption_yields_lower_mdl` **(V1.3)** — the corrupted
    variant differs in the expected direction on a one-step-learnable task.
    Failure ⇒ the test task can't detect the corruption (test is toothless) or
    both paths are wrong.
  - `test_epoch1_ids_pass_coverage_checker` **(V1.6 via V0.3)** — loop output
    satisfies `check_epoch1_coverage`. Failure ⇒ loop mis-enumerates the
    stream.
  - `test_label_token_sum_matches_tokenizer_count` **(V1.6)** — Σ
    `label_token_count` == tokenizer-derived dataset label count. Failure ⇒
    masking/counting drift between data pipeline and loop.
  - `test_same_seed_twice_identical_prequential_log` **(V1.7)** — byte-equal
    JSONL, CPU, fixed threads. Failure ⇒ nondeterminism; EDL numbers not
    reproducible, cross-checks impossible.
  - `test_snapshots_written_at_declared_steps` — failure ⇒ the expensive
    failure mode spec 00 §2 warns about (rerunning training for a missing
    checkpoint).
  - `test_gradstats_stride_and_schema` — failure ⇒ §4 records unusable.
  - `test_testloss_hash_matches_train_hash` **(V0.5 producer side)** — failure
    ⇒ guard would fire on every legitimate run, or never.
- **Acceptance:** V1.3, V1.6, V1.7 named tests pass; wrapper writes all four
  artifact kinds in one run on a tiny model in seconds; no `cuda` literals.
- **Depends on:** EDL-2, ZOO-1, ZOO-2.

---

### EDL-4 — EDL end-to-end validation battery (~1 day)

- **Spec sections:** specs/01 §4 "Validation properties" (the scientific
  sanity properties: V1.1, V1.2, V1.4, V1.5 end-to-end).
- **Boundaries:** tests only (`tests/edl/test_edl_properties.py`) plus any
  library fixes they force. No new public API. Tolerances: V1.1 asserts the
  relative comparison (|EDL/D|_random ≪ EDL/D_structured) as primary, an
  absolute bound as secondary — the spec deliberately says "small tolerance",
  so the test pins the *ordering*, and the absolute bound is calibrated once
  on the fixture and frozen with a comment.
- **Adapted vs fresh:** fresh.
- **Tests:**
  - `test_random_labels_edl_near_zero` **(V1.1)** — random-label task: |EDL/D|
    below tolerance, final test loss ≈ ln(label vocab). Failure ⇒ the harness
    manufactures information from noise — a leak (mask, ordering, or
    pre-update violation) that would fake "elicitation" signal.
  - `test_random_vs_structured_edl_separation` **(V1.1/V1.2 joint)** — random
    ≪ structured. Failure ⇒ no discriminative power; EDL as implemented cannot
    distinguish learning from noise.
  - `test_learnable_task_edl_positive_curve_descends` **(V1.2)** — copy-token
    task: EDL/D clearly positive; epoch-1 curve descends toward L_test.
    Failure ⇒ dead metric or broken curve extraction.
  - `test_unmasking_prompts_inflates_mdl` **(V1.4(b))** — corrupted mask
    config (built directly in the test, not via a library bypass flag): MDL
    strictly increases by ≈ the prompt-token loss mass. Failure ⇒ the mask
    isn't actually applied — M1 enforcement is illusory.
  - `test_hash_guard_raises_end_to_end` **(V1.4(a))** — full-pipeline variant
    of EDL-2's unit test. Failure ⇒ guard exists only at unit level, not wired
    through the harness.
  - `test_two_epoch_sum_strictly_exceeds_epoch1` **(V1.5)** — corrupted value
    recomputed manually from JSONL (public API cannot produce it). Failure ⇒
    epochs indistinguishable in logs, M2 unauditable.
  - `test_public_api_has_no_multiepoch_mdl_path` **(V1.5, structural
    end-to-end)** — failure ⇒ M2 regression surface exists.
- **Acceptance:** every §4 property has a passing named test; battery runs on
  CPU within the suite budget; any tolerance constants carry a comment citing
  the property they calibrate.
- **Depends on:** EDL-3.

---

### STEER-1 — Direction types + extractors E1/E2 (~1 day)

- **Spec sections:** specs/02 §1 "Direction extraction" (E1, E2), §5 "Public
  API" (`extract_activation_diff`, `extract_weight_diff`).
- **Boundaries:** `geode/steering/types.py`, `geode/steering/extract.py`.
  ```python
  @dataclass(frozen=True)
  class Direction: hook: str; vector: torch.Tensor; method: str; provenance: dict
  @dataclass(frozen=True)
  class ModuleSVDs:  # per-module (U, S, Vh) + source ("lora" | "merged_minus_base") + scaling
      modules: dict[str, tuple[Tensor, Tensor, Tensor]]; source: str; provenance: dict
  def extract_activation_diff(base, tuned, dataset_key: str, hook: str,
                              positions: str, *, store=None) -> Direction     # E1
  def per_layer_diff_norms(base, tuned, dataset_key: str, hooks: list[str],
                           positions: str, *, store=None) -> pd.DataFrame     # E1 norms
  def extract_weight_diff(base, adapter_or_merged, *, top_k: int = 8) -> ModuleSVDs  # E2
  def truncated_delta(svds: ModuleSVDs, module: str, rank: int) -> torch.Tensor
  ```
  E1 consumes spec 00 §6 through `load_matched_pair` (ZOO-3) — the *only*
  activation entry point, so V0.4 enforcement is inherited, then re-asserted
  at this surface (V2.5).
- **Adapted vs fresh:** fresh (mean-diff and `torch.linalg.svd` are small; the
  reference repo's steering app solves a different problem at a different
  scale).
- **Tests** (`tests/steering/test_extract.py`):
  - `test_planted_shift_recovered_cosine_ge_099` **(V2.1, extraction half)** —
    tiny model + copy with permanent bias u at hook ℓ; E1 recovers v with
    |cos(v,u)| ≥ 0.99. Failure ⇒ mean-diff wrong (positions, ordering, or
    normalization) — every downstream steering result would be built on a
    wrong vector.
  - `test_e1_raises_on_metadata_mismatch` **(V2.5)** — dataset_key / positions
    / tokenizer-hash mismatch each raise. Failure ⇒ unmatched-input directions
    slip through; C2-style artifacts become headline numbers.
  - `test_e2_matches_svd_of_materialized_lora` **(V2.6)** — E2 on a real PEFT
    LoRA adapter == SVD of explicitly materialized α/r·B@A, including scaling.
    Failure ⇒ silent α/r scaling error — every rank-sweep magnitude wrong.
  - `test_planted_rank2_principal_angles_near_zero` **(V2.2, extraction
    half)** — top-2 singular subspace matches planted subspace. Failure ⇒
    SVD/truncation logic wrong; I2 patches would patch the wrong subspace.
  - `test_per_layer_norms_reported_for_all_hooks` — failure ⇒ concentration-
    layer selection (spec §1) has no data.
- **Acceptance:** extraction halves of V2.1/V2.2 + V2.5 + V2.6 pass;
  `Direction.provenance` records dataset_key/positions/method (reviewer
  checks); no path bypasses `load_matched_pair`.
- **Depends on:** ZOO-3, SETUP-0.

---

### STEER-2 — Interventions I1/I2 as restoring context managers (~1 day)

- **Spec sections:** specs/02 §2 "Interventions" (I1, I2), §5
  (`apply_constant_shift`, `apply_lowrank_patch`, restoration guarantee).
- **Boundaries:** `geode/steering/interventions.py`.
  ```python
  @contextmanager
  def apply_constant_shift(model, direction: Direction, alpha: float,
                           positions: str = "all") -> Iterator[None]      # I1
  @contextmanager
  def apply_lowrank_patch(model, svds: ModuleSVDs, rank: int) -> Iterator[None]  # I2
  ```
  Both guarantee restoration of original weights/hooks on exit, including on
  exception (try/finally; weight patches keep pre-images).
- **Adapted vs fresh:** fresh (forward-hook + weight-delta application; small
  and must be exactly restorable — third-party steering code doesn't give that
  guarantee).
- **Tests** (`tests/steering/test_interventions.py`):
  - `test_recovered_shift_alpha1_reproduces_tuned_outputs` **(V2.1, full)** —
    I1 with E1's recovered v at α=1 matches the planted-tuned model's outputs
    to tolerance. Failure ⇒ hook math or hook-point mapping wrong; PGR would
    understate recovery for mechanical reasons, faking a "teaching" signature.
  - `test_rank2_patch_reproduces_tuned_rank1_does_not` **(V2.2, full)** — I2 at
    r=2 matches planted-ΔW model; r=1 measurably does not. Failure ⇒ rank
    ladder can't resolve the quantity it exists to measure (minimal
    sufficient rank).
  - `test_outputs_bit_identical_after_normal_exit` **(V2.3)** — failure ⇒
    state leaks between measurements; every subsequent number in a session is
    contaminated.
  - `test_outputs_bit_identical_after_exception_exit` **(V2.3)** — same, on
    the error path (spec explicitly requires it).
- **Acceptance:** V2.1, V2.2, V2.3 named tests pass; both interventions are
  context managers with exception-safe restoration; device-agnostic.
- **Depends on:** STEER-1.

---

### STEER-3 — Refit ladder, controls, PGR evaluation, results rows (~1 day)

- **Spec sections:** specs/02 §2 I3 "Re-fit ladder", §3 "Evaluation", §4
  "Controls" (C1–C3), §5 (`refit_lowrank`, `sufficiency_ladder`).
- **Boundaries:** `geode/steering/refit.py`, `geode/steering/controls.py`,
  `geode/steering/ladder.py`.
  ```python
  def refit_lowrank(base_acts: Tensor, tuned_acts: Tensor, rank: int
                    ) -> LowRankMap        # I3, least squares on frozen base acts (OQ-9)
  def random_direction_control(model, hook: str, alphas: Sequence[float],
                               eval_fn, seed: int) -> pd.DataFrame          # C1
  def shuffled_pairing_direction(base, tuned, dataset_key: str, hook: str,
                                 positions: str, seed: int, *, store=None
                                 ) -> Direction                              # C2
  def norm_matched(direction: Direction, norm_budget: float) -> Direction    # C3
  def sufficiency_ladder(run_id: str, ranks: Sequence[int],
                         alphas: Sequence[float], eval_fn,
                         *, base=None, tuned=None, store=None) -> pd.DataFrame
  def minimal_sufficient_rank(ladder_df: pd.DataFrame,
                              threshold: float = 0.5) -> int | None
  ```
  Design note: `base`/`tuned` keyword models keep the ladder testable offline
  (tests pass tiny fixture models; the run_id path that loads real HF weights
  is exercised only in budgeted GPU runs). PGR comes from `geode.edl.pgr`
  (spec 02 §3 cites spec 01). I3 is flag-gated per spec (`enable_refit` in the
  ladder). Emits spec 00 §7 rows via ZOO-4: one row per (run_id, extractor,
  intervention, layer, rank, α), `metric_name ∈ {pgr, best_alpha, vector_norm,
  sv_spectrum_k}`.
- **Adapted vs fresh:** fresh (`torch.linalg.lstsq` for I3; controls are
  bespoke to this experimental design).
- **Tests** (`tests/steering/test_ladder.py`):
  - `test_random_control_pgr_near_zero_recovered_near_one` **(V2.4)** — on the
    planted-shift model, C1 PGR ≈ 0 while recovered-direction PGR ≈ 1.
    Failure ⇒ the pipeline can't separate signal from noise — headline PGR
    numbers uninterpretable.
  - `test_shuffled_pairing_yields_low_pgr_direction` **(C2, spec §4)** —
    failure ⇒ pairing doesn't matter to E1, so "directions" may be dataset-
    mean artifacts rather than finetune effects.
  - `test_norm_matched_directions_have_equal_budget` **(C3, spec §4)** —
    failure ⇒ elicit-vs-teach comparisons confounded by intervention scale
    (the exact artifact C3 exists to kill).
  - `test_refit_recovers_planted_linear_map` **(I3)** — least-squares refit at
    the planted rank beats random and matches the planted map. Failure ⇒ I3
    can't distinguish "diff is low-rank" from "low-rank suffices".
  - `test_ladder_rows_conform_to_results_schema` **(spec 00 §7 / spec 02
    §3)** — required columns + metric_name vocabulary. Failure ⇒ EDL bridge
    join breaks.
  - `test_minimal_sufficient_rank_threshold_logic` — smallest r with PGR ≥
    threshold; None when never reached. Failure ⇒ the H3 candidate scalar is
    wrong.
  - `test_refit_is_flag_gated_off_by_default` — spec says optional,
    flag-gated. Failure ⇒ cost surprise in real sweeps.
- **Acceptance:** V2.4 + C2/C3 tests pass; every headline PGR row is
  accompanied by its C1 row (reviewer checks ladder output); results rows join
  against a ZOO-4 table in-test.
- **Depends on:** STEER-2, ZOO-4, EDL-2 (for `pgr`).

---

### SAE-1 — SAE interface + FVU/NFM metrics (~1 day)

- **Spec sections:** specs/03 §1 "Inputs", §2 "Metrics" (FVU, baseline FVU,
  NFM), §4 (`load_sae`, `fvu`, `nfm`).
- **Boundaries:** `geode/saediff/sae.py`, `geode/saediff/metrics.py`, and
  `tests/saediff/conftest.py` (synthetic SAE fixture: orthogonal decoder,
  encoder built to invert exactly on in-span inputs, per spec §5 preamble).
  ```python
  class SAEProtocol(Protocol):
      def encode(self, x: Tensor) -> Tensor
      def decode(self, a: Tensor) -> Tensor
      @property
      def w_dec(self) -> Tensor
  def load_sae(model_key: str, hook: str, *, store=None) -> SAEProtocol
      # thin wrapper over sae_lens.SAE.load_from_disk, read-only (OQ-12)
  def fvu(sae: SAEProtocol, acts: Tensor) -> float
      # Σ‖x−x̂‖² / Σ‖x−mean(X)‖²  (exact spec §2 formula)
  def nfm(sae: SAEProtocol, acts_base: Tensor, acts_ft: Tensor,
          meta_base: ActivationMeta | None = None,
          meta_ft: ActivationMeta | None = None) -> float
      # FVU(X_ft) − FVU(X_base); raises on metadata mismatch when metas given (V3.5)
  ```
  Consumes spec 00 §6 activations + SAELens-format SAEs from `saes/` (§1
  layout). All metric functions take any `SAEProtocol` — tests never load real
  SAEs.
- **Adapted vs fresh:** fresh. `sae-lens` is a declared dependency used for
  *loading only* (spec §6: SAE training out of scope); the reference repo is
  crosscoders, not SAEs, and is untouched.
- **Tests** (`tests/saediff/test_metrics.py`):
  - `test_in_span_reconstruction_fvu_near_zero` **(V3.1)** — sparse positive
    combinations of dictionary directions ⇒ FVU ≈ 0. Failure ⇒ FVU formula or
    synthetic SAE wrong; every downstream number miscalibrated.
  - `test_nfm_of_data_against_itself_near_zero` **(V3.1)** — failure ⇒ NFM has
    a built-in bias; the headline scalar reads nonzero on null data.
  - `test_nfm_monotone_in_planted_orthogonal_energy` **(V3.2)** — NFM strictly
    increases with ε. Failure ⇒ metric not even ordinal in novel-feature
    energy — H2 untestable.
  - `test_nfm_approx_epsilon_for_small_epsilon` **(V3.2)** — calibration:
    NFM ≈ ε for small ε. Failure ⇒ magnitudes uninterpretable; can't compare
    across layers/checkpoints.
  - `test_nfm_raises_on_metadata_mismatch` **(V3.5)** — failure ⇒ unmatched
    comparisons pass silently at this surface (V0.4's guarantee lost at the
    API that matters).
  - `test_baseline_fvu_is_base_under_base` **(spec §2 baseline)** — failure ⇒
    dictionary imperfection conflated with novel features.
  - `test_load_sae_roundtrip_local_dir` **(OQ-12)** — save a tiny SAE locally
    in SAELens format in-process, load, compare. Failure ⇒ loader broken or
    network-dependent.
- **Acceptance:** V3.1, V3.2, V3.5 named tests pass; `sae_lens` import is
  function-local; synthetic SAE fixture documented as the §5 test apparatus.
- **Depends on:** ZOO-3.

---

### SAE-2 — Latent re-weighting stats + residual PCs (~1 day)

- **Spec sections:** specs/03 §2 "Latent re-weighting stats", "Residual
  direction analysis", §4 (`latent_stats`, `reweighting_report`,
  `residual_pcs`).
- **Boundaries:** `geode/saediff/stats.py`.
  ```python
  def latent_stats(sae: SAEProtocol, acts: Tensor) -> pd.DataFrame
      # per latent j: f_j = P(a_j > 0), m_j = E[a_j | a_j > 0]
  def reweighting_report(sae: SAEProtocol, acts_base: Tensor, acts_ft: Tensor,
                         top_k: int) -> pd.DataFrame
      # per-latent deltas, ranked movers, L1 over frequency deltas
  def residual_pcs(sae: SAEProtocol, acts: Tensor, k: int
                   ) -> tuple[Tensor, Tensor]   # directions, variance fractions (OQ-13)
  ```
- **Adapted vs fresh:** fresh (elementary statistics + eigendecomposition).
- **Tests** (`tests/saediff/test_stats.py`):
  - `test_pure_reweighting_nfm_stays_zero_stats_move` **(V3.3)** — rescale
    coefficients of existing directions: NFM ≈ 0 while f_j/m_j deltas match
    construction. Failure ⇒ the H1-vs-H2 discriminator is broken — the
    project's central contrast (elicitation = re-weighting vs teaching = new
    features) cannot be measured.
  - `test_latent_freq_and_magnitude_match_construction` **(V3.3)** — f_j, m_j
    equal analytically known values on constructed activations. Failure ⇒
    per-latent stats wrong at the definition level.
  - `test_top_residual_pc_matches_planted_direction` **(V3.4)** — in the V3.2
    setup, top residual PC has |cos| ≥ 0.99 with the planted orthogonal
    direction. Failure ⇒ the "candidate new features" handed to
    crosscoder/steering follow-ups are noise.
  - `test_reweighting_report_ranks_true_movers_topk` — failure ⇒ analysis
    surfaces the wrong latents.
  - `test_l1_frequency_delta_summary_matches_manual` — failure ⇒ summary
    distance wrong.
  - `test_residual_pc_varfracs_sum_le_one_and_ordered` — failure ⇒ variance
    accounting broken.
- **Acceptance:** V3.3 + V3.4 named tests pass; DataFrames carry latent index
  + both models' stats + deltas (reviewer checks columns).
- **Depends on:** SAE-1.

---

### SAE-3 — Streaming equivalence + `run_saediff` orchestrator (~1 day)

- **Spec sections:** specs/03 §3 "Outputs", §4 (`run_saediff`), §5 V3.6.
- **Boundaries:** `geode/saediff/orchestrate.py` + streaming variants in
  `metrics.py`/`stats.py` (batch-iterator inputs; accumulate sufficient
  statistics: sums/counts for FVU and latent stats; residual mean + d×d second
  moment for PCs, per OQ-13).
  ```python
  def run_saediff(run_id: str, hooks: list[str], checkpoints: list[int],
                  *, store=None, batch_size: int = 4096) -> pd.DataFrame
      # orchestrates fvu/nfm/latent/residual metrics over cached activations,
      # emits spec00 §7 long-format rows:
      # metric_name ∈ {fvu, nfm, latent_freq_delta_l1, residual_pc_varfrac_k, ...}
  ```
  Consumes spec 00 §6 activations, `saes/` SAEs, ZOO-1 manifests (regime,
  dataset_size); produces spec 00 §7 rows via ZOO-4.
- **Adapted vs fresh:** fresh. (The reference repo streams a 4TB $DATASTORE
  cache with wandb-coupled infrastructure — wrong shape and wrong scale for
  our spec 00 §6 store; the streaming here is ~50 lines of sufficient-
  statistics accumulation.)
- **Tests** (`tests/saediff/test_orchestrate.py`):
  - `test_streaming_fvu_equals_inmemory`, `test_streaming_nfm_equals_inmemory`,
    `test_streaming_latent_stats_equal_inmemory`,
    `test_streaming_residual_pcs_equal_inmemory` **(V3.6, one per metric
    family)** — batched == in-memory to float tolerance, incl. uneven final
    batch. Failure ⇒ the at-scale code path (real activations won't fit RAM)
    computes different numbers than the validated path — validation theater.
  - `test_run_saediff_emits_long_format_results_rows` **(spec 03 §3 / spec 00
    §7)** — required columns, metric_name vocabulary, keyed by
    run_id/layer/checkpoint_step/dataset_size. Failure ⇒ the H3 bridge join
    (EDL table ⋈ this table on run_id) breaks.
  - `test_run_saediff_joins_with_edl_table_on_run_id` — construct both tables
    in-test and join. Failure ⇒ §3's "no special code path" promise unmet.
- **Acceptance:** all four V3.6 tests pass; orchestrator handles multiple
  hooks × checkpoints; results rows verified against ZOO-4 schema.
- **Depends on:** SAE-2, ZOO-4, ZOO-1.

---

### ADAPT-1 — Crosscoder adaptation plan (document, no code) (~1 day)

- **Spec sections:** specs/04, all headings: "Goal", "What the adaptation plan
  must decide" (items 1–5), "Attribution reminder".
- **Deliverable:** `docs/crosscoder-adaptation-plan.md`. **No code.**
  `reference/` untouched (read-only policy).
- **Prerequisite:** OQ-2 met — reference clone present (plus reading their
  `dictionary_learning` fork on GitHub; it is where the crosscoder/BatchTopK
  classes actually live).
- **Protocol deviation:** four-stage pipeline doesn't apply to a document.
  Two stages instead: AUTHOR (fresh context, reads specs/03–04 + reference
  clone) → ADVERSARIAL-REVIEWER (fresh context, checks each of the five
  decision points is *decided*, not surveyed; checks attribution against
  CLAUDE.md).
- **The document must decide (from specs/04):**
  1. Consumed as-is vs re-wired: expected split is crosscoder module +
     BatchTopK + latent-scaler closed form (`compute_scalers.py`) as-is from
     their fork; activation loading re-wired from `$DATASTORE` to spec 00 §6
     loaders (`collect_activations.py` replaced by our cache). The document
     confirms or corrects this from the actual code.
  2. Scale-down: dictionary size / L0 / training tokens for 1B models +
     narrow arithmetic finetunes, anchored on the repo's own
     `scripts/llama_1b/` configs (Llama-3.2-1B, resid hook L8, BatchTopK
     k=100, lr 1e-4 — they already ran our model class); explicit smoke-test
     config with estimated cost and the `--confirm-cost` rule from CLAUDE.md.
  3. Output mapping into spec 00 §7 rows: per-latent Δnorm, νε/νr latent-
     scaling diagnostics, exclusive-latent lists — exact metric_name
     vocabulary proposed.
  4. Delta-Crosscoder (arXiv 2603.04426) public-code check; if usable, an
     integration sketch and the selection criterion (integration cost +
     smoke-test behavior on a planted synthetic diff).
  5. Synthetic acceptance test mirroring specs/03 V3.2–V3.3: named test
     skeletons (`test_exclusive_latent_detected_on_planted_diff`,
     `test_pure_reweighting_not_flagged_exclusive`) for the future
     implementation spec.
- **Adapted vs fresh:** the document *is* the adaptation analysis; it plans
  maximal reuse of Minder et al. code (per specs/04 item 1), which is exactly
  why no geode module in this plan adapts reference code directly.
- **Acceptance:** all five decision points carry a decision + rationale (not
  options); smoke-test config is concrete (model, hook, dict size, k, token
  count, est. cost); attribution section correctly separates Minder et al.
  2504.02922 / Jiralerspong & Bricken 2602.11729 / Donoway et al.; a follow-up
  spec (`specs/05`) could be written from this document alone.
- **Depends on:** SAE-1, SAE-2 (soft — the acceptance test mirrors their
  fixture design); OQ-2 clone (hard).

---

## Task DAG and topological order

```
SETUP-0 ──► ZOO-1 ──► ZOO-2 ──────────────► EDL-2 ──► EDL-3 ──► EDL-4
   │          │  │                            ▲          ▲
   │          │  └──► ZOO-3 ──► STEER-1 ──► STEER-2 ──► STEER-3
   │          │          │                    │(EDL-2:pgr) ▲
   │          └──► ZOO-4 ┼────────────────────────────────┘
   │                     │        ▲
   ├──► EDL-1 ───────────┼────────┼──► (EDL-2)
   │                     └──► SAE-1 ──► SAE-2 ──► SAE-3
   │                                       │        ▲
   └───────────────────────────────────────┴──(soft)─► ADAPT-1
                                              (ZOO-4 ──► SAE-3)
```

Edges (blocking): SETUP-0→{ZOO-1, EDL-1}; ZOO-1→{ZOO-2, ZOO-3, ZOO-4};
{ZOO-2, EDL-1}→EDL-2; EDL-2→EDL-3→EDL-4; ZOO-3→{STEER-1, SAE-1};
STEER-1→STEER-2; {STEER-2, ZOO-4, EDL-2}→STEER-3; SAE-1→SAE-2;
{SAE-2, ZOO-4, ZOO-1}→SAE-3; {SAE-1, SAE-2 (soft), OQ-2 clone}→ADAPT-1.

**Topological order (single-track):**
SETUP-0, ZOO-1, ZOO-2, ZOO-3, ZOO-4, EDL-1, EDL-2, EDL-3, EDL-4,
STEER-1, STEER-2, STEER-3, SAE-1, SAE-2, SAE-3, ADAPT-1.

After ZOO-4/EDL-1, the EDL, STEER, and SAE tracks are mutually independent
and can interleave freely. ~14.5 task-days serial.

---

## Execution protocol

Restated from CLAUDE.md; binding for every task except the stated deviations
(SETUP-0: single pass + review; ADAPT-1: AUTHOR → ADVERSARIAL-REVIEWER).

Each task runs as **four sequential subagent stages, each in a fresh
context**:

1. **TEST-WRITER** — writes failing tests from the task's referenced spec
   sections *only* (file + headings listed above). Must not read any
   implementation code. Test names come from this plan's test list.
2. **TEST-AUDITOR** — reviews tests against the spec; flags missing
   properties, spec mismatches, and tests that overfit to an assumed
   implementation. Tests revised until clean.
3. **IMPLEMENTER** — makes the tests pass. Must not modify tests.
4. **CONFORMANCE-REVIEWER** — reviews the diff against spec + tests; reports
   gaps with file/line references.

Stage models (owner decision 2026-07-11; `opus` is the capability ceiling):
TEST-WRITER, TEST-AUDITOR, CONFORMANCE-REVIEWER run on `opus`; IMPLEMENTER
runs on `sonnet`, escalated to `opus` for EDL-3, STEER-2, SAE-3 (pre-update
determinism, bit-identical restoration, streaming math). ADAPT-1's AUTHOR
and ADVERSARIAL-REVIEWER run on `opus`.

A task is **done** only when stage 4 reports no findings and the full suite
passes (`pytest -q`: CPU-only, <~2 min, no network). Commit after each
completed task with the task ID in the message. Additional standing rules:
fixture models are random-init tiny configs built in-process; tests validate
mathematical properties from specs, never scientific claims; schema changes
require editing the spec in the same PR; nothing launches GPU work without
`--confirm-cost`.

---

## Coverage map

Every V-number from every spec, with owning task(s) and named test(s). An
uncovered V-number is a planning error.

| V | Spec | Task(s) | Test name(s) |
|---|------|---------|--------------|
| V0.1 | 00 | ZOO-1 | `test_missing_required_field_error_names_field`, `test_null_allowed_only_where_schema_says` |
| V0.2 | 00 | ZOO-1 | `test_roundtrip_preserves_unknown_extra_fields` |
| V0.3 | 00 | ZOO-2 (+EDL-3 producer side) | `test_epoch1_ids_skip_rejected`, `test_epoch1_ids_repeat_rejected`, `test_epoch1_ids_exact_cover_accepted`; `test_epoch1_ids_pass_coverage_checker` |
| V0.4 | 00 | ZOO-3 (re-asserted: STEER-1 V2.5, SAE-1 V3.5) | `test_matched_pair_dataset_key_mismatch_raises`, `test_matched_pair_position_policy_mismatch_raises`, `test_matched_pair_tokenizer_hash_mismatch_raises`, `test_matched_pair_error_message_is_clear` |
| V0.5 | 00 | ZOO-2 (+EDL-2 consumer, EDL-3 producer) | `test_masking_hash_mismatch_raises`; `test_edl_refuses_on_masking_hash_mismatch`; `test_testloss_hash_matches_train_hash` |
| V1.1 | 01 | EDL-4 | `test_random_labels_edl_near_zero`, `test_random_vs_structured_edl_separation` |
| V1.2 | 01 | EDL-4 | `test_learnable_task_edl_positive_curve_descends`, `test_random_vs_structured_edl_separation` |
| V1.3 | 01 | EDL-3 | `test_preupdate_loss_matches_two_pass_reference`, `test_postupdate_corruption_yields_lower_mdl` |
| V1.4 | 01 | EDL-4 (guard unit: EDL-2) | `test_hash_guard_raises_end_to_end` (a), `test_unmasking_prompts_inflates_mdl` (b); `test_edl_refuses_on_masking_hash_mismatch` |
| V1.5 | 01 | EDL-2, EDL-4 | `test_accumulator_rejects_epoch_gt1`; `test_two_epoch_sum_strictly_exceeds_epoch1`, `test_public_api_has_no_multiepoch_mdl_path` |
| V1.6 | 01 | EDL-2, EDL-3 | `test_mdl_equals_jsonl_recomputation`; `test_label_token_sum_matches_tokenizer_count`, `test_epoch1_ids_pass_coverage_checker` |
| V1.7 | 01 | EDL-3 | `test_same_seed_twice_identical_prequential_log` |
| V1.8 | 01 | EDL-2 | `test_bits_equal_nats_over_ln2` |
| V2.1 | 02 | STEER-1, STEER-2 | `test_planted_shift_recovered_cosine_ge_099`; `test_recovered_shift_alpha1_reproduces_tuned_outputs` |
| V2.2 | 02 | STEER-1, STEER-2 | `test_planted_rank2_principal_angles_near_zero`; `test_rank2_patch_reproduces_tuned_rank1_does_not` |
| V2.3 | 02 | STEER-2 | `test_outputs_bit_identical_after_normal_exit`, `test_outputs_bit_identical_after_exception_exit` |
| V2.4 | 02 | STEER-3 | `test_random_control_pgr_near_zero_recovered_near_one` |
| V2.5 | 02 | STEER-1 | `test_e1_raises_on_metadata_mismatch` |
| V2.6 | 02 | STEER-1 | `test_e2_matches_svd_of_materialized_lora` |
| V3.1 | 03 | SAE-1 | `test_in_span_reconstruction_fvu_near_zero`, `test_nfm_of_data_against_itself_near_zero` |
| V3.2 | 03 | SAE-1 | `test_nfm_monotone_in_planted_orthogonal_energy`, `test_nfm_approx_epsilon_for_small_epsilon` |
| V3.3 | 03 | SAE-2 | `test_pure_reweighting_nfm_stays_zero_stats_move`, `test_latent_freq_and_magnitude_match_construction` |
| V3.4 | 03 | SAE-2 | `test_top_residual_pc_matches_planted_direction` |
| V3.5 | 03 | SAE-1 | `test_nfm_raises_on_metadata_mismatch` |
| V3.6 | 03 | SAE-3 | `test_streaming_fvu_equals_inmemory`, `test_streaming_nfm_equals_inmemory`, `test_streaming_latent_stats_equal_inmemory`, `test_streaming_residual_pcs_equal_inmemory` |

Non-V coverage worth naming: spec 00 §7 (results tables — no V-number) is
covered by ZOO-4's tests + `test_ladder_rows_conform_to_results_schema` +
`test_run_saediff_emits_long_format_results_rows`; spec 02 §4 C2/C3 (controls
without V-numbers) by `test_shuffled_pairing_yields_low_pgr_direction` and
`test_norm_matched_directions_have_equal_budget`. Specs/04 produces no code
and has no V-numbers; its acceptance is ADAPT-1's document criteria.
