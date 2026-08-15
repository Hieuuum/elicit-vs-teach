# specs/00-interfaces.md — Shared Interfaces and Storage Schemas

Everything downstream (EDL harness, training runs, probe extraction, analysis
drivers) reads and writes through these schemas. External code
(Donoway repo, Minder repo) is wrapped behind adapters that emit these
formats; nothing else in `geode` may depend on external repo internals.

## 1. Directory layout

```
$GEODE_STORE/
  runs/{run_id}/
    manifest.json
    model/                    # final save_pretrained checkpoint:
                              #   model.safetensors = the complete state_dict
                              #   (base + adapter tensors — the FINAL checkpoint
                              #   stays self-contained; zoo.load_model V0.9)
                              #   adapter.safetensors = OPTIONAL sidecar (LoRA
                              #   runs only, 2026-07-31): A/B tensors only,
                              #   metadata base_model/base_revision/lora_rank/
                              #   lora_alpha — see "Adapter sidecar" below
    model_merged/             # OPTIONAL: a LoRA install run's adapter folded
                              #   into plain weights (scripts/merge_adapter.py,
                              #   geode.train.merge_lora) for cross-stage parent
                              #   handoff — loadable by plain from_pretrained,
                              #   never zoo.load_model (it is not method-tagged)
    train_log.jsonl           # per-step trainer log (spec 02 §6.1 contract)
    eval_log.jsonl            # periodic held-out evals (same contract)
    training_meta.json        # trainer config echo + stop record
    snapshots/                # runs 5-6 (adapter-only format 2026-07-22,
                              # supersedes the 2026-07-18 self-contained one):
      base/model.safetensors  #   frozen base + buffers, written once per run
                              #   (tied aliases stored once, restored on load)
      step_{k}/               #   adapter.safetensors = exactly the trainable
                              #   (A/B) tensors at θ_k; reassemble via
                              #   geode.edl.load_snapshot (bit-exact, V1.11;
                              #   legacy full model.safetensors still loads)
    sft_snapshots/            # OPTIONAL: train_sft.py `train.snapshot_steps`
                              #   (2026-08-15, V5.77):
      step_{k:07d}/           #   full save_pretrained dirs at the listed
                              #   steps (wrapped state for LoRA runs — load
                              #   ONLY via zoo.load_model(run, checkpoint=dir));
                              #   probe/diagnostic artifact, expendable,
                              #   replay-derivable. Distinct from `snapshots/`
                              #   above (different trainer, different
                              #   contract) — hf_checkpoint.py's default
                              #   ignore patterns name `snapshots/*`, NOT this
                              #   dir, so a plain push ships every byte here;
                              #   `--no-weights` excludes only
                              #   `model.safetensors` (adapter sidecars still
                              #   ride) and `--metadata-only` excludes every
                              #   `*.safetensors` under it, same as elsewhere.
    probe/step_{k}/           # offline probe dumps (spec 02 §7):
                              #   acts.safetensors + grads.safetensors (bf16,
                              #   n_layers+1 named residual tensors each),
                              #   probe_data.safetensors (ids/masks + fp32
                              #   per-example loss), meta.json sidecar
    logs/
      prequential.jsonl       # per-batch first-epoch label losses (§3)
      gradstats.jsonl         # per-step gradient statistics (§4)
    eval/
      test_loss.json          # final held-out loss (§5)
      task_metrics.json       # accuracy etc., task-defined
  activations/{model_key}/{dataset_key}/{hook_name}.safetensors   # §6
  saes/{model_key}/{hook_name}/                                   # SAELens format
  results/                    # analysis outputs, parquet (§7)
```

Flat run layout (2026-07-21): the final checkpoint lives at
`runs/{run_id}/model/` and the trainer's files at the run root. Before
this, runs nested everything under a phase dir named for how they were
trained (`pretrain/` for run 1, `sft/` for runs 2-4), forcing every
consumer to hardcode or guess the phase name. Legacy support: readers
(`geode.zoo.checkpoint_dir`, monitor.py, hf_checkpoint.py) accept both
layouts; a store converts only explicitly, via
`experiments/training-run/scripts/migrate_store_layout.py`, after its
runs finish — never implicitly, and never mid-training.

Adapter sidecar (2026-07-31): every LoRA run — whether launched via
`train_target.py` (LoRA target runs) or `train_sft.py` (LoRA install runs,
e.g. the fig-2 Llama installer, 2026-07-31) — writes an OPTIONAL
`runs/{run_id}/model/adapter.safetensors` immediately after the
self-contained `model.safetensors` save above — full-FT runs never write
this file. It holds exactly the trainable `.A.weight`/`.B.weight` tensors
(`train.lora_adapter_state_dict`, the shared filter both launchers import),
unmerged, same dtype as trained — the same filter `geode.edl.loop` uses for
`snapshots/step_{k}/adapter.safetensors`. Safetensors metadata (str -> str):
`base_model` (the run's resolved base identifier, `manifest.base_model.hf_id`),
`base_revision` (`manifest.base_model.revision`; `"none"` means the default
revision, not a literal ref to pass through), `lora_rank`, `lora_alpha`.
Reconstruction contract is NOT `geode.train.lora.reapply_lora` — that
strict-loads a FULL state dict, and this sidecar holds only the A/B
tensors: `from_pretrained(base_model[, revision=base_revision])`, then
`geode.train.lora.apply_lora(rank=lora_rank, alpha=lora_alpha)`, then
`load_state_dict(these A/B tensors, strict=False)` — the same base-then-
adapter assembly `load_snapshot` does for
`snapshots/step_{k}/adapter.safetensors` (there `strict=True`, because its
paired base file supplies every other key; here there is no base file, so
`strict=False`). Purpose is cheap recoverability (~90MB vs ~2.5GB) for
`hf_checkpoint.py push --no-weights`, which excludes `model.safetensors`
(plus, always, the derivable `model_merged/` — 2026-07-31) so this sidecar
still ships on metadata-scale pushes;
`model.safetensors` remains the pinned, canonical self-contained checkpoint
every loader (`zoo.load_model` V0.9) targets — this sidecar is never a
substitute for it.

`$GEODE_STORE` is an environment variable; no absolute paths in code.
When it is unset, launch scripts (`experiments/`) default it to
`<repo-root>/geode-store/` — gitignored, so artifacts sit beside the
clone (2026-07-20 owner decision; on rental boxes this lands on the
same volume as the checkout). Library code (`geode.zoo.store`) stays
strict: explicit `store` argument or the env var, never a guessed path.

## 2. Run manifest (`manifest.json`)

Required fields. Unknown extra fields are permitted and preserved.

```json
{
  "schema_version": 1,
  "run_id": "str, unique, filesystem-safe",
  "created_utc": "ISO 8601",
  "git_commit": "str",
  "regime": "elicit | teach | unknown",
  "base_model": {"hf_id": "str", "revision": "str"},
  "task": {"name": "str", "format_version": "str"},
  "dataset": {"name": "str", "n_unique_examples": "int", "seed": "int"},
  "training": {
    "method": "lora | sparse_lora | full_ft",
    "lora": {"rank": "int|null", "alpha": "float|null",
              "target_modules": ["str"], "dropout": "float|null",
              "sparse_param_count": "int|null"},
    "optimizer": {"name": "str", "lr": "float", "batch_size": "int",
                   "micro_batch_size": "int|null", "betas": "[float]|null",
                   "weight_decay": "float", "grad_clip": "float|null"},
    "lr_schedule": "constant | cosine",
    "min_lr": "float|null",
    "precision": "fp32 | bf16",
    "eval_every": "int|null",
    "max_steps": "int|null",
    "stopping": {"eps_nats": "float|null", "k": "int|null",
                  "min_steps": "int|null"}
                | {"metric": "format_validity", "threshold": "float",
                   "k": "int", "n_prompts": "int", "prompt_seed": "int"}
                | {"metric": "train_loss", "eps_nats": "float",
                   "k": "int", "min_steps": "int|null"},
    "epochs_total": "int",
    "seed": "int"
  },
  "trainable_param_count": "int",
  "snapshot_steps": ["int"],
  "cost": {"gpu_type": "str|null", "est_usd": "float|null",
            "actual_usd": "float|null"},
  "status": "planned | running | complete | failed"
}
```

Optional lifecycle metadata (2026-07-26) — two extra fields may follow
`status`:

- `lifecycle`: `"canonical | superseded | pilot | invalid"` — the
  scientific standing of a completed run, orthogonal to `status` (which
  records process completion and stays load-bearing for parent gating).
  Absent ⇒ treat as canonical.
- `superseded_by`: `run_id` of the replacement; present only when
  `lifecycle` is `superseded` or `invalid`.

Both are advisory archive metadata for humans and analysis scripts; zoo
gating never consults them.

Target-run result extras (2026-07-30) — the LoRA target launcher
(`train_target.py`) records an `experiment.target_result` object at finalize
(an unknown-extra field, spec-preserved but not required by every trainer):

```json
"target_result": {
  "final_step": "int",
  "stop_reason": "converged | max_steps",
  "best_val_nats": "float|null",
  "min_val_nats": "float",
  "edl_epoch1_nats": "float",
  "edl_per_label_token_nats": "float",
  "edl_per_example_nats": "float",
  "epoch1_examples": "int, optional"
}
```

All loss/EDL values are **nats**; bits only at reporting boundaries (V1.8).

- `min_val_nats`: the **global minimum** `val_loss_nats` over every row of
  `eval_log.jsonl` — stopping evals AND curve evals alike — NOT the value at
  the run's stop time. (`best_val_nats` stays the ε-gated stopping-tracker
  quantity used for the convergence decision itself; it can differ from
  `min_val_nats` whenever a denser curve eval dips lower between stopping
  evals.) Computed by `geode.edl.metrics.min_val_nats_from_eval_log`.
- `edl_epoch1_nats`: EDL over the run's own epoch-1 `prequential.jsonl`
  stream, floored against `min_val_nats` rather than the held-out
  `eval/test_loss.json` loss that `geode.edl.metrics.edl_nats` (specs/01 §1)
  uses — i.e.
  `(Σ epoch-1 loss_sum_nats) − (Σ epoch-1 label_token_count) · min_val_nats`.
  The floor is **this run's own global minimum in-loop val loss per label
  token** (the quantity above), not `n_val_examples · min_val_nats` — a
  training-stream MDL is dimensioned in training-stream label tokens, not
  held-out eval tokens (Eq. 3, owner-accepted 2026-07-30). Computed by
  `geode.edl.metrics.edl_epoch1_nats`.
- `edl_per_label_token_nats`: `edl_epoch1_nats` divided by the epoch-1
  label-token count (token-weighted). Computed by
  `geode.edl.metrics.edl_epoch1_per_label_token`.
- `edl_per_example_nats`: `edl_epoch1_nats` divided by the epoch-1 example
  count (example-weighted, not token-weighted — distinct denominator from
  the field above). Computed by `geode.edl.metrics.edl_epoch1_per_example`.
- `epoch1_examples` (added 2026-08-14, ts38-mini guard 1): examples consumed
  in epoch 1 — the third element `geode.edl.metrics.epoch1_totals` returns,
  summing `len(example_ids)` over epoch-1 `prequential.jsonl` records (§3).
  Equals `n_examples` **iff** epoch 1 ran to completion, per the §3
  enumeration invariant (concatenating epoch-1 `example_ids` enumerates each
  unique training example exactly once). Written for every run
  `train_target.py` finalizes, not only runs that opt into the guard below.
  **Optional-on-read**: manifests written before this field existed lack it;
  a validator or reader must not require its presence. When the launch-time
  flag `experiment.require_full_epoch1` is set (specs/02 V5.75), the
  launcher additionally requires `epoch1_examples == n_examples` before
  flipping the manifest's `status` to `"complete"` — a mismatch means epoch
  1 was truncated, and the run is left at `status: "running"` rather than
  recorded a clean success (specs/02 V5.76).

Rationale: `regime` is recorded at creation from experimental design, so
analysis code can group runs without re-deriving it. `snapshot_steps` is
declared up front — the checkpoint schedule is designed before training
(rerunning training to recover a missing checkpoint is the expensive
failure mode).

The full training recipe (`lr_schedule` through `stopping`, plus the
optimizer extras; added 2026-07-20) is required so the manifest alone
answers "how exactly was this trained" — before this, telling a cosine
run from a constant-LR run meant digging through
`pretrain/training_meta.json`. Record **resolved** values (e.g. the
precision actually used after any CPU fallback, `micro_batch_size` after
defaulting to `batch_size`). Fields are `|null` only where a training
mode has no such concept (SGD has no `betas`; the prequential EDL loop
has no held-out eval, step cap, or plateau stopping). Run-1 manifests
written before this change were backfilled from their
`training_meta.json`. `stopping` is a union (2026-07-21): loss-stopped
runs record the ε/k rule; the behavior-stopped format installers (runs
3–4, spec 02 §6) record the in-loop format-validity rule instead.
Extended 2026-07-26 to dispatch on the `metric` *value*, adding a
`train_loss` branch for the new-phase dose installers (spec 02 §6): the
same ε/k fields, but labelled, because the metric is the full-dose
**training** loss and not a held-out one — an unlabelled ε/k record would
read as a val curve. A `metric` value outside the listed branches is a
validation error, so the discriminant stays closed.

## 3. Prequential log (`prequential.jsonl`)

One JSON object per optimizer batch, **first epoch records are mandatory**;
later epochs may be logged but are never used for MDL.

```json
{"step": int, "epoch": int, "example_ids": [int],
 "label_token_count": int, "loss_sum_nats": float}
```

- `loss_sum_nats`: sum (not mean) of cross-entropy over **label tokens
  only**, evaluated with the parameters as they were **before** the update
  on this batch.
- `label_token_count`: number of tokens the loss was summed over.
- Invariant: concatenating `example_ids` over epoch 1 enumerates each unique
  training example exactly once.

## 4. Gradient statistics (`gradstats.jsonl`)

One object per logged step (logging stride configurable):

```json
{"step": int, "global_grad_norm": float,
 "per_module_grad_norm": {"module_name": float},
 "topk_grad_subspace_overlap": float | null}
```

The last field (cosine overlap of the current gradient with a running top-k
subspace) may be null when not computed; its precise definition belongs to
the analysis spec that consumes it, not here.

## 5. Test loss (`eval/test_loss.json`)

```json
{"n_test_examples": int, "label_token_count": int,
 "loss_sum_nats": float, "loss_per_label_token_nats": float,
 "masking_config_hash": "str"}
```

`masking_config_hash` must equal the hash recorded by the training loop —
this is the mechanical guard for the train/test mask-parity footgun.

## 6. Activation storage

Safetensors, one file per (model, dataset, hook). Tensor name = hook name
(TransformerLens convention, e.g. `blocks.7.hook_resid_post`). Shape
`[n_samples, n_positions_kept, d_model]`, dtype float16 unless specified.
A sidecar `{hook_name}.meta.json` records: model_key, hf_id + revision,
dataset_key, position policy (`all | answer_only | last`), sample count,
tokenizer hash. Matched-input comparisons across models require identical
dataset_key, position policy, and tokenizer hash; loaders must enforce this.
A matched pair must also have identical row counts; if per-row example
identifiers are stored in the sidecar, they must match element-wise.
`load_matched_pair` refuses row-count or per-row-id mismatches the same way
it refuses the other matched-input mismatches.

## 7. Results tables

Analysis outputs are parquet files in `results/`, long format, one row per
measurement, with at minimum the columns:
`run_id, base_model_key, regime, dataset_size, checkpoint_step, layer,
metric_name, metric_value`.
This is the join key structure for the EDL-vs-internals bridge plots: EDL
values (from the harness) and internal quantities (from analysis drivers)
land in the same table shape and join on `run_id`/`dataset_size`.

## 8. Public API surface (module `geode.zoo`)

```python
class RunManifest:          # load/validate/save manifest.json (schema §2)
def register_run(...) -> RunManifest
def load_run(run_id: str) -> RunManifest
def iter_runs(regime: str | None = None, task: str | None = None,
              status: str = "complete") -> Iterator[RunManifest]
def prequential_records(run_id: str) -> Iterator[PrequentialRecord]
def test_loss(run_id: str) -> TestLoss
def load_model(run_id: str, *, store=None, device="cpu",
               checkpoint=None) -> nn.Module   # V0.9 — the ONLY way to
               # load a run checkpoint for eval/analysis; dispatches on
               # the manifest's training.method
```

## 9. Decisions (owner-approved 2026-07-11)

Noted here from the deleted PLAN.md (2026-07-17 cut) because they explain
why the shipped code behaves as it does. `OQ-n` labels are kept only
because existing docstrings already use them — they are a reference note,
not a contract.

- **OQ-3 — manifest validation:** all listed keys required recursively;
  `null` only where the schema says `|null`; primitive type checks; errors
  name the dotted path (e.g. `training.optimizer.lr`).
- **OQ-4 — `example_ids` universe:** ids are `0..n_unique_examples-1` from
  the run's manifest; checker rejects skips, repeats, out-of-range.
- **OQ-5 — position policy:** one shared enum `{all, answer_only, last}`;
  "generated positions" ≡ `answer_only`.
- **OQ-6 — results layout:** one parquet per analysis,
  `results/{analysis_name}.parquet`, overwrite-by-name; `read_results`
  concatenates the directory; joins in pandas on mandated key columns.
- **OQ-7 — hashes:** `tokenizer_hash` = sha256 of canonical tokenizer JSON;
  `masking_config_hash` = sha256 of canonical JSON of
  `{task name, format_version, mask-rule parameters, tokenizer_hash}`.
- **OQ-8 — label boundary:** dataset builder emits per-example label-token
  spans; `task_format` declares the span rule; `label_mask` is the single
  construction path for training loop and test evaluator (M1's shared fn).
- **OQ-14 — `topk_grad_subspace_overlap`:** always `null` for now; field
  reserved for a future analysis spec.

## Validation properties (tests derive from these)

- **V0.1** A manifest missing any required field fails validation with an
  error naming the field.
- **V0.2** Round-trip: `save(load(m)) == m` including unknown extra fields.
- **V0.3** A prequential log whose epoch-1 `example_ids` skip or repeat an
  example is rejected by the invariant checker.
- **V0.4** Loader refuses matched-input activation comparison when
  dataset_key, position policy, or tokenizer hash differ, with a clear error.
- **V0.5** `masking_config_hash` mismatch between train log and test loss is
  surfaced as an error by `geode.zoo` consistency check.
- **V0.6** Parent-gate refusal (EXPERIMENTS.md §3.1, added 2026-07-20 with
  the runs-2–4 launch surface): `require_parent_ready` raises
  `ConsistencyError` when the parent manifest is missing or invalid, its
  `status` is not `"complete"`, any recorded `experiment.gates` entry lacks
  `pass: true`, or a caller-required gate has no recorded verdict. Gate
  records are objects with at least a boolean `pass` field.
- **V0.7** `training.stopping` union (2026-07-21, extended 2026-07-26): an
  object without `metric` validates against the val-loss ε/k field set only;
  one with `metric` validates against the branch that value selects
  (`format_validity` behavioral, `train_loss` full-dose ε/k) and against no
  other. No branch demands another's fields, and an unlisted `metric` value
  is a validation error.
- **V0.8** Checkpoint resolution (2026-07-21 flat-layout migration):
  `geode.zoo.checkpoint_dir(run_id)` returns `runs/{run_id}/model` when it
  contains `model.safetensors`, else the single legacy `{phase}/model`
  (glob `*/model/model.safetensors`). Exactly one candidate must exist
  across both patterns; zero or several raises an error naming the run
  dir — never a guess. Snapshot files (`snapshots/step_{k}/`, no `model/`
  component) are never candidates.
- **V0.9** Method-faithful checkpoint loading (2026-07-22 G5 incident):
  `geode.zoo.load_model(run_id)` loads the final checkpoint the way the
  run's manifest says it was trained — `training.method: "full_ft"` via
  plain `from_pretrained`; `"lora"` by rebuilding the `apply_lora` module
  tree from `training.lora` and loading the wrapped state dict (bit-exact
  with the model that was saved, V5.51). A wrapped state dict
  (`*.base.weight`/`*.A.weight`/`*.B.weight` keys) under a `full_ft`
  manifest — or a plain one under `"lora"` — is a loud refusal: plain
  `from_pretrained` on a wrapped checkpoint does not error, it silently
  random-initializes every wrapped projection. Tied-embedding checkpoints
  (`tie_word_embeddings: true`, where `save_pretrained` drops the
  `lm_head.weight` duplicate from safetensors) load bit-exactly on the
  lora path too: the tied alias is restored before the strict load.
