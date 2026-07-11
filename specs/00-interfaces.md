# specs/00-interfaces.md — Shared Interfaces and Storage Schemas

Everything downstream (EDL harness, steering library, base-SAE pipeline,
external repo adapters) reads and writes through these schemas. External code
(Donoway repo, Minder repo) is wrapped behind adapters that emit these
formats; nothing else in `geode` may depend on external repo internals.

## 1. Directory layout

```
$GEODE_STORE/
  runs/{run_id}/
    manifest.json
    adapter/                  # PEFT adapter (final)
    merged/                   # optional merged weights, safetensors
    snapshots/step_{k}/       # within-run PEFT adapter snapshots
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

`$GEODE_STORE` is an environment variable; no absolute paths in code.

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
                   "weight_decay": "float"},
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

Rationale: `regime` is recorded at creation from experimental design, so
analysis code can group runs without re-deriving it. `snapshot_steps` is
declared up front — the checkpoint schedule is designed before training
(rerunning training to recover a missing checkpoint is the expensive
failure mode).

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

## 7. Results tables

Analysis outputs are parquet files in `results/`, long format, one row per
measurement, with at minimum the columns:
`run_id, base_model_key, regime, dataset_size, checkpoint_step, layer,
metric_name, metric_value`.
This is the join key structure for the EDL-vs-internals bridge plots: EDL
values (from the harness) and internal quantities (from steering/saediff)
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
```

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
