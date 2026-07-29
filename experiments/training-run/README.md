# training-run — experiment card

Mechanistic comparison between a model whose target-task capability is
**latent** (Arm A, `armA_elicit`: algorithm pre-taught, format installed)
and one where it is **absent** (Arm B, `armB_teach`: format installed
only). Design source of truth: [`specs/02-training-run.md`](../../specs/02-training-run.md).
Never reuse the paper's "pre-elicit" (E.1.1) for Arm A's E.2-style
pre-teaching.

## Run DAG

```mermaid
graph LR
  R1[run1 pretrain\nTinyStories-1B] --> R2[run2 armA algo\nNL add/sub 1M]
  R2 --> R3[run3 armA installer\nmult random labels]
  R1 --> R4[run4 armB installer\nidentical to run3]
  R3 --> R5[run5 armA target\nLoRA op-notation add/sub]
  R4 --> R6[run6 armB target\nLoRA identical data+order]
```

## Status

Live run/gate status lives in [`EXPERIMENTS.md`](../../EXPERIMENTS.md);
decision history in `notes/decisions.md`.

## Layout (lifecycle split, 2026-07-24)

- `configs/` — one YAML per run; `common.yaml` shared blocks;
  `pilot/` overlays (pilot uploads go to the separate `-pilot` HF repo).
- `datagen/` — one-time dataset + tokenizer generation (outputs frozen:
  datasets on HF, tokenizer committed under `tokenizer/`).
- `scripts/` — GPU/box operations: trainers, launchers, gates, relay,
  monitoring, box provisioning. Cost paths **refuse without
  `--confirm-cost`**. Paths in here are load-bearing (box paste sheets
  in `docs/`, sibling imports, the vast.ai template) — don't move files.
- `analysis/` — CPU post-hoc drivers (→ `geode-store/results/`) and
  plotting; all figures land in `analysis/figures/` (gitignored).
- `notes/` — `decisions.md` running log; pilot outcomes close spec 02
  OPEN items there first, then the spec.
- `_lib/` — shared trainer boilerplate (`REPO_ROOT`, `load_config`, the
  phase banner, `git_commit`), single-sourced from `scripts/train.py`;
  the forward-looking import home for the next trainer/driver.
- `scripts/lib/launch_common.sh` — sourceable launcher glue (`notify`,
  `fail`, `milestone`, `status_of`, `gate_recorded`, `stop_reason_of`) for
  the next launcher to `source` instead of re-declaring inline.

### Deliberate non-goal: per-trainer manifest skeletons

The `manifest_fields` builders in `train.py`, `train_sft.py`,
`train_target.py`, and `train_embedding_warmstart.py` stay **duplicated on
purpose**. Each trainer's manifest carries a different resolved shape
(pretrain vs. SFT vs. LoRA-target vs. embedding warm-start), and those
shapes drift independently as the experiment evolves. A shared builder
would couple unrelated schemas and force every trainer to move in lockstep;
the boilerplate hoisted into `_lib/` is only the truly-common glue, not the
manifest bodies.

## Launching run 1 (once OPEN(8)/OPEN(11) are pinned)

```bash
export GEODE_STORE=/path/to/store
python scripts/train.py --config configs/run1_pretrain.yaml            # prints cost, refuses
python scripts/train.py --config configs/run1_pretrain.yaml --confirm-cost
```
