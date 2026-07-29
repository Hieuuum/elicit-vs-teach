# Arm-B 1M LR sweep — box paste sheet (pilot for runs 7/8)

> **Historical runbook** — the runs this file walks through (the runs 7/8
> pair, the Arm-B 1M LR sweep) are closed. Paths were updated 2026-07-29
> for the archive reorg. The launchers it references are frozen
> byte-identical in `scripts/archive/` and reference pre-reorg paths
> internally, so they are **not re-runnable as-is** — see the path-mapping
> table at the top of `notes/decisions.md`.
> Paths are relative to `experiments/training-run/` unless they start with
> `specs/` or `docs/`.

Re-pins `train.lr` before the runs-7/8 pair can launch (owner 2026-07-23/24).
Grid points as overlays on `archive/runs/run8_target_1m.yaml`, each capped at **one
epoch** of the full 1M (`max_steps` 7813) with **no snapshots**. First pass
(2026-07-24): 3e-4 → 0.1063, 1e-3 → 0.0397, 3e-3 → 0.0343 min_val nats —
3e-3 won at the top edge, so the grid was EXTENDED to {…, 1e-2} per the
edge rule below. Extension result (same day): 1e-2 → **1.8674** nats,
"converged" at step 5500 — the ε/k rule firing on a plateau at garbage.
**PIN RESOLVED 2026-07-24: lr 1e-3 everywhere.** The Arm-A 100K pilot
pair overrode the B-sweep's capped-budget winner: 3e-3 → 0.0122 nats vs
1e-3 → 0.0039 (identical overlays, only the LR differs). All four run
yamls + `configs/lr_pin.yaml` carry the pin; decisions.md has the full
record. §2/§3 below are history. For sweep runs `stop_reason=max_steps` is the
expected outcome, not a bug signal (documented exception, decisions.md
2026-07-24).
Design notes live in `configs/sweeps/target_1m/target_sweep_1m_lr3e-4.yaml`.
State (owner 2026-07-24): the old box was DELETED — the four sweep run
dirs and the runs-5/6 snapshots + final weights went with it
(owner-accepted; runs-5/6 manifests + all logs + θ_T test loss survive on
the relay and laptop, so the headline numbers stand). Extraction now
targets runs 7/8 exclusively. Everything below runs on the new chain box.

## 1. Sync the box

Laptop: push, note `git rev-parse --short HEAD`. Box (already set up):

```bash
cd /workspace/elicit-vs-teach && git pull && git log --oneline -1  # hash must match laptop
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store          # store INSIDE the clone
export NTFY=ntfy.sh/<your-topic>
cd experiments/training-run/scripts
```

Fresh box instead (e.g. a new chain box for `scripts/archive/launch_chain_7_10.sh`)?
Template runs `box_onstart.sh` (clone + install + CPU suite + exports; see
its header for the HF_TOKEN/NTFY template vars — READ token, from the
Meta-licensed account if the Llama chain runs here). Then in an SSH
session, after the laptop-hash check above:

```bash
hf auth login --force            # READ token; --force always (owner 2026-07-24)
python3 hf_checkpoint.py pull --run-id evt-run3-armA-inst   # run 7's + pilot's parent
python3 hf_checkpoint.py pull --run-id evt-run4-armB-inst   # run 8's parent
python3 verify_llama_tokenizer.py  # early Meta-license/token check
```

(No sweep pull — those run dirs were lost with the old box. The chain's
LR guard verifies the pin against the committed `configs/lr_pin.yaml`
instead, created at pin time in §3. The sweep runner `sweep_1m.sh` was
DELETED 2026-07-24 — foot-gun removal: with nothing complete in a fresh
store it would have retrained all four points.)

## 2. Launch — 7,813 steps per point, sequential

```bash
for lr in 3e-4 1e-3 3e-3 1e-2; do
  python3 train_target.py --config ../configs/archive/runs/run8_target_1m.yaml \
      --override ../configs/sweeps/target_1m/target_sweep_1m_lr${lr}.yaml \
      --init-from $GEODE_STORE/runs/evt-run4-armB-inst/model --confirm-cost \
    ; curl -d "1M sweep lr=${lr} done (exit $?)" $NTFY
done
```

Or use the runner: `./sweep_1m.sh --confirm-cost` — same grid, skips
already-completed points on re-run (so after the first pass it launches
only the 1e-2 extension), stops on a failed point, and ends with the
winner summary + edge-rule verdict.

Watch from a second window: `python3 monitor.py --run-id evt-run8-sweep-lr<X>`
(prints steps/s + ETA — the first point calibrates wall-clock for the other
two). Disk is a non-issue here: no snapshots, <~1 GB per run (final model +
logs).

## 3. Winner → pin (laptop)

Winner = lowest stopping-block `min_val_nats` at the shared 1-epoch budget
(a point that ε/k-converges earlier is fine — compare its floor the same
way). **Edge rule:** an edge win extends the grid one step in that
direction before pinning — fired once (3e-3 → added 1e-2) and RESOLVED:
1e-2 plateaued at 1.8674, 3e-3 is the interior winner. **Remaining step:**
the 100K-prefix Arm-A sanity pilot before pinning — 3e-3 is the only grid
point never proven on Arm A (3e-4 and 1e-3 are). It runs on the NEW box
as its first GPU job (a good end-to-end shakedown before the chain);
parent pull is in §1.

```bash
python3 train_target.py --config ../configs/archive/runs/run7_target_1m.yaml \
    --override ../configs/sweeps/target_pilot/target_pilot_100k_armA_lr3e-3.yaml \
    --init-from $GEODE_STORE/runs/evt-run3-armA-inst/model --confirm-cost \
  ; curl -d "armA 3e-3 pilot done (exit $?)" $NTFY
```

PASS = `stop_reason=converged` **AND** a small min_val (run-5 reference
floor ~0.0025 nats). "converged" alone is not a pass — the ε/k rule fires
on any plateau (the 1e-2 point "converged" at 1.867). `max_steps` or a
high plateau = do NOT pin 3e-3; fall back toward 1e-3 (~0.005 nats behind)
and bring the numbers to the owner.

Then, laptop: set `train.lr` in **ALL FOUR** run yamls (run7/run8/run9/
run10 — one LR everywhere, owner 2026-07-24) **and commit the pin record
`configs/lr_pin.yaml`** (lr + provenance — the chain's LR-guard evidence
now that the sweep manifests are gone), record it in decisions.md,
commit + push; box `git pull` and confirm the hash before the pair
launches. Launch order and gate mechanics for the pair are in the configs'
headers (run 7 first — G7).

Then the pair runs unattended: `./archive/launch_pair_1m.sh --confirm-cost` (retired; frozen copy) (or
`./archive/launch_chain_7_10.sh --confirm-cost` to continue straight into the Llama
chain — it skips completed runs, so it also picks up after the pair) — checks
the pin equals the sweep winner in the store, ≥200 GB free, parents present;
run 7 then run 8 with NTFY pings; skips a completed run on re-run.

## 4. Extraction — runs 7/8 snapshots → probe dumps → alignment plot

The internals pass (spec 02 §7). Disk math (2026-07-24): one dump is
~0.5 GiB (1024 probe examples × 28 tokens × d512 × 9 hooks, acts+grads
bf16); full density would be ~0.6 TiB/run, so extraction runs at
`--limit 128` per run (~129 GiB both, evenly index-spaced over the
materialized snapshot list, first + final kept). Check
`df -h /workspace` ≥ 150 GB free first. Resumable — existing dumps are
skipped; a later re-run with a larger `--limit` only adds density.
Dumps are NOT relayed: they regenerate from the hub snapshots for ~$1
of GPU, unlike the snapshots themselves.

```bash
cd /workspace/elicit-vs-teach && git pull  # hash must match laptop
cd experiments/training-run/scripts
python3 extract.py --config ../configs/archive/runs/run7_target_1m.yaml \
    --run-id evt-run7-armA-target-1m \
    --probe-hash 2b6d51c27fce0e69b6b0b7d2f033fcc720e39ade287bb31350cb6b2f6fb562e2 \
    --limit 128 --device cuda --confirm-cost \
  ; curl -d "run7 extraction done (exit $?)" $NTFY
python3 extract.py --config ../configs/archive/runs/run8_target_1m.yaml \
    --run-id evt-run8-armB-target-1m \
    --probe-hash 2b6d51c27fce0e69b6b0b7d2f033fcc720e39ade287bb31350cb6b2f6fb562e2 \
    --limit 128 --device cuda --confirm-cost \
  ; curl -d "run8 extraction done (exit $?)" $NTFY
```

Then the first analysis driver (CPU, minutes):

```bash
python3 ../analysis/alignment.py
```

Writes `geode-store/results/gradient_alignment.parquet` (ZOO-4
long-format) and `analysis/figures/gradient_alignment.png`. Copy both
back to the laptop (scp — a few hundred KB); the box is teardown-safe
once they're off and all four runs are verified on the hub.

## Troubleshooting

| Symptom | Fix |
|---|---|
| box `git log` ≠ laptop hash | forgot to push, or stale clone — `git pull` |
| `parent run ... has no manifest` | stale `$GEODE_STORE` (new tmux window) — re-export |
| `order_hash mismatch` | downloaded parquet ≠ frozen file — stop, don't work around |
| `register_run: already running` | double-launch guard — `pgrep -f train_target` |
| stops at 7,813 with `stop_reason=max_steps` | expected — the 1-epoch cap IS the sweep design |
