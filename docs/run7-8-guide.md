# Arm-B 1M LR sweep — box paste sheet (pilot for runs 7/8)

Re-pins `train.lr` before the runs-7/8 pair can launch (owner 2026-07-23/24).
Grid points as overlays on `run8_target_1m.yaml`, each capped at **one
epoch** of the full 1M (`max_steps` 7813) with **no snapshots**. First pass
(2026-07-24): 3e-4 → 0.1063, 1e-3 → 0.0397, 3e-3 → 0.0343 min_val nats —
3e-3 won at the top edge, so the grid is EXTENDED to {…, 1e-2} per the
edge rule below. For sweep runs `stop_reason=max_steps` is the expected
outcome, not a bug signal (documented exception, decisions.md 2026-07-24).
Design notes live in `configs/pilot/target_sweep_1m_lr3e-4.yaml`.
Sequencing (owner 2026-07-24): extraction is deferred — this sweep goes
first; the box must stay alive regardless (runs-5/6 snapshots live only
there).

## 1. Sync the box

Laptop: push, note `git rev-parse --short HEAD`. Box (already set up):

```bash
cd /workspace/elicit-vs-teach && git pull && git log --oneline -1  # hash must match laptop
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store          # store INSIDE the clone
export NTFY=ntfy.sh/<your-topic>
cd experiments/training-run/scripts
```

Fresh box instead? run5-6-guide.md §1–2 first (clone, install, suite), then
pull the parent: `python hf_checkpoint.py pull --run-id evt-run4-armB-inst`.

## 2. Launch — 7,813 steps per point, sequential

```bash
for lr in 3e-4 1e-3 3e-3 1e-2; do
  python train_target.py --config ../configs/run8_target_1m.yaml \
      --override ../configs/pilot/target_sweep_1m_lr${lr}.yaml \
      --init-from $GEODE_STORE/runs/evt-run4-armB-inst/model --confirm-cost \
    ; curl -d "1M sweep lr=${lr} done (exit $?)" $NTFY
done
```

Or use the runner: `./sweep_1m.sh --confirm-cost` — same grid, skips
already-completed points on re-run (so after the first pass it launches
only the 1e-2 extension), stops on a failed point, and ends with the
winner summary + edge-rule verdict.

Watch from a second window: `python monitor.py --run-id evt-run8-sweep-lr<X>`
(prints steps/s + ETA — the first point calibrates wall-clock for the other
two). Disk is a non-issue here: no snapshots, <~1 GB per run (final model +
logs).

## 3. Winner → pin (laptop)

Winner = lowest stopping-block `min_val_nats` at the shared 1-epoch budget
(a point that ε/k-converges earlier is fine — compare its floor the same
way). **Edge rule:** an edge win extends the grid one step in that
direction before pinning (already fired once: 3e-3 → added 1e-2). **If
3e-3 stands after the extension:** run the 100K-prefix Arm-A sanity pilot
before pinning (`pilot/target_pilot_100k_armA_lr3e-3.yaml`; needs
`hf_checkpoint.py pull --run-id evt-run3-armA-inst` first) — 3e-4 and 1e-3
are already proven on Arm A; PASS = `stop_reason=stopping_rule`. **If 1e-2
wins:** stop — consult the owner (one LR everywhere would carry 1e-2 into
the Llama chain).

Then, laptop: set `train.lr` in **ALL FOUR** run yamls (run7/run8/run9/
run10 — one LR everywhere, owner 2026-07-24), record it in decisions.md,
commit + push; box `git pull` and confirm the hash before the pair
launches. Launch order and gate mechanics for the pair are in the configs'
headers (run 7 first — G7).

Then the pair runs unattended: `./launch_pair_1m.sh --confirm-cost` (or
`./launch_chain_7_10.sh --confirm-cost` to continue straight into the Llama
chain — it skips completed runs, so it also picks up after the pair) — checks
the pin equals the sweep winner in the store, ≥200 GB free, parents present;
run 7 then run 8 with NTFY pings; skips a completed run on re-run.

## Troubleshooting

| Symptom | Fix |
|---|---|
| box `git log` ≠ laptop hash | forgot to push, or stale clone — `git pull` |
| `parent run ... has no manifest` | stale `$GEODE_STORE` (new tmux window) — re-export |
| `order_hash mismatch` | downloaded parquet ≠ frozen file — stop, don't work around |
| `register_run: already running` | double-launch guard — `pgrep -f train_target` |
| stops at 7,813 with `stop_reason=max_steps` | expected — the 1-epoch cap IS the sweep design |
