# Runs 7–8 (LoRA targets @ full 1M) — box paste sheet

Rerun of runs 5/6 on the **full 1M** `D_target` order with a re-pinned LR
(owner 2026-07-23, decisions.md). Configs: `run7_target_1m.yaml` /
`run8_target_1m.yaml` (n 1M, ceiling 46,878 steps — the ε/k rule is the
real stop; **LR is provisional until §3 pins it**). Sequencing (owner,
one-box plan): **extraction over the runs-5/6 snapshots runs FIRST** —
they exist nowhere else — then the LR sweep, then the pair.

## 0. Disk reality check (before anything)

```bash
df -h /workspace | tail -1
du -sh /workspace/elicit-vs-teach/geode-store/runs/evt-run{5,6}*/snapshots
```
Runs 5/6 snapshots hold ~75 GB and the box was last seen with ~31 GB
free (decisions.md 2026-07-22). Runs 7/8 want **≥120 GB free** (adapter
snapshots ≈ 48 MB × up to 1024 per run, and run 8 may run long at 1M).
If the free space isn't there after extraction, STOP and ask the owner:
options are resizing the volume, or relaying/archiving the runs-5/6
snapshots first — never delete them without an explicit owner decision.

## 1. Laptop first — push, or the box pulls stale configs

```bash
cd ~/Github/geode && git add -A && git commit -m "..." ; git push origin cut-to-core
git rev-parse --short HEAD    # note this hash — the box must match it
```

On the (already set up) box: `git pull` inside
`/workspace/elicit-vs-teach`, confirm the hash, and re-export in every
tmux window:

```bash
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store      # store INSIDE the clone
export NTFY=ntfy.sh/<your-topic>
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
```

Fresh box instead? Follow run5-6-guide.md §1–2 (clone, install, suite,
pull `evt-run3-armA-inst` + `evt-run4-armB-inst` parents), then return here.

## 2. LR sweep — 3 points, Arm B @ full 1M (~2–4 h total)

Bracket around the 500K-era winner; each point runs to its ε/k stop.

```bash
for lr in 3e-4 1e-3 3e-3; do
  python train_target.py --config ../configs/run8_target_1m.yaml \
      --override ../configs/pilot/target_sweep_1m_lr${lr}.yaml \
      --init-from $GEODE_STORE/runs/evt-run4-armB-inst/model --confirm-cost \
    ; curl -d "1M sweep lr=${lr} done (exit $?)" $NTFY
done
```

Winner = lowest `min_val_nats` among `stop_reason=converged` runs
(`python monitor.py --run-id evt-run8-sweep-lr<X> --once` or the
manifests). **If an edge point (3e-4 or 3e-3) wins, extend one step in
that direction before pinning** — the 50K sweep's first decade came back
monotone; don't pin an unbracketed winner.

## 3. Pin the winner (laptop), then sync the box

Laptop: set `train.lr` in **both** `run7_target_1m.yaml` and
`run8_target_1m.yaml` to the winner, record the pin in decisions.md,
commit + push. Box: `git pull`, confirm the hash. (Shared-vs-per-arm LR
policy is deferred — one shared pin unless decisions.md says otherwise.)

## 4. Dry run (free — no `--confirm-cost`, must end in a refusal)

```bash
python train_target.py --config ../configs/run7_target_1m.yaml \
    --init-from $GEODE_STORE/runs/evt-run3-armA-inst/model
```
Expect: `order_hash verified` → parent gates pass → cost estimate →
`refusing to train (budget rule)`. Exit 1 here is correct.

## 5. Run 7 (Arm A), then run 8 (Arm B) — sequential; G7 needs run 7 registered

```bash
python train_target.py --config ../configs/run7_target_1m.yaml \
    --init-from $GEODE_STORE/runs/evt-run3-armA-inst/model --confirm-cost \
  ; curl -d "run7 armA-1m done (exit $?)" $NTFY

python train_target.py --config ../configs/run8_target_1m.yaml \
    --init-from $GEODE_STORE/runs/evt-run4-armB-inst/model --confirm-cost \
  ; curl -d "run8 armB-1m done (exit $?)" $NTFY
```
Watch from a second window: `python monitor.py --run-id evt-run7-armA-target-1m`.
No reference stops exist at 1M — the 500K pair stopped at 6,000 (A) and
12,500 (B); expect A similar-or-earlier and B possibly much later.
`stop_reason=max_steps` = the rule never fired = investigate, not a result.

## 6. G5 evidence (both runs, minutes)

```bash
for r in evt-run7-armA-target-1m evt-run8-armB-target-1m; do
  python gates.py g5 --run $r --config ../configs/eval_target_data.yaml --device cuda
done ; curl -d "G5 recorded (exit $?)" $NTFY
```
500K references for orientation: A 0.9980 zero-shot / θ_T 0.00194 nats,
B 0.9502 / 0.03558. 16-shot ≈ 0 is expected (metric invalidated).

## 7. Archive the small artifacts (NOT the snapshots)

Same two-paste pattern as run5-6-guide.md §6 (guarded file-list push to
`mhieuuu/geode-store`), with the run list changed to
`("evt-run7-armA-target-1m", "evt-run8-armB-target-1m")` — sweeps are
disposable but cheap to include if wanted
(`evt-run8-sweep-lr*`). `hf_checkpoint.py push` still uploads whole
folders including snapshots — don't use it here.

## 8. Tear down — owner call only

After runs 7/8, the box holds the only copies of **four** runs' snapshots
(5/6 post-extraction + 7/8 pre-extraction). Keep it alive until the owner
decides the extraction/relay plan for the new pair; destroy (never stop)
on vast.ai when cleared. Store lives inside the clone — never
`git clean -dfx` on this box.

## Troubleshooting

| Symptom | Fix |
|---|---|
| box `git log` ≠ laptop hash | forgot to push, or stale clone — `git pull` |
| `parent run ... has no manifest` | stale `$GEODE_STORE` (new tmux window) — re-export, re-pull |
| `order_hash mismatch` | downloaded parquet ≠ frozen file — stop, don't work around |
| `register_run: already running` | double-launch guard — `pgrep -f train_target` |
| `data_order_hash` refusal on run 8 | run 7 not registered yet, or n_examples differs (G7) |
| launcher rejects lr / config | §3 pin not pulled onto the box — `git pull`, check hash |
| disk full mid-run | snapshots — see §0; stop and re-plan, don't delete by hand |
| HF push 401/403 | read token in a write path — see run5-6-guide.md §6 |
