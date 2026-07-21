# Run-1 v3 guide — constant-LR retrain (`evt-run1-base-v3`)

From-scratch retrain of the run-1 base under the **paper's exact
recipe** (decisions.md 2026-07-20): the v2 cosine schedule was
off-protocol, and gate G0 is removed — **whatever this run converges to
is floor 1**. Config: `configs/run1_pretrain.yaml` (already pinned; no
sweep, no pin step, no gate verdict this time).

Tmux survival kit and ntfy phone-ping setup: see `run1-guide.md` — run
everything below inside `tmux new -s train`.

## The recipe (fixed — do not override anything)

| | |
|---|---|
| Optimizer | AdamW, β 0.9/0.999, weight decay 0.01 |
| LR | **1e-3 constant** (2026-07-19 sweep pin; no warmup, no decay) |
| Grad clipping | 1.0 (global norm) |
| Precision | bfloat16 |
| Batch | 128 (grad-accum 4×32) |
| Stopping | val-loss convergence: ε 0.005 nats / k 3 / eval_every 1000 — stop when val improves < 0.005 nats over 3,000 steps |
| Step ceiling | 30,000 (**cost ceiling ~$1, not a target** — `stop_reason=max_steps` means "did not converge": stop and investigate, don't ship) |

Expected: convergence somewhere past v1's 17k steps (v1 stopped under a
2× looser rule at 1.1464 nats); ~2–3.5 h on an RTX 4090, ~$0.6–1.1.

## Phase 0 — box setup (~15 min, no GPU cost)

Same box class as run 1: RTX 4090, ≥32 GB RAM, ≥8 vCPU, ~30 GB disk,
CUDA 12.x image with Python ≥3.11.

```bash
nvidia-smi && python --version && free -g | head -2   # sanity
git clone -b cut-to-core https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach
pip install -e ".[dev]"
export GEODE_STORE=$HOME/geode-store    # re-export in every new tmux window
mkdir -p $GEODE_STORE
python -m pytest -q; echo "suite exit: $?"   # expect 0 after ~2 min
cd experiments/training-run/scripts
```

## Phase 1 — de-risk pilot (~10 min, ~$0.01) — recommended

The launcher's manifest code changed on 2026-07-20 (records the full
recipe now); one tiny end-to-end pass makes any pipeline bug cost cents:

```bash
python train.py --config ../configs/run1_pretrain.yaml \
    --override ../configs/pilot/run1_pretrain.yaml --device cuda --confirm-cost
```

✅ `done:` line prints, and
`$GEODE_STORE/runs/evt-pilot-run1-base/manifest.json` shows
`"lr_schedule": "constant"` under `training`.

## Phase 2 — warm the pack cache (~10–40 min CPU, no GPU cost)

Run **without** `--confirm-cost`; it packs all 2.7M stories, writes the
cache, prints the estimate, and exits. **"refusing to train" + exit 1
at the end is intended.**

```bash
python train.py --config ../configs/run1_pretrain.yaml --device cuda \
    --packed-cache $HOME/packed_full.pt ; \
    curl -d "v3 pack done (exit $?)" ntfy.sh/<your-topic>
```

✅ `wrote packed cache`, `ls -lh $HOME/packed_full.pt` ≈ 1.1 GB.

## Phase 3 — production launch (~2–3.5 h, ≤ ~$1.1)

The printed estimate quotes the 30k-step **ceiling** (~4 epochs); the
plateau rule should stop it earlier.

```bash
python train.py --config ../configs/run1_pretrain.yaml --device cuda \
    --packed-cache $HOME/packed_full.pt --confirm-cost ; \
    curl -d "run1 v3 finished (exit $?)" ntfy.sh/<your-topic>
```

Detach (`Ctrl+b` `d`) and monitor from a second window (re-export
`GEODE_STORE` there):

```bash
cat $GEODE_STORE/runs/evt-run1-base-v3/pretrain/eval_log.jsonl   # val curve
tail -f $GEODE_STORE/runs/evt-run1-base-v3/pretrain/train_log.jsonl
```

✅ Done when the `done:` line prints with **`stop_reason=converged`**
and a min val figure (compare to v2-ext's 1.1066 nats — informational
only, nothing gates on it). `max_steps` instead ⇒ did not converge:
keep the box, paste the eval log to Claude.

## Phase 4 — push to the HF relay (~10 min)

```bash
hf auth login    # write token
python hf_checkpoint.py push --run-id evt-run1-base-v3
```

## Phase 5 — laptop pull, verify, destroy (~15 min)

On the **laptop**:

```bash
cd ~/Github/geode/experiments/training-run/scripts
python hf_checkpoint.py pull --run-id evt-run1-base-v3
ls -laR $GEODE_STORE/runs/evt-run1-base-v3/    # model/ + logs + manifest?
```

Then **destroy** (not stop) the instance on vast.ai.

Also pending from 2026-07-20 (independent of the box): re-push the
backfilled v2/v2-ext manifests so the relay matches the local store:

```bash
python hf_checkpoint.py push --run-id evt-run1-base-v2
python hf_checkpoint.py push --run-id evt-run1-base-v2-ext
```

## After v3

Run 2 launches per `docs/run2-guide.md` — its config already points at
`parent_run_id: evt-run1-base-v3`, and the launcher refuses until the
pulled v3 manifest says `status: complete`. No gate verdict in between.

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| "refusing to train" without `--confirm-cost` | Intended — budget guard |
| `--packed-cache mismatch` ValueError | Config drifted since packing — delete `$HOME/packed_full.pt`, redo Phase 2 |
| `register_run: ... already running` | Double-launch guard — `pgrep -f train.py` before retrying; if the run is dead, edit the stale manifest's `status` |
| stale `GEODE_STORE` in a new shell | Re-export; scripts default to `<repo>/geode-store` if unset — keep it explicit on the box |
| `stop_reason=max_steps` at 30,000 | Not converged under ε 0.005/3,000 steps — do NOT ship as floor 1; investigate with Claude |
