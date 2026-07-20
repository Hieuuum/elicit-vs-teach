# Run-1 checklist — TinyStories pretrain (floor 1)

Owner-facing steps for the first training run: the shared 38.7M-param base
model (spec 02 §6, run_id `evt-run1-base`) that every later run builds on.
Everything is frozen except the OPEN(11)/OPEN(3) hyperparameters (LR, eval
cadence, stopping ε/k), which this checklist closes via a short LR sweep
before the production launch. Total GPU cost ≈ **$1–2** including box idle
time; no step trains without your explicit `--confirm-cost`.

Phases in order. Steps marked **[you]** need your hands or judgment;
**[Claude]** steps happen back on your laptop between box sessions.

| Phase | What | GPU cost | Wall clock |
|---|---|---|---|
| 0 | Rent + set up box | — | ~15 min |
| 1 | De-risk pilot (tiny end-to-end) | ~$0.01 | ~10 min |
| 2 | Warm the full-corpus pack cache | CPU only | ~30–60 min |
| 3 | LR sweep, 4 runs | ~$0.30 | ~45 min |
| 4 | Pin hyperparameters (decision) | — | — |
| 5 | Production pretrain | ~$0.55–0.85 | ~1.5–2 h |
| 6 | Gate G0: judge 20 story samples | — | ~15 min |
| 7 | Archive artifacts, destroy box | — | ~15 min |

## Phase 0 — rent and set up the box **[you]**

Rent an **RTX 4090** (vast.ai / RunPod, ~$0.35–0.50/h) with **≥32 GB system
RAM**, ≥8 vCPU, ~30 GB free disk, a CUDA 12.x image with Python ≥3.11.
Why 4090: the model is tiny (38.7M params, <8 GB VRAM at batch 128), so
you're buying bf16 throughput per dollar, not memory — A100/H100 cost 3–6×
for capacity this job can't use, and pre-Ampere cards (V100/T4) lack bf16.

```bash
git clone -b cut-to-core https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach
pip install -e ".[dev]"
export GEODE_STORE=$HOME/geode-store   # all run artifacts land here
mkdir -p $GEODE_STORE
python -m pytest -q; echo "suite exit: $?"   # expect 0, ~2 min, CPU-only
```

All training commands below run from `experiments/training-run/scripts/`:

```bash
cd experiments/training-run/scripts
```

## Phase 1 — de-risk pilot **[you]**

A shrunk end-to-end pass (20K docs, 2000 steps, batch 32) that exercises
the exact production path — same arch, tokenizer, seq_len 512 — so any
pipeline bug costs cents, not the production run. First view the estimate
(the script refuses to train without the flag), then confirm:

```bash
python train.py --config ../configs/run1_pretrain.yaml \
    --override ../configs/pilot/run1_pretrain.yaml --device cuda
# prints estimate + "refusing to train" — that's the budget guard working
python train.py --config ../configs/run1_pretrain.yaml \
    --override ../configs/pilot/run1_pretrain.yaml --device cuda --confirm-cost
```

Outputs land in `$GEODE_STORE/runs/evt-pilot-run1-base/`:
`pretrain/train_log.jsonl` (per-step loss), `pretrain/eval_log.jsonl`
(val loss per eval), `pretrain/model/` (checkpoint),
`pretrain/training_meta.json`, `manifest.json`.

Check before moving on: val loss in `eval_log.jsonl` is decreasing and
finite; the run printed a `done: ... best val ... nats` line; and the
sampler runs on the pilot checkpoint (output will be near-gibberish from
20K docs — you're testing the pipeline, not the prose):

```bash
python sample_stories.py \
    --checkpoint $GEODE_STORE/runs/evt-pilot-run1-base/pretrain/model --device cuda
```

## Phase 2 — warm the pack cache **[you]**

Tokenizing all 2.7M stories takes tens of CPU-minutes. Do it once, into a
cache the sweep and production launches all reuse. Run **without**
`--confirm-cost` — it packs, writes the cache (~4.3 GB `.pt`), prints the
production estimate, and exits without training (exit 1 is intended):

```bash
python train.py --config ../configs/run1_pretrain.yaml --device cuda \
    --packed-cache $HOME/packed_full.pt
# expect: "... wrote packed cache ..." then "refusing to train"
```

The cache is keyed to (data file, seq_len, max_documents, tokenizer); a
config mismatch raises loudly instead of training on stale rows.

## Phase 3 — LR sweep **[you]**

Why: the config's LR (3e-4), eval cadence, and stopping thresholds are
placeholders — published values from 1B-param papers don't transfer to
38.7M. Four short runs at production batch (128) and production data,
2000 steps each (~10 min, ~$0.07 apiece), differing only in LR:

```bash
for lr in 1e-4 3e-4 1e-3 3e-3; do
  python train.py --config ../configs/run1_pretrain.yaml \
      --override ../configs/pilot/run1_sweep_lr${lr}.yaml \
      --packed-cache $HOME/packed_full.pt --device cuda --confirm-cost
done
```

Watch live if you like: `tail -f $GEODE_STORE/runs/evt-pilot-run1-lr*/pretrain/train_log.jsonl`.
A NaN/exploding run just means that LR is too hot — let the loop continue.

Summarize the four when done:

```bash
python - <<'EOF'
import json, pathlib, os
for p in sorted(pathlib.Path(os.environ["GEODE_STORE"]).glob("runs/evt-pilot-run1-lr*/pretrain/eval_log.jsonl")):
    evals = [json.loads(l) for l in p.read_text().splitlines()]
    best = min(e["val_loss_nats"] for e in evals)
    print(f"{p.parent.parent.name:26s} best_val={best:.4f} nats  final={evals[-1]['val_loss_nats']:.4f}")
EOF
```

## Phase 4 — pin hyperparameters **[you + Claude]**

Send Claude the summary table (or the four `eval_log.jsonl` files). Rule of
thumb: pick the LR with the lowest best-val that's clearly stable (no
spikes); prefer the more conservative of two ties. Claude then pins LR,
`eval_every`, and stopping ε/k in `run1_pretrain.yaml` + spec 02 §12
(closing OPEN(11)/OPEN(3)), records the outcome in `notes/decisions.md`,
and pushes. On the box: `git pull`.

## Phase 5 — production pretrain **[you]**

~1.2 GPU-h of compute (≈$0.55) plus eval overhead; expect ~1.5–2 h wall
clock. No step cap — the stopping rule ends it when val loss plateaus:

```bash
python train.py --config ../configs/run1_pretrain.yaml --device cuda \
    --packed-cache $HOME/packed_full.pt --confirm-cost
```

During: `tail -f $GEODE_STORE/runs/evt-run1-base/pretrain/train_log.jsonl`,
`nvidia-smi` for utilization, and glance at `eval_log.jsonl` occasionally —
val loss should fall steadily then flatten. Losses are in **nats**
(÷0.693 for bits/token). Done line reports stop reason + best val.

## Phase 6 — Gate G0: story coherence **[you]**

The pre-registered check that floor 1 exists — a model that genuinely
writes TinyStories English — before anything is built on it (spec 02 §8):

```bash
python sample_stories.py \
    --checkpoint $GEODE_STORE/runs/evt-run1-base/pretrain/model --device cuda
```

Read the 20 samples (archived to `.../model/floor1_samples.txt`). **Pass =
≥16/20 coherent** under the rubric: grammatical sentences, narrative
continuity, no repetition loops. Tell Claude the verdict (and rough count);
it's recorded in `decisions.md` + the run manifest. If it fails, stop —
don't launch runs 2–4; diagnose with Claude first (usual suspects: needs
more epochs, or the pinned LR).

## Phase 7 — archive and tear down **[you]**

From your laptop, pull everything the experiment will ever need from this
box (the pack cache is regenerable — skip it):

```bash
rsync -avz <user>@<box>:geode-store/ ~/geode-store/
ls -laR ~/geode-store/runs/evt-run1-base/   # verify model/ + logs arrived
```

Then **destroy the instance** — billing stops only when the box does.

## After run 1

Handoff: with G0 passed and OPEN(11)/OPEN(3) pinned, runs 2–4 (the SFT
arms, label-masked loss already in `geode.train`) are next; their launch
gets its own checklist.
