# Run-1 step-by-step guide (beginner edition)

Companion to `run1-checklist.md` — same phases, same commands, but written
assuming you've never used tmux or a rented GPU box. The checklist is the
protocol of record; this is the hand-holding version. Total: ~$1–2,
~4–5 hours wall clock (most of it waiting), two check-ins with Claude.

---

## Before anything: the tmux survival kit

If your SSH connection drops while a run is going, the run **dies with
it** — unless it's inside tmux, which keeps programs alive on the box
independent of your connection. Four commands are all you need:

| What | How |
|---|---|
| Start a session (do this first) | `tmux new -s train` |
| Detach — leave it running, get your shell back | press `Ctrl+b`, release, press `d` |
| Reattach (incl. after SSH dropped: ssh back in, then) | `tmux attach -t train` |
| Scroll up inside tmux | `Ctrl+b` then `[`, arrows to scroll, `q` to quit scrolling |

Run **everything below inside the tmux session**.

---

## Phase 0 — set up the box (~15 min, no GPU cost)

Goal: repo cloned, dependencies installed, test suite green.

You should already have: an RTX 4090 instance (~$0.35–0.50/h) with
≥32 GB system RAM, ≥8 vCPU, ~30 GB free disk, a CUDA 12.x PyTorch image
with Python ≥3.11.

**0.1 — sanity-check the hardware:**

```bash
nvidia-smi                 # one RTX 4090 listed
python --version           # ≥ 3.11 — if lower, stop and tell Claude
free -g | head -2          # "total" column ≥ 32 (GB RAM)
```

**0.2 — clone and install:**

```bash
tmux new -s train          # if not already inside tmux
git clone -b cut-to-core https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach
pip install -e ".[dev]"
export GEODE_STORE=$HOME/geode-store   # all run artifacts land here
mkdir -p $GEODE_STORE
```

> `export` only lives in the current shell. Any *new* terminal or tmux
> window needs `export GEODE_STORE=$HOME/geode-store` again before
> running anything.

**0.3 — prove the install works:**

```bash
python -m pytest -q; echo "suite exit: $?"
```

Expect `suite exit: 0` after ~2 minutes (CPU-only, no network). Anything
else: stop, paste the output to Claude.

**0.4 — move to the launch directory** (every command below runs here):

```bash
cd experiments/training-run/scripts
```

✅ Done when: suite exit 0, you're in `experiments/training-run/scripts/`.

---

## Phase 1 — de-risk pilot (~10 min, ~$0.01)

Goal: one tiny end-to-end run (20K docs, 2000 steps, batch 32) through
the exact production path, so any pipeline bug costs cents instead of
the production run.

**1.1 — see the budget guard work.** This first command is *supposed* to
refuse — no `--confirm-cost`, no training, ever:

```bash
python train.py --config ../configs/run1_pretrain.yaml \
    --override ../configs/pilot/run1_pretrain.yaml --device cuda
```

Expect: a printed cost estimate, then "refusing to train". Good.

**1.2 — run it for real:**

```bash
python train.py --config ../configs/run1_pretrain.yaml \
    --override ../configs/pilot/run1_pretrain.yaml --device cuda --confirm-cost
```

Expect: a few minutes of tokenizing/packing, then training steps, ending
in a `done: ... best val ... nats` line.

**1.3 — health check.** Val loss should be decreasing and finite:

```bash
cat $GEODE_STORE/runs/evt-pilot-run1-base/pretrain/eval_log.jsonl
```

**1.4 — prove the sampler runs** (output will be near-gibberish from
20K docs — you're testing the pipeline, not the prose):

```bash
python sample_stories.py \
    --checkpoint $GEODE_STORE/runs/evt-pilot-run1-base/pretrain/model --device cuda
```

✅ Done when: `done:` line printed, val loss decreasing and finite,
sampler produced 20 samples without error.

---

## Phase 2 — warm the pack cache (~30–60 min, CPU-only, no GPU cost)

Goal: tokenize all 2.7M stories **once** into a ~4.3 GB cache file that
the sweep and production launches all reuse — saves ~40 CPU-minutes per
launch afterward.

Run **without** `--confirm-cost`; it packs, writes the cache, prints the
production estimate, and exits *without training*. **Exit code 1 and
"refusing to train" at the end are intended.**

```bash
python train.py --config ../configs/run1_pretrain.yaml --device cuda \
    --packed-cache $HOME/packed_full.pt
```

Expect: `... wrote packed cache ...` then `refusing to train`.

Good moment to practice detaching: `Ctrl+b` `d`, come back later with
`tmux attach -t train`.

The cache is keyed to (data file, seq_len, max_documents, tokenizer); if
a config ever drifts, the mismatch raises loudly instead of silently
training on stale rows.

✅ Done when: `$HOME/packed_full.pt` exists (`ls -lh $HOME/packed_full.pt`,
~4.3 GB).

---

## Phase 3 — LR sweep (~45 min, ~$0.30)

Goal: the config's LR, eval cadence, and stopping thresholds are
placeholders (1B-paper values don't transfer to 38.7M). Four short runs
at production batch (128) and production data, 2000 steps each, differing
only in learning rate — the results pick the real values.

**3.1 — launch all four** (runs sequentially, ~10 min each):

```bash
for lr in 1e-4 3e-4 1e-3 3e-3; do
  python train.py --config ../configs/run1_pretrain.yaml \
      --override ../configs/pilot/run1_sweep_lr${lr}.yaml \
      --packed-cache $HOME/packed_full.pt --device cuda --confirm-cost
done
```

If a run goes NaN or explodes, that LR is just too hot — **let the loop
continue**; a dead run is a data point, not a problem.

**3.2 — summarize when the loop finishes:**

```bash
python - <<'EOF'
import json, pathlib, os
for p in sorted(pathlib.Path(os.environ["GEODE_STORE"]).glob("runs/evt-pilot-run1-lr*/pretrain/eval_log.jsonl")):
    evals = [json.loads(l) for l in p.read_text().splitlines()]
    best = min(e["val_loss_nats"] for e in evals)
    print(f"{p.parent.parent.name:26s} best_val={best:.4f} nats  final={evals[-1]['val_loss_nats']:.4f}")
EOF
```

✅ Done when: the summary prints one line per LR (dead runs may be
missing or ugly — fine).

---

## Phase 4 — pin hyperparameters (box idles ~$0.02, Claude works)

Goal: turn the sweep into pinned config values.

**4.1 — paste the Phase-3 summary table to Claude** (the four
`eval_log.jsonl` contents too, if convenient). Rule of thumb Claude
applies: lowest best-val that's clearly stable, conservative on ties.

**4.2 — Claude pins** LR, `eval_every`, and stopping ε/k in
`run1_pretrain.yaml` + spec 02 §12 (closing OPEN(11)/OPEN(3)), records
the decision, and pushes.

**4.3 — pull on the box:**

```bash
git -C ~/elicit-vs-teach pull
```

✅ Done when: `git pull` shows the config commit arriving.

---

## Phase 5 — production pretrain (~1.5–2 h, ~$0.55–0.85)

Goal: the real run — `evt-run1-base`, the shared floor-1 model every
later run builds on. No step cap; the stopping rule ends it when val
loss plateaus.

**5.1 — launch:**

```bash
python train.py --config ../configs/run1_pretrain.yaml --device cuda \
    --packed-cache $HOME/packed_full.pt --confirm-cost
```

**5.2 — detach and monitor.** `Ctrl+b` `d` to detach; check in with
`tmux attach -t train`, or from a second SSH window
(re-export `GEODE_STORE` there first):

```bash
tail -f $GEODE_STORE/runs/evt-run1-base/pretrain/train_log.jsonl  # per-step loss
nvidia-smi                                                        # GPU busy?
cat $GEODE_STORE/runs/evt-run1-base/pretrain/eval_log.jsonl       # val curve
```

Val loss should fall steadily then flatten. Losses are in **nats**
(divide by 0.693 for bits/token).

✅ Done when: the `done:` line prints with a stop reason
(`converged` expected) and best val loss.

---

## Phase 6 — Gate G0: story coherence (~15 min, your judgment)

Goal: the pre-registered check that floor 1 actually exists — a model
that genuinely writes TinyStories English — before anything is built on
it (spec 02 §8).

**6.1 — generate the 20 samples:**

```bash
python sample_stories.py \
    --checkpoint $GEODE_STORE/runs/evt-run1-base/pretrain/model --device cuda
```

Samples are printed and archived to `.../model/floor1_samples.txt`.

**6.2 — judge them.** Read all 20. A sample is *coherent* if it has
grammatical sentences, narrative continuity, and no repetition loops.
**Pass = at least 16 of 20.**

**6.3 — tell Claude the verdict and rough count** — it goes in
`decisions.md` + the run manifest.

If it **fails**: stop. Don't launch runs 2–4. Diagnose with Claude first
(usual suspects: needs more epochs, or the pinned LR).

✅ Done when: verdict reported and recorded.

---

## Phase 7 — archive and tear down (~15 min)

Goal: everything the experiment will ever need is off the box, then the
meter stops.

**7.1 — from your laptop** (not the box), pull the store down. Get user,
host, and port from the instance's connect button on vast.ai:

```bash
rsync -avz -e "ssh -p <port>" <user>@<host>:geode-store/ ~/geode-store/
```

(Skip `packed_full.pt` — the cache is regenerable.)

**7.2 — verify before destroying:**

```bash
ls -laR ~/geode-store/runs/evt-run1-base/   # model/ + logs arrived?
```

**7.3 — destroy the instance on vast.ai.** Billing stops only when the
box does. Stopped-but-not-destroyed instances still bill for storage.

✅ Done when: artifacts verified locally, instance destroyed.

---

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| SSH dropped mid-run | Run is fine if it was in tmux: ssh back in, `tmux attach -t train` |
| "refusing to train" without `--confirm-cost` | Intended — that's the budget guard |
| `KeyError: GEODE_STORE` / runs land nowhere | New shell lost the export: `export GEODE_STORE=$HOME/geode-store` |
| Sweep run goes NaN | That LR is too hot — a valid data point, let the loop continue |
| `--packed-cache mismatch` ValueError | Config drifted since the cache was built — delete `$HOME/packed_full.pt` and rerun Phase 2 |
| `CUDA out of memory` | Something else is on the GPU — `nvidia-smi`, kill stray processes; this job needs <8 GB |
| Pilot samples are gibberish | Expected at Phase 1 (20K docs); only Phase 6 judges prose |

## After run 1

With G0 passed and OPEN(11)/OPEN(3) pinned, runs 2–4 (the SFT arms —
label-masked loss already lives in `geode.train`) are next; their launch
gets its own checklist.
