# Run-1 extension guide (v2-ext) — bare box to convergence run

Continues `evt-run1-base-v2` (cosine, stopped at its fixed 17k horizon,
best val 1.1140 nats) to *measured* convergence: warm-start from the v2
checkpoint, constant LR 1e-4, plateau stopping live. Decision record:
`experiments/training-run/notes/decisions.md` 2026-07-20. Config:
`configs/run1_extend.yaml`.

Prerequisite on your **laptop**: the v2 artifacts at
`~/geode-store/runs/evt-run1-base-v2/` (pulled when the old box was
archived). The tmux survival kit and the ntfy phone-ping setup from
`docs/run1-guide.md` apply here unchanged.

Cost picture: ≤ ~$0.55 GPU + ~30–60 min CPU-only packing. Wall clock
≤ ~2 h of training (less if it converges before the 17k ceiling).

---

## Phase 0 — set up the box (~15 min, no GPU cost)

Rent the same class of instance as run 1: one RTX 4090 (~$0.35–0.50/h),
≥32 GB RAM, ≥8 vCPU, ~30 GB free disk, CUDA 12.x PyTorch image,
Python ≥ 3.11.

```bash
nvidia-smi                 # one RTX 4090 listed
python --version           # ≥ 3.11 — if lower, stop
free -g | head -2          # "total" ≥ 32

tmux new -s train
git clone -b cut-to-core https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach
git log --oneline -1       # must show the run-1 extension commit
                           # (SCRIPTS+DOCS: run-1 extension ...) or later —
                           # older clones lack --init-from
pip install -e ".[dev]"
export GEODE_STORE=$HOME/geode-store
mkdir -p $GEODE_STORE
python -m pytest -q; echo "suite exit: $?"     # expect: suite exit: 0, ~2 min
cd experiments/training-run/scripts
```

> `export GEODE_STORE=...` only lives in the current shell — every new
> tmux window needs it again.

✅ Done when: suite exit 0, you're in `experiments/training-run/scripts/`.

---

## Phase 1 — push the v2 checkpoint up (~1–5 min, from your LAPTOP)

The box needs the v2 checkpoint to warm-start from. User, host, and port
come from the instance's connect button on vast.ai. Same transfer rules
as the §7.1 pull: no `-z` (tensor bytes don't compress), `-W`
(checkpoints are all-new binaries, delta-transfer is wasted work):

```bash
rsync -avP -W -e "ssh -p <PORT>" ~/geode-store/runs/ <USER>@<HOST>:geode-store/runs/
```

Then verify **on the box**:

```bash
ls $GEODE_STORE/runs/evt-run1-base-v2/pretrain/model/
# expect: config.json  generation_config.json  model.safetensors  floor1_samples.txt
```

✅ Done when: `model.safetensors` is on the box.

---

## Phase 2 — dry run: pack + validate, spend nothing (~30–60 min, CPU)

Without `--confirm-cost` the script does everything *except* train:
downloads TinyStories (~2.2 GB from HF), packs the corpus, writes the
uint16 cache, loads the v2 checkpoint through the arch guard, prints
the cost estimate, then refuses. Any config/path mistake dies here for
free.

```bash
python train.py --config configs/run1_pretrain.yaml \
    --override configs/run1_extend.yaml \
    --packed-cache $GEODE_STORE/packed_full.pt \
    --init-from $GEODE_STORE/runs/evt-run1-base-v2/pretrain/model \
    ; curl -d "pack done (exit $?)" ntfy.sh/<your-topic>
```

Expect, in order: `loading + packing` → `wrote packed cache` →
`loading init checkpoint` → `run_id=evt-run1-base-v2-ext docs=2717495 ...`
→ the cost estimate → `refusing to train (budget rule)`.

✅ Done when: you saw the refusal line (exit 1 is correct here).

---

## Phase 3 — launch (~≤2 h, ≤ ~$0.55)

Same command plus `--confirm-cost`. The cache now hits, so it goes
straight to the GPU (first line: `pid=... loading packed cache`, ~20 s):

```bash
python train.py --config configs/run1_pretrain.yaml \
    --override configs/run1_extend.yaml \
    --packed-cache $GEODE_STORE/packed_full.pt \
    --init-from $GEODE_STORE/runs/evt-run1-base-v2/pretrain/model \
    --confirm-cost \
    ; curl -d "v2-ext finished (exit $?)" ntfy.sh/<your-topic>
```

Watch progress from a second tmux window (`Ctrl+b` then `c`; re-export
`GEODE_STORE` there):

```bash
tail -f $GEODE_STORE/runs/evt-run1-base-v2-ext/pretrain/eval_log.jsonl
```

One line per 1000 steps. `val_loss_nats` starts near v2's 1.1140 (the
first eval or two may wobble slightly — fresh optimizer state) and
should drift down. The run ends on its own; the final line of the
console tells you which way:

- **`done: converged at step N`** — the plateau rule fired: three
  consecutive 1000-step windows improved by less than 0.005 nats. This
  is the goal: genuine convergence at LR 1e-4.
- **`done: max_steps at step 17000`** — budget ceiling hit while still
  improving. Not a failure — it means the model was *still* gaining
  >5 millinats per 3k steps at the end. Read the last eval deltas and
  decide with Claude whether another extension is worth it.

✅ Done when: the done-line printed and the manifest says
`"status": "complete"`.

---

## Phase 4 — G0 samples, archive, tear down (~20 min)

**4.1 — generate the coherence samples on the box** (the judgment can
happen later, but sampling here is instant):

```bash
python sample_stories.py \
    --checkpoint $GEODE_STORE/runs/evt-run1-base-v2-ext/pretrain/model --device cuda
```

**4.2 — pull everything down, from your LAPTOP** (the packed cache is
regenerable — it lives outside `runs/`, so this skips it automatically):

```bash
rsync -avP -W -e "ssh -p <PORT>" <USER>@<HOST>:geode-store/runs/ ~/geode-store/runs/
ls -laR ~/geode-store/runs/evt-run1-base-v2-ext/    # model/ + logs + manifest arrived?
```

**4.3 — destroy the instance on vast.ai.** Billing stops only when the
box does (stopped-but-not-destroyed still bills storage).

**4.4 — tell Claude the outcome** (stop_reason, best val, and your G0
read of the samples) — it goes in `decisions.md` and the manifest, and
decides whether v2-ext or v2 is the floor-1 candidate.

✅ Done when: artifacts verified locally, box destroyed, verdict recorded.

---

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `--init-from arch mismatch: ...` | Wrong `--init-from` path (must be the `.../pretrain/model` dir of v2). |
| `--packed-cache mismatch: ...` | Cache built under a different data/tokenizer config — delete `$GEODE_STORE/packed_full.pt` and re-run Phase 2. |
| `register_run: ... already running` | A live launch with this run_id exists (double-launch guard). Check `pgrep -f train.py` before retrying. |
| First eval slightly *above* 1.1140 | Normal: fresh AdamW moments. Worry only if it's still above after 2–3 evals. |
| `KeyError: 'GEODE_STORE'` | You're in a new shell/tmux window — re-export `GEODE_STORE`. |
