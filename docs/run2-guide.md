# Run-2 guide (Arm A pre-teach) — bare box to G1 verdict

Runs the full run-2 sequence on one rented box: the 4-LR full-length
sweep, the owner's LR pin, the canonical `evt-run2-armA-algo` relaunch,
and the G1 gate. Decision record: `notes/decisions.md` 2026-07-20 (run-2
tooling + v3 retrain + "G0 removed" entries). Configs:
`configs/run2_algo.yaml` + `configs/pilot/run2_sweep_lr*.yaml`.

Parent is pinned: **`evt-run1-base-v3-ext`** (floor 1 = the extension's
convergence point; v3 itself hit its 30k ceiling still descending,
2026-07-21). The launcher checks the parent manifest says
`status: complete` via `require_parent_ready` — no manual verification
needed, but v3-ext must have finished and been pushed to the HF relay
first. **Every `--init-from` below points at the v3-ext checkpoint** —
plain v3 also reads `status: complete` (clean exit at the ceiling), so
warm-starting from v3 by mistake would NOT be caught by the launcher.

Run 2 is *much* lighter than run 1: no TinyStories download, no corpus
packing. The data is the frozen `D_algo.parquet` (1M NL add/sub rows,
public HF repo), tokenization is ~1–2 min, and one epoch is 7,773 steps
of 33-token rows through the 38.7M model.

Cost picture: 4 sweep arms ≈ $0.25 total + canonical ≈ $0.06 + G1 evals
≈ pennies. **Total ≤ ~$0.35 GPU**; wall clock well under 2 h including
setup. The tmux survival kit and ntfy phone-ping setup from
`docs/run1-guide.md` apply unchanged.

---

## Phase 0 — set up the box (~15 min, no GPU cost)

Same instance class as run 1: one RTX 4090 (~$0.35–0.50/h), ≥8 vCPU,
~20 GB free disk (no packed corpus this time), CUDA 12.x PyTorch image,
Python ≥ 3.11. The ≥32 GB RAM rule from run 1 doesn't apply — 16 GB is
plenty.

```bash
nvidia-smi                 # one RTX 4090 listed
python --version           # ≥ 3.11 — if lower, stop

tmux new -s train
git clone -b cut-to-core https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach
grep -m1 parent_run_id experiments/training-run/configs/run2_algo.yaml
                           # must print evt-run1-base-v3-ext — an older
                           # clone pins plain v3 and would warm-start
                           # from the unconverged checkpoint WITHOUT the
                           # launcher refusing (a stale checkout is what
                           # mislaunched v3-ext as v2-ext, 2026-07-21)
pip install -e ".[dev]"
export GEODE_STORE=$PWD/geode-store   # scripts default here anyway; the
                                      # export is for hand-typed commands
python -m pytest -q; echo "suite exit: $?"   # expect 0, ~2 min
cd experiments/training-run/scripts
```

> vast.ai: login shell starts in `/workspace`, so the clone is
> `/workspace/elicit-vs-teach`. Re-export `GEODE_STORE` in every new
> tmux window — a *stale* export pointing elsewhere beats the script
> default (run-1 incident).

✅ Done when: suite exit 0, you're in `experiments/training-run/scripts/`.

---

## Phase 1 — pull the parent checkpoint (~2–5 min)

`mhieuuu/geode-store` is private; log in once with a **write** token
(read would cover the pull, but Phase 6 pushes results back):

```bash
hf auth login
python hf_checkpoint.py pull --run-id evt-run1-base-v3-ext
```

The pull sha256-verifies `model.safetensors` and brings the manifest —
whose `status: complete` is what the launcher checks.

```bash
ls $GEODE_STORE/runs/evt-run1-base-v3-ext/pretrain/model/
# expect: config.json  generation_config.json  model.safetensors
```

✅ Done when: the pull printed `verified model.safetensors sha256 ...`.

---

## Phase 2 — dry run: spend nothing (~5 min, CPU + download)

Without `--confirm-cost` the launcher does everything except train:
downloads `D_algo.parquet`, recomputes and verifies `order_hash`,
tokenizes with span conversion, loads the parent through the arch guard,
checks the parent's gates, prints the cost estimate, then refuses. Any
mistake dies here for free.

```bash
python train_sft.py --config ../configs/run2_algo.yaml \
    --override ../configs/pilot/run2_sweep_lr3e-5.yaml \
    --init-from $GEODE_STORE/runs/evt-run1-base-v3-ext/pretrain/model
```

Expect, in order: `order_hash verified` → `tokenized: max ≈33
tokens/example` → `loading init checkpoint` → `parent
'evt-run1-base-v3-ext' complete, gates pass` → the cost estimate →
`refusing to train (budget rule)`.

✅ Done when: you saw the refusal line (exit 1 is correct here).

---

## Phase 3 — the sweep: 4 full-length arms (~30–60 min, ≈ $0.25)

Each arm is a *complete* candidate run — 1 epoch, ε/k stopping live —
not a short pilot (decisions.md 2026-07-20: short-horizon extrapolation
is what misled run 1). Sequential is fine at these sizes:

```bash
for lr in 3e-5 1e-4 3e-4 1e-3; do
  python train_sft.py --config ../configs/run2_algo.yaml \
      --override ../configs/pilot/run2_sweep_lr${lr}.yaml \
      --init-from $GEODE_STORE/runs/evt-run1-base-v3-ext/pretrain/model \
      --confirm-cost || break
done ; curl -d "run-2 sweep done (exit $?)" ntfy.sh/<your-topic>
```

Watch from a second tmux window (re-export `GEODE_STORE` there):

```bash
tail -f $GEODE_STORE/runs/evt-run2-sweep-lr1e-4/sft/eval_log.jsonl
```

One line per 500 steps. Each arm ends on its own (`converged` or
`max_steps` at 7,773 — either is fine for a sweep arm) and writes
`min_val_nats` into its manifest.

✅ Done when: four manifests under `$GEODE_STORE/runs/evt-run2-sweep-*/`
say `"status": "complete"`.

---

## Phase 4 — pick the winner (~10 min, GPU eval only)

Score every arm with the G1 protocol (1,024 seeded val examples, greedy,
`exact_match` — provably never trained on, via `split_indices`):

```bash
for lr in 3e-5 1e-4 3e-4 1e-3; do
  python gates.py g1 --run evt-run2-sweep-lr${lr} \
      --config ../configs/run2_algo.yaml --device cuda
done
```

Each prints `G1 accuracy X.XXXX ... -> PASS|FAIL` and records the
verdict in that arm's manifest (harmless — sweep arms are never
parents; exit 1 on a FAIL arm is expected, the loop continues).

**Decision rule (owner 2026-07-20):**

- **Winner = highest accuracy**; tiebreak = lower `min_val_nats`
  (in each sweep manifest under `experiment.sft_result`).
- **Hard stop: if NO arm reaches 0.95** — do *not* launch the canonical
  run. Push the sweep artifacts (Phase 6), destroy the box, and reassess
  with Claude. A sub-ceiling plateau across all four LRs is a finding,
  not a tuning problem.

✅ Done when: winner identified, or hard-stop taken.

---

## Phase 5 — pin the LR, launch canonical (~10–20 min, ≈ $0.06)

Pin from the **laptop** so the manifest's `git_commit` stays honest
(a dirty on-box edit would be invisible in provenance):

```bash
# laptop: edit configs/run2_algo.yaml -> train.lr: <winner>, drop the
# PLACEHOLDER comment; commit + push.
# box:
git -C /workspace/elicit-vs-teach pull
```

Then relaunch the winner's exact config as the canonical run, and render
the official G1 verdict on it:

```bash
python train_sft.py --config ../configs/run2_algo.yaml \
    --init-from $GEODE_STORE/runs/evt-run1-base-v3-ext/pretrain/model \
    --confirm-cost \
    ; curl -d "evt-run2-armA-algo finished (exit $?)" ntfy.sh/<your-topic>

python gates.py g1 --run evt-run2-armA-algo \
    --config ../configs/run2_algo.yaml --device cuda
```

Exit 0 = G1 pass, recorded in the canonical manifest — this is the gate
run 3 will check at launch.

✅ Done when: canonical manifest `"status": "complete"` and G1 printed
PASS.

---

## Phase 6 — archive, tear down, report (~15 min)

**6.1 — push everything to the HF relay** (from the box; ~150 MB per
run, write token from Phase 1):

```bash
for r in evt-run2-sweep-lr3e-5 evt-run2-sweep-lr1e-4 \
         evt-run2-sweep-lr3e-4 evt-run2-sweep-lr1e-3 \
         evt-run2-armA-algo; do
  python hf_checkpoint.py push --run-id $r
done
```

**6.2 — pull to the laptop** (skip sweep arms if you only want the
canonical run locally; the relay keeps them all):

```bash
python experiments/training-run/scripts/hf_checkpoint.py pull --run-id evt-run2-armA-algo
```

**6.3 — destroy the instance on vast.ai** (stopped-but-not-destroyed
still bills storage).

**6.4 — tell Claude the outcome**: per-arm accuracies + `min_val_nats`,
winner, canonical G1 verdict. It goes in `notes/decisions.md` and
unblocks run 3 (Arm A format install).

✅ Done when: relay verified, box destroyed, outcome recorded.

---

## Troubleshooting

| Symptom | Meaning / fix |
|---|---|
| `parent run ... has no manifest in the store` | Phase 1 pull missing or `GEODE_STORE` stale — `echo $GEODE_STORE`, re-export, re-pull. |
| `parent_run_id is null — pin it` | Clone predates the run-1 closure commit — `git pull`. |
| `parent run ... status ...` (not complete) | v3-ext hasn't finished (or an old manifest copy) — wait for the run-1 extension, re-pull evt-run1-base-v3-ext. |
| `order_hash mismatch` | Downloaded parquet ≠ frozen 2026-07-19 file. Do NOT work around — the dataset is the experiment. Stop and investigate. |
| `--init-from arch mismatch` | Wrong path — must be `.../evt-run1-base-v3-ext/pretrain/model`. |
| `register_run: ... already running` | Double-launch guard: check `pgrep -f train_sft` before retrying. |
| g1 accuracy exactly 0.0 on a trained arm | Wrong `--run`/checkpoint pairing (e.g. the smoke run) — g1 defaults to `runs/<run>/sft/model`. |
| HF push 401/403 | Token is read-only — re-login with a write token. |
| `$GEODE_STORE` empty in a shell command | New tmux window — re-export. Scripts don't need it (default to `<repo>/geode-store`); hand-typed commands do. |
