# HANDOFF — launch + post-process the n = 46 416 probe trajectory (fresh session, 2026-08-22)

Everything is BUILT, pre-registered and pushed (commit `1ea5946` on
`ts38-mini`). Nothing has been launched. Design + reads: `EXPERIMENTS.md`
§6.24 and `notes/decisions.md` "2026-08-22 — probe trajectory at
n = 46 416". Project memory entry: `project-probe-traj-n46416-2026-08-22`.

## 0. What this run is (plain language)
One dataset size (46 416 English arithmetic examples). Four models start
from different places and learn the same task: **base** (untrained),
**pp** (pre-taught arithmetic in symbolic form), **fmt** (pre-taught the
English format with wrong answers), **k7** (a fully trained model with its
last block reset — a known "hidden answer", the control; trained here).
At every saved checkpoint we ask a linear probe: on the carry/borrow
problems — the ones a digit-copying shortcut cannot solve — can the
answer's first digit be read from the model's internals? And: does a probe
trained on pp's *symbolic* inputs transfer to its *English* inputs (same
representation) or not? Output: one curve per model, plus the transfer
curve. Owner reads the probe half (tests 1/6/10/4/5); the teammate reads
the dynamics half (tests 8/9/7 + Tier-3 2/3) using the k7 row this run
adds at 46 416.

## 1. Box prep (owner's rental — never destroy it, never `rm -rf /workspace`)
```bash
# 1a. connect with the owner's ssh line; sanity
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace            # need >= 20 GB free (3 x ~1.3 GB snapshot sets + k7 run + data)

# 1b. venv (non-interactive shells do NOT get it) + the CUDA ld.so.conf bug
source /workspace/venv/bin/activate
python3 -c "import torch; print(torch.cuda.is_available())"
#   if False ("Error 804"): ls /etc/ld.so.conf.d/ ; then
#   mv /etc/ld.so.conf.d/00-compat-<id>.conf /etc/ld.so.conf.d/zz-compat-<id>.conf && ldconfig
#   Do NOT `source /etc/environment` wholesale (owner rule).

# 1c. repo at the pushed commit
cd /workspace/elicit-vs-teach && git fetch -q && git checkout -q ts38-mini && git pull -q
git log --oneline -1        # must be 1ea5946 or a descendant
python3 -c "import huggingface_hub, torch, geode; print('imports ok')"

# 1d. HF identity with WRITE access (k7 pushes to geode-store + geode-internals; results upload)
hf auth login --force       # owner's token; the ambient one may be read-only
python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"   # mhieuuu

# 1e. data (~3 min; the chain refuses to start without D_algo_eval_bare.parquet)
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
python3 ../datagen/make_data.py --scale full --out ../data/full --seed 20260717
python3 ../datagen/make_data.py --scale full --out ../data/full --seed 20260717 --eval-set
python3 ../datagen/make_data.py --scale full --out ../data/full --seed 20260717 --nl-eval-set
python3 ../datagen/make_bare_sets.py --out ../data/full --skip-dose-mult   # no D_dose_mult on ts38-mini
ls ../data/full/D_algo_bare.parquet ../data/full/D_algo_eval_bare.parquet

# 1f. optional: shellcheck (never run locally)
apt-get install -y shellcheck >/dev/null 2>&1 && shellcheck -S warning run_probe_traj.sh
```

## 2. Launch (one command, detached tmux)
```bash
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
tmux new-session -d -s probetraj \
  'source /workspace/venv/bin/activate; \
   NTFY=https://ntfy.sh/geode-run1-kx83q1 NTFY_AUTO=1 \
   bash run_probe_traj.sh --confirm-cost > /workspace/probe_traj.log 2>&1'
sleep 60; grep MILESTONE /workspace/probe_traj.log | tail     # one-shot, not a poll loop
```
Env knobs (prefix the bash line): `N=46416` (size), `LIMIT=6000` (probe
rows), `SKIP_K7=1` (no control training, ~40 min instead of ~1.5 h).

Expected milestone order in `/workspace/probe_traj.log`:
`pull_complete` ×3 parents → `pull_complete run=evt-ts38mt-{base,pp,fmt}-n46416
n_snapshots=26–29` → `pull_complete run=evt-ts38tr-k7-parent` →
`prep_complete` → (inner `[ts38tr]` lines: `parent_skip`, `theta0_scored`
or `theta0_skip`, `train_start/train_complete run=evt-ts38tr-k7-n46416`,
`convergence_check … stop_reason=converged`, `gate_recorded`,
`push_complete` ×2) → `k7_complete` → `probe_done arm=base/pp/fmt/k7`
(~8 min each) → `probe_done theta0 models=4` → `probe_complete` →
`mech_complete run=evt-ts38tr-k7-n46416` → `upload_complete` → `ALL_DONE`
→ `TERMINAL_SUCCESS`. Every stage is idempotent: if the chain dies,
relaunch the SAME tmux command.

## 3. If something stops
| symptom | what it means | do |
|---|---|---|
| `FAILED: pull left no weights/snapshots` | HF token read-only or repo path changed | re-do 1d; check `hf_checkpoint.py pull --run-id <id> --repo-id mhieuuu/geode-internals --with-snapshots` by hand |
| `CONVERGENCE CHECK … stop_reason='max_steps'` on k7 | bug signal, NOT a result (owner policy) | stop, report; do not raise max_steps |
| `LEAKED` verdict for k7 parent | the pulled parent answers zero-shot (em0 > 0.05) | stop, report — the parent on the relay is wrong |
| probe OOM | two GPU jobs at once | make sure nothing else runs on the card; relaunch |
| `ALL_DONE` never, tmux gone | box restarted | relaunch; skips everything already done |

## 4. After `ALL_DONE` (laptop or box)
```bash
# receiver check — list what landed, never trust the sender
python3 -c "from huggingface_hub import HfApi; print('\n'.join(f for f in HfApi().list_repo_files('mhieuuu/geode-internals') if f.startswith('results/ts38mt_probe_traj/')))"
# expected: probe_traj_{base,pp,fmt,k7}.csv + the 4 *_transfer.csv, probe_traj_theta0{,_transfer}.csv,
#           grad_dynamics/weight_diff/resid_shift/jacobian_lens/cross_patch for evt-ts38tr-k7-n46416

# pull + plot (laptop)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('mhieuuu/geode-internals', allow_patterns=['results/ts38mt_probe_traj/*'], local_dir='/tmp/pt')"
cd experiments/training-run/analysis
PYTHONPATH=../../.. python3 plot_probe_traj.py --results-dir /tmp/pt/results/ts38mt_probe_traj
#   -> figures/probe_traj_n46416.png + figures/probe_traj_n46416_summary.csv (one row per model x step)
```
Then score the pre-registered reads (decisions.md entry, verbatim):
- **R-T0** θ0 rows (`checkpoint_step == -1`): base, fmt ≤ chance + 0.05;
  pp ≥ chance + 0.10 AND ≥ base + 0.10 ⇒ gate reopens; ≤ chance + 0.05 ⇒ noise.
  (chance = `max(majority_affected_acc, token_baseline_acc_affected)`.)
- **R-T1** k7 best-layer `acc_affected` ≥ 0.80 by step ≤ 8, else stop at R-T1.
- **R-T2** `t50` = first step with best-layer `acc_affected` ≥ 0.50:
  pp ≤ ½ base ⇒ elicit; 0.8–1.25× ⇒ teach; fmt must be 0.8–1.25× base.
- **R-T3** pp `op_to_task` at θ0 ≥ majority_affected + 0.10 ⇒ shared
  representation; over training, `op_to_task` rising with `task_to_task`
  ⇒ reuse, lagging ≥ 3 snapshots ⇒ rebuild. Failure ≠ no latent sum.
- **R-T4** (teammate) k7-n46416 vs base-n46416 on `rel_fro`, effective
  rank, `first_layer_ge_half`, `grad_early_mass_frac`, cos-to-J —
  base/pp/fmt rows already in `analysis/ts38mt_mech_summary.csv`
  (`n == 46416`); fold the k7 row with
  `ts38mt_mech_summary.py --results-dir <dir> --run-prefix evt-ts38tr --arms k7 --sizes 46416`.
Write the Outcome under the decisions.md entry (replace "Outcome:
pending"), flip EXPERIMENTS.md §6.24 to DONE, commit + push, update memory.

## 5. Teammate hand-off (dynamics half)
Give them: this file's §4 R-T4 line, `analysis/ts38mt_mech_summary.csv`
(rows at n = 46 416), the k7 tables from `results/ts38mt_probe_traj/`, and
decisions.md "2026-08-22 — ts38mt follow-ups" R-B2 table for the
definitions. Tier-3 (tests 2 `circuit_jaccard.py`, 3 `node_edge_delta.py`)
have never run anywhere; their command loop is in
`handoff-ts38mt-launch.md` §5b (`--batch-size ≥ --limit` for the mean
ablation); run them on the θ0/θ_T pairs at n = 46 416 only.
