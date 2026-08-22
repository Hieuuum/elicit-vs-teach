# HANDOFF — launch + post-process ts38mt (for a fresh agent, 2026-08-21)

You are executing a family that is already BUILT, pre-registered and pushed
(commit `93cddfe` on `ts38-mini`). Nothing has been launched. The owner will
give you an SSH line for THEIR box. This file is the authoritative runbook;
design background is `EXPERIMENTS.md` §6.22 and `notes/decisions.md`
"2026-08-21 (night) — ts38mt pre-registration". Read `MEMORY.md` (project
memory) first; the entry `project-mech-tests-plan-2026-08-21` points here.

## 0. Standing rules (owner policies — do not re-litigate)
- The box is the OWNER'S rental: **never `vastai destroy` it, never `rm -rf
  /workspace`**. When everything is done, leave it idle and say so.
- GPU spend for THIS launch is authorized ("do the training first" +
  handing over the box). Anything beyond the launcher (re-runs at a new LR,
  extra sizes, seeds) needs an explicit owner go-ahead.
- Run-until-convergence: `stop_reason=max_steps` anywhere = bug signal, the
  launcher `fail`s on it; do not "fix" by raising ceilings without asking.
- Notifications: ONE ntfy ping at the end (or on a true block), never per
  run/stage. Topic: `https://ntfy.sh/geode-run1-kx83q1`. The launcher pings
  itself on TERMINAL_SUCCESS/FAILED when `NTFY`+`NTFY_AUTO=1` are set.
- Verify the RECEIVER (list files on the HF repos), never trust push logs.
- No poll loops over SSH (`while sleep`): one-shot `tail`, or the Monitor tool.
- Subagents: Sonnet 5, max 4 concurrent. Never `git add -A`.
- Don't touch ts38fs / fig2nl3 / ts1b work; they are other threads.

## 1. Box prep (≈10 min) — do ALL of these before launching
```bash
# 1a. connect (owner's line, e.g. ssh -p <port> root@<ip> -L 8080:localhost:8080)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
df -h /workspace            # need >= 60 GB free (snapshots ~48 GB worst case + data)

# 1b. CUDA forward-compat ld.so.conf bug (hit on BOTH previous boxes from this template)
source /workspace/venv/bin/activate      # non-interactive shells do NOT get the venv
python3 -c "import torch; print(torch.cuda.is_available())"
#   if False ("Error 804"):
ls /etc/ld.so.conf.d/
#   mv /etc/ld.so.conf.d/00-compat-<id>.conf /etc/ld.so.conf.d/zz-compat-<id>.conf && ldconfig
python3 -c "import torch; x=torch.randn(64,64,device='cuda'); print((x@x).sum().item())"

# 1c. repo at the pushed commit
cd /workspace/elicit-vs-teach && git fetch -q && git checkout -q ts38-mini && git pull -q
git log --oneline -1        # must be 93cddfe or a descendant
python3 -c "import huggingface_hub, torch, geode; print('imports ok')"

# 1d. HF identity + WRITE access (pushes go to TWO public repos)
hf auth login --force       # paste the owner's token if the ambient one is read-only
python3 -c "from huggingface_hub import HfApi; w=HfApi().whoami(); print(w['name'])"   # must print mhieuuu
#   quick write probe (harmless): HfApi().upload_file(path_or_fileobj=b'ok', path_in_repo='_write_probe', repo_id='mhieuuu/geode-internals') ; then delete_file it

# 1e. optional but recommended: shellcheck the launcher (never run locally — not installed)
apt-get install -y shellcheck >/dev/null 2>&1 && shellcheck -S warning experiments/training-run/scripts/launch_ts38mt_family.sh
```
If 1c is behind `93cddfe`, STOP — someone pushed over it; check `git log`.

## 2. Launch (one command, detached tmux, log tee'd)
```bash
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
tmux new-session -d -s ts38mt \
  'source /workspace/venv/bin/activate; set -a; . /etc/environment; set +a; \
   NTFY=https://ntfy.sh/geode-run1-kx83q1 NTFY_AUTO=1 \
   bash launch_ts38mt_family.sh --confirm-cost > /workspace/ts38mt_launch.log 2>&1'
sleep 60; tail -40 /workspace/ts38mt_launch.log      # one-shot; expect MILESTONE lines
```
Expected milestone order: `repo …` → `store=…` → stage 2 pulls (base, ts38pp
parent, 10 anchor manifests) → stage 3 data preflight (regenerates
D_algo_bare / D_algo_eval_bare / D_preteachfmt — several minutes, hashes
asserted) → stage 4 overlay guard (34 files) → stage 5 `train_start
run=evt-ts38mt-fmt-parent` (≤ 3340 steps ≈ 4–6 min) → `parent_verified` →
stage 6 `format_acquisition_check verdict=LEARNED` → `format_acquired` →
stage 7 parent pushes → stage 8 per size: `size_start n=1000` … base → pp →
fmt, each `train_complete` / `convergence_check … stop_reason=converged` /
`gate_recorded G5` / two `push_complete` → … `size_complete n=316228` →
stage 9 summary table → stage 10 receiver checks on both repos →
`family_complete runs=31` + `TERMINAL_SUCCESS`. ETA ≈ 2.5–3 h on a 4090.

Monitoring: arm ONE Monitor on `tail -f /workspace/ts38mt_launch.log`
filtered to `TERMINAL_SUCCESS|HALT|FAILED|LAUNCHER_EXIT|Traceback|Error`
(NOT on MILESTONE). Progress checks = one-shot `tail`, or
`ls $GEODE_STORE/runs | grep ts38mt | wc -l` (GEODE_STORE defaults to
`/workspace/elicit-vs-teach/geode-store`). ETAs from measured
`manifest.json` mtimes, not borrowed tables.

## 3. If it stops early — decision table
| log says | meaning | what you do |
|---|---|---|
| `FORMAT-ACQUISITION CHECK … NOT_LEARNED` (stage 6) | full-FT format parent at lr 3e-5 didn't lower bare-NL loss ≥10% vs base | **STOP. Report** the numbers in `$GEODE_STORE/results/ts38mt_family_theta0.json` to the owner. The pre-registered fallback is a parent re-run at 1e-4 — NOT built, needs owner go-ahead (new config `ts38mt_fmt_parent` override + rerun; the 30 overlays are unaffected). |
| `… LEAKED` | parent em0 > 5% — permutation control failed | STOP, report; do not proceed. |
| `CONVERGENCE CHECK … stop_reason='max_steps'` | a ceiling bound | inspect that run's `eval_log.jsonl` tail; report; do not raise ceilings on your own. |
| `G7 refusal` | data order mismatch vs anchor | stage 3 hash passed so this means a stale `runs/evt-ts38-base-n<N>` manifest — re-pull it (`hf_checkpoint.py pull --run-id evt-ts38-base-n<N> --no-weights`) and relaunch. |
| `push_warn` lines | best-effort upload failed | stage 10 retries once; if runs are still missing at the end, push them by hand (below). |
| SSH drops / tmux gone | box restarted | relaunch the SAME command: `train_or_skip` skips every `status: complete` run, so it resumes. A run left `status != complete` must be inspected then removed (`rm -r $GEODE_STORE/runs/<rid>`) before relaunch. |

Manual push for a missing run (repo names matter):
```bash
python3 hf_checkpoint.py push --run-id <rid> --repo-id mhieuuu/geode-store
python3 hf_checkpoint.py push --run-id <rid> --repo-id mhieuuu/geode-internals --with-snapshots --public
```

## 4. After TERMINAL_SUCCESS (box side, ≈15 min)
1. Independent receiver check from the LAPTOP side (not the box's own log):
   list `runs/evt-ts38mt-*` on both repos — 31 run dirs each; on
   `geode-internals` every target has `snapshots/step_*/adapter.safetensors`
   (≥1) and the parent has `sft_snapshots/step_*/model.safetensors` (≥8 —
   steps past convergence are legitimately absent).
2. Pull metadata (no weights) to the laptop mirror, as ts38dense did: one
   `snapshot_download(repo_id='mhieuuu/geode-store', allow_patterns=['runs/evt-ts38mt-*/manifest.json','runs/evt-ts38mt-*/logs/*','runs/evt-ts38mt-*/*.json','runs/evt-ts38mt-*/*.jsonl'], local_dir=<REPO_ROOT>/geode-store)`.
3. Register the family in the analysis drivers — `ts38mt` is NOT yet in
   `analysis/edl_converged_val_floor.py::FAMILIES` / `dataset_size_sweep.py`
   (3 arms: `evt-ts38mt-base-n<N>`, `evt-ts38mt-pp-n<N>`, `evt-ts38mt-fmt-n<N>`).
   Add it following the `ts38pp` entries (tests in
   `tests/experiments/analysis/test_edl_converged_val_floor_families.py`),
   run `--family ts38mt`, commit the CSV `edl_converged_val_floor_ts38mt.csv`.
4. Reproducibility check (pre-registered): base/pp cells vs the shipped
   `edl_converged_val_floor_ts38pp.csv` values, OCV floor, ≤5% relative. Out
   of tolerance = snapshot-hook side effect, report it, do not "fix".
5. Write the Outcome: `decisions.md` → replace "Outcome: pending" in the
   ts38mt entry (table: n × {base, pp, fmt} EDL/label-token OCV floor + test
   floor, parent HALT-gate numbers, repro check, storage actually used);
   `EXPERIMENTS.md` §6.22 status line → DONE + one-paragraph summary.
   Commit + push. Update memory (`project-mech-tests-plan-2026-08-21`).
6. ONE ntfy ping with the headline. Box stays up, idle — say so explicitly.

## 5. Phase-0 analysis (can run in a 2nd tmux while the grid trains; VRAM-light)
Targets: θ0 = `evt-run1-base-v3-ext`, θ_T = `evt-ts38pp-parent` (+ its
`sft_snapshots/step_{7773,15546,23319}`; the launcher already pulls the
parent without snapshots — pull them: `python3 hf_checkpoint.py pull
--run-id evt-ts38pp-parent --with-snapshots`).
```bash
cd /workspace/elicit-vs-teach/experiments/training-run/analysis
S=$GEODE_STORE/runs   # after `export GEODE_STORE=/workspace/elicit-vs-teach/geode-store`
# generic text: 2000 held-out TinyStories, one per line
python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="validation")
with open("/workspace/tinystories_val_2000.txt", "w") as f:
    for r in ds.select(range(2000)): f.write(r["text"].replace("\n", " ").strip() + "\n")
PY
for M in "theta0:dir:$S/evt-run1-base-v3-ext/model" "s7773:dir:$S/evt-ts38pp-parent/sft_snapshots/step_0007773" \
         "s15546:dir:$S/evt-ts38pp-parent/sft_snapshots/step_0015546" "s23319:dir:$S/evt-ts38pp-parent/sft_snapshots/step_0023319" \
         "thetaT:dir:$S/evt-ts38pp-parent/model"; do
  name=${M%%:*}; spec=${M#*:}
  python3 logit_lens.py --model "$spec" --model-name "$name" --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task --device cuda --limit 2000 --out $GEODE_STORE/results/ts38mt_phase0/logit_lens_${name}_task.csv
  python3 logit_lens.py --model "$spec" --model-name "$name" --prompt-parquet ../data/full/probe.parquet            --set-name op   --device cuda --limit 1024 --out $GEODE_STORE/results/ts38mt_phase0/logit_lens_${name}_op.csv
  [[ $name == theta0 ]] && continue
  python3 weight_diff.py --model-a "dir:$S/evt-run1-base-v3-ext/model" --model-b "$spec" --device cuda --out $GEODE_STORE/results/ts38mt_phase0/weight_diff_${name}.parquet
  python3 resid_shift.py --model-a "dir:$S/evt-run1-base-v3-ext/model" --model-b "$spec" --task-parquet ../data/full/D_algo_eval_bare.parquet --generic-local /workspace/tinystories_val_2000.txt --device cuda --limit 2000 --out $GEODE_STORE/results/ts38mt_phase0/resid_shift_${name}.csv
done
```
(`sft_snapshots` dir names are `step_%07d` — check with `ls`.) Each script
prints a summary; read `--help` for flags. Upload the results dir to
`mhieuuu/geode-internals` under `results/ts38mt_phase0/` (`HfApi().upload_folder`).
Test 1 (linear probe) now has its own driver, `analysis/resid_probe.py`
(built 2026-08-21 late; does NOT need `extract.py` dumps — it forwards the
prompts itself). Add to the loop above, for every `$M`:
```bash
  python3 resid_probe.py --model "$spec" --model-name "$name" --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task --prompt-parquet ../data/full/probe.parquet --set-name op --device cuda --limit 2000 --out $GEODE_STORE/results/ts38mt_phase0/resid_probe_${name}.csv
  python3 jacobian_lens.py --model-a "dir:$S/evt-run1-base-v3-ext/model" --model-b "$spec" --prompt-parquet ../data/full/D_algo_eval_bare.parquet --set-name task --device cuda --limit 2000 --out $GEODE_STORE/results/ts38mt_phase0/jacobian_lens_${name}.csv   # skip for theta0
```
Report Phase-0 numbers (emergence layer per checkpoint, task vs op; ΔW
rel_fro / effective rank / overlap_32 per layer; task/generic shift ratio;
probe layer-0 FLOOR vs best layer, task vs op) in the same decisions.md
Outcome. The pre-registered test-1 discriminator and the headline question
are in `EXPERIMENTS.md` §6.22 — quote against them, don't invent new bars.

## 5b. The other mechanistic tests (after the grid; all ten drivers exist)
All drivers live in `analysis/`, CPU-or-GPU, `--help` for flags; every one
writes a table via `--out` and prints an `[evt]` summary. Candidate
readouts for the Tier-2/3 tests are in each module docstring and in
decisions.md "2026-08-21 (late night) — ts38mt mechanistic-test drivers"
— they were registered BEFORE any grid data existed; quote against them.
**LoRA target runs must be `run:<id>` specs** (`dir:` on a wrapped
checkpoint now refuses loudly). Per size N and arm A ∈ {pp, fmt, base},
with P = the arm's θ0 dir (`evt-ts38pp-parent/model`,
`evt-ts38mt-fmt-parent/model`, `evt-run1-base-v3-ext/model`):
```bash
R=evt-ts38mt-${A}-n${N}; T=../data/full/D_algo_eval_bare.parquet; O=$GEODE_STORE/results/ts38mt_mech
python3 grad_dynamics.py --run-id $R --out $O/grad_dynamics_$R.csv                                   # test 8 (logs + snapshots)
python3 resid_probe.py --run-id $R --prompt-parquet $T --set-name task --device cuda --limit 2000 --out $O/resid_probe_$R.csv   # test 1 across snapshots
python3 weight_diff.py --model-a dir:$S/$P --model-b run:$R --device cuda --out $O/weight_diff_$R.parquet                # test 9 (LoRA path)
python3 resid_shift.py --model-a dir:$S/$P --model-b run:$R --task-parquet $T --generic-local /workspace/tinystories_val_2000.txt --device cuda --limit 2000 --out $O/resid_shift_$R.csv   # test 10
python3 jacobian_lens.py --model-a dir:$S/$P --model-b run:$R --prompt-parquet $T --set-name task --device cuda --limit 2000 --out $O/jacobian_lens_$R.csv   # test 7 (+ bridge to 10)
python3 cross_patch.py --model-a dir:$S/$P --model-b run:$R --prompt-parquet $T --device cuda --limit 1000 --out $O/cross_patch_$R.csv   # test 4
python3 node_edge_delta.py --model-a dir:$S/$P --model-b run:$R --prompt-parquet $T --device cuda --limit 1024 --batch-size 1024 --out $O/node_edge_delta_$R.csv   # test 3
python3 dcm.py --model-a dir:$S/$P --model-b run:$R --prompt-parquet $T --device cuda --limit 512 --lambdas 0.001,0.01,0.1 --steps 200 --out $O/dcm_$R.csv   # test 5
```
Test 2 takes several models at once (one call per N):
```bash
python3 circuit_jaccard.py --model base=dir:$S/evt-run1-base-v3-ext/model --model pp0=dir:$S/evt-ts38pp-parent/model --model ppT=run:evt-ts38mt-pp-n$N \
  --model fmt0=dir:$S/evt-ts38mt-fmt-parent/model --model fmtT=run:evt-ts38mt-fmt-n$N --model baseT=run:evt-ts38mt-base-n$N \
  --prompt-parquet $T --device cuda --limit 1024 --batch-size 1024 --out $O/circuit_jaccard_n$N.csv   # test 2
```
Mean-ablation reference is batch-local in `circuit_jaccard.py` — keep
`--batch-size >= --limit` there (or pass `--ablation zero`);
`node_edge_delta.py` buckets by token length and is batch-size-invariant.
Tier 3 (tests 2/3/5) is pre-registered as GATED on Tier 1 finding a latent
sum — run Tier 1 first, and say in the Outcome whether the gate opened.
Upload `results/ts38mt_mech/` to `mhieuuu/geode-internals` like §5.

## 6. Report back to the owner (template)
- Launch time, commit on box, box GPU; HALT-gate numbers (parent vs base
  em0/loss, verdict); runs converged 31/31? any `push_warn`; receiver check
  both repos; storage used on `geode-internals`; EDL table (3 arms × 10);
  repro check pass/fail per cell; Phase-0 headline (did the sum emerge
  earlier in the parent? is ΔW low-rank? task-vs-generic ratio); what is
  NOT done; box status (left up, idle). One ntfy ping, then stop.
