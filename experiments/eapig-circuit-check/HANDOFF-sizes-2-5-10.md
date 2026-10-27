# Handoff: rerun the EAP-IG check at 2%, 5%, 10% of edges

For a fresh agent. Read this file, `PLAN.md` (the frozen plan) and the vast-box
skill (`~/.claude/skills/vast-box/SKILL.md`) before doing anything. The owner
asked for this run on 2026-10-07; renting the box for it is approved (state the
prices in chat before `race --confirm-cost`, per the skill).

## Why

The first run (2026-10-01, results on HF `mhieuuu/geode-internals` under
`results/eapig_check/`) evaluated circuits at 0.1%, 0.2%, 0.5% and 1% of the
195,865 edges. No model passed all five tests at any size. The best was 2/5
(sufficiency and partial necessity). Equivalence, consistency and specificity
failed everywhere, even for `elicit_child`, which is accurate (EM 0.953, m_full
42.4). PLAN.md pre-registered this case: "No circuit passes at ≤ 1%: extend to
2% and 5% before concluding anything." The owner added 10%.

First-run numbers to compare against:

| model | EM | m_full | m_empty | f @0.1% | f @1% |
|---|---|---|---|---|---|
| elicit_parent | 0.000 | 0.207 | −0.534 | 0.000 | 0.017 |
| fmt_parent | 0.000 | 0.030 | 0.028 | −0.764 | 0.382 (noise: denominator 0.002) |
| elicit_child | 0.953 | 42.36 | −42.37 | 0.003 | 0.725 |
| teach_child | 0.145 | 28.02 | −27.71 | 0.023 | 0.609 |

## What is the same and what changes

Same as the first run. Do not change any of these.

- The data files under `data/` (committed).
- EAP-IG scores. Reuse each model's `scores.pt` from the first run rather than
  rescoring. Scoring does not depend on circuit size.
- `sanity.json`. Reuse the first run's file (the EM-fixed version, commit
  24b3d48). It carries `sec_per_circuit_eval` (4.4 s), which drives the cut
  rule: parents get 50 random draws, children 100, and TinyStories 10. Reusing
  it keeps the draw counts identical to the first run. **Never let a new
  sanity.json into the results dir.** A faster batch size could push the
  timing under 2 s and silently change the draw counts.
- All tests, thresholds, the stopping rule, the TinyStories check,
  parent-in-child, real patching of the top 20, and compare.py. The frozen
  decisions table in PLAN.md applies unchanged.

What changes:

- Sizes are 0.02, 0.05 and 0.1, so k = 3,917, 9,793 and 19,586 edges. The
  stopping rule runs 2% → 5% → 10%.
- Random-circuit seeds are derived from the size, so the draws are new.
- Chance edge Jaccard rises with size: about 0.010, 0.026 and 0.053, against at
  most 0.005 before. compare.py reports it per size. Always quote overlaps
  against it.
- Outputs go to `results_large/` locally (gitignored) and to HF
  `results/eapig_check_large/`. **Never write to `results/eapig_check/`.** It
  holds the first run.

## Code (already done; branch `eapig-circuit-check`)

`run.py` and `compare.py` take `--sizes`, with the default unchanged. The
`run_box.sh` script reads these env vars:

| env var | effect |
|---|---|
| `SIZES` | circuit sizes for evaluate and compare |
| `RES_NAME` | local results dir |
| `HF_SUBDIR` | push target under `results/` |
| `BATCH_SIZE` | batch size: changes speed and VRAM, not results |

These are tested in `tests/experiments/test_eapig_compare.py` (CLI end-to-end
at the large sizes). A CPU smoke test of all three stages at 2/5/10% on a tiny
random 16L/32h/8kv model passed. Check that the box's `git log -1` equals the
laptop's HEAD.

## Steps

1. **Box.** Follow the vast-box skill: preflight, then
   `race 5 --template 47d6874fc604ad50603ae6c334e859c8`. Use the default bar of
   at least 24 GB and Ampere or newer. Keep the first `ready:` box and destroy
   the other four at once. Label the winner `claude-eapig`.
   - On the winner, check out the branch:
     `git fetch && git checkout eapig-circuit-check && git pull`.
   - Install the circuits extra: `pip install -q -e ".[circuits]"`. It provides
     scipy; onstart does not install it.
   - Confirm HF whoami returns `mhieuuu`.

2. **Seed the results dir** from the first run, in
   `experiments/eapig-circuit-check/` on the box with the venv active:
   ```
   HF_HUB_DISABLE_XET=1 python - <<'EOF'
   import pathlib, shutil
   from huggingface_hub import hf_hub_download
   for t in ["elicit_parent", "fmt_parent", "elicit_child", "teach_child"]:
       d = pathlib.Path("results_large") / t; d.mkdir(parents=True, exist_ok=True)
       for f in ["sanity.json", "scores.pt"]:
           shutil.copy(hf_hub_download("mhieuuu/geode-internals",
                                       f"results/eapig_check/{t}/{f}"), d / f)
   EOF
   python -c "import json; print(json.load(open('results_large/elicit_child/sanity.json'))['exact_match'])"   # must print 0.953125
   ```
   With these files in place, `run_box.sh` logs `skip ... sanity` and
   `skip ... score` for all four models. If it ever runs `sanity` or `score`,
   stop it: the seeding failed.

3. **Pick the batch size** (about 3 min). The first run used bs 16, held about
   10 GB of 32 GB, and showed about 99% GPU utilization. A larger batch may
   still cut wall time, because there is less Python overhead per mask. Probe
   in a **scratch** dir that is never `results_large`:
   ```
   for bs in 16 32 64; do
     python run.py --tag elicit_child --model podhajskimarcin/evt-ts1b-elicit-ft-n4000000 \
       --stage sanity --results /workspace/bsprobe/bs$bs --batch-size $bs 2>&1 | grep '\[sanity\]'
   done   # while it runs, in another ssh: nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 5
   ```
   - Pick the bs with the lowest `eval` seconds whose peak VRAM stays at or
     below about 80% of total. Leave headroom: evaluate builds more masks than
     sanity.
   - If 32 and 64 are no faster than 16 (within about 5%), keep 16.
   - `m_full` and `m_empty` must match across batch sizes to about 1e-4.

4. **Launch** in tmux, recording the `EXIT=` marker per the skill:
   ```
   SIZES="0.02 0.05 0.1" RES_NAME=results_large HF_SUBDIR=eapig_check_large BATCH_SIZE=<bs> \
     bash run_box.sh --confirm-cost
   ```
   The log is `results_large/run_box.log`. Within about 10 min,
   `elicit_parent` should print `[eval] size 0.02: ...`.

5. **Watch utilization closely for the whole run.** The owner wants the box
   fully used but not overloaded. Arm a Monitor and re-arm it on every 30-min
   expiry until `EXIT=`. It should print one line about every 2 min with
   these fields:
   - from `nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks_throttle_reasons.active --format=csv,noheader`:
     GPU util %, VRAM used/total, temperature, power, and throttle reasons
   - from `/proc/loadavg`: the 1-min load against `nproc`
   - from `free -m`: RAM and swap used
   - from `top -b -n1 -p $(pgrep -f "[r]un.py")`: %CPU of the run.py process

   Have it also grep the log for `done|FAILED|EXIT=|Traceback|Error|OOM|Killed`.

   Rules:

   - **Healthy:** GPU util is at least 90% during evaluate, VRAM at most 90%,
     load average at most nproc, swap 0, and temperature under about 85 °C with
     no thermal or power throttling. run.py at about 100% of one core is normal:
     one Python thread drives the GPU.
   - **Not fully used:** GPU util below about 80% for more than 5 min while a
     stage is running. Model loading and HF downloads between models are short
     dips; ignore them. The fix is one of the following:
     - **(a) Raise the batch.** Kill the tmux job, then relaunch with a larger
       `BATCH_SIZE`. A finished model's `evaluate.json` is kept (skip-if-done),
       so only the current model restarts.
     - **(b) Run models side by side.** Use this when VRAM headroom is at least
       twice one process's usage. All four evaluates are independent, since
       every `scores.pt` is pre-seeded. Run each model's evaluate as its own
       `python run.py --tag <t> --model <repo> --stage evaluate --results results_large --sizes 0.02 0.05 0.1 --batch-size <bs> [--parent-tag <p>]`
       in its own tmux window, with `OMP_NUM_THREADS=$(( $(nproc) / 2 ))`.
       Children need `--parent-tag` (`elicit_child` → `elicit_parent`,
       `teach_child` → `fmt_parent`); the model repos are in `run_box.sh`.
       When all four `evaluate.json` exist, run the same `run_box.sh` command
       once more. It skips everything and only runs compare and the push.
       Never run two `run_box.sh` at once: they share one log and both push.
   - **Overloaded:** any of VRAM above 95%, a CUDA OOM, load average above
     nproc, swap above 0, RAM above 90%, or throttling. Lower `BATCH_SIZE` or
     stop the second process. An OOM kills the stage, and `run_box.sh` exits
     with `FAILED`. Relaunch with a smaller batch; finished models are skipped.
   - Log every change you make (time, what, why, util before and after) in
     PLAN.md's Log.

6. **Stop conditions** (pre-registered or obvious):
   - **`fmt_parent` passes all five tests at any size.** The pipeline is
     broken. Stop, keep the box, and report. This is the one case for an
     early ntfy, as "blocked".
   - **k is not 3917/9793/19586 in `evaluate.json`, or any `sanity`/`score`
     stage runs.** The setup is wrong. Stop and fix it.

7. **Finish.** Follow the skill's §7:
   - Push. `run_box.sh` does this and prints `verify: missing on hub: none`.
   - Verify from the laptop: list `results/eapig_check_large/` with
     `env -u HF_TOKEN`, and open one `evaluate.json` to check that `k` matches.
   - Destroy the box and confirm none of ours are left.
   - Send exactly **one** ntfy with the headline.
   - Watchers die with the session. On any new session, check the box's log
     for `EXIT=` first.

8. **Write-up** (laptop, no GPU).
   - Pull `results/eapig_check_large/` into `results_large/` and run
     `python3 compare.py --results-dir results_large --figures-dir results_large/figures --sizes 0.02 0.05 0.1`.
   - In PLAN.md's Results, add a "2026-10 rerun at 2/5/10%" block. It holds the
     per-model table (f per size, tests passed per size, selected size), the
     route overlaps against chance, and the accuracy-gap flag. Put it next to
     the first run's 0.1–1% numbers so f is visible across all 7 sizes.
   - Commit to `eapig-circuit-check` and push. Do not open a PR; the owner
     decides that.
   - Update the memory file `project-eapig-circuit-check-2026-10-01.md`.

## Reading the results (guardrails)

- Larger circuits make sufficiency and f easier by construction. Read f
  against each size's random band, not on its own.
- The accuracy gap is still flagged (EM 0.953 vs 0.145). Child-vs-child
  f/sufficiency is not a fair comparison without the matched subset, which the
  saved artifacts cannot rebuild.
- `fmt_parent`'s m_full − m_empty is 0.002, so its f is noise even though
  `f_degenerate` reads False. The parents' f ceilings are flagged unreliable.
- If equivalence (TOST, ε = 10% of m_full) still fails at 10% for
  `elicit_child`, say so plainly. Do not loosen ε or any threshold
  post hoc. The frozen table is the owner's.
- Report in the owner's style: plain, short, and anchored to the run. Define
  each label once.

## Cost and time

The first run's evaluate took about 39 min per parent and 40–80 min per child
over 4 sizes at bs 16. Three sizes should take about 2–2.5 h end to end on a
similar GPU, roughly $1–1.5 including the race. If utilization tuning
(step 5a or 5b) shortens it, record the actual time in the Log.
