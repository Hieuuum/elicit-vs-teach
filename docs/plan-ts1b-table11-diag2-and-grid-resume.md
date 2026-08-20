# Plan: ts1b Table-11 follow-up diagnostics → grid resume

Written 2026-08-20 (session: paper re-read + Table-11 mismatch analysis).
Audience: the fresh session the owner will start to execute the remaining
ts1b pipeline. Read alongside memory
`project-ts1b-scaleup-plan-2026-08-19.md` (SESSION UPDATE 5/6 at the
bottom are this session's) and decisions.md's ts1b pre-registration entry.

## 0. What happened in the authoring session (2026-08-20 midday UTC)

1. **Paper re-read verdict on Table 11** (full detail in chat + memory SU5):
   most of Table 11 already replicates (base 0/0 ✓, pre-teach-format 0/0 ✓,
   pp 0-shot ≈ replicates under install-native render: block nl_scaffolded
   k=0 = 1.76% vs paper 2.0%). The one hard-failing cell is pre-taught
   16-shot (paper 11.9%, ours 0.0) — and it fails even in the parent's own
   exact training render (block op 0.96 → 0.011 at k=1), so it is decided
   purely by context-generality, which the paper's protocol under-specifies
   (packing of the 4M-example install; few-shot separator convention).
2. **pp-arm target-grid LR sweep KILLED mid-rung-3** (owner instruction
   "stop the lr sweep" + "kill everything, decide later"). Results at kill:
   - rung 1e-4: complete, min_val **2.2908** nats
   - rung 3.53e-4 (paper's Table-3 LoRA LR): complete, min_val **1.1799**
   - rung 1e-3: **killed incomplete** — partial run dir
     `runs/evt-ts1b-pp-target-lrsweep-1e-3` left on the box, NOT cleaned.
   No target runs started. tmux `pptgt` is gone.
3. **Two diagnostics built, tested (97 tests green), committed (`d6cbe6b`),
   smoked on GPU, and launched full** (tmux `diag2`, sequenced):
   - **EOS-separator few-shot** (`--shot-separator eos`): shots joined by
     per-exemplar tokenization + one EOS each (the training-row/document-
     boundary shape) instead of the blank-line text join. Tests whether the
     few-shot collapse is an eval-composition convention. Output:
     `results/ts1b_theta0_fewshot_diag_eos.json`, log
     `/workspace/ts1b_diag_eos.log`.
   - **DM-mixture eval at 1B** (`--dm-mixture-only --dm-mixture-renders
     bare block`): the paper's 19-template DM question mixture under the
     install-native block scaffold AND the bare (paper-likely) render,
     k={0,16}. Output: `results/ts1b_theta0_dm_mixture.json`, log
     `/workspace/ts1b_diag_dm.log`.
   - Smoke preview (n=64, noisy): block-scaffold DM **mixture k=0 = 0.25**
     (⋙ paper's 2.0%; sym_q 0.27, bare_op 0.97); **bare render = 0.0
     everywhere** (the `Answer:` handle-lock is total); k=16 = 0 under
     blank separator; EOS k=1 block-op 0.047 vs 0.011 blank (weak, n=64).
4. `results/ts1b_theta0_fewshot_diag.json` (the main Table-11 diag, was
   box-local) pushed to `mhieuuu/geode-store` + receiver-verified.

## 1. PAUSE POINT — owner decisions before anything below runs

- **D1 (Table-11 interpretation / next experiment).** After reading the two
  diag JSONs: if EOS-separated shots recover few-shot EM, the Table-11
  16-shot cell is reachable eval-side — rerun the DM mixture with
  `--shot-separator eos` for the paper-comparable k=16 number. If EOS does
  NOT recover, the only remaining lever is a **packed-install pp parent**
  (~$3, ~3h: same 4M block data, several EOS-separated examples per
  sequence) — this reverses the earlier "no packing" P2 ruling, so it is
  explicitly the owner's call.
- **D2 (target-stage LR).** Recommendation: pin **3.53e-4** (clear sweep
  leader, and the paper's own LoRA LR) into BOTH `configs/ts1b_pp_target.yaml`
  and `configs/ts1b_pf_target.yaml` (same value in both is load-bearing —
  the arms must differ only in θ0). Alternative: re-run rung 1e-3 first
  (~$0.02) for completeness.
- **D3 (proceed with grids).** ~4h pp grid + ~4h pf grid at $0.7426/h ≈ $6
  GPU total, then push/verify/teardown per the standing checklist.

## 2. Execution steps (after the owner's go)

1. **Verify box state directly** (never trust this doc's snapshot):
   `ssh -p 16375 root@212.13.234.23` — instance `48125506`, tracked vast
   acct. `tmux ls`, `tail /workspace/ts1b_diag_*.log`, `nvidia-smi`.
   Venv: `. /workspace/venv/bin/activate`. Never `source /etc/environment`
   wholesale; extract HF_TOKEN with
   `sed -n 's/^HF_TOKEN=//p' /etc/environment | tr -d '"' | head -1`.
2. **Confirm both diag JSONs are pushed + receiver-verified**
   (`HfApi().list_repo_files('mhieuuu/geode-store')` must contain
   `results/ts1b_theta0_fewshot_diag.json`, `..._diag_eos.json`,
   `..._dm_mixture.json`). The authoring session pushes the latter two once
   the runs finish — verify anyway.
3. **Pin the LR (D2)** from the LAPTOP repo (never a box-side
   `git checkout --` dance): edit both target configs' `train.lr`, commit,
   push, box `git pull`.
4. **Clean the killed sweep rung**: delete
   `geode-store/runs/evt-ts1b-pp-target-lrsweep-1e-3` on the box (partial,
   killed mid-run; its manifest was registered pre-training and would
   confuse the launcher's skip logic).
5. **Sweep bypass**: `launch_ts1b_pp_target_grid.sh` unconditionally runs
   its LR sweep (it only skips COMPLETE rungs — with rung 1e-3 deleted it
   would retrain it). Either (a) small patch: skip the whole sweep phase
   when the config `train.lr` is already pinned (mirror the pf launcher's
   PLACEHOLDER check), or (b) let it retrain rung 1e-3 (~$0.02, ~10 min)
   and auto-pick — but then an auto-pick ≠ D2's pinned value could
   diverge; (a) is recommended and matches the owner's "stop the sweep".
6. **Relaunch pp grid** (tmux, `--confirm-cost`, GPU otherwise idle — NEVER
   concurrent with any diag; the 47x contention lesson): 5 target runs,
   push-as-you-go. ~4h.
7. **pf grid** (`launch_ts1b_pf_target_grid.sh`): verifies the pinned LR,
   5 runs. ~4h.
8. **OCV/EDL curves** (checklist step 5): add a ts1b family to
   `analysis/edl_converged_val_floor.py` — needs a design decision
   (two-arm family, no base comparator; `match_data_order_with` stays
   null). Local CPU, box not needed. Do NOT rush it at 5am.
9. **Final receiver verification** of every run file, then ntfy
   `https://ntfy.sh/geode-run1-kx83q1`, then **destroy instance 48125506**
   (`vastai destroy instance 48125506` — terminate, not stop; only after
   step 9's verification).

## 3. Known breakage to reconcile (small FIX commit, any time)

`tests/experiments/scripts/test_config_completeness.py` has **6 pre-existing
failures** (verified present without this session's diff): they still assert
the pre-override world — pp parent 36,093-step one-epoch pin
(`test_ts1b_pp_parent_one_epoch_step_count`,
`test_ts1b_pp_vs_pf_differs_only_in_labels_and_step_count`), pf
convergence-regime stopping (`test_ts1b_pf_parent_stopping_regime`), and the
deleted install-LR sweep overlays (`test_ts1b_pp_lrsweep_overlay_values`).
Reality: pp = `evt-ts1b-pp-parent-contdiag5000` (dead config
`ts1b_pp_parent.yaml` kept but unused), pf = one-epoch pin (owner override,
commit `c9f5f7e`), overlays superseded. Update the tests to the current
config reality (or delete the dead config + its tests in one commit).

## 4. Diag reproduction lines (for reference)

```bash
# EOS-separator few-shot (pp parent only)
python3 experiments/training-run/analysis/theta0_fewshot_diag.py \
  --model-family ts1b --runs evt-ts1b-pp-parent-contdiag5000 \
  --shot-separator eos --skip-label-loss --device cuda \
  --out $GEODE_STORE/results/ts1b_theta0_fewshot_diag_eos.json

# DM 19-template mixture, block + bare renders, k={0,16}
python3 experiments/training-run/analysis/theta0_fewshot_diag.py \
  --model-family ts1b --runs evt-ts1b-pp-parent-contdiag5000 \
  --dm-mixture-only --dm-mixture-renders bare block --skip-label-loss \
  --device cuda --out $GEODE_STORE/results/ts1b_theta0_dm_mixture.json
```
