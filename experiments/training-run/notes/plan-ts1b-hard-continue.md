# Plan: continue the two ts1b children on the hard-weighted set (2026-11-11)

Status: **planned, nothing built or run.** Owner asked for this plan before a /clear.
Read with `STATUS.md` and decisions.md entries 2026-11-07 / 2026-11-11.

## Goal

Train the matched children further on more hard problems, under two stopping
rules, to see what each rule gives:

- **A. Loss stops improving** (less strict than the parents' rule, owner).
- **B. Higher accuracy target** (matched-skill stop, harder bar).

## Fixed inputs (done)

| what | value |
|---|---|
| elicit start | `mhieuuu/evt-ts1b-elicit-ft-fresh-b256-c500` (test EM 0.953, 4x4+ 0.829) |
| taught start | `mhieuuu/evt-ts1b-teach-ft-fresh-b256` (test EM 0.952, 4x4+ 0.906) |
| data | `data/full/D_algo_bare_hard4m.parquet`, order_hash db1bb820… (gitignored; scp to the box). 4x4 40%, 3x3/3x4/4x3 15%, 2x4/4x2 5%, small 1.25% each. 95% new to both children |
| test | unchanged `D_algo_eval_bare_trainmix_n20000` (order_hash 6be24318…), scored per cell by `score_em.py` |
| reserve | `data/full/D_eapig_reserve_3x4.parquet` — 3x4 pairs for future circuit folds; in no training set |

## The four runs

Same for all: full FT, batch 256, lr 4e-5 constant, bf16, seed 316, compile,
checks every **500 steps** (128K examples) for both children, val/check rows
held out from hard4m by the trainer (`val_fraction`, 4,000 rows).

| run id | start | stop rule |
|---|---|---|
| `evt-ts1b-elicit-hard-loss` | elicit `-c500` | A |
| `evt-ts1b-teach-hard-loss` | taught `-b256` | A |
| `evt-ts1b-elicit-hard-acc` | elicit `-c500` | B |
| `evt-ts1b-teach-hard-acc` | taught `-b256` | B |

- **Rule A (proposed values):** stop when held-out val loss has not improved by
  more than **0.005 nats** for **3 checks in a row** (parents used 0.002 / 5:
  this fires sooner, as asked).
- **Rule B (proposed values):** exact match **≥ 0.95** on 1,040 check prompts,
  3 checks in a row. Check prompts are stratified to the TRAINING mix, so they
  are 40% 4x4 — a harder bar than the old 90% on the old mix.
- **Step cap (proposed): 10 passes = 156,250 steps** (~13 h per run on an
  L40S). Below the usual 20-pass rule, to bound cost; hitting it = "did not
  converge / never reached 95%", report it, don't ship as converged.

## Hardware and schedule

- One box with **2 GPUs**, L40S or H100 (not 300 W workstation cards;
  check `nvidia-smi` power vs clock in the first minute).
- One run per GPU at a time (`CUDA_VISIBLE_DEVICES`):
  - GPU 0, rule A: elicit-hard-loss, then teach-hard-loss.
  - GPU 1, rule B: elicit-hard-acc, then teach-hard-acc.
- Cost estimate: one pass over 4M rows = 15,625 steps ≈ 1.25 h per run on an
  L40S. Likely 1–3 passes per run → ~4–10 h of box time at ~$1.2–1.6/h for
  2× L40S ≈ **$5–16**; worst case (all four hit the cap) ≈ 26 h ≈ $40.

## Build before renting (laptop)

1. Four configs `configs/ts1b_{elicit,teach}_hard_{loss,acc}.yaml`, copied from
   `ts1b_*_ft_fresh.yaml`; data = hard4m pin; a test pins that they differ only
   in run id, start model and stop rule.
2. Box script `scripts/run_ts1b_children_hard.sh --confirm-cost`: pull the two
   start models from HF, run the two lanes in parallel on GPU 0 / GPU 1, score
   each with `score_em.py`, push each to `mhieuuu/<run id>` (private), print
   `EXIT=`. Reuse `run_ts1b_children_fresh.sh`'s `child()` pattern.
3. **Accuracy logging for rule-A runs (decision, see below).**
4. Run the touched tests; push the branch (commit 0b2e4ab is not pushed yet).

## Open decisions (defaults in brackets)

1. **Log accuracy during rule-A runs?** [yes] Today a loss-stopped run logs no
   accuracy, so A vs B can't be compared along the way, only at the end. Fix:
   an optional "log-only" accuracy check in the train loop. It lives in
   `geode/train/sft.py`, which Marcin's branch also edits — check
   `git log origin/unlearn-elicit-teach -- geode/train/sft.py` first; keep the
   change small and additive.
2. Rule A values [0.005 nats, 3 checks]. Rule B target [0.95].
3. Step cap [10 passes].
4. Save intermediate checkpoints? [no — 5 GB each; only the stop model].

## After the runs

- Table per run: stop step, examples, wall time, test EM overall + per cell
  (esp. 4x4+), and for B whether the two children still match.
- Verify each HF push from the laptop (`env -u HF_TOKEN`), destroy the box, one
  ntfy, update STATUS.md / decisions.md / memory.
- Then: circuit suite on the chosen pair (main cell 3x4+, second 4x4+; needs
  `build_folds.py` to support 3x4 and read the reserve).
