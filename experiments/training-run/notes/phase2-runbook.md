# New-phase runbook — role-matched installers + the elicit dose curve

Design ratified by the owner 2026-07-26; rationale and the five numbered
decisions live in `notes/decisions.md` (entry "new-phase installer redesign
ratified"), the rules in `specs/02-training-run.md` §5/§6. This file is the
operational side: what runs, in what order, on what hardware, and what has to
be true before each step. Everything here is pre-registered — nothing in the
sequence below is a decision left to launch time.

## The twelve runs

| run_id | stage | data | stop rule | where |
|---|---|---|---|---|
| `evt-p2-armA-dose{1,2,4,8,16}` | elicit installer | `D_dose_mult` prefix (n) | ε/k 0.0002/5 on full-dose train loss | CPU, pinned |
| `evt-p2-armB-instperm` | teach installer | `D_inst_perm` (200K) | G4 ≥ 0.90, k=3, per step | GPU |
| `evt-p2-armA-target-dose{1,2,4,8,16}` | EDL measurement | `D_target` 1M | ε/k 0.002/5 on the stopping block | GPU |
| `evt-p2-armB-target-perm` | EDL measurement | `D_target` 1M | same | GPU |

Configs: `configs/p2_armA_dose.yaml` (**is** the dose-1 run) with the doses
2/4/8/16 as overlays in `configs/p2/`; `configs/p2_armA_target_dose.yaml`
(**is** the dose-1 target, and the G7 anchor) with the same overlay pattern;
`configs/p2_armB_instperm.yaml` and `configs/p2_armB_target_perm.yaml`
standalone. Calibration pilots are `configs/pilot/p2_dose_cal_n{1,16}.yaml`.

Every dose is a **prefix** of the frozen `D_dose_mult` order, so
dose 1 ⊂ 2 ⊂ 4 ⊂ 8 ⊂ 16 — the curve is a dose curve, not five samples.

## Order, and why it is this order

```
5 dose installers  ──► G4 + G2 + G5 each ──► 5 dose targets (anchor first)
teach installer    ──► G4 + G3 + G5      ──► leak bar ──► teach target
```

1. **Dose installers first**, in any order — they have no dependency on the
   teach arm and need no GPU (≤ 16 examples).
2. **Every installer is gated before its target.** The target configs list
   the required gates, so `require_parent_ready` (V0.6) refuses the launch if
   one is missing or failing. There is no "gate it later" path.
3. **The dose-1 target is launched before the other five targets** — it is
   the G7 anchor and their launch-time check reads its registered manifest.
   (Registration happens at launch, so the others can start as soon as it
   *starts*, not when it finishes.)
4. **The leak bar runs between the teach installer and the teach target**
   (below).

`./launch_phase2.sh --confirm-cost [--stage doses|teach|targets|all]` does all
of it, skips anything already complete, and refuses to start unless the
artifacts hash-match, the dose ε/k is pinned non-null, and the target LR both
equals the committed pin and differs from the installer LR.

## Gates, and which number is load-bearing

| gate | on | meaning here |
|---|---|---|
| G4 ≥ 0.90 | both arms' installers | The phase's **shared** format bar. Dose runs carve no val split, so theirs is scored on a fixed slice of the frozen `D_target_eval` (`gates.py g4 --prompt-config … --threshold 0.90`); the teach arm's is its own in-loop metric re-scored. |
| G2 ≥ 0.95 | dose installers | **Retention** on `D_algo`. This is what re-validates the inherited 3e-6 installer LR for the elicit arm (decision 4). A failure means extend the LR downward, as the run-9 fix did — never proceed. |
| G3 ≤ 0.02 | teach installer | NL add/sub leak, recorded for continuity with run 4. **Cross-notation**, so a pass alone proves little here. |
| **G5 zero-shot ≤ 0.02** | teach installer | **The operative leak measure.** `D_inst_perm` is operator-notation add/sub — the target task's own surface form — so the matched-notation, matched-op, question-disjoint check is G5 on `D_target_eval`. A leak deflates teach's EDL and inflates the headline ratio, so `launch_phase2.sh` enforces the number itself before the teach target; G5 records `pass: true` by protocol and must never be read as a verdict. |
| G5 | every run | Zero/16-shot + shared-set test loss, evidence. Required to be *recorded* before each target (so the measurement provably exists). |

## Preconditions that are easy to get wrong

- **The two new artifacts are not on the HF hub** (publish is owner-held as of
  2026-07-26). Both configs carry `data.local_path`, repo-root-relative, and
  the hash is still verified. On a fresh box either publish them first and
  delete those lines, or copy the two parquets into
  `experiments/training-run/data/full/`.
- **The dose installers must be pushed to the relay before the box runs their
  targets** (`hf_checkpoint.py`): a target's `--init-from` and its G7 check
  both need the parent present in the box's store. Dose checkpoints are
  ~155 MB each (full FT at 38.7M, fp32).
- **fp32 everywhere in this phase**, both arms and both stages. The target
  harness is fp32 already; pinning the installers to it removes an
  arm-asymmetric precision (the dose runs are CPU-run, the teach installer
  GPU-run) and makes the two numerically comparable.
- **`stop_reason=max_steps` on any run in this phase is a bug signal**, not a
  budget outcome: every ceiling here is ≥ 4× the expected stop (dose ceiling
  6000 vs the n=16 stop at 1281). The sole exception is the two `eps_nats: 0.0`
  calibration pilots, which are *designed* to run to their ceiling so the whole
  trajectory gets recorded.
- **A dose run's `sft_result.min_val_nats` is NOT a val loss.** The trainer
  reuses that field name for whatever metric drove the stop, and a dose run
  has no val split at all — the number is the minimum **full-dose training**
  loss. What disambiguates it is `training.stopping.metric == "train_loss"` in
  the same manifest (spec 00 §2, the labelled union branch added for exactly
  this reason). Never plot it on the same axis as a target run's val curve.
- **"Absorbed" means trained to convergence on the dose, not shown it once.**
  At n=1 that is memorisation of a single example (8.9963 → ~3e-4 nats over
  ~600 steps at lr 3e-6). The dose is a dose of *information*, delivered until
  the model has it; it is not a dose of *steps* (owner decision 3, and the
  run-until-convergence policy).
- **Step 0 is recorded for every run** (`experiment.step0` in the manifest) —
  format validity for a behavioral run, the full-dose loss for a dose run.
  This is the phase-0 lesson: without it, "the rule fired" and "the rule was
  already satisfied before training" are indistinguishable.

## Two things to do first, on the box — BOTH DONE 2026-07-26

1. **Pin the dose ε/k.** DONE: **ε 0.0002, k 5**, from both pilots rerun on
   one device (box CPU, `OMP_NUM_THREADS=16`). Table and selection rule in
   decisions.md. Both ends agree to 0.01pp of descent; the inherited 0.002/5
   would have split them by 25× more. `max_steps` went 4000 → 6000 in the same
   pass, because the measurement showed n=16 (not n=1) is the slowest dose to
   absorb. Every launch path refuses while ε is null, so this could not have
   been skipped by accident.
2. **Record the parent's G4 baseline.** DONE: **1.0000** on
   `evt-run2-armA-algo`, n=512 external prompts, `--no-record`. The launcher
   re-runs it at the start of the doses stage. Read every dose G4 against it:
   the parent is already saturated, so for this arm G4 detects damage only.

**Device is part of the pin.** `DOSE_DEVICE` (default `cpu`) drives every dose
training call and `experiment.device` is recorded in every manifest; a resume
that would mix devices fails loudly. `train_sft.py` defaults to cuda-when-
present, so on a GPU box an unpinned dose stage would silently have run the
production installers on a different device from the pilots that calibrated
their stopping rule.

## Post-run check the dose curve needs

ε/k is calibrated at the two ENDS of the grid (n=1 and n=16, the
`eps_nats: 0.0` pilots replayed by `analysis/dose_stop_calibration.py`). The
middle doses inherit it, so after all five runs, verify the rule treated them
alike: for each, the fraction of descent achieved at the stop is
`1 − L_stop / L0` (both in `experiment.step0.dose_loss_nats` and the run's
`min_val_nats`; the floor is taken as 0, which the pilots justify — both
bottomed out at ~3e-4 nats). If one dose is an outlier, that is a finding
about the rule and belongs in decisions.md before any EDL number is quoted —
not a reason to re-tune ε after the fact.

## Cost

Twelve runs at 38.7M. Dose installers: ~$0 (CPU, ≤16 examples). Targets: the
`max_steps` ceiling quotes $0.08 each at 4090 rates (runs 7/8 printed exactly
that for the identical ceiling), so ~$0.50 for all six, and no snapshot disk
(`snapshots.n: 0`). The teach installer is the only open-ended one: 200K
examples with a per-step behavioral eval, expected to stop in the low hundreds
of steps. Nothing in this phase moves the ~$2k budget meaningfully; the
`--confirm-cost` flag is still required everywhere.
