# New-phase runbook — one installer, two EDL measurements

> **STATUS 2026-07-26: the dose grid ran, and its result retired the elicit
> installer.** All five dose installers converged and were gated;
> `launch_phase2.sh` stopped at dose 16 on G2 retention 0.8467 < 0.95, as
> written. Every dose was damage and none helped, so the curve had no interior
> optimum and its argmax was n = 0 — **the owner dropped the elicit arm's
> installer entirely** (decisions.md, "no elicit installer"). The phase is now
> two runs: the teach installer, and one target per arm. The five dose runs
> and their curve are kept as the negative control that licenses n = 0.

Design ratified by the owner 2026-07-26 and revised the same day after the
dose grid ran; rationale in `notes/decisions.md` (entries "new-phase installer
redesign ratified", "dose grid RUN", "no elicit installer"), the rules in
`specs/02-training-run.md` §5/§6. This file is the operational side: what
runs, in what order, on what hardware, and what has to be true before each
step. Everything here is pre-registered — nothing in the sequence below is a
decision left to launch time.

## The three runs

| run_id | stage | data | stop rule | where |
|---|---|---|---|---|
| `evt-p2-armB-instperm` | teach installer | `D_inst_perm` (200K pool) | G4 ≥ 0.90, k=3, per step | GPU |
| `evt-p2-armA-target-noinst` | EDL measurement | `D_target` 1M | ε/k 0.002/5 on the stopping block | GPU |
| `evt-p2-armB-target-perm` | EDL measurement | `D_target` 1M | same | GPU |

**The elicit arm has no installer row, and that is the design.** Its target
inits straight from `evt-run2-armA-algo`, whose only recorded gate is G1
(0.9961). What the arms must share is the *state* at the start of the target
stage — format-valid, holding the true answer-shape prior, no target mapping —
not the *procedure* of having had an installer. Arm A's parent measures that
state already (G4 1.0000; mean answer digits 3.726 vs true 3.746, phase 0b);
Arm B's does not (G4 0.0039), which is why it still takes `D_inst_perm`.

Configs: `configs/p2_armB_instperm.yaml`,
`configs/p2_armA_target_noinst.yaml` (the G7 anchor) and
`configs/p2_armB_target_perm.yaml`, all standalone. Retired but kept, with
banners: `configs/p2_armA_dose.yaml` + `configs/p2/dose*.yaml` (ran — the
negative control), `configs/p2_armA_target_dose.yaml` + `configs/p2/
target_dose*.yaml` (never launched). Calibration pilots
`configs/pilot/p2_dose_cal_n{1,16}.yaml` are likewise closed.

## Order, and why it is this order

```
(elicit: no installer)                   ──► Arm A target  (G7 anchor)
teach installer ──► G4 + G3 + G5 ──► leak bar ──► Arm B target
```

1. **The teach installer first** — it is the only installer, and the Arm B
   target cannot launch until it is gated.
2. **The installer is gated before its target.** The target config lists the
   required gates, so `require_parent_ready` (V0.6) refuses the launch if one
   is missing or failing. There is no "gate it later" path.
3. **The Arm A target is launched before the Arm B target** — it is the G7
   anchor and Arm B's launch-time check reads its registered manifest.
   (Registration happens at launch, so Arm B can start as soon as it
   *starts*, not when it finishes.)
4. **The leak bar runs between the teach installer and the teach target**
   (below).

`./launch_phase2.sh --confirm-cost [--stage teach|targets|all]` does all of
it, skips anything already complete, and refuses to start unless the artifact
hash-matches and the target LR both equals the committed pin and differs from
the installer LR. `--stage doses` is refused with a pointer here.

## Exposure is the phase's one asymmetry — report it, don't assume it

With Arm A at zero installer examples, everything the teach installer sees is
unbilled warm-up on the target task's own surface form (mapping-only EDL,
decision 5). The owner's instruction is to keep it **low**, and the config
already does the only thing that can: it stops at the *first* step format is
installed (G4 ≥ 0.90, k=3, `eval_every: 1` ⇒ a 3-step / 384-example floor).
The 200K file is a **pool** and `max_steps` a cost ceiling — neither is a
budget. So the exposure is an *output*: the teach stage prints
`final_step × batch_size` and that number goes in decisions.md. If it comes
back large the lever is `batch_size`, never `max_rows` — starving the pool
risks the format never installing, which fires `stop_reason=max_steps` as a
bug signal and measures nothing. Note the direction: this asymmetry favours
**teach** (unbilled warm-up can only shrink its EDL and the ratio with it), so
it cannot manufacture the elicit result.

## Gates, and which number is load-bearing

| gate | on | meaning here |
|---|---|---|
| G4 ≥ 0.90 | teach installer | The phase's format bar, and this arm's in-loop stopping metric re-scored. Arm A has no installer to gate; its parent measures 1.0000 on the same external prompts (`gates.py g4 --prompt-config … --threshold 0.90 --no-record`), which is *why* it has none. |
| G2 ≥ 0.95 | — | Retention on `D_algo` gated the retired dose installers (decision 4) and is what failed at n=16. With no elicit installer there is nothing to retain *through*: the Arm A target inits from the algo checkpoint whose own G1 is 0.9961. |
| G3 ≤ 0.02 | teach installer | NL add/sub leak, recorded for continuity with run 4. **Cross-notation**, so a pass alone proves little here. |
| **G5 zero-shot ≤ 0.02** | teach installer | **The operative leak measure.** `D_inst_perm` is operator-notation add/sub — the target task's own surface form — so the matched-notation, matched-op, question-disjoint check is G5 on `D_target_eval`. A leak deflates teach's EDL and inflates the headline ratio, so `launch_phase2.sh` enforces the number itself before the teach target; G5 records `pass: true` by protocol and must never be read as a verdict. |
| G5 | every run | Zero/16-shot + shared-set test loss, evidence. Required to be *recorded* before each target (so the measurement provably exists). |

## Preconditions that are easy to get wrong

- **`D_inst_perm` is not on the HF hub** (publish is owner-held as of
  2026-07-26). `p2_armB_instperm.yaml` carries `data.local_path`,
  repo-root-relative, and the hash is still verified. On a fresh box either
  publish it first and delete that line, or copy the parquet into
  `experiments/training-run/data/full/`.
- **Both target parents must be in the box's store before the targets run**
  (`hf_checkpoint.py pull`): `--init-from` and the G7 check both read them.
  Arm A's parent is `evt-run2-armA-algo` and Arm B's is
  `evt-p2-armB-instperm`; the teach installer's own parent
  `evt-run1-base-v3-ext` has to be there before *it* can start. Checkpoints
  are ~155 MB each (full FT at 38.7M, fp32).
- **fp32 everywhere in this phase.** The target harness is fp32 already and
  the teach installer is pinned to it, so nothing here carries an
  arm-asymmetric precision.
- **`stop_reason=max_steps` on any run in this phase is a bug signal**, not a
  budget outcome: every ceiling here is ≥ 4× the expected stop. The sole
  exception was the two `eps_nats: 0.0` calibration pilots, which were
  *designed* to run to their ceiling so the whole trajectory got recorded;
  both are closed.
- **Step 0 is recorded for every run** (`experiment.step0` in the manifest) —
  format validity for a behavioral run. This is the phase-0 lesson: without
  it, "the rule fired" and "the rule was already satisfied before training"
  are indistinguishable. It is what proved Arm A needed no installer.
- **The Arm A target's parent is a SHARED checkpoint.** `evt-run2-armA-algo`
  parents runs 3 and 5 as well, so nothing may be recorded onto its manifest:
  a gate verdict written there gates every existing child via
  `require_parent_ready` (V0.6). Its G4 (1.0000) and G5 numbers are measured
  and live in decisions.md, scored `--no-record`; the target config asks only
  for the G1 that is already there.

## Closed, kept for the record

- **Dose ε/k pin** (ε 0.0002, k 5, box CPU, `OMP_NUM_THREADS=16`) and the
  device rule (`DOSE_DEVICE`, `experiment.device`) — both did their job: the
  replay predicted the production stops exactly, and the curve below was not
  confounded by its own stopping rule (%descent spread 0.015pp across all
  five doses). Table and selection rule in decisions.md.
- **Parent G4 baseline 1.0000** on `evt-run2-armA-algo` (n=512 external
  prompts, `--no-record`) — the measurement that turned "the dose did
  nothing visible" into "G4 was saturated before any dose", and one of the
  two numbers that license n = 0.
- **`analysis/dose_curve.py`** still reads the five dose manifests and
  reproduces the damage curve. It is now a control-experiment tool, not a
  gating step.

## Cost

Three runs at 38.7M. Targets: the `max_steps` ceiling quotes $0.08 each at
4090 rates (runs 7/8 printed exactly that for the identical ceiling), so
~$0.16 for both, and no snapshot disk (`snapshots.n: 0`). The teach installer
is the only open-ended one: a 200K pool with a per-step behavioral eval,
expected to stop in the low hundreds of steps. Nothing in this phase moves the
~$2k budget meaningfully; the `--confirm-cost` flag is still required
everywhere.
