# Phase-3 runbook — notation swap (stub)

> **Historical runbook stub** — every phase-3 run that launched is complete
> and closed; the teach arm was built but never launched. This file maps the
> phase's moving parts to their full records. The launchers are frozen
> byte-identical in `scripts/archive/` and reference pre-reorg paths
> internally, so they are **not re-runnable as-is** — see the path-mapping
> table at the top of `notes/decisions.md`.
> Paths are relative to `experiments/training-run/` unless they start with
> `specs/` or `docs/`.

## Design in one paragraph

Addition-only notation swap, the reverse direction of runs 2/5–8: an
operator-notation ADDITION parent, then an NL-addition target through the
prequential EDL harness. 1–8 digit operands over 64 cells, 500K parent /
500K target streams, positive operands, seed 20260727, elicit arm only
(the elicit installer was cut to n=0 in the prior phase — generous to
elicit, not conservative). Datasets are frozen in `data/phase3/`
(**local-only, NOT on HF**) — hashes, row counts, and provenance in
`manifests/data_phase3.md`. Pre-exposure parent↔target: **5.30% direct /
6.00% incl. answer-identical commuted twins** (quote both or neither).
Full design + rationale: `notes/decisions.md` entry **"2026-07-27 —
phase 3: the notation swap, and the EDL floor that made the signature
unreadable"** and `EXPERIMENTS.md` §3b.

## What ran, in order

| stage | launcher (frozen) | run ids | outcome |
|---|---|---|---|
| parent + G1 | `scripts/archive/launch_phase3.sh --stage parent` | `evt-p3-elicit-parent` | complete; G1 caveats in decisions.md entry "Launch, 2026-07-27" |
| format-install conditional | same launcher, `--stage target` preamble | (decision only) | G4 on NL-add prompts, `--no-record`, threshold 0.90, decision parsed from the PRINTED RATE (never the exit code); ≥0.90 ⇒ no installer ran |
| no-bridge target (control) | same launcher, `--stage target` | `evt-p3-elicit-target` | complete, converged; entry "2026-07-27 — phase 3 elicit arm COMPLETE" |
| translation bridge | same launcher, `--stage bridge` | `evt-p3-elicit-bridge` | G6 0.9993 PASS, but G2 op-add retention 0.3018 **FAIL** → phase halted; entry "Phase 3 bridge RAN and FAILED its retention gate" |
| recovery detour | `scripts/archive/launch_phase3_recover.sh` | `evt-p3-elicit-recover` → `evt-p3-elicit-recover-target` | op-add restored (G1 0.9941) but repair erased translation (G6 → 0.0000); recover-target zero-shot saturates identical to control |
| bridged target | `scripts/archive/launch_phase3_bridge_target.sh` | `evt-p3-elicit-target-bridge` | trained directly on the G2-failed bridge via the `external_base` bypass (preserves the bridge's recorded failing G2); converged step 4500 |
| teach arm | `scripts/archive/launch_phase3_teach.sh` | — | **built, never launched**; entry "2026-07-28 — Phase-3 teaching arm built (unrun)" |
| embedding warm-start family | `scripts/archive/launch_phase3_warmstart.sh` | 12 candidates + 3 exact-100K targets | complete, relay-verified; entries "practical Phase-3 embedding warm-start pre-registration" and "… RESULTS" |

## Headline result (bridge question)

At matched n = 384K with the **fixed test floor**, EDL/token: control
0.01954 < recover 0.02900 < bridge 0.03891 bits/token — the bridge helps in
NO form (≈2× control), doubly confounded, no causal claim. Endpoint accuracy
saturates at matched capability — read EDL and always name the floor.
Full tables: decisions.md entries "Phase 3 bridge RAN and FAILED…" and
"2026-07-27 (later) — the bridged target itself".

## Configs

- Training configs (all archived): `configs/archive/phase3/p3_*.yaml` +
  `configs/archive/phase3/p3/*.yaml` (12 overlays).
- **Still live** (not archived — used by evaluation, not training):
  `configs/eval_p3_data.yaml`, `configs/eval_p3_bridge_data.yaml`.
- LR pin: `configs/lr_pin.yaml` (never moved). The launcher refuses to start
  when the target LR ≠ pin or the installer LR = target pin (the run-9 scope
  leak); the live guard is `geode.train.assert_lr_scope` (V5.71) behind the
  `scripts/phase3_guards.py` shim.

## Gate quirks worth knowing before reading any phase-3 record

- The G4 format conditional and bridge G6 are scored `--no-record` /
  score-first: a recorded sub-threshold gate on a shared parent is V0.6 death
  for every child. Decisions are parsed from the printed rate, never exit
  codes.
- `stop_reason=max_steps` anywhere in this phase = bug signal, never a done
  state (runs end on ε/k val convergence).
- Bridge-descended runs (`recover_from_bridge`, `target_on_bridge`) declare
  `parent_run_id: null` + `external_base` to leave the bridge's recorded
  failing G2 intact. This bypass lives in `scripts/train_target.py`.
- G4/G5 refuse bridge configs (those gates parse integer answer slots; the
  bridge is answer-free translation).

## Where the rest lives

- Decision log: `notes/decisions.md`, entries dated 2026-07-27 → 2026-07-28
  (see the phase TOC at the top of that file).
- Experiment plan status: `EXPERIMENTS.md` §2 (DAG) and §3b (datasets).
- Warm-start selection (sum LR .1 / colon 1.0 / sum-colon 1.0, raw MDL vs
  control 0.7247 bits/example): decisions.md entry "… warm-start RESULTS";
  warm-start exposure is unbilled teaching, not elicitation.
