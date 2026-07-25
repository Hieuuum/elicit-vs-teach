# EXPERIMENTS.md — live experiment plan

Status: **executing** (updated 2026-07-24). This is the current state
and remaining work of the elicit-vs-teach training-run experiment.
`specs/02-training-run.md` is the detailed design source;
`experiments/training-run/notes/decisions.md` is the running decision
log; `docs/run*-guide.md` are the box walkthroughs. (This file replaced
the 2026-07-17 cut plan — the cut is executed history; see git history
if you need its record.)

## 1. Question & design

Does eliciting a latent capability differ *mechanistically* from
teaching it? Two arms train the same target task (operator-notation
add/sub, LoRA) from the same base model; they differ **only** in whether
the model was pre-taught the algorithm in another format (Arm A: 1M
correct-label NL add/sub examples; Arm B: none). Internals — gradient
alignment, representation drift, adapter structure — are compared across
performance-aligned snapshots. Single seed (recorded limitation,
spec 02 §13).

## 2. Runs, DAG, status

| # | run_id | Role | Init | Method | Data | Status |
|---|---|---|---|---|---|---|
| 1 | `evt-run1-base-v3-ext` | pretrain (floor 1) | v3 @ 30k | full FT | TinyStories-v2 | **DONE — converged**, `min_val_nats` 1.0718 (owner confirmed 2026-07-21); floor 1 |
| 2 | `evt-run2-armA-algo` | pre-teach | run 1 | full FT | `D_algo` (NL add/sub, correct labels) | **DONE — converged** (2026-07-21, lr 3e-4, step 19k, `min_val_nats` 0.0037); G1 0.9961 PASS |
| 3 | `evt-run3-armA-inst` | format install | run 2 | full FT | `D_inst` (op-notation mult, random labels), to behavioral stop (spec 02 §6) | **DONE** (2026-07-22, lr 3e-6, behavioral stop @ 750); G2/G4/G5 recorded |
| 4 | `evt-run4-armB-inst` | format install | run 1 | full FT | identical `D_inst` + order as run 3, same behavioral stop; counts emergent | **DONE** (2026-07-22, behavioral stop @ 750); G3/G4/G5 recorded |
| 5 | `evt-run5-armA-target` | target | run 3 | LoRA | `D_target` 500K prefix (op-notation add/sub; OPEN(2) closed 2026-07-22) | **DONE — converged** (2026-07-22, step 6,000, min_val 0.00245; 711 snapshots + base); G5 recorded |
| 6 | `evt-run6-armB-target` | target | run 4 | LoRA | identical data, identical order as run 5 | **DONE — converged** (2026-07-22, step 12,500, min_val 0.02301; 832 snapshots + base); G7 verified; G5 recorded (stop-wobble caveat → decisions.md) |
| 7 | `evt-run7-armA-target-1m` | target @ full 1M | run 3 | LoRA | full 1M `D_target` (owner 2026-07-23; supersedes OPEN(2) for this pair only) | **DONE — converged** (2026-07-24 chain, lr 1e-3, step 6,000, min_val 0.0027); G5 recorded; on relay with full snapshots |
| 8 | `evt-run8-armB-target-1m` | target @ full 1M | run 4 | LoRA | identical data + order as run 7 (G7) | **DONE — converged** (2026-07-24 chain, step 11,000, min_val 0.0237 — 8.7× the run-7 floor); G5 recorded; on relay with full snapshots |
| 9 | `evt-run9-llama1b-inst` | format install, external validity | `meta-llama/Llama-3.2-1B` (base) | LoRA r=64, merged after | same frozen `D_inst` + behavioral stop | **INVALIDATED 2026-07-25 — superseded by v2.** Ran at the target pin 1e-3 (scope leak, decisions.md): behavioral stop @ 750, G4 1.0, but **retention 0.0000** vs base Llama's 0.3271 on NL add/sub — format installed, arithmetic destroyed. Dir kept on the relay as the record of the defect |
| 9-v2 | `evt-run9-llama1b-inst-v2` | format install, external validity | `meta-llama/Llama-3.2-1B` (base) | LoRA r=64, merged after | same frozen `D_inst` + behavioral stop | **DONE** (2026-07-25, **lr 3e-6** from the installer retention sweep): behavioral stop @ 1,000, **G4 1.0000**, **G2 retention 0.3193 = 97.6% of base** (bar 0.29). Merged → `model_merged/`, re-verified there: 112/147 tensors moved (max │Δ│ 2.44e-04), format 0.9902, retention 0.3242. **G5 zero-shot 0.2969 / 16-shot 0.5342 / 1.5010 nats — the capability is PRESENT and ACCESSIBLE in the parent, which is what makes run 10-v2 elicitation** (v1's parent: 0.0000 / 0.0 / 9.26) |
| 10 | `evt-run10-llama1b-target` | target, external validity | run 9 (merged) | LoRA r=64 | full 1M `D_target` | **INVALIDATED 2026-07-25 — superseded by v2.** Converged (2026-07-24 chain, step 7,500, min_val 0.0156), G5 0.9844 / 0.0232 nats, on relay — but its parent had **zero** arithmetic, so this run measured **teaching, not elicitation**. Consistent with min_val 0.0156 sitting between run 7 (0.0027) and run 8 (0.0237), nearer teach |
| 10-v2 | `evt-run10-llama1b-target-v2` | target, external validity | run 9-v2 (merged) | LoRA r=64 | full 1M `D_target` | **DONE — converged** (2026-07-25, lr 1e-3 unchanged, step **5,500**, **min_val 0.01323**); G5 zero-shot **0.9883** / 16-shot 0.1426 / test loss **0.0129 nats**. Elicits 0.2969 → 0.9883 EM and 1.5010 → 0.0129 nats from a parent that demonstrably HELD the capability — the first valid external-validity elicitation measurement in the project. vs v1 (same model/tokenizer/data, only the parent differs): floor 0.0156 → 0.01323 (−15%), convergence 7,500 → 5,500 steps (−27%) |

DAG: `1 → 2 → 3 → 5` (Arm A, elicit) and `1 → 4 → 6` (Arm B, teach);
the 1M rerun pair reuses the installers: `3 → 7`, `4 → 8`. External
validity: `Llama-3.2-1B → 9-v2 → 10-v2` (Llama's own pretraining stands
in for the pre-teach stage; elicit-only — a real pretrained model can't
be un-taught, so there is no Llama teach arm). The v1 chain
`9 → 10` is invalidated: run 9 ran at the target-stage LR pin and
destroyed the very capability run 10 was meant to elicit. **The Llama
arm only means "elicitation" if its installer PRESERVES arithmetic —
that is now enforced, not assumed: run 10's `parent_required_gates` is
`[G4, G2]`, so a parent that fails retention cannot launch a target.**
Every run registers in zoo before launch; `require_parent_ready`
(spec 00 V0.6) makes a child refuse to start unless its parent is
complete, every recorded parent gate has `pass: true`, and the gates
named in the config's `parent_required_gates` are recorded.

**Run-1 lineage** (all under `runs/` in the store + HF relay
`mhieuuu/geode-store`): `evt-run1-base` (constant LR 1e-3, min val
1.1464) → `evt-run1-base-v2` (cosine 1e-3→1e-4, fixed 17k horizon, min
val 1.1125) → `evt-run1-base-v2-ext` (warm start, constant 1e-4,
converged at step 4k, min val 1.1066) → `evt-run1-base-v3` (completed
2026-07-21): from-scratch retrain under the paper's exact recipe —
constant LR 1e-3, stop on validation-loss convergence (ε 0.005 / k 3 /
eval_every 1000). Hit the 30k-step cost ceiling still descending
(`stop_reason=max_steps`, min val 1.1020, ~2.5–3.0 mnat/1k in the tail;
samples clean, no repetition loops) → **`evt-run1-base-v3-ext` (launched
2026-07-21, running): warm start from the v3 checkpoint, identical optimizer
recipe, tightened convergence rule ε 2 mnat / k 5 (abandons descent
slower than 0.4 mnat/1k vs 1.7 under 0.005/3; owner 2026-07-21), runs
to convergence — `max_steps` 81k is a 10-epoch ETA bound, not a stop,
with a 5k-step stopping grace (V5.42) absorbing the warm-start
transient (`configs/run1_extend.yaml`) — floor 1 is this run's
convergence point, and children init from its checkpoint.** Compare
runs via `min_val_nats` — manifest `best_val_nats` is ε-gated by the
stopping rule and reads stale.

The model: custom ~38.7M-param Llama-style arch (hidden 512, 8 layers,
MHA, tied embeddings) with a frozen 10K byte-level BPE tokenizer
(digits 0–9 single-token; `experiments/training-run/tokenizer/`).
Spec 02 (2026-07-18 downscale decision) has the details.

## 3. Datasets — DONE (frozen 2026-07-19)

Public HF repo **`mhieuuu/elicit-vs-teach-arith`**, seed 20260717:
`D_algo` / `D_inst` / `D_target`, 1M unique examples each, plus the
1,024-example stratified probe set. `order_hash` per file and
`probe_set_hash` are frozen in `experiments/training-run/data/full/report.json`;
launchers re-verify the downloaded file and refuse on mismatch. Probe
exclusion is question-level (the (a, b, op) triple appears in no
training set, any format). Known + accepted: ~0.07% of `D_inst`'s
random labels coincide with the true answer (won't-fix, decisions.md
2026-07-19).

## 4. Gates

Recorded in each run's manifest under `experiment.gates` by
`scripts/gates.py`; enforced at child launch (§2). Full definitions:
spec 02 §8.

G0 (run-1 story coherence) was **removed 2026-07-20** (owner, spec 02
§8): run 1 trains with the paper's exact recipe and its convergence
point is floor 1 unconditionally; `sample_stories.py` stays as an
ungated inspection tool.

| Gate | After | Check | Status |
|---|---|---|---|
| G1 | run 2 | Arm A ≥95% on NL add/sub — 1,024 seeded val examples, greedy, `exact_match` (`gates.py g1`) | **PASS 0.9961** (2026-07-22) |
| G2 | run 3 | Arm A still ≥95% on NL add/sub post-install — same bar as G1, no separate δ (owner 2026-07-21); drop from G1 reported | **PASS 0.9531** (drop 0.043 from G1; installer-LR length-prior finding → decisions.md 2026-07-22) |
| G3 | run 4 | Arm B ≈ 0% on real add/sub (random labels didn't leak) | **PASS 0.0000** |
| G4 | runs 3–4 | op-notation format validity ~≥99%, both arms; same metric is the installers' in-loop stopping signal (spec 02 §6) | **PASS 1.0 both arms** |
| G5 | runs 3–6 | zero/16-shot op add/sub + shared-set test loss, fixed slices of `D_target_eval` (final protocol 2026-07-22) | recorded (evidence-only): parents A 0.0117 / 2.30 nats, B 0.0000 / 3.75 (latent as required); **finals: run 5 (A) 0.9980 / 0.00194 vs run 6 (B) 0.9502 / 0.03558 — 18× θ_T loss gap** (quote with the stop-wobble caveat, decisions.md 2026-07-22). Pilots B@500K 0.9805 / 0.0140 vs A-ref@50K 0.9941 / 0.0059 → OPEN(2) = 500K. **1M pair: run 7 (A) 0.9971 / 0.0025 vs run 8 (B) 0.9551 / 0.0312 — 12.4× θ_T gap. Llama: run 9 parent 0.0000 / 9.26, run 10 0.9844 / 0.0232 — measured with `eval_target_data_llama.yaml` (an eval's tokenizer must match the model under eval; decisions.md 2026-07-24)**. **Caution (2026-07-25): run 9's 0.0000 was read as "latent as required" — it is not that. It is op-notation after random-label training, not the retention probe. The retention probe is NL add/sub vs a base baseline, and on it run 9 v1 scored 0.0000 against base Llama's 0.3271: the capability was destroyed, not latent. See runs 9-v2/10-v2.** 16-shot ≈ 0 everywhere incl. the 1.24B Llama (collapse — invalidated as a metric; decisions.md) |
| G6 | data gen | V5.1/V5.2 integrity on the real sets | partial — generation-time evidence in `report.json`; formal re-run when `gates.py` grows the subcommand |
| G7 | before run 6 | `data_order_hash`(run 5) == (run 6) | enforced at launch |

## 5. Workflow

- **Tested core** (`geode/*`): code + property tests written together,
  single pass. A change to core math updates its property tests in the
  same commit. Property lists live in specs 01 and 02 (V-numbers); name
  tests after the property (e.g. `test_v5_1_no_probe_leakage`).
- **Scripts** (`experiments/*`): single pass, self-reviewed; smoke test
  only where cheap. `--confirm-cost` on any GPU path.
- **Promotion rule:** logic used by two or more scripts, or whose
  silent failure would corrupt results, moves into `geode/` and gains
  property tests. Nothing else does.
- Suite stays CPU-only, < 2 minutes, no network, tiny in-process
  fixture models (non-negotiable).
- Decisions worth recording go in `notes/decisions.md` (experiment) or
  this file (structure/plan).

Built so far: `geode.edl` (incl. the pinned-adapter prequential loop,
V1.9/V1.10), `geode.train` (full FT + SFT + `apply_lora`), `geode.zoo`,
`geode.arith`, `geode.probe` (schedule V5.55–V5.61, extraction
V5.9–V5.12, analysis metrics V5.13–V5.16 + V5.63 linear CKA) — all
with property tests; launchers `train_sft.py` + `train_target.py`,
`gates.py` g1–g5, `scripts/extract.py` (resumable, `--limit` disk
cap), and ten analysis drivers (`alignment.py`, `drift.py`,
`adapters.py`, `matching.py`, `cka.py`, `learning_curves.py`,
`act_rank.py`, `probes.py`, `trajectory.py`, `steering.py` —
2026-07-24, design notes in decisions.md). Not yet built:
`export_hf.py`, `gates.py` g6.

## 6. Remaining work, in order

1. ~~Run 2~~ / ~~Runs 3–4~~ — **DONE 2026-07-22** (§2 table; outcomes
   + gate values in decisions.md 2026-07-22).
2. ~~OPEN(2) pilot~~ — **DONE 2026-07-22**: phase 1 (target-LR
   mini-sweep) pinned lr 1e-3; phase 2 (the B grid @ 10K/50K/200K/500K
   + A-ref @ 50K, re-scored under the fixed shared eval protocol)
   closed **OPEN(2) = 500K** (§4 G5 row; decisions.md). OPEN(4) closed
   same day (mechanical: schedule over max_steps 23442) and OPEN(10)
   closed **no** (owner) — runs 5/6 are launch-ready; snapshots are
   adapter-only (owner 2026-07-22, ~48 MB/step + one base file ⇒
   ~75 GB both runs; decisions.md sizing).
3. **Extraction over runs 7/8** — tooling DONE 2026-07-24
   (`geode.probe` V5.9–V5.13 + `scripts/extract.py` +
   `analysis/alignment.py`; the runs-5/6 snapshots died with the old
   box, so runs 7/8 are the only extraction targets). Protocol:
   `--limit 128` snapshots/run (one dump ≈ 0.5 GiB; full density
   ≈ 0.6 TiB/run does not fit the 300 GB box), resumable, dumps stay
   on-box (regenerable from the hub snapshots). Paste sheet:
   `docs/run7-8-guide.md` §4; target: one real gradient-alignment
   plot (spec 02 §11).
4. ~~Runs 5–6~~ — **DONE 2026-07-22**: both converged, G5 recorded;
   snapshots + final weights LOST 2026-07-24 with the old box
   (owner-accepted; logs/manifests survive on relay + laptop) —
   headline numbers stand, internals evidence moves to runs 7/8.
5. ~~1M LR re-pin + runs 7–8~~ — **DONE 2026-07-24**: pin lr 1e-3
   everywhere (Arm-A tie-break pilot overrode the B-sweep edge;
   decisions.md); chain ran 7 → 8 to convergence (§2 table); both on
   the relay with full snapshot sets.
6. ~~Llama chain (runs 9–10)~~ — **DONE 2026-07-24** via
   `launch_chain_7_10.sh` (§2 table; four chain bugs + the G5
   eval-tokenizer incident fixed en route — decisions.md). All four
   runs on the relay, run 10 LFS-hash-verified 2026-07-24.
7. **Analyses + publication** — metrics V5.14–V5.16 + V5.63 and all
   ten drivers **built 2026-07-24** (suite 532; decisions.md design
   notes). All ten **run and harvested**; every parquet + figure is on
   the relay. Headline internals results so far: elicit gradients more
   coherent (late phase-mean cos 1.78×, top-PC EV 1.94×); weight-space
   trajectory shows elicit aims at its final displacement ~2.8× more
   sharply in the first 30 steps and keeps step-to-step consistency far
   longer (late 0.51 vs 0.13, teach ending *anti*-correlated at −0.12)
   while **path efficiency is identical across arms** — the difference
   is aim and consistency, not wasted motion; **steering square
   2026-07-25** (§ below). Remaining: cross-metric synthesis, then
   `export_hf.py` to the HF dataset repo (spec 02 §9–10).

   **Steering square (2026-07-25, `steering_square.parquet`).** The 2×2
   of direction source × injection target, one matched alpha grid
   extended to 4. Best EM into an untrained parent, no weight change:
   elicit direction → arm A parent **0.1406** (random 0.0273; an
   interior maximum for its own hook, but early hooks are still
   climbing at the grid edge — `blocks.1` reaches 0.117 at alpha 4, so
   this is the best *found*, not the maximum recoverable) but →
   arm B parent **0.0000**; teach direction → arm A parent 0.0352
   (= random, 9 vs 7 hits/256) and → arm B parent 0.0000. Reading: a
   steering vector **cannot install a capability that is not there**
   (the right-column null is bracketed — at alpha 3–4 the same
   injections drive that parent's loss 3.90 → 8–16 nats without one
   correct answer), and only the elicit arm's shift is causally potent
   where the capability *does* exist. Teach's direction lowers
   teacher-forced loss as much as elicit's (1.58 vs 1.54 nats) while
   producing 4× fewer correct answers — distributional movement is not
   capability surfacing. Basis-comparability control in decisions.md.

   **Direction emergence (2026-07-25, `direction_emergence.parquet`).**
   `analysis/emergence.py` times the vector the square injected across
   all 128 dumps per run. Split-half reliability is 0.997–0.999 at every
   snapshot, so the curve is signal end to end. At the final hook point
   elicitation's direction is **fixed from the first snapshot** — cos to
   its final direction 0.9824 at step 10 and never below that across
   6000 steps, while only its magnitude grows — whereas teaching's
   rotates away to **cos 0.117** by step 52, recovers past 0.5 only at
   step 279, and settles in the last ~10% (mean cos over the middle of
   the run 0.612 vs elicit's 0.999). The depth pattern inverts: elicit's
   deepest hooks settle first (0–2% of the run), teach's last (96%).
   Persistent-robust hook-mean thresholds: cos ≥ 0.5 at 0.2% of the run
   (elicit) vs 3.7% (teach); progress ≥ 0.5 at 0.5% vs 9.5%. **Caveat
   carried in decisions.md**: at matched capability (V5.16) elicit leads
   on progress at only 31/128 points, all early — the late comparison is
   confounded because both metrics saturate at each run's own endpoint
   and arm B sits 45.7% through its run against arm A's 3.4% at matched
   points. Quote the per-layer contrast and the persistent thresholds,
   not the raw step ratios.

## 7. Budget

~$2k total, tracked in the external sheet — this repo never spends it
silently (`--confirm-cost` everywhere). Spent to date: ≈ $2–3 (run-1
family + runs 2–4 incl. sweeps — the 38.7M scale keeps whole run
families under a dollar). The real spend is runs 5–6 extraction
storage (~1 TB HF PRO, re-estimated at pilot) and GPU time for
2 × 1,024-snapshot LoRA runs.
