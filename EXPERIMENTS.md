# EXPERIMENTS.md — live experiment plan

Status: **executing** (updated 2026-07-27). This is the current state
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

### 3b. Phase-3 datasets — DONE (frozen 2026-07-27, seed 20260727)

`experiments/training-run/data/phase3/`, **not yet published to HF**, so
the configs read them via `local_path` (hash still verified). Addition
only, positive operands, so no phase-3 example contains a `-` at all —
the `'Ġ-'` collision that carried the old +/− asymmetry cannot occur.
The notation is **reversed**: `D_p3_on_add` (500K, operator) is the
pre-intervention set and `D_p3_nl_add` (500K, natural language) is the
target. Plus `D_p3_nl_eval` (100K), `D_p3_nl_mult` (200K, permuted —
the conditional installer pool) and a 4,042-row NL probe.

**Operands run 1–8 digits, evenly across all 64 stratification cells**
(owner 2026-07-27). At the old 4-digit ceiling the six smallest cells
held only 63–7,032 questions in total, so parent and target overlapped
in them by force: the same 500K/500K split measured **31.95% / 41.09%**
pre-exposure, worse than the 29.18% that drew the 2026-07-26 criticism.
Widening the bands is what bought the room — 58 of 64 cells now land
within one question of the fair share. Bands 5–8 are additive; `CELLS`
stays 4×4, so every pre-phase-3 dataset hash is unchanged.

Probe and eval are carved **first** with a per-cell ceiling of cap/8, so
an eval generated afterwards cannot end up with zero rows in the cells a
training set consumed whole. Measured pre-exposure of target by parent:
**5.30% direct, 6.00% including commuted twins** (for addition the twin
is answer-identical — quote both or neither), of which 22,465 of 26,476
shared questions are the six structurally saturated cells. Per-cell
figures in `data/phase3/report.json`.

Two consequences: the prequential stream ends at **n = 500,000**, so
phase 3 has no value at the n = 576,000 every earlier ratio is quoted
at — it compares against its own teach arm, or at n ≤ 500,000. And ~75%
of training now has an operand of 5+ digits, a harder task than any run
1–10; the parent's G1 gate is what catches that before the target
spends anything.

**Answer-free elicit bridge (built 2026-07-27, unrun).** From the same
operator-addition parent, `evt-p3-elicit-bridge` full-FTs on a frozen
200K-row bidirectional question-rewriting corpus (100K positive addition
pairs; no computed sums), then must pass operator-add retention (G2), NL
integer-format validity (G4), and held-out bidirectional translation exact
match (G6). `evt-p3-elicit-target-bridge` then consumes the same 500K
natural-language target artifact, order, LR, seed, LoRA settings, epsilon/k
rule, and ceiling as the unchanged no-bridge `evt-p3-elicit-target`; its
overlay changes only run identity and parent/gate lineage. The bridge train
hash is `d68ec2d…4997`, held-out eval hash `d0ddc0b1…4719`; direct overlap
with target is zero (3.441% including commuted twins). The corpus is frozen
for a future teach arm, but no teach branch is built here.

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
| G6 | phase-3 bridge | held-out bidirectional translation exact text match; aggregate and both directions ≥95% | built, unrun (`gates.py g6`; frozen 4,096-row eval) |
| G7 | before matched target | identical frozen target `data_order_hash` + prefix against its anchor | enforced at launch |

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
`gates.py` g1–g6, `scripts/extract.py` (resumable, `--limit` disk
cap), and ten analysis drivers (`alignment.py`, `drift.py`,
`adapters.py`, `matching.py`, `cka.py`, `learning_curves.py`,
`act_rank.py`, `probes.py`, `trajectory.py`, `steering.py` —
2026-07-24, design notes in decisions.md). Not yet built:
`export_hf.py`.

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

8. **New phase — role-matched installers + dose runs (design ratified
   2026-07-26, decisions.md; NOT LAUNCHED).** Runs *alongside* the closed
   chains — nothing supersedes runs 7/8 or 9-v2/10-v2 (owner). Design:
   teach installer = `D_inst_perm` (permuted add/sub; correct marginals
   install the true shape prior), stop G4 ≥ 0.90, k=3, per-step eval,
   step-0 recorded; elicit installer = dose from `D_dose_mult` (1 real
   mult example at the smallest dose), ε/k plateau on the full-dose
   training loss; LR 3e-6 both with G2 retention gating; target EDL
   redefined **mapping-only** (spec 02 §5/§6, edited same commit). Landed
   2026-07-26: both artifacts generated + hash-pinned (`label_coincidence`
   0.0145%), `permute_labels` (V5.64), trainer train-loss stopping mode
   (V5.65/V5.66), launcher wiring, runs-index + manifest `lifecycle`
   metadata (spec 00 §2).

   **Built out the same day** (decisions.md "new phase built out";
   operational detail in `notes/phase2-runbook.md`). Owner set the dose grid
   at **n ∈ {1, 2, 4, 8, 16}**, so the phase is **twelve runs**: 5 dose
   installers (`evt-p2-armA-dose{n}`, prefix-nested doses), 1 teach shape
   installer (`evt-p2-armB-instperm`), and **one target per installer** —
   5 + 1 EDL measurements on the identical frozen 1M order as runs 7/8, with
   `snapshots.n: 0` (≈$0.08 each at the ceiling runs 7/8 printed). Landed:
   all twelve configs (`configs/p2_*` + `configs/p2/` overlays),
   `launch_phase2.sh` (three resumable stages, guards before spend, and the
   teach-arm leak bar), `gates.py g4 --prompt-config` for no-val runs
   (verified end-to-end), launcher-side **step-0 recording** for every SFT
   run, the fp32 phase pin, and a `train_loss` branch in the spec-00
   stopping union (V0.7, dispatching on the metric *value*).

   Remaining before launch: (a) rerun the n=16 calibration pilot to its floor
   and **pin the dose ε/k** — a coarse rule was measured to fire at 99.08% of
   descent at n=1 but 93.70% at n=16, so it is not inheritable; the config
   holds a null ε and every launch path refuses until it is set; (b) publish
   `D_inst_perm`/`D_dose_mult` to the hub (owner-held) or copy them to the
   box; (c) the runs themselves. Compute policy (owner 2026-07-26): the phase
   runs on a rented GPU box over SSH, not on the laptop.

   **BOTH TARGETS DONE 2026-07-26** — after the owner cut Arm A's installer
   to n=0 the phase is **two** target runs, not six, and both have now run at
   the sweep-confirmed pin **lr 1e-3, seed 316**, on the identical frozen 1M
   order (G7 verified), leak bar 0.0000 ≤ 0.02:

   | run | stop | min_val | L_test (97,952) | G5 0-shot |
   |---|---|---|---|---|
   | `evt-p2-armA-target-noinst` (elicit) | converged @ **4,500** | 0.00225 | **0.0024** nats | 0.9971 |
   | `evt-p2-armB-target-perm` (teach) | converged @ **12,000** | 0.01959 | **0.0159** nats | 0.9697 |

   **Headline: teach/elicit EDL = 12.1× at matched n = 576,000** (per label
   token 0.41215 / 0.03399 bits; per example 2.03280 / 0.16766 — the two
   agree because both arms share a tokenizer at 4.93 label tokens/example).
   Figure: `analysis/figures/edl_per_token_p2.png`. Both confounds cut
   AGAINST this result, not for it: arm B entered 3.12 nats ahead (its
   installer ran, arm A's did not), and it still needed 12× the excess code
   length. LR-pin caveat and the arm-A/undetermined-sign note: decisions.md
   2026-07-26.

   **vs runs 7/8, at the same n = 576,000** (recomputed — the two ratios must
   be read at one n, since both curves are still descending): runs 7/8 give
   0.02285 / 0.44480 = **19.5×**, this phase **12.1×**. The drop is real, not
   a marker artifact, and is an arm-A effect: teach moved −7%, elicit rose
   **+49%** (0.02285 → 0.03399). Both arms' parent chains differ between the
   designs, so this bounds rather than isolates the installer's contribution.
   Caveat: arm B converged at 1.54 epochs, so its MDL — epoch-1 only, by
   footnote 1 — stops accumulating while it is still descending (val 0.03074
   at the boundary vs its 0.01959 floor). See decisions.md.

   Preceded by the **8-point LR sweep** (both arms × {3e-4, 1e-3, 3e-3} +
   seed twins at 1e-3). The arms disagree — arm A's optimum is 3e-4, arm B's
   is 1e-3 — so rule 11's shared-LR band is empty and rule 8 takes arm B's
   optimum. The pin therefore does not move, and phase 2 stays comparable
   with runs 7/8, which executed under the same value.

9. **Phase 3 — the notation swap (owner 2026-07-27). BUILT, NOT LAUNCHED.**
   Addition only, positive operands; operator notation becomes the
   pre-intervention task and natural language the target — the reverse of
   runs 2 and 5–8. Elicit arm only; the teach arm is deferred.

   **Read the metric finding in decisions.md before reading any phase-3
   curve.** `plot_edl_per_token.py` subtracts a *moving* val-loss floor, so
   no run can be monotone under it — including run 10-v2, which has the
   most rising steps of any run in the project. The canonical fixed-test
   floor (`geode/edl/metrics.py:68`) makes every run monotone or nearly so.
   **Monotonicity does not discriminate elicit from teach.** The script now
   takes `--floor {val,test}` (default `val`, existing figures unchanged)
   and prints `rising k/n` so a shape claim is checkable. What separates the
   arms is the level at matched n (0.034 vs 0.412 bits/token) and where the
   information lands (1.5K vs 152K).

   Phase 3's actual aim is narrower than the original ask: shrink the elicit
   arm's 3.46-bit early spike, which the 2026-07-27 unlock result implicates
   as an *addressing* cost rather than algorithm acquisition.

   Ready: datasets (§3b), `configs/p3_elicit_{parent,inst,target}.yaml` +
   `configs/eval_p3_data.yaml` + `configs/p3/target_after_inst.yaml`, and
   `scripts/launch_phase3.sh --stage parent|target|all` (four refusal paths
   negative-tested). The format install is **conditional** on G4 format
   validity < 0.90 zero-shot on NL prompts, and is expected not to fire;
   when it does fire it uses NL **multiplication**, because permuted-label
   NL addition would train wrong sums into a parent that already knows
   addition (the run-9 retention failure).

   Blocked on: a GPU. Nothing here has been run.

## 7. Budget

~$2k total, tracked in the external sheet — this repo never spends it
silently (`--confirm-cost` everywhere). Spent to date: ≈ $2–3 (run-1
family + runs 2–4 incl. sweeps — the 38.7M scale keeps whole run
families under a dollar). The real spend is runs 5–6 extraction
storage (~1 TB HF PRO, re-estimated at pilot) and GPU time for
2 × 1,024-snapshot LoRA runs.
