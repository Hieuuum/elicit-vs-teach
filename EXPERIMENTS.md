# EXPERIMENTS.md — live experiment plan

Status: **executing** (updated 2026-07-28). This is the current state
and remaining work of the elicit-vs-teach training-run experiment.
`specs/02-training-run.md` is the detailed design source;
`experiments/training-run/notes/decisions.md` is the running decision
log; `docs/runbooks/*.md` are the box walkthroughs. (This file replaced
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
transient (`configs/archive/runs/run1_extend.yaml`) — floor 1 is this run's
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
the conditional elicit installer pool), `D_p3_nl_add_perm` (200K,
permuted — the role-matched teach installer pool), `D_p3_nl_warmstart`
(4,096 correct NL-add rows: 512 fixed-dose + 3,584 LR selection), and a
4,042-row NL probe. The teach pool is question-disjoint from target, eval,
and probe; its order hash is `0e58ba91…c535` and label coincidence is
0/200K. The warm-start pool additionally excludes parent questions and
answer-identical commuted twins of all four frozen sets; its order hash is
`3a8383e6…9e32`.

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
- **Structure (2026-07-29 reorg):** closed material lives byte-identical in
  archive subtrees — `configs/archive/{runs,phase2,phase3}/`,
  `configs/sweeps/<family>/`, `scripts/archive/`, `notebooks/archive/`;
  operator runbooks in `docs/runbooks/`; tracked artifact manifests in
  `experiments/training-run/manifests/`; tests mirror the source tree
  (`tests/lib/**`, `tests/experiments/**`). `notes/decisions.md` opens with
  the old→new path map. Naming: archived files keep historical names; NEW
  files follow the live-tree conventions (`run<N>_<stage>.yaml`,
  `p<phase>_<arm>_<stage>.yaml`, `launch_<family>.sh`,
  `eval_<slice>_data.yaml`, `test_v<N>_<M>_<property>.py`).

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
   `docs/runbooks/run7-8-guide.md` §4; target: one real gradient-alignment
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
   operational detail in `docs/runbooks/phase2-runbook.md`). Owner set the dose grid
   at **n ∈ {1, 2, 4, 8, 16}**, so the phase is **twelve runs**: 5 dose
   installers (`evt-p2-armA-dose{n}`, prefix-nested doses), 1 teach shape
   installer (`evt-p2-armB-instperm`), and **one target per installer** —
   5 + 1 EDL measurements on the identical frozen 1M order as runs 7/8, with
   `snapshots.n: 0` (≈$0.08 each at the ceiling runs 7/8 printed). Landed:
   all twelve configs (`configs/archive/phase2/p2_*` + `configs/archive/phase2/p2/` overlays),
   `scripts/archive/launch_phase2.sh` (retired; three resumable stages, guards before spend, and the
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

9. **Phase 3 — the notation swap (owner 2026-07-27). PARENT + CONTROL TARGET +
   BRIDGE RAN; recovery detour closed.** Addition only, positive operands;
   operator notation becomes the pre-intervention task and natural language the
   target — the reverse of runs 2 and 5–8. The elicit arm has run; the teach
   arm is now built but unrun. **Outcome (see decisions.md 2026-07-27):** the translation bridge
   passed G6 (0.9993) but destroyed op-add retention (G2 0.3018); an owner-
   directed recovery retrained op-add on the bridge checkpoint (G1 restored to
   0.9941) but the repair erased the equivalence (G6 → 0.0000). The plain NL
   target on the recovered base has byte-identical zero-shot (0.9912, saturation
   — not a finding), but its EDL/token curve DIVERGES from the control: ~1.5×
   HIGHER excess description length at matched n (0.029 vs 0.020 bits/token,
   fixed-test floor) despite a lower final loss — the opposite of an elicitation
   benefit. **Then the bridged target itself ran** (`evt-p3-elicit-target-bridge`,
   NL-add trained DIRECTLY on the G2-failed bridge with translation still intact —
   `train_target.py` gained the `external_base` bypass; G7-matched to the control;
   converged step 4500, G5 zero-shot 0.9961). It is the WORST of the three: at
   matched n=384K, fixed-test floor, EDL/token **0.03891** (bridge) > **0.02900**
   (recover) > **0.01954** (control) — ~2× the control, above both curves at every
   step, with a HIGHER early spike (6.22 vs 5.23 bits). No evidence the bridge
   helped in ANY form; the intact-translation hypothesis is not supported. The
   comparison is doubly confounded (NL-format exposure biases EDL down, damaged
   op-add up); net is decisively up. See decisions.md 2026-07-27 for the confounds
   and the three-way logic; figures `edl_bridge_threeway_{test,val}_floor.png`.

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

   Elicit setup: `configs/archive/phase3/p3_elicit_{parent,inst,target}.yaml` +
   `configs/eval_p3_data.yaml` + `configs/archive/phase3/p3/target_after_inst.yaml`, and
   `scripts/archive/launch_phase3.sh --stage parent|target|all` (retired). Its format install is
   **conditional** on G4 < 0.90 and uses NL multiplication if it fires, because
   permuted-label NL addition would train wrong sums into a parent that already
   knows addition (the run-9 retention failure).

   **Teach arm DONE 2026-07-28.** `evt-p3-teach-inst` reached its behavioral
   stop at step 75 (9,600 examples), with G5 zero-shot 0.0000; the matched
   `evt-p3-teach-target` converged at step 16,000 with G5 zero-shot 0.9443 and
   reporting loss 0.02029 nats/token. At matched n=448K its fixed-test-floor
   EDL/example is 5.7767 bits versus the control's 0.1130: the teach comparator
   needs ~51x more excess description length. Both runs are relay-verified.

   **Practical embedding warm-start family DONE 2026-07-28.** This is a
   diagnostic/control family, not Arm A: each parent saw 512 correct NL-addition
   examples for 200 optimizer steps (102,400 presentations) before the target,
   so its EDL is residual after unbilled teaching exposure. All 12 candidates
   were persisted and relay-verified. Target-free held-out selection chose
   `sum` LR 0.1 (NL EM 0.5845, operator retention 0.9717), `colon` LR 1.0
   (0.5511, retention 0.5605), and `sum-colon` LR 1.0 (0.8156, retention
   0.7305). This confirms that the prompt-general colon row can be a strong but
   destructive mode switch; retention was report-only as pre-registered.

   Each selected parent then consumed the exact first 100K target rows once
   (782 updates, intentional `max_steps`); sibling G7 wiring and all 15 Hub
   checkpoint hashes were verified. At exact n=100K, raw MDL/example is
   **0.1138 / 0.1273 / 0.09344 bits** for sum / colon / sum-colon, versus
   **0.7247** for the control's interpolated first-100K prefix. Own-final-
   test-floor residual EDL/example is **0.05447 / 0.03830 / 0.03616 bits**
   (control **0.6406**); reporting losses are **0.00579 / 0.00867 / 0.00558
   nats/token** (control endpoint 0.00820). The raw, floor-free MDL result is
   decisive: all three warm-starts remove 82–87% of the first-100K codelength,
   and sum-colon is best. This does **not** compare steady-state sample efficiency:
   the 100K children intentionally stop before convergence while the control floor
   comes from its converged 448K endpoint. But colon alone is nearly as effective despite heavy
   operator damage, so this is evidence for broad prompt routing, not a
   sum-specific lexical lock and not clean elicitation. Moving-validation-floor
   curves tell the same level story but remain floor-dependent diagnostics.

10. **Fig-2 Llama dataset-size sweep — BOTH ARMS DONE 2026-07-31.** The
   paper's Figure-2 protocol on D_target, Llama-3.2-1B, 19 log-spaced
   prefix-nested sizes (1,000 → 1,000,000) × 2 conditions (base vs
   1-example format-installed parent), LoRA r64/α32 @ 3.53e-4 targets,
   each run fresh to val convergence. **All 38 converged** (per-size
   stopping schedule never bound). Result
   (`results/dataset_size_sweep.parquet`, 228 rows;
   `analysis/figures/dataset_size_sweep.png`, 2-curve EDL/D): the
   format-install pays off at SMALL n — EDL/label-token 0.14714 (inst)
   vs 0.23049 (noinst) nats at n=1000, −36% — the advantage decays
   through n≈4642, the curves interleave mid-range, and converge by
   n=1M (0.03050 vs 0.03277). G5 zero-shot EM never separates the arms
   (≈0.63–0.66 at n=1000; ≥0.99 from n≈100K in both) — endpoint
   accuracy is blind to the installed format; early-data codelength
   shows it. Both arms share a non-monotone EDL bump at n≈6813 (ee
   schedule steps 5→10 there); cross-arm deltas at matched n stay fair
   (G7-matched data order). The **installer took three designs**
   (decisions.md 2026-07-31 ×2): full-FT @ 3.53e-4 diverged (G4
   0.0000); full-FT @ 2.0e-5 absorbed the dose but destroyed retention
   (G2 0.0732 vs bar 0.29); the owner-picked **LoRA r64/α32 @ 3.0e-6**
   (run-9-v2 recipe) passed everything — absorption 0.00893 nats, G4
   0.9531, G2 0.3447 (above base's 0.3271, zero forgetting) — and its
   merged checkpoint parented the inst sweep. Relay: 39 records; full
   weights for both n=1M runs (sha-verified); installer = adapter
   sidecar only (stale full-FT relay record deliberately deleted
   pre-push). Total sweep cost ≈ $5.6 across two boxes.

11. **Fig-2 NL replication sweep (fig2nl) — PLANNED 2026-08-03.** A
    replication of the paper's Figure-2 dataset-size-sweep protocol
    (§6.10's design, same 19 log-spaced sizes and base
    `meta-llama/Llama-3.2-1B`), retargeted from the shipped op-notation
    sweep onto a **natural-language add/sub target task** on our frozen
    `D_algo` (2026-07-19). Two conditions, arm-serial ascending so G7
    data-order matching is pre-satisfied by construction (noinst runs
    fully before inst, never interleaved): noinst = base Llama → fresh
    LoRA r512/α32 → `D_algo[:n]`; inst = merged installer checkpoint →
    fresh LoRA r512/α32 → `D_algo[:n]`. Installer: LoRA r512/α32 @
    3.53e-4 on row 0 of `D_dose_mult` (operator-notation MULT — the same
    dose source as §6.8/§6.10, unchanged), gated on absorption ≤0.1 nats
    training loss, G4 format ≥0.90 scored on NL prompts, and **G2
    retention ≥0.31** (raised from §6.10's 0.29 — rationale in
    decisions.md 2026-08-03). Targets: LR 3.53e-4 unchanged, local batch
    128 (no gradient accumulation), bf16, seed 316 (single seed,
    deferred not dropped), ε/k 0.002/k5/min0, per-size `max_steps`
    ceilings **doubled** vs §6.10 (rationale: decisions.md 2026-08-03).
    New eval set `D_algo_eval.parquet`, 100K NL add/sub,
    question-disjoint from `D_target ∪ D_algo ∪ D_target_eval ∪ probe`.
    Run ids: `evt-llama-fig2nl-installer`,
    `evt-llama-fig2nl-{noinst,inst}-n<size>` over the 19 sizes (1,000 →
    1,000,000); 39 runs total. The shipped §6.10 op-notation sweep is
    untouched, immutable history — fig2nl runs alongside it, never in
    place of it.

    **Deliverable, narrowed twice by the owner on 2026-08-03.** The first
    pass cut analysis entirely; the second restored exactly one figure:

    - **ONE figure: EDL/D vs. n**, computed the same way as §6.10 —
      `edl_per_label_token_nats` off each run's
      `experiment.target_result`, converted to bits at the reporting
      boundary, log-x, one curve per condition.
      `analysis/dataset_size_sweep.py` gained `--family {op,nl}` for it;
      `--family nl` reads the `evt-llama-fig2nl-` prefix and writes
      `results/dataset_size_sweep_nl.parquet` +
      `analysis/figures/dataset_size_sweep_nl.png`. The distinct `_nl`
      stem is load-bearing: `write_results` is overwrite-by-name (OQ-6),
      so a shared stem would let this family silently replace §6.10's
      shipped table. Default stays `op`, so every existing path and
      output name is unchanged. Nothing else is analysed — no
      per-token/floor work, no cross-family comparison.
    - **Relay push is METADATA ONLY** — manifests, train logs, gate
      records, `eval/test_loss.json`; not one byte of `*.safetensors`,
      adapter sidecars included. `hf_checkpoint.py push` gained
      `--metadata-only` for this (`--no-weights` deliberately keeps the
      sidecar, which at r512 is ~0.72 GB × 39 ≈ 27 GB — *larger* than the
      full checkpoints it excludes, which is what made the distinction
      worth a flag). Consequence, stated because it is irreversible at
      teardown: **no run in this family is recoverable from the relay**;
      re-running the sweep is the only route back to any of these
      weights. The figure is unaffected — every field it reads is
      manifest-side, so it regenerates from a `pull --no-weights` on any
      machine. The launcher spares the two n=1,000,000 runs from its
      local prune so a late reversal can still push one by hand before
      the box dies.

    **Deviation register** (fig2nl vs the paper's Figure-2 protocol; all
    owner-accepted 2026-08-03 unless noted otherwise):
    1. Effective batch 128 vs the paper's 1024 (no gradient
       accumulation) ⇒ optimization behavior differs; 3.53e-4 is 8×
       hotter per example than the paper's rate at matched batch.
       Owner-accepted ("assume full utilization on a 4090").
    2. 1 seed (316) vs the paper's 3. Deferred, not dropped.
    3. Installer = LoRA r512 @ 3.53e-4, not the paper's full-FT @ 2e-5.
       r512 = **360,710,144 trainable params = 29% of the 1.24B model
       (37% of non-embedding)** and is **FULL RANK for k_proj/v_proj**
       (out_features 512) — nearer full-FT than "LoRA" suggests. Stated
       plainly here; it is the deviation most likely to be misread.
    4. Dataset = our `D_algo` (4×4-digit operand grid), not DeepMind
       Mathematics. Its signed subtraction convention (the NL prompt
       reads as |a−b| but the label is a−b) puts a hard ceiling on any
       model answering with absolute values, equal to the non-negative
       share of whichever set is scored. Quote the set, never a single
       number: **0.7383** is the G1/G2 1,024-question set (268 negative,
       decisions.md 2026-07-25) — that is the set the 0.31 G2 bar and
       the 0.3271 base-ref live on. `D_algo` itself is 250,110/1,000,000
       negative (ceiling **0.7499**) and the new `D_algo_eval` is
       25,004/100,000 (ceiling **0.7500**) — both MEASURED 2026-08-03,
       and the latter is what caps G5 exact-match for this family.
    5. `D_algo_eval` is **empty in exactly 6 of 16 operand cells** —
       `1x1, 1x2, 1x3, 2x1, 2x2, 3x1` — because the frozen 1M sets
       exhausted the small-operand question space. MEASURED, not
       predicted: the generation dry-run allocated the full 100,000
       evenly as 10,000 across the 10 non-empty cells, and adding
       `D_target_eval` to the exclusion killed no additional cell.
       Consequence: EDL floor / test loss / G5 are scored on
       large-operand questions only — shared by both arms, so cross-arm
       comparisons stay fair. Row order is shuffled, verified 2026-08-03:
       the ε/k stopping block (rows 0–2047) and G4's 512 prompts (rows
       2048–2559) each draw all 10 non-empty cells in near-equal
       proportion at ~50/50 ops, so no stage is scored on a single
       operand cell. `D_target_eval` behaves identically.
    6. No fp32 master weights anywhere in the Llama chain: checkpoints
       load and save bf16, so the V5.62 "fp32 measurement" note holds
       for the loss *compute* dtype, not the weights (decisions.md
       2026-07-31). This is **also true of the shipped §6.10 op
       sweep** — it is recorded, not fixed, because fixing it would
       fork comparability. `eps_nats 0.002` sits near bf16 resolution.
    7. Denser 19-point size grid vs the paper's ~3 points per decade.
    8. **Commuted-twin exposure.** `D_algo_eval` shares **zero** exact
       `(a, op, b)` triples with `D_algo` — the exclusion is
       exact-triple, and 0 collisions were verified over the full
       1M × 100K product — but **12,652 of its 100,000 rows (12.65%)**
       are answer-identical commuted twins (`b+a` for a training `a+b`).
       Quote both numbers or neither (the phase-3 norm, decisions.md
       2026-07-27). Subtraction contributes none: `b−a` has a different
       answer, so it is not an answer leak. This is **inherited from the
       capacity-capped water-fill, not new to fig2nl** — the shipped op
       sweep's `D_target`/`D_target_eval` pair measures 0% / 12.64% on
       the identical test (both MEASURED 2026-08-03), so it does not
       differentiate the two families.

## 7. Budget

~$2k total, tracked in the external sheet — this repo never spends it
silently (`--confirm-cost` everywhere). Spent to date: ≈ $2–3 (run-1
family + runs 2–4 incl. sweeps — the 38.7M scale keeps whole run
families under a dollar). The real spend is runs 5–6 extraction
storage (~1 TB HF PRO, re-estimated at pilot) and GPU time for
2 × 1,024-snapshot LoRA runs.
