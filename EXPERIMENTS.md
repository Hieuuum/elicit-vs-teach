# EXPERIMENTS.md — live experiment plan

Status: **executing** (updated 2026-07-20). This is the current state
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
| 1 | `evt-run1-base-v3-ext` | pretrain (floor 1) | v3 @ 30k | full FT | TinyStories-v2 | **NEXT — extension queued** (v3 hit its 30k ceiling still descending; same recipe to convergence, 2026-07-21) |
| 2 | `evt-run2-armA-algo` | pre-teach | run 1 | full FT | `D_algo` (NL add/sub, correct labels) | blocked on run 1, then LR sweep → canonical; `docs/run2-guide.md` |
| 3 | `evt-run3-armA-inst` | format install | run 2 | full FT | `D_inst` (op-notation mult, random labels), count OPEN(1) | todo |
| 4 | `evt-run4-armB-inst` | format install | run 1 | full FT | identical `D_inst` slice + count as run 3 | todo |
| 5 | `evt-run5-armA-target` | target | run 3 | LoRA | `D_target` (op-notation add/sub), count OPEN(2) | todo |
| 6 | `evt-run6-armB-target` | target | run 4 | LoRA | identical data, identical order as run 5 | todo |

DAG: `1 → 2 → 3 → 5` (Arm A, elicit) and `1 → 4 → 6` (Arm B, teach).
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
samples clean, no repetition loops) → **`evt-run1-base-v3-ext` (queued
2026-07-21): warm start from the v3 checkpoint, identical optimizer
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
| G1 | run 2 | Arm A ≥95% on NL add/sub — 1,024 seeded val examples, greedy, `exact_match` (`gates.py g1`) | pending run 2 |
| G2 | run 3 | Arm A still near ceiling (installer didn't corrupt; δ at pilot) | todo |
| G3 | run 4 | Arm B ≈ 0% on real add/sub (random labels didn't leak) | todo |
| G4 | runs 3–4 | op-notation format validity ~≥99%, both arms | todo |
| G5 | runs 3–4 | zero/16-shot op add/sub (expect A ~2%/12%, B 0%/0%) | todo |
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

Built so far: `geode.edl`, `geode.train` (full FT + SFT), `geode.zoo`,
`geode.arith` — all with property tests (suite ≈ 310). Not yet built:
`geode.probe` (schedule / extraction / metrics, V5.8–V5.16),
`scripts/extract.py`, `analysis/` drivers, `export_hf.py`,
`gates.py` g2–g7 subcommands.

## 6. Remaining work, in order

1. **Run 2** — 4-LR full-length sweep, owner pins `train.lr`, canonical
   relaunch, G1 (`docs/run2-guide.md`; ≤ ~$0.35). Hard stop if no sweep
   arm reaches 0.95 accuracy.
2. **Runs 3–4** — pilot pins the installer count (OPEN(1)); configs +
   `gates.py` g2–g5; both runs are `train_sft.py` launches with the
   op-notation task format.
3. **`geode.probe`** + `scripts/extract.py` (V5.8–V5.12): snapshot
   schedule, offline probe pass (activations + activation-gradients at
   9 residual points), matched-load guards.
4. **Pilot** (spec 02 §11) at toy scale through train → extract →
   upload → one real gradient-alignment plot; closes the remaining
   OPEN items (target count, stopping ε/k, snapshot schedule).
5. **Runs 5–6** — LoRA prequential with 1,024 full-model snapshots;
   G7 order-hash guard at run-6 launch.
6. **Analyses + publication** — metrics V5.13–V5.16, the four drivers
   (alignment, drift, adapters, matching), `export_hf.py` to the HF
   dataset repo (spec 02 §9–10).

## 7. Budget

~$2k total, tracked in the external sheet — this repo never spends it
silently (`--confirm-cost` everywhere). Spent to date: ≈ $2 (run-1
family + sweeps). Runs 2–4 are pennies at the 38.7M scale; the real
spend is runs 5–6 extraction storage (~1 TB HF PRO, re-estimated at
pilot) and GPU time for 2 × 1,024-snapshot LoRA runs.
