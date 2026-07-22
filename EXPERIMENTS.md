# EXPERIMENTS.md — live experiment plan

Status: **executing** (updated 2026-07-22). This is the current state
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
| 5 | `evt-run5-armA-target` | target | run 3 | LoRA | `D_target` 500K prefix (op-notation add/sub; OPEN(2) closed 2026-07-22) | ready to launch — lr 1e-3 + n 500K pinned; awaits OPEN(4)/OPEN(10) |
| 6 | `evt-run6-armB-target` | target | run 4 | LoRA | identical data, identical order as run 5 | same — G7 order guard at launch |

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
| G5 | runs 3–6 | zero/16-shot op add/sub + shared-set test loss, fixed slices of `D_target_eval` (final protocol 2026-07-22) | recorded (evidence-only): parents A 0.0117 / 2.30 nats, B 0.0000 / 3.75 (latent as required); pilots B@500K 0.9805 / 0.0140 vs A-ref@50K 0.9941 / 0.0059 → OPEN(2) = 500K. 16-shot ≈ 0 everywhere (collapse — invalidated as a metric; decisions.md) |
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
`geode.arith`, `geode.probe.schedule` — all with property tests (suite
≈ 491); launchers `train_sft.py` + `train_target.py`, `gates.py` g1–g5.
Not yet built: `geode.probe` extraction + metrics (V5.9–V5.16, extraction
+ V5.13 in progress 2026-07-22), `scripts/extract.py`, `analysis/`
drivers, `export_hf.py`, `gates.py` g6.

## 6. Remaining work, in order

1. ~~Run 2~~ / ~~Runs 3–4~~ — **DONE 2026-07-22** (§2 table; outcomes
   + gate values in decisions.md 2026-07-22).
2. ~~OPEN(2) pilot~~ — **DONE 2026-07-22**: phase 1 (target-LR
   mini-sweep) pinned lr 1e-3; phase 2 (the B grid @ 10K/50K/200K/500K
   + A-ref @ 50K, re-scored under the fixed shared eval protocol)
   closed **OPEN(2) = 500K** (§4 G5 row; decisions.md). Next: OPEN(4);
   OPEN(10) is an owner call before run 5.
3. **`geode.probe` extraction + `scripts/extract.py`** (V5.9–V5.13:
   offline probe pass — activations + activation-gradients at 9
   residual points, matched-load guards, alignment metric) — in
   progress 2026-07-22; validate on the pilot's snapshots before that
   box is destroyed (train → extract → one real gradient-alignment
   plot, spec 02 §11).
4. **Runs 5–6** — LoRA prequential with 1,024 full-model snapshots;
   G7 order-hash guard at run-6 launch. Box needs ≥400 GB disk
   (fp32 snapshots, 148 MB × 1,024 × 2 arms ≈ 300 GB) unless the
   owner opts for bf16 snapshots.
5. **Analyses + publication** — metrics V5.14–V5.16, the four drivers
   (alignment, drift, adapters, matching), `export_hf.py` to the HF
   dataset repo (spec 02 §9–10).

## 7. Budget

~$2k total, tracked in the external sheet — this repo never spends it
silently (`--confirm-cost` everywhere). Spent to date: ≈ $2–3 (run-1
family + runs 2–4 incl. sweeps — the 38.7M scale keeps whole run
families under a dollar). The real spend is runs 5–6 extraction
storage (~1 TB HF PRO, re-estimated at pilot) and GPU time for
2 × 1,024-snapshot LoRA runs.
