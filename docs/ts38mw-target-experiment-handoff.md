# ts38mw target-stage experiment — information packet for planning

**Scope note (read first):** this document is a fact-gathering handoff per
owner instruction 2026-08-15 ("Give me all of the information needed so
that the next model can do the planning... You are just collecting the
information"). It contains **no experiment design** — no dataset-size grid,
no run-id scheme, no launcher, no recommendation on open questions. §5 lists
every decision explicitly left open. The planning session should read this
document plus its three citations (`docs/bits-that-count.md`,
`docs/bits-that-count-experiments.md`, `docs/ts38-vs-bits-that-count.md`,
`docs/plan-ts38mw-multiwrap-install.md`) before designing anything.

## 1. The question

Owner request (2026-08-15, this session): using **LoRA only** (not
full-FT — see §4.4 for why), confirm whether the **EDL-per-label-token
(EDL/D) signature vs. dataset size** shows the paper's teaching→elicitation
shift (Fig. 2/Fig. 3, §4.1–4.2, §5, Table 5 of *Bits That Count*) for the
target phrasing **"What is the sum of a and b?"** ("sumof" — the paper's
literal word-only NL rendering, `geode.arith.formats._NL_PHRASE`), comparing:

- **arm base**: untouched TinyStories-38.7M base (`evt-run1-base-v3-ext`) →
  target-task LoRA fine-tune on the NL target, across a dataset-size grid.
- **arm pretaught-mw**: the ts38mw wrapper-diverse installed parent
  (`evt-ts38mw-parent-probe-lr3e-4`, verdict GO-B, §3.4 below) →
  **same** target-task LoRA fine-tune, same grid.

This is a minimal version of the paper's causal intervention (§5, Fig. 3):
does converting a capability from absent to (partially) latent shift the
EDL/D scaling signature from increasing-then-decreasing ("teaching") to
monotonically decreasing ("elicitation")?

## 2. Paper's exact protocol for this measurement (citations, not restated)

- **EDL/MDL formulas, floor definition**: `docs/bits-that-count.md` §2.2–2.4
  (eq. 1–3); `docs/bits-that-count-experiments.md` §1.1–1.5.
- **Fig. 2 protocol** (the exact figure this experiment approximates):
  `docs/bits-that-count-experiments.md` §5.1. "All experiments shown use
  LoRA rank 512" (paper). This repo's whole ts38 family uses **r128/α32**
  throughout (already-accepted deviation, not new — see
  `docs/ts38-vs-bits-that-count.md` §3.2 table).
- **Formal elicitation/teaching classification rule** (App. J.1):
  elicitation predominates where `∂/∂n[EDL(n)/n] < 0`; teaching where
  `∂/∂n[EDL(n)/n] > 0`. Table 5's legend: "↓ = monotonically decreasing
  (elicitation-dominated). ↑↓ = non-monotonic with initial increase
  (teaching-dominated, then elicitation)."
- **Table 5's TinyStories–1B add/sub rows** (the literal target this
  experiment approximates): base ↑↓ peak ≈300K; pre-teach format ↑↓ peak
  ≈150K; **pre-teach add/sub ↓** ("converts to elicitation").
- **Table 6's actual numbers for this row** (§5.4 of the experiments doc):
  base TS-1B add/sub EDL/P* = 2.21 (EDL-col) / 2.23 (PGR-col) bits/param →
  pre-teach add/sub (ID) 1.81 / 1.50 bits/param. **This stays above 1
  bit/parameter — it does NOT cross into the elicitation regime (~0.01–0.1
  bits/param).** The paper's dramatic, clean 20× conversion was
  **multiplication** (0.70→0.06 / 1.02→0.05), not add/sub. Add/sub was
  already the paper's *weaker* demonstration case. Any planning should
  calibrate expectations to "modest signature shift, if any" for this
  specific row, not a dramatic one.
- **Hyperparameters** (Table 1, Table 3): AdamW, β1 0.9, β2 0.999, weight
  decay 0.01, constant LR, grad clip 1.0, bf16, LoRA LR 3.53e-4, batch 128,
  eff. batch 1024 (8×H100), stopping = validation-loss convergence.

## 3. What already exists in this repo — reusable, zero new cost

### 3.1 Base arm — already fully trained and measured, all 5 sizes

Run ids on the relay (`mhieuuu/geode-store`, confirmed present via
`HfApi().list_repo_files`): `evt-ts38-base-n1000`, `-n4642`, `-n21544`,
`-n100000`, `-n316228`. Trained on `D_algo_bare.parquet` (bare NL, no
`Question:/Answer:` scaffold), config `configs/ts38_base.yaml` + per-size
overlays `configs/sweeps/ts38/ts38_base_n<size>.yaml`.

Its EDL/D curve (three floors) is **already computed and written up** in
`docs/ts38-vs-bits-that-count.md` §2 — reproduced here verbatim:

| n | MDL/D (nats) | L_conv | L_min | L_test | EDL/D OCV | EDL/D min-val | EDL/D test | steps (epochs) | zero-shot EM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 4.647 | 1.539 | 1.465 | 1.533 | **4.484** | 4.590 | 4.492 | 135 (17.3) | 0.3% |
| 4,642 | 2.638 | 1.299 | 1.239 | 1.308 | **1.932** | 2.019 | 1.919 | 270 (7.4) | 1.9% |
| 21,544 | 1.734 | 0.196 | 0.170 | 0.191 | **2.219** | 2.256 | 2.226 | 1,825 (10.8) | 74.6% |
| 100,000 | 1.262 | 0.065 | 0.052 | 0.065 | **1.728** | 1.746 | 1.727 | 5,000 (6.4) | 92.5% |
| 316,228 | 0.620 | 0.037 | 0.023 | 0.038 | **0.841** | 0.861 | 0.841 | 10,875 (4.4) | 94.5% |

Reading (already established): base shows a **rising span** 4,642→21,544
(+15% under all three floors) — the paper's teaching signature fires
formally, though compressed ~15× earlier in n than the paper's ~300K peak
(reasons: harder/8× fewer updates/higher LR — `docs/ts38-vs-bits-that-count.md`
§3.2 table; **not chaseable, not a bug**).

**This means: if the new "pretaught-mw" arm is trained on the identical
target dataset, config, and dataset-size grid, only ONE new arm needs
training — the base curve does not need to be reproduced.**

### 3.2 Target dataset — already built, pinned, order-hash-verified

- `D_algo_bare.parquet` (bare, no scaffold): `order_hash
  946b5d02a8f9260fec00ce68a4db42a12f16966f6b49f685269382ae7b4b6ace`,
  `local_path experiments/training-run/data/full/D_algo_bare.parquet`,
  1,000,000 rows, `hf_id mhieuuu/elicit-vs-teach-arith`. Phrasing =
  `geode.arith.formats._NL_PHRASE` ("What is the sum of {a} and {b}?" /
  "What is the difference between {a} and {b}?"), rendered bare:
  `<question>\n<answer>`.
- `D_algo_eval_bare.parquet` (eval counterpart): `order_hash
  e419baa213bbe07dfeb50f46fe17b464056cd18c2a7302238a66682d7c594631`, rows
  0–2047 = stopping block, 2048+ = reporting block.
- `D_algo.parquet` / `D_algo_eval.parquet` (**scaffolded** variant,
  `Question:.../Answer:...` wrapper): also already exist as base artifacts
  (produced by the same `make_data.py` base run that produces
  `D_target`/`D_algo_bare`'s source) but **have never been used as a
  target-training set for any family** — only the bare variant has family
  precedent (§3.1 above used bare exclusively).
- Both regenerate deterministically from seed 20260717 via
  `datagen/make_data.py --scale full --seed 20260717` (+`--nl-eval-set` for
  the eval file) then `datagen/make_bare_sets.py` for the bare derivations
  — same mechanism the ts38mw launcher fix (commit `e711c93`) now runs
  automatically on a fresh box.

### 3.3 Config templates — the existing matched-pair pattern

`configs/ts38_base.yaml` (arm from base) and `configs/ts38_pretaught.yaml`
(arm from `evt-ts38-pretaught-parent`) are **byte-identical except
`parent_run_id`, `parent_required_gates`, and `experiment.arm`** — this is
enforced by `tests/.../test_config_completeness.py`'s ts38 section (an
allowed-diff set). Shared fields, verbatim:

```yaml
model: {hidden_size: 512, num_hidden_layers: 8}
tokenizer: {path: ../tokenizer}
task: {name: arith_bare_addsub, format_version: v1}
data:
  hf_id: mhieuuu/elicit-vs-teach-arith
  file: D_algo_bare.parquet
  order_hash: 946b5d02a8f9260fec00ce68a4db42a12f16966f6b49f685269382ae7b4b6ace
  local_path: experiments/training-run/data/full/D_algo_bare.parquet
  eval_file: D_algo_eval_bare.parquet
  eval_order_hash: e419baa213bbe07dfeb50f46fe17b464056cd18c2a7302238a66682d7c594631
  eval_local_path: experiments/training-run/data/full/D_algo_eval_bare.parquet
  seed: 316
lora: {r: 128, alpha: 32}
train:
  lr: 1.0e-3            # target-stage pin, 2026-07-22 runs-5/6 sweep — SAME both arms
  precision: bf16
  batch_size: 128
  stopping: {eps_nats: 0.002, k: 5, min_steps: <ceil(n/128), per-overlay>}
  snapshots: {n: 0, dense_until: 30}   # snapshots OFF in this family
  seed: 316
```

Per-size overlays live at `configs/sweeps/ts38/ts38_{base,pretaught}_n<size>.yaml`
for n ∈ {1000, 4642, 21544, 100000, 316228} (5 log-spaced points) — these
are the exact 5 sizes §3.1's table reports. `experiment.match_data_order_with`
pairs each `ts38_pretaught_n<size>` overlay to the same-size `ts38_base_n<size>`
run so both arms see identical data order (G7 anchor — `ts38_base` IS the
anchor, `match_data_order_with: null` on the base side).

`experiment.require_full_epoch1: true` forces `train.stopping.min_steps =
ceil(n/128)` per overlay — MDL needs the full first epoch (guard 1,
specs/02 V5.75/V5.76).

### 3.4 The GO-B parent (today's run) — full state

Run id `evt-ts38mw-parent-probe-lr3e-4`. Manifest (full dump, pulled live
from the box):

```json
{
  "run_id": "evt-ts38mw-parent-probe-lr3e-4",
  "git_commit": "e711c936ab57ec306995a4dbb6c1c0b5e23a8ba5",
  "base_model": {"hf_id": "zoo-run/evt-run1-base-v3-ext"},
  "task": {"name": "arith_op_mw_addsub"},
  "dataset": {"name": "mhieuuu/elicit-vs-teach-arith:D_target_mw.parquet",
              "n_unique_examples": 1000000, "seed": 316},
  "training": {
    "method": "lora",
    "lora": {"rank": 128, "alpha": 32,
             "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]},
    "optimizer": {"name": "adamw", "lr": 0.0003, "batch_size": 128},
    "precision": "bf16", "max_steps": 160000,
    "stopping": {"eps_nats": 0.002, "k": 5, "min_steps": 5000}
  },
  "trainable_param_count": 12058624,
  "snapshot_steps": [8000,12000,16000,20000,24000,28000,32000,36000,40000,48000,56000,64000,80000,96000],
  "status": "complete",
  "experiment": {
    "arm": "A", "role": "pre_teach",
    "parent_run_id": "evt-run1-base-v3-ext",
    "parent_required_gates": [],
    "gates": {},
    "init_from": ".../runs/evt-run1-base-v3-ext/model",
    "data_order_hash": "bf0b28bde9636d0ef4a7ccfc753de5aec3067109903d65e7a8c3f2677144e5d7",
    "step0": {"val_loss_nats": 5.285502295988615},
    "sft_result": {"final_step": 28000, "best_val_nats": 0.021177572263184622,
                    "min_val_nats": 0.02113788780244003, "stop_reason": "converged"}
  }
}
```

**Note: `gates: {}` — every gate call this session used `--no-record`
(deliberate, per the pre-registered probe design). This run has NEVER had
a gate recorded, pass or fail.** `trainable_param_count: 12058624` (12.06M
— relevant for any future EDL/P capacity calculation).

Snapshots on disk (box `38.246.237.140:32489`, `/workspace/elicit-vs-teach/
geode-store/runs/evt-ts38mw-parent-probe-lr3e-4/sft_snapshots/`):
`step_0008000` through `step_0028000` (6 total, ~240MB each). `step_0028000`
is also copied to `.../model` (the run's default/final checkpoint).

**Per-snapshot probe results** (from `analysis/ts38mw_probe.json`,
`gates.py g5 --no-record` on 6 held-out pins, 1024 questions each; `canonical_em`
= `bare_op` zero-shot EM = the run's own op-format capability check):

| step | canonical_em | g1_own | g8_val_loss | sym_q EM/loss | **sumof EM/loss** | word_q EM/loss |
|---:|---:|---:|---:|---:|---:|---:|
| 8000 | 0.8164 | 0.8096 | — | 0.607/0.447 | **0.032/5.168** | 0.016/6.810 |
| 12000 | 0.9053 | 0.8955 | — | 0.844/0.113 | **0.047/5.623** | 0.029/7.535 |
| 16000 | 0.9355 | 0.9258 | — | 0.831/0.202 | **0.064/5.700** | 0.030/7.899 |
| 20000 | 0.9648 | 0.9531 | 1.2406 FAIL | 0.940/0.041 | **0.075/5.514** | 0.032/7.745 |
| 24000 | 0.9414 | 0.9307 | — | 0.898/0.114 | **0.164/4.425** | 0.058/7.408 |
| 28000 | 0.9805 | 0.9619 | 1.2694 FAIL | 0.955/0.031 | **0.171/4.517** | 0.041/6.924 |
| **base** | — | — | 1.0718 | 0.000/5.166 | **0.000/5.194** | 0.000/5.126 |

**The `sumof` column is the exact target phrasing this experiment is about.**
Loss goes 5.19→4.52 nats (~13% reduction) at best; EM peaks at 0.171. This
is a real but modest gap, nowhere near `sym_q`'s collapse (5.17→0.03 nats,
~170×). Verdict recorded: **GO-B** (`sym_q` persists ≥0.90 EM with loss <
base at two adjacent qualifying snapshots 20000/28000; `sumof`/GO-A never
persists ≥0.20 EM). Full detail: `experiments/training-run/notes/decisions.md`
"2026-08-15 ts38mw Stage 1 outcome" entry; `EXPERIMENTS.md` §6.15.

**G8 (TinyStories retention) FAILs at both scored points** (1.2406, 1.2694
vs. bar ≤1.1718; base 1.0718) — worse retention than the single-wrapper
certified `evt-ts38-pretaught-parent` (G8=1.163 at S=15000), at the same LR.

**Durability status: CONFIRMED on the relay (receiver-side verified, not
just sender-side).** The launcher's own push was `--metadata-only`
(manifest/eval_log/train_log/training_meta only, excluding all
`.safetensors`); this session additionally ran `hf_checkpoint.py push
--with-snapshots` and verified via `HfApi().list_repo_files('mhieuuu/geode-store',
repo_type='model')` that both `adapter.safetensors` and `model.safetensors`
exist for `.../model/` and all 6 `.../sft_snapshots/step_00{08,12,16,20,24,28}000/`
directories (32 files total under the run's prefix). The box
(`38.246.237.140:32489`, owner's rental, never destroy) still has the
original copy, but is no longer the only copy.

### 3.5 Analysis tooling — what exists, what it does NOT already handle

- `experiments/training-run/analysis/edl_converged_val_floor.py --family ts38`
  — computes EDL/D under three floors (OCV/min-val/test) for a family.
  **Its `FAMILIES` dict hardcodes the ts38 regex to
  `^evt-ts38-(base|pretaught)-n(\d+)$`** (line ~107). A new arm's run ids
  will **not** be picked up unless they match this exact pattern or the
  regex/FAMILIES dict is extended — `pretaught` is already taken by the
  existing single-wrapper parent's family, so a new arm cannot reuse that
  literal token without ambiguity.
- `experiments/training-run/analysis/dataset_size_sweep.py` — companion
  tool referenced alongside the floor script in `docs/ts38-vs-bits-that-count.md`'s
  header (exact CLI shown there).
- `experiments/training-run/analysis/plot_edl_per_token.py` — plotting.
- All three are laptop-side, CPU-only, reproducible from the manifests/logs
  already on the relay — no GPU needed to re-derive the base arm's numbers.

## 4. Mechanics the planner must account for (facts, not decisions)

### 4.1 `require_parent_ready` — the parent-gating check in `train_target.py`

`geode/zoo/checks.py::require_parent_ready(parent_run_id, required_gates, store)`:
raises `ConsistencyError` unless (a) the parent manifest exists and loads,
(b) `manifest.status == "complete"`, (c) every gate **already recorded** in
`experiment.gates` has `pass: true`, and (d) every gate named in
`required_gates` has been recorded as passing.

The GO-B parent's manifest has `status: "complete"` (✓ condition b) and
`gates: {}` (✓ condition c, vacuously — nothing recorded to fail). But
`ts38_pretaught.yaml`'s existing pattern sets `parent_required_gates: [G1, G8]`
— reusing that verbatim against the GO-B parent would fail condition (d)
immediately, since **neither gate has ever been recorded** on this run.
`ts38_base.yaml`'s pattern (`parent_required_gates: []`, used because its
parent is a plain base model with no gates) is the closer precedent for a
parent with no recorded gates.

### 4.2 Never record G8 on the GO-B parent

Per `feedback-gate-thresholds-are-task-scoped` (project memory): **a
recorded FAIL blocks every child of that parent, permanently** ("V0.6
death"). The GO-B parent is already known (measured `--no-record`) to FAIL
G8 (1.2694 vs 1.1718). If a future launcher or config ever runs `gates.py
g8` on this parent **without** `--no-record`, that FAIL becomes permanent
and un-recoverable for any downstream arm. This is a hard constraint on
however the new arm is wired up, not a design preference.

### 4.3 Merging the LoRA snapshot into a standalone parent checkpoint

`scripts/merge_adapter.py` / `geode.train.merge_lora()` (`geode/train/lora.py:160`)
is the existing mechanism — used previously to produce the standalone,
loadable `evt-ts38-pretaught-parent` checkpoint from its own certified LoRA
snapshot. Its docstring: "For cross-stage parent handoff only... never for
snapshots" — read in context, this means don't merge every probe-scoring
snapshot repeatedly; merging the ONE selected snapshot for a genuine
parent-handoff use (which is what any new arm needs) is exactly the
intended case. `geode.zoo.load_model` is the only supported loader for a
LoRA-wrapped checkpoint — `feedback-lora-checkpoints-load-via-zoo-load-model`
(project memory): never add a new bare `from_pretrained(checkpoint)` call
site for a LoRA run.

### 4.4 Why LoRA, not full-FT (owner decision this session, already made)

The paper's App. E.2 pre-teach protocol is full fine-tuning, "until strong
performance," no retention check. This repo's full-FT ladder for the
**single-wrapper** parent failed G8 at **every** LR tested (design result,
`EXPERIMENTS.md` §6.14, decisions.md "ladder CLOSED" 2026-08-15) — the
strongest rung reached G1=0.9883 (our best-ever single-wrapper op-notation
capability) before being rejected for G8=9.9579. **That checkpoint's
weights were pushed metadata-only and are gone from the relay** — its NL
transfer was never measured and cannot be measured now without retraining
(`runs-failed/evt-ts38-pretaught-parent-lr3.0e-4` has only
`eval_log.jsonl`/`manifest.json`/`train_log.jsonl`/`training_meta.json` on
the relay, confirmed via `HfApi().list_repo_files`, no `model/` directory).
The LoRA equivalent (G1≈0.978, weights still on the relay,
`runs-failed/evt-ts38-pretaught-parent-lora-lr3e-4/model/adapter.safetensors`)
falls within the already-measured 10k–24k snapshot sweep, which found
NL-phrasing transfer flat/near-zero across that entire range regardless of
training amount. The paper itself states LoRA and full-FT give "similar
results" for the analogous Llama pre-elicitation step (App. E.1.1) — not
proof for the TinyStories algorithm-teaching step specifically, but the
closest textual evidence available. Net: no measured or documented reason
to expect full-FT would behave differently from LoRA here, and LoRA is
what today's positive (GO-B) result was actually built on.

## 5. Open decisions — explicitly left to the planner

1. **Which snapshot** of the GO-B parent to use as theta0 (28000 = final,
   strongest `sym_q`/`sumof` signal, also worst G8; 20000 = first
   qualifying snapshot, slightly less trained).
2. **Bare vs. scaffolded target rendering.** Bare (`D_algo_bare`) has full
   family precedent and lets the new arm's curve sit directly alongside
   §3.1's already-measured base curve. Scaffolded (`D_algo`, no family
   precedent yet) showed better transfer in the probe (`sumof` 4.517 nats
   vs `sumof_bare` 5.040 at step 28000) — but using it would mean the base
   arm's curve is NOT directly reusable (§3.1's base runs are on
   bare-only) and would need its own new base-arm training on scaffolded
   data for a fair comparison.
3. **Dataset-size grid** — reuse the existing 5 log-spaced points (1000,
   4642, 21544, 100000, 316228) for direct overlay onto §3.1's curve, or a
   different grid.
4. **run_id naming scheme** for the new arm — must not collide with, and
   will not be auto-discovered by, the existing `FAMILIES["ts38"]` regex
   (§3.5). Extending that regex/dict (or adding a parallel entry) is
   required either way for the analysis tooling to find the new runs.
5. **`parent_required_gates` value** for the new arm's config — §4.1 shows
   `[]` is the only value that won't hard-fail against this parent's
   manifest; whether that's an acceptable protocol deviation (vs. e.g.
   finding some other way to certify readiness) is a call for the planner
   to make explicit, not silently inherit from the `ts38_pretaught.yaml`
   template.
6. **LR/rank for the target stage** — reusing `ts38_base.yaml`/
   `ts38_pretaught.yaml`'s exact recipe (LoRA r128/α32, LR 1e-3) keeps
   "algorithm A" fixed between arms (EDL's own requirement, and the
   existing family's explicit rationale for using an identical recipe
   across arms) and is directly comparable to §3.1's numbers; a different
   recipe would need its own justification and would NOT be directly
   comparable to the existing base curve.
7. **Whether to compute all three floors** (OCV/min-val/test, as §3.1 did)
   or just one, for the new arm.

## 6. Sources

- `docs/bits-that-count.md` — full paper text (tidied transcription).
- `docs/bits-that-count-experiments.md` — per-experiment protocol reference
  (§5.1 Fig. 2, §5.3–5.4 Fig. 3/Table 6 causal intervention).
- `docs/ts38-vs-bits-that-count.md` — prior comparison of the single-wrapper
  ts38 family against Table 5 (why it failed to separate; the base arm's
  full measured curve is in its §2).
- `docs/plan-ts38mw-multiwrap-install.md` — the ts38mw Stage 0/1 plan (GO-B
  verdict bands, wrapper templates, probe pins).
- `experiments/training-run/notes/decisions.md` — "2026-08-15 ts38mw Stage
  1 outcome — verdict GO-B" entry (full table + reasoning).
- `EXPERIMENTS.md` §6.14 (ts38 family, full-FT/LoRA ladder closures) and
  §6.15 (ts38mw).
- `analysis/ts38mw_probe.json` — raw per-snapshot/per-pin numbers.
