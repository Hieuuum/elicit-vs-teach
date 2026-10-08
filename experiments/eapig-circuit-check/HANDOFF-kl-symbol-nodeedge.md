# Handoff: KL scoring (#1), symbol-form circuits (#2), edge-vs-node change (#4)

Branch `eapig-circuit-check`. Written 2026-10-08. **Not started.** The owner
will start this in a fresh session. Nothing is built and no box is rented.

## Why

The 10% parent → child analysis (PLAN.md, "2026-10 circuit change at 10%")
found that the parents' scores are mostly noise, so the parent → child
comparison mostly measures parent noise. Both parents score EM 0 on the word task, and the LD metric needs the
model to prefer the right answer.

Three ideas from Wang et al.'s fine-tuning circuit paper apply. The owner
picked #1, #2 and #4:

- **#1 KL scoring.** Score edges by how well the model reproduces its own
  clean output, right or wrong. This gives the parents a defined circuit for
  "what the parent computes on this prompt".
- **#2 Symbol form.** Measure `elicit_parent` on the task it can do,
  `7465 + 8497 = ` → `15962`. Then ask whether the child's circuit reuses it.
  Prior node-level evidence: decisions.md stage 14 (2026-09-17). There,
  `elicit_parent` performs on this surface (0-shot logit-diff 11.0) and
  `fmt_parent` does not (−1.46).
- **#4 Edge vs node change rates.** Measure these **with a null**. Edges
  outnumber nodes by hundreds to one, so edges changing more is nearly
  automatic. The paper had no control for this.

## Labels

- **performing**: the model's m(full) − m(empty) is clearly above 0, and f at
  10% is above the 95th percentile of the random band. This is the same
  criterion as the stage-14 node maps. Read each overlap against the
  split-half ceiling.
- **LD**: logit difference toward the correct answer (current metric).
- **KL**: −KL(p_clean ‖ p_x) over the full vocabulary at the two answer
  positions, teacher-forced. A positive score means the edge supports
  reproducing the model's own clean output.
- **score set**: one (task, metric) pair, e.g. LD-word (exists), KL-word,
  LD-symbol, KL-symbol.

## Decisions

| Item | Value | Status |
|---|---|---|
| GPU | **one box with 2× RTX 3090**, race 3 | **owner, 2026-10-08** |
| Score children on symbol too | yes (+~10 min GPU) | proposed; confirm |
| Random draws in the probe stage | 20 parents / 10 children | proposed; confirm |
| Full KL test rerun | all five tests, KL-judged, on the KL-word circuits, sizes **0.1 / 0.2 / 1 / 2 / 5%** | **owner, 2026-10-08** |
| KL equivalence bound | mean per-example KL(full ‖ circuit) < 0.10 × mean KL(full ‖ empty), one-sided t-test, α = 0.05 | proposed; confirm |
| KL parent gate | mean KL(full ‖ empty) > 0.1 nats, and sufficiency passes at some size | proposed; confirm |

## Steps

| Step | What | Where |
|---|---|---|
| 0 | Build the code (below), CPU smoke on a tiny random model, then commit and push | laptop |
| 1 | Symbol sanity per model: EM, m(full), m(empty) | GPU |
| 2 | Score the 3 new sets per model: LD-symbol, KL-word, KL-symbol (~45 s each) | GPU |
| 3 | Probe stage per new set: f at 2/5/10% against the random band. **No frozen tests.** | GPU |
| 4 | Run `circuit_change.py` on each set, plus the cross-surface pairs parent-symbol → child-word and parent-symbol → child-symbol | laptop |
| 5 | Edge vs node change rates at 0.1/0.2/0.5/1% and at the top 100/500/1,000 edges. The null is random edge sets of the same size; references are parent-vs-parent and split-half. Build and run it on the existing LD-word scores during step 0, then rerun on the new sets | laptop |

**Build list (step 0):**
- `data.py`: re-render the existing 512 discovery + 256 validation pairs
  (and their counterfactuals) in symbol form (`D_algo_eval_op` render:
  `"{a} + {b} = "`). Pairs stay identical across surfaces. They are already
  leakage-clean: `D_algo_op` was in the exclusion set.
  - **Check tokenization first.** The prompt ends in `"= "` (trailing
    space). The prompt must be a token-prefix of the full training text (see
    memory `feedback-eval-decode-must-match-training-tokenization`).
  - Check the two-token answer split and the "differs at both positions"
    counterfactual assert on the new surface.
- `geode/circuits/eapig.py` (tested core): a KL metric, with property tests:
  - KL = 0 at the clean input;
  - non-negative;
  - the metric is maximal at clean;
  - EAP-IG with KL on a tiny random model gives finite scores of the right
    shape.
- `run.py`: add `--task word|symbol`, `--metric ld|kl`, and a `probe`
  stage (f + random band only). Output dirs are keyed by score set, e.g.
  `results_kl/<tag>/`. **Never write to `results/` or `results_large/`.**
  Keep the frozen-test path unchanged.
- `circuit_change.py`: take score-set dirs as arguments, and allow
  cross-set pairs.
- `run_box.sh`: a two-GPU variant (next section). Keep the existing
  `--confirm-cost` / skip-if-exists behaviour.

## Full KL test rerun (step 6, owner 2026-10-08)

Rerun all five frozen tests with KL in place of LD, at **0.1%, 0.2%, 1%,
2% and 5%** (k = 196 / 392 / 1,959 / 3,917 / 9,793). It runs on the word
task only, because the copy task and the test set are defined there.
- Circuits are the top-k of the **KL-word** scores from step 2. This is
  level B in the chat: KL for scoring and for judging.
- Exploratory: write to `results_kltests/<tag>/`, never to `evaluate.json`
  in `results/` or `results_large/`. The frozen LD tests stay the record.
- Draw counts are the frozen ones (100 children / 50 parents; TinyStories
  10), so the results are comparable with the LD runs. The stopping rule
  is unchanged: once a size passes all five, larger sizes are skipped.

**KL definitions.** KL(full ‖ X) is per example, at the two answer
positions, over the full vocabulary. Here "full" means the unpatched clean
run, and X is the patched run.

| Test / quantity | LD version (frozen) | KL version (this step) |
|---|---|---|
| f | (m(C) − m(∅)) / (m(full) − m(∅)) | 1 − KL(full ‖ C) / KL(full ‖ ∅) |
| Sufficiency | f(C) beats ≥ 90% of random, binomial p < 0.05 | same, on KL f |
| Partial necessity | f(without C) below random | same, on KL f |
| Equivalence | TOST, LD(C) − LD(full) within ±10% m(full) | KL(full ‖ C) < 0.10 × KL(full ‖ ∅), one-sided (proposed) |
| Consistency | coverage ≥ 0.70, and ablating shared beats random | coverage from per-example KL scores; ablation on KL f |
| Specificity | relative addition drop ≥ 3× copy drop, above random p95 | relative damage KL(full ‖ without C) / KL(full ‖ ∅) per task, same 3× and p95 rules |
| Parent gate | m(full) > 0 and sufficiency somewhere | KL(full ‖ ∅) > 0.1 nats and sufficiency somewhere (proposed) |
| Parent in child, real patching of the top 20 | LD | KL |

**Build:** a metric switch in `run.py`'s evaluate path. Add `eval_kl`
alongside `eval_ld`; it caches the clean full-model log-probs at the two
answer positions once per model and returns per-example KL for each keep
mask. Add the KL equivalence test to `geode/circuits/edge_tests.py` with
property tests: KL(full ‖ full) = 0 passes; a circuit equal to ∅ fails.

**Time per size** (estimate). The basis is the measured ~3.9 s per circuit
evaluation at bs 32 on a 3090: 3,720 s / ~960 evals for `elicit_child` at
3 sizes, and 1,866 s / 464 evals for `elicit_parent`. A size costs 3N + 4
evaluations: sufficiency, partial necessity and the consistency ablation,
each 1 + N, plus the copy-task ablation. The per-size cost does not depend
on k. The KL read-out is a log-softmax at 2 positions, which is negligible.

| size | k | child (N = 100) | parent (N = 50) | all 4 models, 1 GPU | per GPU on 2× 3090 |
|---|---|---|---|---|---|
| 0.1% | 196 | ~20 min | ~10 min | ~60 min | ~30 min |
| 0.2% | 392 | ~20 min | ~10 min | ~60 min | ~30 min |
| 1% | 1,959 | ~20 min | ~10 min | ~60 min | ~30 min |
| 2% | 3,917 | ~20 min | ~10 min | ~60 min | ~30 min |
| 5% | 9,793 | ~20 min | ~10 min | ~60 min | ~30 min |
| once per model | – | ~3 min (full/copy refs, real patching of the top 20) | <1 min | ~7 min | ~4 min |
| **total** | | **~1 h 43 min** | **~51 min** | **~5 h 10 min** | **~2 h 35 min** |

If a parent fails the KL gate, its tests stop after sufficiency and
partial necessity, at about 7 min per size. If a size passes all five, the
remaining sizes are skipped.

## Two-GPU layout

Run one process per GPU, each loading one model at a time:
- `CUDA_VISIBLE_DEVICES=0`: `elicit_parent`, then `elicit_child`.
- `CUDA_VISIBLE_DEVICES=1`: `fmt_parent`, then `teach_child`.

Each process runs that model's sanity → 3 scores → probe stages → full KL
tests (step 6), then moves to the next model.

**Box minimums:** ≥ 4 CPU cores (run.py uses ~1 core per process), ≥ 32 GB
RAM (two 4.94 GB fp32 models loading at once), and ≥ 60 GB disk.

Each process writes `EXIT_<gpu>=` to the log. The job is done when both
markers are present; then push once.

Measured on a 1× 3090 at bs 32: peak 11.3 GB VRAM, ~4.4 s per circuit eval,
memory-bandwidth bound. Use bs 32.

## Cost (estimate, live search 2026-10-08)

- 2× 3090 offers: $0.27–0.30/h search price (US, rel ≥ 0.994). Storage adds
  ~$0.08/h.
- Steps 1–3: GPU work is ~55 min on one 3090, so ~28 min per GPU on two.
- Step 6 (full KL tests): ~2 h 35 min per GPU on two.
- Fixed overhead is ~28 min: race/onstart, pip, model downloads, push.
- Wall-clock ≈ **~3.5 h**. Cost ≈ **~$1.3–1.5**, including the 3-box race.
- Team credit was $76.40.

## Pre-registered readings

- **#2, elicit parent performing on symbol.** Child circuits (word or
  symbol) that overlap the parent near both ceilings mean edge-level reuse.
  Overlap at the parent-vs-parent level (~0.13 at 10%) means no evidence of
  reuse.
- **#2, `fmt_parent` not performing on symbol (expected).** Report "the
  teach parent has no circuit to keep". That is an absence, not a measured
  change.
- **#1, KL gives the parents a performing, repeatable word-task circuit (split-half ceiling well above today's ~0.2 at 10%).** Redo the
  parent → child analysis on KL-word.
- **#1, KL does not.** The parents have no structured word-task computation.
  #2 becomes the only parent-side measure.
- **#4.** Only an edge/node gap above the random-set null counts as
  "fine-tuning rewires edges more than nodes".

## Guardrails

- Exploratory: don't touch the frozen tests, the thresholds,
  `evaluate.json`, or the HF `results/eapig_check*/` folders.
- Push new results to HF `mhieuuu/geode-internals:results/eapig_kl_symbol/`.
  The full KL tests go under `results/eapig_kl_tests/`.
- Caveats to carry into the write-up:
  - one training seed per child;
  - the children's EM gap (0.953 vs 0.145);
  - `fmt_parent`'s word-task m(full) − m(empty) is 0.002.
- Box: follow the vast-box skill. Export the team key, race 3 with
  `--num-gpus 2` and `gpu_name=RTX_3090`, and kill any box that crawls.
  Watch downloads actively. Push, verify on the laptop, destroy.
- Finish:
  - write a PLAN.md Results block and a Log entry;
  - update memory `project-eapig-circuit-check-2026-10-01.md`;
  - send **one ntfy ping** when done.
