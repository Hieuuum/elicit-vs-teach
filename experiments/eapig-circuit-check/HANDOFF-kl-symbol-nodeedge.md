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
- **#2 Symbol form: ELICIT ROUTE ONLY** (owner, 2026-10-08). The elicit
  route was trained on symbol arithmetic (`7465 + 8497 = ` → `15962`). The
  teach route never was, so it gets no symbol form anywhere in this plan.
  - The teach route stays on NL (word-form) add/sub. NL add/sub is the
    default for `fmt_parent`.
  - **`elicit_parent`'s plan is PENDING the owner** ("I'll decide the plan
    for elicit parent later"). Do not run any `elicit_parent` symbol job
    until the owner says what to do.
  - Prior node-level evidence: decisions.md stage 14 (2026-09-17).
    `elicit_parent` performs on the symbol form there (0-shot logit-diff
    11.0).
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
| Surface per route | elicit route: symbol form (+ NL for reference). Teach route: **NL add/sub only, no symbol form** | **owner, 2026-10-08** |
| `elicit_parent` | plan for it (symbol and/or NL, which steps) **PENDING the owner**. Run no `elicit_parent` symbol job until decided | **owner, 2026-10-08: pending** |
| `fmt_parent` | NL add/sub (default) | **owner, 2026-10-08** |
| Score `elicit_child` on symbol | **decided after step S** | **owner, 2026-10-08: pending step S** |
| Score `teach_child` on symbol | **no** (teach route has no symbol form) | **owner, 2026-10-08** |
| Random draws in the probe stage | 20 parents / 10 children | proposed; confirm |
| Full KL test rerun | all five tests, KL-judged, on the KL-word circuits. **First pass: 2% and 5% only. Then STOP and report to the owner.** 0.1 / 0.2 / 1% run only after the owner says go | **owner, 2026-10-08** |
| KL equivalence bound | mean per-example KL(full ‖ circuit) < 0.10 × mean KL(full ‖ empty), one-sided t-test, α = 0.05 | **owner, 2026-10-08** |
| KL parent gate | mean KL(full ‖ empty) > 0.1 nats, and sufficiency passes at some size | **owner, 2026-10-08** |

## Step S (FIRST): performance check, by route

The owner wants to see `elicit_child`'s symbol-form performance before
deciding whether to score it on that form (owner, 2026-10-08). Each route
is measured only on its own surface.

| model | what step S measures |
|---|---|
| `elicit_child` | symbol add (50,000) + NL add/sub on the same problems (100,000) |
| `fmt_parent` | NL add/sub (100,000) |
| `teach_child` | NL add/sub (100,000) |
| `elicit_parent` | **not run until the owner decides** (pending) |

- **Data (considerable).**
  - Symbol: all 50,000 addition rows of `D_algo_eval_op` (op `+`, 10 cells,
    5,000 each; prompt `"{a} + {b} = "`).
  - NL add/sub: all 100,000 rows of `D_algo_eval_bare`. That is 50,000 `+`
    (`"What is the sum of {a} and {b}?\n"`) and 50,000 `−`
    (`"What is the difference between {a} and {b}?\n"`).
  - All are re-renders of the frozen `D_algo_eval` triples. The `+` rows are
    the same 50,000 problems in both forms (verified 2026-10-08: same `idx`,
    `a`, `b`, cell).
  - Headline: the 4x4 addition cell, which is the circuit task. Also report
    4x4 on the leakage-clean subset, using the same exclusion set as
    `data.py`'s pair pool.
- **NL subtraction caveat.** The label is signed `a − b`, and 50% of rows
  have `a < b` (memory `project-nl-difference-sign-ambiguity`). Report
  strict EM **and** |a − b|-lenient EM.
- **Measures, per model × surface × op × cell:**
  - greedy exact match (max 6 new tokens; cut at EOS before decoding, as in
    `run.py` sanity);
  - for 4x4 addition, teacher-forced log-prob of the answer.
- **`elicit_child` only:** m(full) and m(empty) (LD) on the 768 circuit
  pairs (512 discovery + 256 validation) re-rendered in symbol form. This
  tells whether its LD-symbol circuit would have signal.
- **Tokenization check first, on the laptop (tokenizer only, no model).**
  - Is `"{a} + {b} = "` a token-prefix of the full training text
    `"{a} + {b} = {answer}"`?
  - How does the trailing space tokenize?
  - If the prompt is not a clean prefix, use the training tokenization (see
    memory `feedback-eval-decode-must-match-training-tokenization`).
- **Script:** `perf_eval.py --surface symbol|nl --ops + -`, one model per
  process, `--device`.
  - Output `results_perf/<tag>/perf_eval.json`, pushed to HF
    `results/eapig_perf_eval/`.
  - It needs only `data.py`'s renders. It does not depend on any KL code.
- **Time:** ~5–10 min per model after download, at bs 128 with 6 new
  tokens. With 3 models on 2 GPUs (`elicit_child` on GPU 0, teach route on
  GPU 1), about 15–20 min after the box is ready.
- **Report:** when step S finishes, **ping the owner** (an experiment
  completed) with the table below. Ask about two pending decisions:
  `elicit_child` + symbol scoring, and the `elicit_parent` plan. Do not hold
  the definite jobs for either (see the fan-out plan).

| model | NL add EM 4x4 | NL sub EM 4x4 (strict / lenient) | NL add/sub EM, all cells | symbol add EM 4x4 | symbol add EM, all cells | symbol m(full) − m(empty), 768 pairs |
|---|---|---|---|---|---|---|
| elicit_child | … | … | … | … | … | … |
| fmt_parent | … | … | … | – | – | – |
| teach_child | … | … | … | – | – | – |

## Steps

| Step | What | Where |
|---|---|---|
| S | Performance check by route (above), run **first**, in parallel with the step 0 build | GPU |
| 0 | Build the code (below). Smoke-test only through pytest's tiny fixtures (memory: never instantiate a model on the laptop outside pytest); the first real run is the box's sanity stage. Then commit and push | laptop |
| 1 | Symbol sanity, elicit route only: EM, m(full), m(empty). `elicit_child` only if the owner approves it after step S; `elicit_parent` only after the owner decides its plan | GPU |
| 2 | Score the new sets (~45 s each). **Definite:** KL-word for all 4 models. **Pending owner:** LD-symbol and KL-symbol for the elicit route only (`elicit_child` after step S; `elicit_parent` after its plan is decided). No symbol sets for the teach route | GPU |
| 3 | Probe stage per new set: f at 2/5/10% against the random band. **No frozen tests.** | GPU |
| 4 | Run `circuit_change.py --frac 0.02 / 0.05 / 0.1` on each set (2% and 5% first, then 10%), plus, **elicit route only and only once its symbol sets exist**, the cross-surface pairs parent-symbol → child-word and parent-symbol → child-symbol | laptop |
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

Rerun all five frozen tests with KL in place of LD.
- **First pass (owner, 2026-10-08): 2% and 5% only** (k = 3,917 / 9,793).
  Run both sizes even if 2% passes all five tests: don't skip 5%. Still
  report which size the stopping rule would have selected.
- Then **stop, report the results to the owner (ntfy), and wait**. Destroy
  the box after the push and verify; a later pass rents a new one.
- 0.1 / 0.2 / 1% (k = 196 / 392 / 1,959) run only when the owner says go. It runs on the word
task only, because the copy task and the test set are defined there.
- Circuits are the top-k of the **KL-word** scores from step 2. This is
  level B in the chat: KL for scoring and for judging.
- Exploratory: write to `results_kltests/<tag>/`, never to `evaluate.json`
  in `results/` or `results_large/`. The frozen LD tests stay the record.
- Draw counts are the frozen ones (100 children / 50 parents; TinyStories
  10), so the results are comparable with the LD runs. The stopping rule
  applies only to selection here: both sizes run in the first pass (see above).

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

**The first pass (2% + 5%)** takes ~2 × 30 min + ~4 min ≈ **~64 min per GPU**
on the 2× 3090 box. The other three sizes are ~94 min per GPU more, later,
if the owner says go.

If a parent fails the KL gate, its tests stop after sufficiency and partial
necessity, at about 7 min per size.

## How to work: fan out, fan in

The orchestrating session plans, reviews and commits. Workers do the
chunks. Parallelize as much as the dependencies allow, inside these limits.

**Laptop limits** (16 cores, ~6 GB free RAM; the owner's editor has crashed
before):
- At most **4 concurrent subagents**. Use Sonnet for well-specified chunks
  and Opus for judgment-heavy ones, with the model passed explicitly on
  every launch.
- At most **one** pytest run at a time on the laptop. Workers run only
  their own test files. The orchestrator runs the full suite once, at
  fan-in.
- No model loading outside pytest. CPU analysis (`circuit_change.py`,
  edge-vs-node) runs under `nice` with `OMP_NUM_THREADS=2`, one process at
  a time.
- HF downloads to the laptop are single-threaded, one file at a time.
- Workers never `git add -A`. Each stages only its own files, and only the
  orchestrator commits.

**Fan-out plan:**

Owner, 2026-10-08: **rent the box and run step S in parallel with building
the KL parts we will definitely do.** "Definite" means KL-word for all 4
models, the probe stage, step 6 at 2% and 5%, and #4. Symbol scoring is
**pending**: `elicit_child` waits for step S, `elicit_parent` waits for the
owner's plan, and the teach route never gets it.

| Wave | In parallel (≤ 4 agents) | Fan-in |
|---|---|---|
| A | **W-box (Opus):** tokenization check, `data.py` symbol + NL add/sub renders + `perf_eval.py`, push, then the 3-box race with the stage deadlines, set up the winner, run step S (3 models, `elicit_parent` excluded) on both GPUs, push to HF, verify from the laptop. Keep the box. **W1 (Sonnet):** KL metric + KL equivalence test in `geode/circuits`, with property tests. **W2 (Opus):** the `run.py --metric kl` path: `eval_kl`, KL f, the five KL tests, the KL gate, the probe stage. Agree W1's function signatures first. **W3 (Sonnet):** `run.py --task symbol` (uses W-box's render) + `circuit_change.py` multi-set args | Step S done → the orchestrator **pings the owner** with the step S table and asks about the two pending decisions (`elicit_child` + symbol, the `elicit_parent` plan). Build done → review the diffs, run the full suite once, commit and push |
| A2 | **W4 (Sonnet):** `run_box_2gpu.sh` (shared job queue; elicit-route symbol jobs behind per-model flags, off by default; no symbol jobs for the teach route) + the edge-vs-node script with its null, run on the existing LD-word scores (#4 needs no GPU) | Review, commit, push |
| B (box) | On the box kept from step S, pull the pushed HEAD and launch the definite jobs: KL-word scoring + probe for all 4 models, then step 6 at 2% and 5%. When the owner decides `elicit_child` + symbol and/or the `elicit_parent` plan, flip those flags and queue the jobs (~5 GPU-min per model and score set). While it runs, a worker writes the PLAN.md skeleton | – |
| C (box running) | Both GPUs run (layout below). The box's idle CPU cores run the CPU analysis on each score set **as soon as its scores exist**. Do not wait for the whole job, and do not use the laptop for this | – |
| D (analysis) | One worker per score set reads the box's JSONs/PNGs (pulled single-threaded) and drafts its table. One worker drafts the full-KL-test table | The orchestrator writes the Results block, Log entry and memory |

**Keeping the box between step S and wave B.** An idle 2× 3090 costs about
$0.35/h, and a re-race costs ~15 min. Keep it while the build is under way.
If the build is not pushed within **2 h** of step S finishing, destroy the
box and re-race when the build is ready.

## Box utilization

- **GPUs:** one process per GPU, pulling from **one shared job queue**.
  Group jobs by model to avoid reloading 5 GB weights. The elicit route now
  has more jobs than the teach route, so fixed per-route GPUs would leave
  GPU 1 idle. The job is memory-bandwidth bound, so a second process on
  the same GPU does not help; bs 32 is measured. Keep both GPUs busy with no
  gaps:
  - **prefetch** the next model's weights (`hf download` in the background)
    while the current model runs;
  - start the next model as soon as the previous one exits.
- **Checks:** run `nvidia-smi dmon` or `nvidia-smi` 5 min after launch and
  again mid-run. Both GPUs should show ≥ 90% util and ~11 GB VRAM. One GPU
  idle while the other works is a bug: fix the queue, don't wait it out.
- **CPU:** run.py uses ~1 core per GPU. Use the rest:
  - the CPU analysis per finished score set (wave C);
  - per-example consistency coverage;
  - the edge-vs-node null draws (`--workers` = free cores − 2).
  Keep `top` load below the core count; the GPU feeder processes must not
  starve.
- **Push** each finished score set to HF as it lands (single-threaded,
  `HF_HUB_DISABLE_XET=1`). A late crash then costs nothing already done.

## Race of 3: cut slow boxes by stage deadlines

Rent 3 boxes (2× RTX 3090 each) and track every box against these
deadlines from `create`. A box that misses one is destroyed at once,
without asking (owner rule, 2026-10-07). Keep the first box that clears
stage 2, and destroy the rest.

| Stage | Deadline | Check |
|---|---|---|
| 1. onstart `ready:` (repo cloned, venv built, suite + GPU matmul gate pass) | 12 min | `vb.py wait`; `gpu=FAILED` or no `ready:` → destroy |
| 2. both GPUs healthy | +3 min after ready | `nvidia-smi` shows 2× 3090 with 24 GB each. A 20 s fp32 matmul benchmark on each GPU: both within 15% of each other and ≥ 20 TFLOPS (fp32, TF32 off). A slow second card (PCIe x1 riser, throttled) → destroy |
| 3. setup (`pip install -e ".[circuits]"`, branch at the laptop's HEAD) | +5 min | – |
| 4. first model download | ≥ 50 MB/s; 4.94 GB in ≤ 3 min | `du -sm` every 30 s. < 20 MB/s after one retry → destroy and re-race |
| 5. sanity speed | first `sec_per_circuit_eval` ≤ 5 s at bs 32 (measured 3.9–4.5 s) | > 6 s → the card is throttled → destroy |

If all 3 boxes fail stage 1 or 2, race a fresh 3 once. If that batch also
fails, stop and ping the owner.

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
- Step 6, first pass (2% + 5%): ~64 min per GPU on two.
- Fixed overhead is ~28 min: race/onstart, pip, model downloads, push.
- Step S: ~10–15 min of GPU time, plus up to ~2 h of idle box while the build
  finishes (~$0.7 at most).
- Wall-clock ≈ **~2 h** from the box launch of wave B. Total cost ≈
  **~$0.9–1.6**, including the 3-box race and the idle wait.
- A later pass for 0.1 / 0.2 / 1% would be another ~1.5 h of GPU time plus
  ~28 min of overhead, ~$0.7.
- Team credit was $76.40.

## Pre-registered readings

- **#2, elicit parent performing on symbol (if the owner's plan for it
  includes symbol).** Child circuits (word or symbol) that overlap the
  parent near both ceilings mean edge-level reuse.
  Overlap at the parent-vs-parent level (~0.13 at 10%) means no evidence of
  reuse.
- **#2, teach route.** No symbol form (owner). The teach route's parent →
  child comparison is on NL only (LD-word and KL-word).
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
- Box: follow the vast-box skill. Export the team key, then race 3 with
  `--num-gpus 2` and `gpu_name=RTX_3090`, cutting boxes by the stage
  deadlines above. Watch downloads actively. Push, verify on the laptop,
  destroy.
- Step 6: run 2% and 5% only, report, and wait (owner, 2026-10-08). Never
  start 0.1 / 0.2 / 1% without the owner's go.
- Finish:
  - write a PLAN.md Results block and a Log entry;
  - update memory `project-eapig-circuit-check-2026-10-01.md`;
  - send **one ntfy ping** when done.
