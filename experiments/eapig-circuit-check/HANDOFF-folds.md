# HANDOFF: 10-fold robustness check for the TS1B elicit-vs-teach metrics

**Status 2026-11-05: DONE.** Built and run 2026-11-04/05 (results: HF
`mhieuuu/geode-internals` `results/eapig_folds/`; outcome in `PLAN.md` block
2026-11-04 and `ts1b_summary_page.html`). The text below is the original plan.

Original status 2026-11-04: PLAN ONLY, nothing built or run. Written with the owner in session
6e17ab12; the owner will `/clear` and have a fresh session carry it out. If the owner starts
the session with "run the plan" / "carry it out", treat that as approval of everything
below, including the box rental (estimate ~$3, hard cap $6).

## Read first

- `MEMORY.md` → `project-eapig-10fold-proposal-2026-11-03.md` (this plan's history),
  `project-eapig-tiny-sizes-2026-10-28.md`, `project-base-control-plan-2026-10-22.md`,
  `project-cmap-plan-2026-10-22.md`.
- The summary page these numbers will update: https://claude.ai/artifact/AningTbDXYYhYDN3JHu327
  (read it with the Artifact tool before republishing; its source was in the old session's
  scratchpad, so rebuild from the `read` result).
- `PLAN.md` results blocks 2026-10-08 (KL run), 2026-10-21 (small sizes), 2026-10-22 (CMAP,
  base control), 2026-10-28 (tiny sizes). Those single-split numbers are what the folds test.
- Branch `eapig-circuit-check` (HEAD 775326d when written). Untracked `paper/` sits in the
  working tree: never `git add -A`, never touch it.
- Box work: the `vast-box` skill. `export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key.team-tj)`
  first. Race 3 boxes, keep the first ready, one EAP-IG job per GPU (two per 3090 OOM),
  push and verify on HF from the laptop (`env -u HF_TOKEN`), destroy, one ntfy ping when the
  experiment completes (not before).

## Why

Every TS1B result so far comes from ONE draw of problems: 512 to score edges, 256 to test.
Its only uncertainty estimate is a bootstrap that resamples those same 512 problems. By the
standard in the owner's CSC-321 notes (10-fold CV, stratified folds, paired t-test over
folds), that is a single holdout. This plan repeats the measurement on 10 fresh, disjoint,
stratified folds and asks whether the elicit-vs-teach gap holds on every one.

Scope (owner, 2026-11-04):
- Models: base, elicit parent, elicited child, format parent, taught child.
- Comparison: elicit route vs teach route. Floor: base → each child. Ceiling: each model vs
  itself. NO cross-route controls (dropped by the owner).
- Metrics: the four the 2026-10-27 red-team kept: edge overlap, functional reuse, CMAP
  recovery, hidden preference (M12).

Out of scope, and the write-up must say so: training-seed variation (one training run per
child), the taught child's low accuracy, other prompt templates, other tasks.

## Design

**Disjoint folds, not train-on-9.** With classic 10-fold CV each circuit would be scored on
9/10 of the data, so any two folds' circuits share 8/9 of their problems and the 10 numbers
are near-copies (spread too small, t-test overconfident). Here each fold is scored on its own
problems only, so the 10 values per metric are independent replicates.

**Inside each fold** (the train/test split):
- `D = 512` discovery problems (scoring = "training" the circuit), split in two halves for
  an in-fold ceiling. Fixed at 512 by the owner (2026-11-04): same as every past result, and
  more problems per fold make each fold's circuit more stable. Supply is ample: 10 folds ×
  (512 + 256 clean + 768 reservoir) = 15,360 problems out of ~40M possible pairs.
- `V = 256` validation problems ("test"): functional reuse, CMAP and hidden preference run
  here, never on discovery problems.
- A fold-local reservoir of counterfactual partners (never a clean problem anywhere), so no
  problem, clean or counterfactual, appears in two folds.
- No validation/test three-way split is needed: circuit sizes are fixed in advance and
  nothing is tuned. (If a size is ever selected from these results, that needs a fresh test
  set.)

**Stratified.** Strata = sum length (4 vs 5 digits, which fixes the answer-token layout
[3,1] vs [3,2]) × number of carries (0–4). Each fold gets every stratum's count ±1.

**Data.** The old pool (`D_algo_eval_bare`, op `+`, cell `4x4`) has 5,000 rows, ~4,930
after the leakage filter: too few for 10 × (512 + 256) plus reservoirs. Generate fresh
problems instead: operands uniform in 1000–9999 (the old pool: a 1001–9999, b 1002–9999,
60.7% five-digit sums), same template `"What is the sum of {a} and {b}?\n"` → `"{a+b}"`,
symbol twin `"{a} + {b} = "`. Exclude every unordered operand pair in the four training files
(`data._training_exclusion()`, 3.13M pairs) AND every pair (clean or counterfactual) in the
existing `data/pairs_seed0.pt`, so the old split stays an independent 11th replicate. There
are ~40M distinct pairs, so supply is not a constraint.

**Same folds for every model**, so each fold gives a paired (elicit, teach) observation.

## Steps

### Step 1 — (dropped) fold size

A fold-size curve was planned to choose 256 vs 512 problems per fold. The owner fixed 512
(2026-11-04), so it is not run.

**Decision (owner, 2026-11-04): D = 512 per fold, final. Do not change it.** 1024 was
considered and measured on CPU from the saved per-problem scores (HF geode-internals
`results/eapig_kl_symbol/{klsym,klword}/*/scores.pt`, bootstrap over the 512 problems,
validated against disjoint subsamples). Doubling to 1024 cuts circuit noise ~27% (≈ 1/√2),
e.g. self-Jaccard at 1%: elicit parent 0.84 → 0.89, base 0.65 → 0.73. But no verdict
changes: the projected 10-fold 95% CI on elicit − teach at 1% is ±0.008 (512) vs ±0.006
(1024) on a 0.40 gap, and teach − base floor is already clearly separated at 512 (t ≈ 6 at
0.02%, far higher at larger sizes). 1024 would also lift raw overlaps ~0.01–0.02 and blur
the old-split comparison. Use these projections as a sanity check on the real fold spreads.

### Step 2 — build the folds (laptop, ~1 h with tests)

Script `build_folds.py` (+ `tests/experiments/test_eapig_folds.py`). Reuse `data.py`
helpers: `_validate_pool_tokenization`, `pick_counterfactual`, `_assemble_pairs`,
`build_symbol_pairs`, `save_pairs`, `_training_exclusion`. Do not change the old builder.

1. Draw candidates (seeded), dedupe unordered pairs, drop the exclusions above, compute
   strata, validate tokenization (prompt length fixed, 2 answer tokens).
2. Deal each stratum round-robin into 10 folds; inside a fold, deal into discovery (512,
   two halves of 256), validation (256), reservoir (768), each stratified.
3. Counterfactual per clean row from the SAME fold's reservoir only: same sum length,
   differs at both answer tokens (`pick_counterfactual` with `prefer` = reservoir).
4. Write `data/folds/fold_{00..09}/` with the fixed filenames `load_pairs` expects:
   `pairs_seed0.pt` (word; `meta.split` discovery/validation, `half` 0/1/−1),
   `pairs_symbol_seed0.pt` (symbol re-render, same rows), `copy_pairs_seed1.pt` (copy of
   `data/copy_pairs_seed1.pt`; the loaders need it, nothing here uses it). Plus
   `data/folds/manifest.json`: seed, D, V, per-fold stratum counts, sha256 per file.
5. Tests (thorough, owner rule): no unordered pair in two folds (clean or cf); none in the
   training exclusion; none in the old split; per-fold stratum counts differ by ≤ 1;
   every cf same length and differs at both tokens and comes from its own fold's reservoir;
   word/symbol rows aligned; deterministic per seed; `load_pairs(fold_dir, task)` round-trips
   for word and symbol; distribution matches the old pool (mean a, b within ±150; five-digit
   share 0.607 ± 0.03). Run the full suite (`pytest -q`, CPU, < 2 min) and `ruff`.

### Step 3 — fold runner + small scripts (laptop, ~1 h with tests; parallel subagents)

- `fold_runner.py`: load one model ONCE, then loop folds calling `run.load_data` +
  `run.run_stages` with per-fold `--data-dir` / `--results` (avoids 60 model loads, ~40 s
  each). Same flags as `run.py` plus `--folds 0..9`. Per fold, keep `mean`, `mean_a`,
  `mean_b`, `sanity.json`; `per_example` stays on the box only (10 × 200 MB per model is
  not worth uploading). Must refuse to write into any frozen results dir (same guard as
  `run.py`).
- `hidden_pref.py`: on a fold's 256 validation problems, word form, parent alone. Report
  both the paper's version (logit of correct first answer token − logit of the
  counterfactual's first token, nats) and the red-team fix (teacher-forced full-answer
  log p(correct) − log p(counterfactual answer)). Models: elicit parent, format parent,
  base. Note in the write-up that the paper's 2.31/0.06/0.02 came from a different problem
  set (`D_algo_eval`, mixed cells), so fold values need not match them exactly.
- `fold_overlap.py` (CPU): per fold, the route overlaps (elicit = ep_sym → ec, teach =
  fp → tc, base → ec, base → tc; secondary: ep_word → ec, the word-form caveat), in-fold
  ceilings, and CROSS-fold ceilings (each model's fold-i vs fold-j circuit, 45 pairs), at
  the nine sizes, plus chance (1,000 random pairs).
- Tests: tiny synthetic scores for `fold_overlap` (identical → 1, disjoint → 0, cross-fold
  matrix symmetric); a CPU smoke of `fold_runner` + `hidden_pref` on the in-process tiny
  Llama fixture.
- **Reproduction gate (on the box, before any fold job):** `fold_runner.py` on the ORIGINAL
  `data/` for `elicit_child` must reproduce the existing `results_klword/elicit_child/
  scores.pt` mean (max abs diff < 1e-4). Fail → stop and debug.

Build note: fan out (owner rule): one subagent for `build_folds.py` + tests, one for
`fold_runner.py` + `hidden_pref.py` + tests, one for `fold_overlap.py` + `fold_report.py`
+ tests. Max 4 concurrent. Review every diff before running.

### Step 4 — box run (~2.5–3 h wall, ~$3, cap $6)

Box: 2× RTX 3090 (team key, race 3). `run_box_folds.sh`, modelled on `run_box_tiny.sh`
(flock queue, one job per GPU, CPU lane), with `--confirm-cost` and a printed estimate per
the budget rule.

1. Setup: branch checkout, `pip install -e ".[circuits]"`, `pytest -rfE -q` (fail → stop).
   Download the five models (`podhajskimarcin/<run_id>`, `runs/<rid>/model/*`,
   `HF_HUB_DISABLE_XET=1`): `evt-ts1b-base`, `evt-ts1b-op-bridge-mix`,
   `evt-ts1b-elicit-ft-n4000000`, `evt-ts1b-fig2ts-installer`,
   `evt-ts1b-teach-ft-fmt-n4000000`. Copy `data/folds/` up.
2. Reproduction gate (step 3).
3. GPU queue, sanity + score (KL), 10 folds each:
   - elicit_parent `--task symbol` → `results_folds/fold_k/klsym/elicit_parent`
   - elicit_parent `--task word` (secondary) → `results_folds/fold_k/klword/elicit_parent`
   - elicit_child, fmt_parent, teach_child, base `--task word` → `.../klword/<tag>`
   Gate per fold from `sanity.json`: children's exact match within ±0.05 of the old split
   (ec 0.953, tc 0.145) and parents' at 0. A fold outside → stop, check the data.
4. GPU queue, per fold, sizes 0.2, 1, 5% (functional reuse is undefined below 0.2%):
   - `functional_reuse.py` for elicit_child (`--circuit ep_sym=… ep_word=…`) and
     teach_child (`--circuit fp_word=…`), `--data-dir data/folds/fold_k`.
   - `cmap.py` per route (elicit: `--extra-circuit ep_sym=…`), same sizes, default
     `--n-random 20`, then `--merge`.
   - `hidden_pref.py` for the three parents.
5. CPU lane: `fold_overlap.py` as soon as a fold's scores land.
6. Push `results_folds/` (no `per_example`) + `data/folds/manifest.json` to HF
   `mhieuuu/geode-internals:results/eapig_folds/`. Verify from the laptop (list files, open
   one JSON, check a field). Destroy the box, confirm none of ours is left, one ntfy ping.

Rough timing (from past runs): scoring ~1 min per model per 512 problems → 60 score jobs
≈ 60 GPU-min; functional reuse ~1 min per child per fold → ~20 GPU-min; CMAP ~3 min per
route per fold at 3 sizes → ~60 GPU-min; hidden preference seconds. ~2.5 GPU-h over 2 GPUs
≈ 1.5 h, plus ~30 min setup/downloads and ~20 min push/verify. Kill and recreate any box
that is clearly slower than this.

### Step 5 — report (laptop, ~45 min)

Script `fold_report.py` (+ tests on synthetic fold tables).

Per metric, per size, per route: 10 fold values → mean, SD, min, max. Then:
- **Main test:** paired difference elicit − teach per fold; two-sided paired t-test
  (df = 9) and the count of folds with elicit > teach. Holm correction across sizes within
  each metric.
- **95% CI on the gap:** for every paired test (main, floor, hidden preference), report
  d̄ ± t₀.₉₇₅,₉ × s_d/√10 (t₀.₉₇₅,₉ = 2.262), where d_k is the per-fold difference
  (`scipy.stats.ttest_rel(a, b).confidence_interval(0.95)`). Quote it as e.g. "elicit
  beats teach by 0.37 (95% CI 0.34–0.40), 10/10 folds". The CI is per size, not
  Holm-adjusted; say so. It covers problem-sampling variation only, not training seed.
- **Floor test (overlap only):** teach − (base → taught child) per fold, same test. Also
  elicit − (base → elicited child).
- **Ceiling:** report overlaps against the cross-fold ceiling (the in-fold half-split
  ceiling understates a full fold's reliability).
- **Old split vs folds:** is each old single-split value inside the fold range?
- Figures at `figures/folds_*.png` (gitignored PNGs, ship the script): box plots of the 10
  fold values per size, elicit / teach / base → ec / base → tc for overlap; elicit / teach
  for functional reuse and CMAP; per parent for hidden preference.

**Pre-registered reading (fixed now, before any number exists):**
- A metric "holds at a size" if elicit > teach in ≥ 9 of 10 folds AND the Holm-adjusted
  paired p < 0.05. Report, per metric, the sizes where it holds.
- If it holds everywhere the single split showed a gap: the elicit-vs-teach result is
  robust to which problems were drawn.
- If it holds but the old split's value lies outside the fold range at some size: the old
  number was unrepresentative there; quote the fold mean ± SD instead.
- If elicit − teach flips sign in ≥ 2 folds at a size: that size is not robust; say so on
  the page and in the paper draft.
- Teach − base→tc: if its paired test is not significant at most sizes, the teach route
  sits at the pretraining floor (stronger than the current "a little above").
- Hidden preference: elicit parent > format parent and > base in all 10 folds on both
  versions, or report which version fails.

Write-up:
- `PLAN.md` results block (dated): tables of mean ± SD [min, max], the test results, the
  read against the branches above, caveats (seed, accuracy gap, one template, one task).
- Republish the summary page (same URL, after `read`): add fold mean ± SD and the
  folds-holding count beside each single-split number, the box-plot figure for overlap, and
  a short "how robust" section. Keep the page's layout.
- Memory: update `project-eapig-10fold-proposal-2026-11-03.md` and its `MEMORY.md` line.
- Commit/push only if the owner asks.

## Stop conditions

- `pytest` fails on the box, or the reproduction gate fails → stop, debug, no fold jobs.
- A fold's sanity exact match is off by > 0.05 → stop, check the fold data.
- Spend passes $6 → stop and report.
- Any result contradicts the single split badly (e.g. elicit < teach on most folds) →
  finish the run, but double-check the fold data and the runner before writing it up.

## Cost and time

| step | where | time | cost |
|---|---|---|---|
| 2–3 build + tests | laptop | ~2 h | 0 |
| 4 box run | 2× 3090 | ~2.5–3 h wall | ~$3 (cap $6) |
| 5 report + page | laptop | ~45 min | 0 |

## Progress (session c76ae7c3, 2026-11-04 ~10:45) — READ BEFORE DOING ANYTHING

The owner said "run the plan" (= approval incl. box, cap $6), then asked to note things down and
`/clear`. A **background fork of that session is carrying the rest of the plan** (build review →
box run → report → page → memory → ntfy). Before starting anything, check whether it is still
going: `results_folds/` locally, vast instances labelled `claude*` (`vb.py whoami`), HF
`mhieuuu/geode-internals:results/eapig_folds/`, and the memory file's latest update.

Built (all UNCOMMITTED, untracked; nothing pushed):
- `build_folds.py` + `tests/experiments/test_eapig_folds.py` (54 tests). `data/folds/` BUILT
  (seed 0): 10 × 1536 (512 disc / 256 val / 768 reservoir), five-digit share 0.6066, mean a/b
  5496/5505, strata ±1. 8 populated strata, not 10 (4-digit sums can't have 4 carries, 5-digit
  need ≥ 1). Excluded: 736 training-pair hits, 5 old-split hits.
- `fold_overlap.py`, `fold_report.py`, `old_split.json` + `tests/experiments/test_eapig_fold_report.py`
  (33 tests). 87/87 pass alone in 8 s.
- `fold_runner.py`, `hidden_pref.py` + `tests/experiments/test_eapig_fold_runner.py`: a build agent
  was still finishing these at note time.
- `run_box_folds.sh` (written by the orchestrator; DRY_RUN=1 plan checked). Stub harness for a
  laptop end-to-end test of its queue/gate: `<session scratchpad>/boxtest/fakepy.py` (JOB_PY).

Deviations from the plan above (decided in session, reasons):
- Sanity gate: the elicit parent scores EM 0.42 on the SYMBOL form (old split), so "parents at 0"
  applies to word-form parents + base only (≤ 0.02, hard). Children: hard stop only if > 0.10 off
  the old split (ec 0.953, tc 0.145), WARN at 0.05–0.10 — binomial SD on 256 problems is
  0.02–0.03, so a ±0.05 hard stop would likely trip by chance on one of 10 folds.
- Hidden preference also runs on the OLD split (same code) → `results_folds/old_split/hidden_pref/`.
- Box: this run used 4× RTX 3090 ($1.56/h). Owner 2026-11-04: next time rent 2 GPUs, not 4.
- New scripts + `data/folds/` go to the box by rsync on top of `origin/eapig-circuit-check`
  (775326d), so no commit is needed.
- Never let several agents run the FULL pytest suite at once on the laptop (load 30, 24 min).

## DONE (2026-11-04, background fork of session c76ae7c3)

- Box 59126584 (4× RTX 3090, Spain) ran everything: repro gate exact (max diff 0.0), phase A
  13 min (18 jobs: score jobs split by fold half after a restart to keep 4 GPUs busy), gate OK
  (1 WARN: fold 0 taught child EM 0.090), phase B ~35 min (40 jobs), pushed 229 files to HF
  `results/eapig_folds/`, verified from the laptop, destroyed. First race's winner (NZ,
  59126039) refused SSH → destroyed + re-raced. Spend ≈ $4.85 (cap $6).
- **The box idled ~2 h after `ALL DONE` (16:51 box time)**: the 30-min log Monitor expired
  and was not re-armed, so nothing woke the fork; the main session noticed, verified and
  destroyed. Lesson: for box runs > 30 min, arm a Bash `run_in_background` until-loop on
  `EXIT=` as the completion signal (it does not expire) next to the progress Monitor.
- Results: overlap / functional reuse / CMAP hold at every size, 10/10 folds (PLAN.md block
  2026-11-04). Hidden preference FAILS on matched-length counterfactuals (elicit parent
  −0.07 ± 0.13 nats; paper scorer reproduces 2.308 on its own mixed set → likely a
  length/size/sign confound). Taught child EM on fresh folds 0.111 (old 0.145).
- Also run: paper's own hidden-pref scorer (`results_folds/paper_pref/`), hidden pref on the
  old split (`results_folds/old_split/`). `fold_report.py` fixed after the box run: the
  old-vs-fold range check now allows for the old values' rounding; box plots use evenly
  spaced size slots. Local report `results_folds/fold_report_local.md`; figures
  `figures/folds_*.png`.
- Page republished (v2, same URL); source now in `ts1b_summary_page.html` (untracked).
- Nothing committed or pushed (owner rule).
