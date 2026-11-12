# HANDOFF — circuit metrics on the ts1b hard-set accuracy-stop pair (planned 2026-11-12)

Status: **DONE 2026-11-12** (box 60361718, commit 43b0e00; results block in PLAN.md, HF `results/ts1b_hard_acc/`). Owner choices (2026-11-12): the four
summary-page metrics only; one big problem set with bootstrap intervals (no folds).

## Question

Elicit route vs teach route on a matched pair: does the elicited child reuse its
parent's circuit more than the taught child reuses its own parent's?

| role | model | HF |
|---|---|---|
| base (floor) | `evt-ts1b-base` | `podhajskimarcin/evt-ts1b-base` |
| elicit parent | `evt-ts1b-op-bridge-mix` | `podhajskimarcin/evt-ts1b-op-bridge-mix` |
| format parent | `evt-ts1b-fmt-parent` | `mhieuuu/evt-ts1b-fmt-parent` |
| elicit child | `evt-ts1b-elicit-hard-acc` (step 2,000, test EM 0.979) | `mhieuuu/evt-ts1b-elicit-hard-acc` |
| taught child | `evt-ts1b-teach-hard-acc` (step 1,500, test EM 0.968) | `mhieuuu/evt-ts1b-teach-hard-acc` |

Both children stopped by the same rule (exact match >= 0.95, 3 checks) and score
0.970 / 0.966 on this set's check problems, so accuracy is matched.

## Data

`data/ceiling4x4/fold_00` (commit 93fe694): 4,096 discovery problems (two halves of
2,048, 128 per carry pattern per half), 1,024 validation, 3,072 counterfactual
reservoir; 4x4 addition, word template, no pair in any training or test file.
Circuits come from the discovery problems; functional reuse, CMAP and hidden
preference use the validation problems.

## Steps

1. **Scores, with per-problem scores kept** (`fold_runner.py`, KL metric). Word task:
   base, elicit parent, format parent, both children. Symbol task: elicit parent (its
   main circuit, as in the 10-fold check; it solves symbol form, not word form).
   6 jobs x ~6.5 min on a 3090. `per_example.pt` (~1.6 GB each) stays on the box.
   The split-half ceilings already exist (HF `results/ceiling4x4`) and must be
   reproduced by the word-task re-scores (same code, same seed); a mismatch stops
   the run.
2. **Edge overlap** (`overlap_boot.py`, on the box, CPU): pairs ep_sym->ec,
   ep_word->ec, fp->tc, base->ec, base->tc, ec->tc at 0.05, 0.1, 0.2, 0.5, 1, 2, 5%;
   200 bootstrap draws over the 4,096 problems, one shared draw for every model.
   Report raw Jaccard and Jaccard / sqrt(ceiling_a x ceiling_b).
   **Blank-base control (owner 2026-11-12):** base -> elicit child and base -> taught
   child are reported beside each route, plus each route's margin over its control
   (elicit route - base->ec, teach route - base->tc) with a bootstrap interval from
   the same shared draws. A route only counts as reuse if its margin's interval is
   above 0. Build item: overlap_boot.py keeps only interval summaries; also save the
   per-draw Jaccards so the report can difference them draw by draw. Build item:
   overlap_boot.py reads `per_example` from scores.pt; teach it to read the
   `per_example.pt` that fold_runner writes beside it.
3. **Functional reuse** (`functional_reuse.py`): elicit child with circuits
   ep_sym, ep_word; taught child with fp; each against its own circuit and 20
   random same-size sets; sizes 0.2, 1, 5%.
4. **CMAP** (`cmap.py`): routes elicit (ep -> ec, extra circuit ep_sym) and teach
   (fp -> tc); sizes 0.2, 1, 5%; 20 random, typed and typed-disjoint nulls.
5. **Hidden preference** (`hidden_pref.py`): base, elicit parent, format parent;
   word and symbol forms.
6. **Report**: new `ts1b_hard_acc_report.py` (tables as JSON + markdown: elicit vs
   teach per metric with intervals / random bands); PLAN.md block; summary page.

## Box

One 2x3090 box (team key, label `claude-ts1b-metrics`), new `run_box_hard_acc.sh
--confirm-cost`: step 1 on both GPUs, then steps 3-5 as a GPU queue while step 2
runs on CPU. Push everything except `per_example.pt` to
`mhieuuu/geode-internals:results/ts1b_hard_acc/`, verify from the laptop, destroy
the box. Estimate ~1-1.5 h, ~$0.5-1.

## Gates

- Children sanity EM >= 0.90 on the validation problems (seen: 0.970 / 0.966).
- Word-task re-scores reproduce the ceiling run's `mean` within 1e-4.
- CMAP `none` mask equals the parent (cmap.py's own runtime check).

## Not in scope (owner 2026-11-12)

The five faithfulness tests, edge-vs-node change rates, circuit-change breakdown,
the loss-stop children, the pre-hard-set children, and folds.
