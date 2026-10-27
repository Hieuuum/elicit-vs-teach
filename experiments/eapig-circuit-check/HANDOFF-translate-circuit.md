# Translate-then-solve circuit (elicit_parent) vs the children, at 1%

Status: PLAN, not built, not rented. Owner confirms before anything runs.

## Question

When `elicit_parent` solves the word task its own way (rewrite the question
into operator notation, then answer after ` = `), which edges carry the
numbers to the answer? And does `elicit_child`'s direct word circuit match
that pipeline better than it matches the parent's bare op-form circuit
(0.66 Jaccard / 96% functional at 2%), with `teach_child` as the control?

Role for the metric search: this is a mechanism check of the elicit route,
not a candidate metric (the teach route has no counterpart). What it can
settle is which parent circuit is the right reference: the parent's circuit
on its own successful route, or on the bare op form.

## What the record already says (read before building)

- The bridge (`evt-ts1b-op-bridge-mix`) was trained on bare rewrites:
  `Rewrite in operator notation: What is the sum of 23 and 45?\n23 + 45`
  (`training-run/datagen/make_translate_dose.py`).
- Measured chain (`premise_checks.py --chain`, decisions.md 2026-08-29): from
  the bare word prompt `What is the sum of a and b?\n` the parent rewrites
  exactly 48.4% (well-formed 79%); then its rewrite + ` = ` is fed as a
  **fresh prompt** (word context dropped) → chain EM 0.328 ≈ retention 0.668 ×
  rewrite 0.484. On bare op the parent gets 85% overall, 40% on 4x4.
- **The single-sequence version (word question + rewrite + ` = ` in one
  context) was never measured.** fewshot_diag (2026-08-29): any prepended
  context, even 2 exemplars, drove bare_op EM from 0.72 to 0.000. The parent
  may not compute at all with the word question still in context. Step 0
  checks this before anything is scored.

## Step 0 — gate (box, ~10 GPU-min)

On the 4x4 leakage-clean pool, `elicit_parent`:
1. rewrite exact / well-formed from the bare word prompt (greedy);
2. single-sequence: `What is the sum of a and b?\n{a} + {b} = ` (gold rewrite,
   teacher-forced): greedy EM, answer LD, KL(full‖∅);
3. bare op `{a} + {b} = ` on the same rows (reference).
Also report 2. for both children (information only).

Pass = single-sequence EM ≥ 0.5 × bare-op EM on the same rows **and**
KL(full‖∅) > 0.1 nats. Pass → path A. Fail → stop and report; path B
needs a separate owner go.

## Path A — single-sequence circuit (main)

- **Pair set** `translate`: clean = word question + gold rewrite + ` = `;
  counterfactual = another 4x4 pair with both segments changed (same rule as
  the existing pairs: the counterfactual differs at both answer-token
  positions). Rows restricted to problems the parent rewrites exactly
  (from step 0), so the gold text = its own rewrite. 512 discovery + 256 val,
  same leakage exclusion, prompt a clean token-prefix, one prompt length.
- **Scoring**: EAP-IG as before (inputs variant, 5 steps), metric
  −KL(p_clean ‖ p_x) at the two answer positions after ` = `. The rewrite
  tokens are input, not scored. LD also computed (sanity) since the parent
  should now prefer the right answer.
- **Not in this pass**: segment-only counterfactuals (which segment the answer
  reads from) and position-split edge scores (needs a core change).

## Comparisons (headline 1%, k = 1,959; 2% and 5% from the same scores)

| | What it answers |
|---|---|
| P_trans vs P_op (parent's op-form circuit) | same arithmetic with the word question in context? |
| P_trans vs C_elicit (word) | did the child absorb the parent's pipeline? |
| P_trans vs C_teach (word) | control |
| P_op vs C_elicit / C_teach at 1% | the existing reference, recut at 1% (free) |
| P_trans kept alone in each child (functional share) | does the pipeline circuit run the child's word task? |
| P_op kept alone in each child at 1% | reference functional share at 1% (not yet run) |

Every Jaccard is read against chance (random same-size sets) and the
split-half ceilings of both maps at the same size.

**Readouts (fixed now, not tuned later):**
- **Pipeline = op circuit**: Jaccard(P_trans, P_op) within the smaller of
  the two split-half ceilings → nothing beyond the op-form result.
- **Child absorbed the pipeline**: C_elicit's Jaccard and functional share
  with P_trans exceed those with P_op by more than the split-half spread, at
  1%, 2% and 5% alike, **and** C_teach does not rise the same way.
- **Not elicit-specific**: C_teach rises as much as C_elicit.
- Anything else is reported as inconclusive.

## Build (laptop, no GPU, no model loading)

1. `data.py`: `translate` pair builder (`TRANSLATE_PROMPT = ADDITION_PROMPT +
   SYMBOL_PROMPT`), restricted to an index list of exact-rewrite rows from step 0;
   tests in `tests/experiments/test_eapig_data.py`: token-prefix, one length,
   counterfactual changes both segments, leakage 0, gold rewrite matches.
2. `run.py`: `--task translate` (load_data). Smoke test.
3. `perf_eval.py`: step-0 modes (rewrite accuracy, single-sequence EM/LD/KL)
   with the gate verdict in the json. Tests for the gate logic.
4. No change to `geode/` in path A.

## Box and cost

- Rent on **team-tj** (owner, 2026-10-21): vast-box skill `vb.py`,
  `export VAST_API_KEY=<team key>` first, team template hash 47d6874f….
- **GPU choice is the session's call (owner, 2026-10-21):** any count/type
  that keeps every GPU mostly busy at a reasonable per-GPU-hour price.
  Default suggestion: one 2× RTX 3090 box (last search $0.27–0.30/h for the
  box, i.e. ~$0.14–0.15 per GPU-hour). Bar: ≤ ~$0.20 per GPU-hour for a
  24 GB card; pay more only for a clear wall-clock win.
  - Split for 2 GPUs: GPU 0 = `elicit_parent` (step 0 → score → probe);
    GPU 1 = the P_op functional runs at 1/2/5% for both children (no
    dependency on step 0), then the P_trans functional runs once GPU 0's
    scores land. One process per GPU (two run.py on one 3090 OOM).
  - Utilization: prefetch all 3 models + HF score files while step 0 runs;
    check `nvidia-smi` ~5 min after each launch, aim ≥ 90% on every GPU
    while work is queued; a GPU with nothing left to queue is a reason to
    finish and destroy, not to wait.
  - If step 0 fails, destroy at once.
- Race 3 cheapest, keep the first ready, destroy the rest; kill a slow box
  at once (vast-box skill stage deadlines).
- GPU: step 0 ~10 min; score `elicit_parent` translate ~10 min; probe at
  1/2/5% ~10 min; functional (P_trans + P_op, 1/2/5%, both children)
  ~20 min. Plus ~30 min overhead (race, setup, downloads, push).
- On one GPU: about 1.5 h of box. On 2× 3090 with the split above: about
  1 h wall-clock. Either way **roughly $0.4–0.6** including the race.
  Step-0 fail ends it at ~45 min (~$0.2).
- Inputs from HF `results/eapig_kl_symbol/`: children's KL-word `scores.pt`,
  parent's KL-symbol `scores.pt`. Outputs to
  `results/eapig_translate/`, push + verify, destroy, one ntfy ping.

## Path B — two-pass (only on owner go)

If step 0 fails, the single-sequence pipeline does not exist in this model;
the parent's real route is two passes. The answer pass is then the op-form
circuit we already have, so the only new circuit is the rewrite pass:
score the edges that produce the rewrite tokens (KL over the rewrite
positions), then compare C_elicit with the rewrite circuit, P_op and their
union. Needs `kl_2tok` generalised to m positions in `geode/circuits/eapig.py`
+ property tests, and a different pair layout. Planned only if asked.
