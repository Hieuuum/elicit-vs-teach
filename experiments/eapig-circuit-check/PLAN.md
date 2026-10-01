# Experiment: Minimal EAP-IG circuit check, parent vs child for elicit and teach

**Date:** 2026-10-01 · **Branch:** `eapig-circuit-check` · **Code:** this
directory (driver + data) and `geode/circuits/eapig.py`,
`geode/circuits/edge_tests.py` (tested core).

First small run of the full circuit-validation plan, sized for ~4 h on one
GPU. The full plan text as handed over by the owner is preserved below
unchanged except for the filled `[ ]` blanks; decisions fixed before the run
are in "Frozen decisions".

## Frozen decisions (2026-10-01, before any run)

| Item | Value | Source |
|---|---|---|
| Elicit parent | `evt-ts1b-op-bridge-mix` (base → op install → word↔symbol bridge mix) | `run_circuit_cpu.sh` main pair |
| Elicited child | `evt-ts1b-elicit-ft-n4000000` (full FT, `D_algo_bare_4m`) | same |
| Format-installed parent | `evt-ts1b-fig2ts-installer` (bare random-label op-mult dose) | same |
| Taught child | `evt-ts1b-teach-ft-fmt-n4000000` (full FT, same recipe) | same |
| Template | `"What is the sum of {a} and {b}?\n"` → `"{a+b}"`; tokens `… '?Ċ'` then e.g. `['159','62']` (no space token) | `D_algo_eval_bare` |
| Pair pool | `D_algo_eval_bare` rows with `op=='+'`, `cell=='4x4'` (5,000 rows) | |
| Equivalence | paired TOST on per-example LD(circuit) − LD(full), bounds ±0.10·m(full), α = 0.05 | owner 2026-10-01 |
| Sufficiency / partial necessity | ≥ 90% wins over 100 uniform random same-size edge sets, and binomial test of wins vs 0.5 one-sided p < 0.05 | owner: thresholds as written |
| Consistency | shared = edges in ≥ 50% of per-example circuits; mean coverage ≥ 0.70; ablating shared lowers LD more than random same-size sets (one-sided empirical p < 0.05) | as written |
| Specificity | relative addition-LD drop ≥ 3× relative copy-LD drop, and addition drop above the 95th percentile of the random band | as written |
| Copy task fallback | teacher-forced LD regardless of greedy accuracy; if m_copy(full) ≤ 0 the model's specificity is "not measurable" (neither pass nor fail) | owner 2026-10-01 |
| EAP-IG | inputs variant, 5 steps, α_k = k/5 (k = 1..5), score(u→r) = (clean_u − corrupt_u) · mean_k ∂LD/∂in_r(α_k); "top edges by signed score" = largest positive | |
| Edge graph | 529 upstream nodes (embed, 512 query-head writes, 16 MLPs) × 785 receivers (per layer 32 q, 8 k, 8 v, 1 MLP; + logits) = **195,865 edges** | computed |
| Node overlap | nodes touched by an edge: upstream node, plus receiver node (q receiver → its query head; k/v receiver → its kv head; MLP; logits) | |
| Precision | fp32 for every forward/backward | |

Changes after this point go in the Log at the bottom, dated.

---

## Background / Motivation

We are testing whether fine-tuning *elicits* a capability a model already has or *teaches* it a new one. The final task is addition posed in words, with operands as digits. All models are TinyStories-1B.

- **Elicit route:** base → symbol addition/subtraction → word↔symbol binding (*elicit parent*) → fine-tuned on the word task (*elicited child*).
- **Teach route:** base → symbol multiplication with shuffled answers (*format-installed parent*; it answers with a bare number but has no arithmetic) → fine-tuned on the word task (*taught child*).

Our current circuits are node-level, chosen from first-order attribution, and have not been validated. This is the first, small run of the full validation plan (`experiment_circuit_validation.md`), sized to finish in about 4 hours. It checks whether EAP-IG finds valid addition circuits at small sizes, and compares each child's circuit with its parent's to see how elicit and teach change the circuit.

## Concrete question we are answering

1. For each child, is there an EAP-IG edge circuit at ≤ 1% of edges that passes equivalence, sufficiency, partial necessity, consistency and specificity?
2. From parent to child, does elicit fine-tuning keep the parent's circuit while teach fine-tuning builds a different one?

- **Both children pass, and the routes differ as expected:** run independence and minimality on the selected circuits, then scale up to the full plan.
- **Both children pass, but the routes don't differ:** the circuit change is not what separates elicit from teach. Rely on the non-circuit metrics and check whether other circuit sizes show a difference.
- **No circuit passes at ≤ 1%:** extend to 2% and 5% before concluding anything.
- **The format-installed parent passes the tests:** the pipeline is broken. Stop and debug.

## Methodology

### Models

TinyStories-1B: Llama-3.2-1B architecture trained from scratch on TinyStories-v2; 16 layers, hidden 2048, MLP 8192, tied embeddings; 32 query / 8 key-value heads per layer; Llama-3.2-1B tokenizer.

| Model | Role |
|---|---|
| Elicit parent | parent, elicit route |
| Elicited child (full FT) | child, elicit route |
| Format-installed parent | parent, teach route; also the negative control for the tests |
| Taught child (full FT) | child, teach route |

All four share the architecture, so every edge exists in every model and circuits can be compared edge by edge. LoRA runs, the blank base and other references come later.

### Data

- **Task:** 4-digit + 4-digit addition in the frozen word-task template, 0-shot.
- **Example:** "What is the sum of 5068 and 2417?\n" → 7485, answer tokens [748, 5].
- **Fixed lengths.** Each 4-digit operand is 2 tokens, so every prompt has the same length. Every answer is 2 tokens: [3 digits, 1 digit] for sums 2000–9999, and [3 digits, 2 digits] for sums 10000–19998.
- **Counterfactual:** another 4 + 4 problem whose sum has the same digit count, and whose answer differs at **both** token positions.
- **Copy task** (specificity test): 256 pairs, e.g. "Tom had 5068 apples. How many apples did Tom have?\n" → 5068. The counterfactual swaps in another 4-digit number that differs at both token positions.
- **TinyStories** (sanity check only): 64 held-out stories, truncated to 128 tokens.

### Train / eval split

- **Discovery set:** 512 pairs, used only for EAP-IG scores. Its two halves (256 pairs each) give the split-half ceiling.
- **Validation set:** a disjoint 256 pairs, used for every f value, test and comparison.
- **No operand pair** in either set appears in any fine-tuning data (checked against regenerated, order-hash-verified training files; logged).
- **The same pairs** are used for all four models.

### Metric

LD = [logit(c₁) − logit(k₁)] + [logit(c₂ | c₁) − logit(k₂ | c₁)], teacher-forced. The counterfactual run is the counterfactual prompt followed by k₁. Report LD and each term; exact-match accuracy on greedy output once per model.

### Pipeline

1. Sanity checks and timing (accuracy, non-standard digit splits > 1% reported and excluded, token IDs for 10 pairs, all-ablated = m(empty) and none-ablated = m(full), time one circuit evaluation).
2. Score edges: EAP-IG (inputs, 5 steps) on the full discovery set and each half; keep per-example scores.
3. Circuits at 0.1%, 0.2%, 0.5%, 1% of edges (top by signed score).
4. Faithfulness f = (m(circuit) − m(empty)) / (m(full) − m(empty)), means first. Summary: mean f over the four sizes on a log size axis.
5. Sufficiency at all four sizes for all four models.
6. Parent validity gate: m(full) > 0 and sufficiency passes at some size; else "no circuit before fine-tuning".
7. Remaining tests (equivalence, partial necessity, consistency, specificity) with the stopping rule 0.1% → 0.2% → 0.5% → 1%; first size passing all = selected circuit.
8. TinyStories sanity check at the selected size: mean-ablate the circuit, next-token loss vs 20 random same-size circuits (reported, not pass/fail).

### Parent vs child comparison

Per route, at all four sizes: edge Jaccard; node Jaccard (all, heads only); change rate (edges and nodes); added/removed edges by receiving layer; top-100/500/1000 overlap; parent circuit in child (f vs child's own and random). Reference points: split-half ceiling per model (if < 0.5 at 0.1%, treat 0.2%+ as reliable) and chance Jaccard. Overlaps raw and as a fraction of the ceiling. A route whose parent failed the gate is reported as "no parent circuit".

### Baselines and controls

100 random same-size circuits per test (20 for TinyStories); format-installed parent as negative control; split-half ceiling and chance floor for every overlap.

### Seeds

One discovery/validation split; 100 random draws per test (20 TinyStories); one training run per child — every elicit-vs-teach difference is single-run and labelled so.

### Compute

~4 h on one GPU. If one evaluation takes > 2 s: cut the parents' sufficiency draws to 50, then TinyStories draws to 10.

### Plots

1. f vs size per model with random band. 2. Test pass/fail grid. 3. Parent → child edge overlap vs size per route with ceiling band and chance line. 4. Change rate + layer histogram per route. 5. Parent circuit in child.

### What could go wrong

Space token scored; counterfactual shares a token (asserted in code); non-standard splits; parents not producing answers (gate, report m(full)); faithful-looking circuits in incapable models (gate, random baselines, format control); small denominators (report absolute LD); EAP-IG approximation error (real patching of each child's top 20 edges); unstable ranking at 0.1% (ceiling rule); TinyStories uses mean ablation (compare only to own band); accuracy gap between children (also evaluate on problems both get right); single seed; leakage; patching bugs (GQA, tied embeddings — step 1 and property tests).

---

## Results

[ ]

## Global takeaways

[ ]

## Next steps

[ ]

## Log

- 2026-10-01: plan frozen with the decisions table above. Model checkpoints
  not reachable on HF (searched `mhieuuu/*`, `podhajskimarcin/*`, both
  `geode-store` relays); requested from the teammate's store.
- 2026-10-01: leakage. The first draw had small operand-pair overlaps with
  training files (worst: `D_algo_bare_4m` 6/256 validation clean, 3 of them
  addition). Fixed by excluding from the pair pool every unordered operand
  pair in the four regenerated, order-hash-verified training files
  (`D_algo_op`, `D_translate_mix`, `D_inst_bare`, `D_algo_bare_4m`;
  3,129,816 pairs) before drawing; `leakage_report.json` now shows 0 for
  clean and counterfactual in both sets.
- 2026-10-01: implementation choices not in the plan. Discovery-half scores
  are means of the full-set per-example EAP-IG scores over each half (same
  numbers as rerunning on the half, at a third of the cost). Random circuits
  are uniform over all 195,865 edges with fixed seeds, shared across models.
  Real patching of the top 20 edges is measured on the discovery set (where
  the scores come from). If any model's digit-split share exceeds 1%, the
  excluded problems are taken from `sanity.json` per model (not yet needed).

- 2026-10-01: checkpoints now on HF as `podhajskimarcin/<run_id>` (runs/<rid>/model/, fp32, 4.94 GB each; configs verified 16L/32q/8kv/2048/tied). Ready to run.
