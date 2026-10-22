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

Two runs, same models, data, scores and tests. The first run (2026-10-01,
HF `results/eapig_check/`) covered 0.1–1% of the 195,865 edges; the
2026-10 rerun (HF `results/eapig_check_large/`) covered 2%, 5% and 10%
(k = 3917 / 9793 / 19586). Labels: **f** = (m(circuit) − m(empty)) /
(m(full) − m(empty)), where m is the mean logit difference (LD) on the 256
validation problems; **tests** = how many of the five frozen tests pass
(sufficiency, equivalence, partial necessity, consistency, specificity).

### 2026-10 rerun at 2/5/10% (with the first run's 0.1–1% for context)

**No model passes all five tests at any of the 7 sizes. No size is selected
for any model.** The best is `elicit_child` at 3/5 from 5% up.

f per size (tests passed in brackets):

| model | EM | 0.1% | 0.2% | 0.5% | 1% | 2% | 5% | 10% |
|---|---|---|---|---|---|---|---|---|
| elicit_parent | 0.000 | 0.000 (2) | 0.022 (2) | −0.002 (1) | 0.017 (2) | 0.052 (2) | 0.082 (2) | 0.113 (2) |
| elicit_child | 0.953 | 0.003 (2) | 0.065 (2) | 0.366 (2) | 0.725 (2) | 0.899 (2) | 0.963 (3) | 0.974 (3) |
| fmt_parent | 0.000 | −0.764 (1) | −0.801 (1) | −0.533 (1) | 0.382 (2) | 1.508 (2) | 1.868 (1) | 2.175 (1) |
| teach_child | 0.145 | 0.023 (2) | 0.177 (2) | 0.419 (2) | 0.609 (2) | 0.748 (2) | 0.784 (2) | 0.779 (2) |

- **Random band.** 100 random same-size circuits give f ≈ 0 for both
  children at every size (95th percentile ≤ 0.0001), and ≤ 0.001 for
  `elicit_parent`. So every child f above is far outside chance;
  sufficiency and partial necessity pass for both children at every size.
- **Equivalence** (LD of the circuit within ±10% of m(full), TOST): `elicit_child`
  now passes at 5% and 10% (mean LD gap −3.16 and −2.22 against a bound of
  ±4.24) and fails at 2% (−8.52). `teach_child` fails at every size: its f
  plateaus at about 0.78 (gap −12 to −14 against ±2.80).
- **Consistency fails for every model at every size**, always on coverage
  (how much of the shared edge set each problem's own circuit contains;
  needs ≥ 0.70). `elicit_child` 0.34 / 0.22 / 0.16, `teach_child` 0.25 /
  0.18 / 0.14 at 2 / 5 / 10%. Coverage falls as the circuit grows. The other
  half of the test (ablating the shared edges beats random) passes everywhere.
- **Specificity fails for every model at every size**, on the ratio: ablating
  the circuit wipes out the copy task as much as addition (relative drops
  ≈ 2.0 for both, ratio ≈ 1 against the 3× bar). Both children's LD goes from
  +m(full) to about −m(full) on both tasks once the top 2% is removed. The
  circuit is shared machinery, not an addition-specific path.
- **Parents.** `elicit_parent` stays at f ≤ 0.11 (EM 0, m(full) 0.21).
  `fmt_parent`'s f is noise (m(full) − m(empty) = 0.002); it never passes
  more than 2/5, so the "format parent passes" broken-pipeline branch did not
  fire. Both parents' split-half ceilings are flagged unreliable.
- **Parent circuit inside the child** (f of the parent's own top-k edges run in
  the child): `elicit` 0.001 at every size, `teach` 0.013–0.021. Above random
  (≈ 0) but ~1–3% of the child's own f. The parent's circuit does not carry
  the child's behavior on either route.

Route overlaps (edge Jaccard, parent → child; chance = random sets of the same size):

| size | elicit | teach | children vs children | chance | parent split-half ceiling (elicit / fmt) |
|---|---|---|---|---|---|
| 2% | 0.226 | 0.198 | 0.370 | 0.010 | 0.237 / 0.186 |
| 5% | 0.197 | 0.177 | 0.285 | 0.026 | 0.220 / 0.171 |
| 10% | 0.179 | 0.162 | 0.231 | 0.053 | 0.207 / 0.174 |

- Both routes overlap 3–23× above chance, and about as much as each parent's
  own split-half ceiling allows (overlap / parent ceiling 0.86–1.07). The two
  routes are within 0.03 of each other at every size: **no elicit-vs-teach
  separation** in parent → child overlap. The two children overlap more with
  each other than either does with its parent.
- Node Jaccard is 0.88–0.997 at these sizes: nearly every node is touched, so
  it no longer says anything.

**Accuracy gap — FLAGGED.** `elicit_child` EM 0.953 vs `teach_child` 0.145
(gap 0.81). Child-vs-child f, sufficiency and equivalence are not a fair
comparison; the matched subset (problems both get right) cannot be rebuilt
from saved artifacts because m(empty) is not stored per example.
Single training seed per child; every elicit-vs-teach difference above is
single-run.

### 2026-10 circuit change at 10% (parent → child)

**Exploratory, CPU only**, from the saved `scores.pt` (no new model runs, no
frozen test touched). Script `circuit_change.py`; output
`results_large/circuit_change_<frac>.json` and `results_large/figures/cc_*_<frac>.png`
(local, gitignored). Circuit = top 10% by signed mean score (k = 19,586).
Edge Jaccards reproduce `compare.json` exactly (asserted).

**Read this first: the parents' circuits are mostly sign noise.** Pick each
model's top 1,000 edges by |score| in one half of the discovery set and check
their sign in the other half (null = 0.5):

| model | sign agreement | split-half Spearman, signed | split-half Spearman, \|score\| | 10% ceiling |
|---|---|---|---|---|
| elicit_parent | 0.63 | 0.12 | 0.45 | 0.207 |
| elicit_child | 0.999 | 0.41 | 0.68 | 0.529 |
| fmt_parent | 0.52 | 0.01 | 0.58 | 0.174 |
| teach_child | 0.99 | 0.55 | 0.73 | 0.548 |

The parents agree on *which* edges are large, but the sign of those edges is
at or near chance. A signed top-10% circuit in a parent is therefore "large
edges whose sign happened to come out positive". Everything below inherits
this.

**How much changed** (a → b; chance Jaccard 0.053):

| pair | edge Jaccard | Spearman, signed | Spearman, \|score\| | \|score\| top-10% Jaccard | b's mass on kept edges |
|---|---|---|---|---|---|
| elicit (elicit_parent → elicit_child) | 0.179 | 0.02 | 0.40 | 0.31 | 0.52 |
| teach (fmt_parent → teach_child) | 0.162 | 0.00 | 0.39 | 0.31 | 0.54 |
| ref: elicit_child vs teach_child | 0.231 | 0.07 | 0.44 | 0.30 | 0.95 |
| ref: elicit_parent vs fmt_parent | 0.129 | 0.01 | 0.32 | 0.30 | 0.41 |

- Signed scores are uncorrelated parent → child (Spearman ≈ 0) on both
  routes. |score| correlates about 0.4 in **every** pair, including the two
  parents with each other. Which edges are large is shared by all four
  models, so it reflects the architecture and inputs, not the route.

**What became important.** For each child's top 1,000 edges, where they sat
in the parent:

| pair | in parent's circuit (top 10%) | in parent's bottom 10% (most negative) | in parent's \|score\| top 10% |
|---|---|---|---|
| elicit | 0.51 | 0.46 | 0.93 |
| teach | 0.49 | 0.47 | 0.93 |
| ref: children | 0.94 | 0.04 | 0.97 |
| ref: parents | 0.48 | 0.48 | 0.90 |

- 93% of each child's top edges were already large in the parent. About half
  had a positive sign there and half a negative one, on both routes and in
  the parent-vs-parent reference alike. This is the coin-flip parent sign,
  not a route effect.
- The top "added" edges on both routes are layer-0 / early-MLP edges
  (`embed→a0.v0`, `embed→a0.v2`, `m0→m1.in`, `m0→m2.in`) that were at
  the parent's very bottom (rank ≈ 195,860 of 195,865). They are large-magnitude
  edges whose parent sign was negative. They are not new edges.

**Shape.** Fine-tuning concentrates the positive score mass. Half of it sits
on 35 edges in `elicit_child` and 48 in `teach_child`, against 1,592
(`elicit_parent`) and 1,431 (`fmt_parent`). 90% sits on about 1,070–1,090
edges in the children and 28k–33k in the parents. The two children's curves
nearly overlap (`cc_mass_concentration.png`). The parent side is weak signal
plus sign noise, so read this as "the children have a sharp circuit and the
parents don't", not as a measured change of shape.

**Where (counts, added − dropped).** By receiver type, elicit nets +1,738
value-input edges and −2,266 query-input edges; teach nets +83 and +591.
Both routes gain MLP-input edges (+1,210 elicit, +886 teach). The layer
heatmaps (`cc_layer_added_minus_dropped.png`) differ by route. Elicit's
layer 0 sends +1,312 more edges, layers 1–3 send about 800 fewer each, and
receivers 14–15 lose 1,349 and 1,189. Teach's layers 0–5 send fewer edges
(−605 at layer 0), layers 6–7 send +510 and +552, receivers 4–5 lose
(−372, −444) and receivers 9–11 gain (+465 at 9). The parent-vs-parent
reference shows shifts of the same size (layer 0 sends +1,970; receivers
13–15 lose 539–901). That reference has no fine-tuning in it, so these maps
do not exceed noise. The "dropped" half of each map is the parent's
positive-by-chance edges.

**Elicit vs teach.** No separation on any whole-circuit number. The two routes
are within 0.02 on Jaccard, Spearman, kept-mass share and every rank-shift
share. Both match the parent-vs-parent reference. At 10%, the parent → child
comparison measures the parents' sign noise more than any circuit change.
Caveats: both parents have EM 0; `fmt_parent`'s m(full) − m(empty) is 0.002;
the children's EM differs (0.953 vs 0.145); there is one training seed per
child.

**Same analysis at 2% and 5% (2026-10-08, owner request).** Same script,
`--frac 0.02` / `--frac 0.05`; Jaccards match `compare.json` (asserted).
The 10% reading holds at both sizes.

| | 2% (k 3,917) | 5% (k 9,793) | 10% (k 19,586) |
|---|---|---|---|
| edge Jaccard elicit / teach | 0.226 / 0.198 | 0.197 / 0.177 | 0.179 / 0.162 |
| ref: children / parents | 0.370 / 0.154 | 0.285 / 0.137 | 0.231 / 0.129 |
| chance | 0.010 | 0.026 | 0.053 |
| split-half ceiling: elicit_parent / fmt_parent | 0.237 / 0.186 | 0.220 / 0.171 | 0.207 / 0.174 |
| split-half ceiling: elicit_child / teach_child | 0.806 / 0.728 | 0.653 / 0.607 | 0.529 / 0.548 |
| child's mass on kept edges, elicit / teach | 0.51 / 0.52 | 0.52 / 0.54 | 0.52 / 0.54 |
| child top-1000 in parent's circuit, elicit / teach | 0.44 / 0.42 | 0.49 / 0.47 | 0.51 / 0.49 |
| child top-1000 in parent's most negative k, elicit / teach | 0.41 / 0.41 | 0.45 / 0.46 | 0.46 / 0.47 |

- Each route's overlap sits at its parent's split-half ceiling at every size.
  That is the most the parent's noise allows, so it is not a measure of how
  much was kept.
- The routes are within 0.03 of each other on every row, and both match the
  parent-vs-parent reference. **No elicit-vs-teach separation at 2% or 5%
  either.**
- The child's top added edges are the same at all three sizes. They are
  layer-0 and early-MLP edges at the bottom of the parent's ranking (e.g.
  `embed→a0.v0`, parent rank 195,865 of 195,865).
- **Two receiver-type shifts repeat at every size. They are not tested
  against a null, so treat them as leads only.**
  - Both routes gain MLP-input edges: elicit +646 / +1,054 / +1,210, teach
    +394 / +696 / +886, against parent-vs-parent +85 / +67 / +53.
  - Elicit also gains value-input edges: +276 / +973 / +1,738. Teach gains
    +172 / +73 / +83, and parent-vs-parent +257 / +460 / +652.

### 2026-10-08 KL metric, op-form parent circuit, functional reuse (exploratory)

Box 54866627 (2× 3090). HF `results/eapig_kl_symbol/` (scores, edge, functional) and
`results/eapig_kl_tests/frac_{0.02,0.05}/`. Deck: claude.ai/artifact/RwoygAuWM2Jvzxrck2cRae.

**Behavior (greedy EM).** `elicit_parent`: op-form addition 0.850 (4x4 0.403), NL add/sub
0.0000 (2.7% of outputs are integers). `elicit_child`: op add 0.886, NL 0.986.
`teach_child`: NL 0.403. `fmt_parent`: 0.000. Symbol subtraction not measurable:
`a - b = ` is not a token-prefix of `a - b = -c` (" -" merges), 25,004 rows.

**KL fixes the parent noise.** Split-half Jaccard of parent circuits at 2%: LD 0.19–0.24,
KL 0.67–0.68. `fmt_parent` fails the KL gate (KL(full‖∅) 0.005 nats).

**Headline: the elicit child reuses the parent's op-form circuit.**

| top-2% circuits | KL Jaccard | LD Jaccard | functional f share, KL / LD |
|---|---|---|---|
| `elicit_parent` op → `elicit_child` word | **0.661** | **0.663** | **0.96 / 0.96** |
| `elicit_parent` op → `teach_child` word (control) | 0.367 | 0.362 | 0.53 / 0.64 |
| `elicit_parent` word → `elicit_child` word | 0.310 | 0.226 | 0.28 / 0.00 |
| `fmt_parent` word → `teach_child` word | 0.299 | 0.198 | 0.31 / 0.02 |
| split-half ceiling | 0.71–0.75 | 0.76–0.81 | |

Functional share = f of the child on word val pairs with only that circuit live, over the
child's own circuit f (`functional_reuse.py`; random sets ≈ 0). At 5%: elicit 0.98 / 0.97,
teach control 0.61 / 0.78. `elicit_child` op vs word circuits: 0.79 (one circuit for both
forms). At 1%, op-parent → child changes 16% of edges (word parent: 52–69%).

**Against the word-task parent the routes look the same** (KL edges replaced at 2%: 53%
elicit, 54% teach; 87–89% of child weight on kept edges; nodes change ~3× more than the
N3 null at 1%, both routes). This comparison cannot see reuse: the parents have no
word-task skill.

**KL tests (2% and 5% same pattern).** Sufficiency + partial necessity pass for all three
gated models; equivalence passes for `elicit_parent` and `elicit_child` (leftover KL 0.63
/ 3.37 nats vs bounds 1.98 / 4.23 at 2%), fails for `teach_child` (7.91 vs 2.78);
consistency fails (coverage 0.21–0.28); specificity fails because removing the circuit
floors both addition and copy (relative damage 1.00). No circuit passes all five.

### 2026-10-21 translate-then-solve, step 0 (gate FAILED, no circuit scored)

Box 57088349 (1× 3090), code 31a9342, HF `results/eapig_translate/`. Plan:
`HANDOFF-translate-circuit.md`. Rows: 4x4 '+' leak-clean, n 4930.

| model | rewrite exact | bare op `a + b = ` EM | single-sequence translate EM |
|---|---|---|---|
| `elicit_parent` | 0.913 (4503/4930) | 0.405 (on the 4503) | **0.000** |
| `elicit_child` | — | — | 0.711 |
| `teach_child` | — | — | 0.000 |

Translate prompt = `What is the sum of a and b?\n` + `a + b = ` (gold rewrite).
The parent outputs a well-formed integer of the right size with wrong digits
(e.g. 7465+8497: bare op 15962 ✓, translate 8375); teacher-forced log-prob of the
gold answer −18.5 nats vs −0.94 bare. KL(full‖∅) 17.2 nats, so the KL half of the
gate passes; the EM half (≥ 0.5× bare-op) fails. Same picture as the θ0
sequence-start lock (fewshot_diag): the parent's addition only fires at the
start of a sequence. The parent's route is two-pass, so a single-sequence
translate circuit does not exist in it. Per the plan: stopped before scoring;
path B (two-pass rewrite circuit, needs `kl_2tok` generalised in geode) only on
owner go. Side fact: `elicit_child` does solve the single-sequence form (0.71),
`teach_child` does not (0.00).

### 2026-10-21 addition at small sizes (0.1–5%), KL tests + overlap + functional reuse

Box 57127284 (4× RTX 3090, ~$2.8/h), code 3f3c6b9, 25 min wall for the jobs. HF
`results/eapig_kl_tests_small/` (16 files: `<tag>/evaluate.json`, `functional/`,
`overlap/edge_node.json`, `box_check/`, `run_log.txt`). Runbook `HANDOFF-small-sizes.md`.
Same KL scores as the 2026-10-08 block (no rescoring); circuit = top-k by signed mean,
k = 196 / 392 / 979 / 1,959 / 3,917 / 9,793. `elicit_parent` on op form (`a + b = `),
the others on NL. Every patched pass used the new sparse path. The box check matched the
dense path to 3.8e-6 nats (mean KL; tolerance 1e-4) at ×3.7 speed (0.1%) and ×2.0 (5%).
Three read-only code reviews during the run found no bug.

**Headline: the small sizes repeat the 2–5% result.** The elicit child's circuit is mostly
the op-form parent's circuit at every size down to 0.1%. The teach control stays at about
a third.

Overlap (signed edge Jaccard; equals the step-0 table in the handoff at every size;
chance = 95th pct of 1,000 random same-size pairs):

| pair | 0.1% | 0.2% | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|---|
| ep op → ec word (elicit) | **0.766** | **0.727** | **0.718** | **0.731** | 0.661 | 0.497 |
| ep op → tc word (control) | 0.395 | 0.320 | 0.307 | 0.343 | 0.367 | 0.270 |
| fp word → tc word (teach route) | 0.425 | 0.338 | 0.312 | 0.308 | 0.299 | 0.252 |
| ep word → ec word | 0.420 | 0.315 | 0.323 | 0.318 | 0.310 | 0.267 |
| ec op → ec word | 0.840 | 0.802 | 0.820 | 0.851 | 0.787 | 0.637 |
| ec word → tc word | 0.436 | 0.331 | 0.319 | 0.351 | 0.377 | 0.284 |
| split-half ceiling ep op / ec word | 0.885 / 0.903 | 0.862 / 0.853 | 0.833 / 0.833 | 0.806 / 0.819 | 0.707 / 0.747 | 0.515 / 0.569 |
| split-half ceiling fp word / tc word | 0.840 / 0.840 | 0.798 / 0.766 | 0.758 / 0.769 | 0.720 / 0.754 | 0.679 / 0.675 | 0.617 / 0.530 |
| chance | 0.003 | 0.003 | 0.005 | 0.007 | 0.012 | 0.027 |

- Elicit overlap / parent ceiling = 0.84–0.91 at 0.1–1% (0.93–0.97 at 2–5%). The control
  is 0.37–0.52 of the same ceiling, and the teach route 0.41–0.51 of `fmt_parent`'s.
- Edges do not change more than nodes beyond the N3 null for any route pair at any
  size (`edges_more_than_nodes_n3` false everywhere).

Functional reuse (child on NL val pairs with only that circuit live; share = f / child's
own f; 20 random sets give f ≈ 0, max 2e-7):

| child, circuit | 0.1% | 0.2% | 0.5% | 1% | 2% † | 5% † |
|---|---|---|---|---|---|---|
| `elicit_child` own f | 0.007 | 0.091 | 0.392 | 0.782 | | |
| elicit: ep op share | (1.26) | **0.92** | **0.86** | **0.89** | 0.96 | 0.98 |
| elicit: ep word share | (0.58) | 0.55 | 0.49 | 0.30 | 0.28 | |
| `teach_child` own f | 0.003 | 0.079 | 0.314 | 0.516 | | |
| teach: ep op share (control) | (0.48) | 0.25 | 0.32 | 0.42 | 0.53 | 0.61 |
| teach: fp word share | (1.31) | 0.25 | 0.24 | 0.29 | 0.31 | |

† From the 2026-10-08 block. At 0.1%, own f ≤ 0.007, so shares there are ratios of noise
(bracketed); read from 0.2% up.

**KL circuit tests.** f (tests passed of 5):

| model (task) | 0.1% | 0.2% | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|---|
| `elicit_parent` (op) | 0.015 (2) | 0.094 (2) | 0.369 (2) | 0.722 (2) | 0.891 (2) | 0.939 (3) |
| `elicit_child` (NL) | 0.007 (2) | 0.091 (2) | 0.392 (2) | 0.782 (2) | 0.920 (3) † | 0.971 (3) † |
| `teach_child` (NL) | 0.003 (2) | 0.079 (2) | 0.314 (2) | 0.516 (2) | 0.715 (2) † | 0.743 (2) † |
| `fmt_parent` (NL) | gate fail | | | | | |

† 2026-10-08 runs (`results_kltests_{02,05}`), same frozen code.

- **No circuit passes all five tests at any size.** Sufficiency and partial necessity pass
  everywhere. Random circuits score f ≈ 0 by construction (a random edge set almost never
  holds a full embed → logits path), so these two passes say little on their own.
- **Equivalence** (leftover KL within 10% of KL(full‖∅)): fails at every size up to 1% for
  every model. `elicit_parent` op passes at 5% (2.05 vs bound 3.34 nats) and just misses
  at 2% (3.62). This is the first test of the op-form parent circuit; the
  2026-10-08 `elicit_parent` tests ran on the NL task (KL(full‖∅) 19.8 nats there,
  33.4 on op).
- **Consistency fails everywhere.** Coverage is below the 0.70 bar and falls with size:
  `elicit_child` 0.50 → 0.32, `teach_child` 0.33 → 0.23, `elicit_parent` 0.44 → 0.16.
- **Specificity fails everywhere.** Removing the circuit floors both addition and copy
  (relative damage ≈ 1.00 on both). This matches 2–5%: the top edges are shared machinery.
- **Real patching vs EAP-IG score** (top 20 edges): Spearman 0.80 `elicit_child`, 0.47
  `teach_child`.
- `fmt_parent` fails the KL gate (KL(full‖∅) 0.005 nats), so only sufficiency and necessity
  were computed. Its f values (−0.17 to 0.38) are noise.

Caveats:
- Random baselines are uniform over all edges, not type-matched, and near zero by
  construction, so "beats random" carries little information. The f values, equivalence,
  overlap vs ceiling and the functional shares carry the result.
- `elicit_parent`'s specificity compares op-form addition with the NL copy task (the copy
  set is NL for every model; parent copy KL(full‖∅) is only 2.3 nats). Its specificity
  result is not comparable to the children's.
- `elicit_child`'s `evaluate.json` has no `parent_in_child` section; `teach_child`'s does
  (fp word). Running it for `elicit_child` would load the NL parent. Compare the two arms
  through `functional_reuse.py`; `compare.py` plot 5 would show only the teach route.
- Single training seed per child. The accuracy gap stands (NL EM 0.986 vs 0.403), so
  child-vs-child f is not a matched comparison.

### 2026-10-22 edge-level CMAP, child → parent, 0.1–5%

Box 57225900 (2× RTX 3090, ~$1/h; race of 3), code 47a6305, 12 per-size jobs on a queue
over both GPUs (GPU util 86–100%), ~13 min wall per pass. Total spend ~$1.5. HF
`results/eapig_cmap/` (`{elicit,teach}.json/.npz`, `shards/`, `summary.md`, `run_log.txt`).
Runbook `HANDOFF-cmap.md`; script `cmap.py`, figure `plot_cmap.py` →
`figures/cmap_recovery.png`. Method after Prakash et al. 2024 (arXiv 2402.14811), at the
edge level: the parent runs on the 256 NL val prompts with every edge in a mask carrying
the CHILD's node output; everything else is the parent. recovery = 1 − KL(child‖patched) /
KL(child‖parent) (0 = parent, 1 = child); acc = two-token teacher-forced accuracy.
Circuits = the 2026-10-08 KL scores (top-k by signed mean), k = 196 … 9,793.

Runtime identity checks pass on the real models (max 5e-5 nats): all-kept = parent,
child self-patch = child, sparse = dense, all embed edges patched = parent on the child's
embeddings, all logits edges patched = all edges patched. 71 CPU property tests. A
read-only math review found no bug.

**Headline: on the elicit route, the parent can use what the child's circuit edges carry;
on the teach route it cannot, and they push it away from the child.** The transferable
change sits on the op-form edges the child reuses from the parent.

Baselines: elicit KL(child‖parent) 23.8 nats, acc parent 0.00 / child 0.95; teach 16.3
nats, acc 0.00 / 0.14. All edges patched: elicit 1.000 (acc 0.96), teach 0.838 (acc 0.10).

Recovery (acc of the patched parent in brackets); (b) = child circuit minus edges into
logits, the main read:

| route, mask | 0.1% | 0.2% | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|---|
| elicit (a) child circuit | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| **elicit (b) minus logits** | **0.768** (0.32) | **0.810** (0.45) | **0.903** (0.60) | **0.917** (0.64) | **0.954** (0.75) | **0.975** (0.86) |
| elicit (d) op-form parent circuit, minus logits | 0.702 (0.27) | 0.841 (0.48) | 0.878 (0.55) | 0.917 (0.66) | 0.958 (0.77) | 0.971 (0.84) |
| elicit (b) ∩ (d) | 0.710 (0.27) | 0.804 (0.41) | 0.869 (0.55) | 0.896 (0.60) | 0.943 (0.72) | 0.966 (0.79) |
| elicit (b) minus (d) | 0.115 (0) | 0.029 (0) | 0.111 (0) | 0.115 (0) | 0.131 (0) | 0.175 (0) |
| elicit null: same strata, outside (b), p95 | 0.025 | 0.128 | 0.227 | 0.239 | 0.236 | 0.210 |
| teach (a) child circuit | 0.843 | 0.837 | 0.833 | 0.836 | 0.837 | 0.838 |
| **teach (b) minus logits** | **−0.830** (0) | **−0.896** (0) | **−1.060** (0) | **−1.140** (0) | **−1.187** (0) | **−1.116** (0) |
| teach null: same strata, outside (b), p95 | 0.060 | 0.057 | 0.045 | 0.125 | 0.172 | 0.183 |
| uniform random minus logits, p95 (elicit / teach) | 0.057 / 0.007 | 0.015 / 0.003 | 0.080 / 0.010 | 0.135 / 0.024 | 0.292 / 0.041 | 0.268 / 0.084 |

Sizes: (b) = k minus 6–221 logits edges; (b) ∩ (d) = 162 / 319 / 790 / 1,589 / 2,975 /
6,308 edges, (b) minus (d) = 26 / 62 / 161 / 295 / 781 / 3,264. 95% bootstrap CIs over
prompts are ±0.01–0.05 on every (b)/(d) value (in `summary.md`).

Read against the pre-registered branches (HANDOFF-cmap.md):
- **Elicit (b) is far above its random bands and above teach (b) at every size** ⇒
  consistent with elicitation: the change sits on the child's circuit edges and the
  parent's downstream weights can turn it into the child's answer (patched-parent acc
  0.32–0.86 from 0.00).
- **(a) is the logits shortcut, as feared**: 1.000 / 0.84 on both routes, equal to the
  all-edges ceiling. Even uniform random sets with logits edges reach 0.95 / 0.74 at 5%.
  Read only (b).
- **Teach (b) is not at the random level but far below it.** The patched parent moves
  AWAY from the child (KL grows 1.8–2.2× over the unpatched parent; log p(answer)
  −33 to −38 vs −19 unpatched), while same-strata edges outside the circuit move it
  slightly toward the child (0.04–0.18). Stronger than "the format parent cannot use the
  child's signals": its downstream turns them into something else.
- **(d) ≈ (b), and the split is not circular**: (b) ∩ (d) carries nearly all of (b)
  (0.71–0.97) with 0.66–0.86 of its edges, while (b) minus (d) stays at 0.03–0.18 with acc
  0, at the null's level (p95 0.03–0.24), even at 5% where it has 3,264 edges (more than (b) ∩ (d) at
  1%, which recovers 0.90).

Null design (found during the run): the first type-matched null (same count per
(upstream kind+layer, receiver kind+layer) stratum, drawn from the whole stratum) shared
40–71% of its edges with (b), because the circuit fills many small strata (e.g. `m_i →
m_j.in` is one edge). It scored ≈ (b) on both routes (elicit 0.77–0.98, teach −0.77 to
−1.25), so it is not a null and is not quoted above. A disjoint version (same strata,
only edges outside (b)) was added and the whole run repeated (same numbers for every
other mask, max |Δ| = 0.0). It is smaller than (b) where strata are fully taken (71–7,264
edges vs 188–9,595), but size does not explain the gap: elicit (b) at 0.2% (381 edges)
recovers 0.81 vs ≤ 0.24 for any disjoint set up to 6,030 edges. That the overlapping
typed null reproduces (b) hints the effect is carried by the small-strata edges (MLP→MLP,
embed→*, layer-to-layer head groups) the circuit takes whole. Not tested.

Caveats:
- Single seed and one training run per child. The teach child is weak (acc 0.14), so its
  KL(child‖parent) mixes what it learned with what it got wrong.
- The circuits are EAP-IG scores on operand-swap counterfactuals, so they hold
  operand-sensitive edges. A constant fine-tuning change (template/routing) is invisible
  to the selection; low recovery means "not on the operand circuit".
- (b) still lets the child's signals into the last layers (e.g. `a14.h* → m15.in`); the
  parent's downstream share of the work is then small at the top of the stack. The
  receiving layer of (b) is ≥ 12 for only 10–26% of its edges (both routes).
- At 0.1% the child's own circuit has f ≤ 0.007 (small-sizes block), yet (b) recovers
  0.77; CMAP and faithfulness measure different things (child signals in the parent vs
  the child's own computation from corrupted inputs).
- Tied embeddings: the elicit "all edges" recovery is 1.000, so the final norm + unembedding
  barely moved in fine-tuning; for teach, 0.838.

## Global takeaways

[ ]

## Next steps

- Small sizes (0.1–5%), addition only: DONE 2026-10-21 (results block above). Nothing
  queued.
- Edge-level CMAP child → parent 0.1–5%: DONE 2026-10-22 (results block above). Possible
  follow-up, not queued: patch only the small-strata edges (b) takes whole, to test the hint
  that they carry the effect.

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
- 2026-10-01: first run done (box 53715523, results on HF `mhieuuu/geode-internals:results/eapig_check/`). No model passes all five tests at 0.1–1% (max 2/5). Sanity EM decode bug (post-EOS text glued on) fixed in 24b3d48 and sanity rerun before the push.
- 2026-10-07: owner asked for the pre-registered "no circuit at ≤ 1%" branch at 2%, 5% and 10% (10% added by owner), same plan otherwise: reuse first-run `scores.pt` + `sanity.json` (same draw counts), outputs to `results_large/` / HF `results/eapig_check_large/`. `--sizes` / `SIZES`, `RES_NAME`, `HF_SUBDIR`, `BATCH_SIZE` added. Runbook for the executing agent: `HANDOFF-sizes-2-5-10.md`.
- 2026-10-08 01:32 UTC: 2/5/10% rerun launched on box 54750972 (RTX 3090 24 GB, 8 cores, Spain; 3-box race, the China box was destroyed for repeated pip download drops). Seeded `results_large/` from first-run `scores.pt` + `sanity.json` (EM 0.953125; `sec_per_circuit_eval` ≈ 4.4 s for all four, so draw counts unchanged); all four models logged `skip ... sanity/score`. Batch-size probe on `elicit_child` sanity (scratch dir): bs 16/32/64 → eval 4.81/4.52/4.39 s, peak VRAM 8.3/11.3/17.2 GB of 24.6, m_full/m_empty equal to 1e-5. Chose **bs 32**: bs 64 is only 3% faster than 32 and its 70% peak leaves little headroom for evaluate; an OOM costs a ~40-min model restart. TinyStories path uses a fixed bs 8 regardless.
- 2026-10-08 01:36 UTC: utilization check during `elicit_parent` evaluate: GPU 100%, VRAM 11.2/24.6 GB, run.py ~100% of one core, load 1.5/8, RAM 3.7/32 GB, swap 942 MB (pre-existing before the run, not growing). Temp peaked 85 °C then settled 81–83 °C with throttle flags alternating SW power cap (0x4) / SW thermal (0x20) at the 300 W limit, SM clock ~1.4 GHz (= 3090 base), fan 62–66%, no HW thermal flag. Treated as normal sustained 3090 load (a smaller batch would not cut power draw at 100% util); no change.
- 2026-10-08 01:53 UTC: checked whether "100% GPU util" is real saturation (nvidia-smi util only means a kernel was resident each sample). `nvidia-smi dmon`: power 298–299/300 W (at cap), memory-controller busy 78–100%, clock ~1.4 GHz power/thermal-limited. The job is memory-bandwidth bound and the card is genuinely saturated; consistent with the bs probe (4× batch → only 9% faster). No side-by-side process (it would split the same power/bandwidth).
- 2026-10-08 04:39 UTC: rerun DONE (`EXIT=0`, box log `verify: missing on hub: none`). Evaluate times at bs 32: `elicit_parent` 1866 s, `fmt_parent` 1849 s, `elicit_child` 3720 s, `teach_child` 3767 s; launch → push 3 h 07 min (vs ~2–2.5 h estimated; children run 100 draws). Verified from the laptop: 24 files under HF `results/eapig_check_large/`, every `evaluate.json` has k = 3917/9793/19586, draws 50 (parents) / 100 (children), TinyStories 10, gate pass, selected size none; seeded `sanity.json` files untouched (01:29 timestamps). Box 54750972 destroyed. `compare.py` ran on the box (summary.md + figures pushed); not re-run on the laptop since it needs the four 200 MB `scores.pt` and gives the same output. Results above.
- 2026-10-08: owner: treat the consistency and specificity failures as a test-design question for later, and for now only examine how the 10% circuit changes from parent to child on each route (CPU only, from saved `scores.pt`; only final checkpoints exist, so "over time" = the two endpoints). Two design notes from the saved numbers, not yet acted on: consistency coverage is capped at n_shared/k (0.26 `elicit_child`, 0.24 `teach_child` at 10%; at or below 0.70 at all 7 sizes, best 0.699 for `elicit_child` at 0.1%), and removing even the top 0.1% floors both addition and copy (relative drop 2.0 at all 7 sizes), so specificity cannot separate the tasks as built. Runbook: `HANDOFF-circuit-change.md`.
- 2026-10-08: parent → child circuit change at 10% (`HANDOFF-circuit-change.md`), CPU only on the laptop from the HF `results/eapig_check_large/` `scores.pt` (pulled single-threaded into `results_large/`). New script `circuit_change.py`; edge Jaccards reproduce `compare.json` (asserted). Added beyond the runbook: a cross-half sign check (top-n by |score| in one half, sign read in the other; null 0.5) and |score| versions of the overlap stats, because the child's top edges turned out to sit at the parent's very top or very bottom in about equal shares. Finding: parent edge signs are at or near chance (0.52 `fmt_parent`, 0.63 `elicit_parent` vs ≥ 0.99 children), so the signed parent → child comparison mostly measures parent sign noise. No elicit-vs-teach separation. Results block above.
- 2026-10-08: owner picked three follow-ups from Wang et al.'s fine-tuning circuit paper: #1 KL-metric scoring (gives the parents a defined circuit), #2 symbol-form circuits (`elicit_parent` performs there), #4 edge-vs-node change rates with a random-set null. GPU: one box with 2× RTX 3090 (owner); ~1 h, ~$0.35–0.45. Not started; the owner starts it in a fresh session. Runbook: `HANDOFF-kl-symbol-nodeedge.md`. Two proposed choices still need a one-line confirm: score children on symbol too, 20/10 random draws. (Owner, same day: the sign-stable gate was dropped from the plan as not important.)
- 2026-10-08: owner added a full KL test rerun to `HANDOFF-kl-symbol-nodeedge.md` (step 6). It runs all five tests with KL in place of LD, on the KL-word circuits, at 0.1/0.2/1/2/5%, with the frozen draw counts. Exploratory, to `results_kltests/`. Estimate: ~60 GPU-min per size for all 4 models (~30 min per GPU on 2× 3090); ~3.5 h and ~$1.3–1.5 for the whole handoff. The KL equivalence bound and KL parent gate are proposed and not yet confirmed. Not started.
- 2026-10-08: reran the parent → child circuit-change analysis at 2% and 5% (CPU, seconds). Outputs are now named by size (`circuit_change_<frac>.json`, `cc_*_<frac>.png`); the 10% files were renamed to match. Same reading as 10%; block above.
- 2026-10-08: owner. Full KL test rerun, first pass = 2% and 5% only (run both even if 2% passes all five), then report and wait; 0.1 / 0.2 / 1% only on the owner's go. KL equivalence bound (KL(full ‖ C) < 0.10 × KL(full ‖ ∅), one-sided) and KL parent gate (KL(full ‖ ∅) > 0.1 nats + sufficiency somewhere) CONFIRMED. Handoff estimate is now ~2 h and ~$0.8–0.9.
- 2026-10-08: owner. New FIRST step S in `HANDOFF-kl-symbol-nodeedge.md`: symbol-form performance of all 4 models on all 50,000 `D_algo_eval_op` addition rows plus the same problems in word form (4x4 headline, leakage-clean 4x4 too, m(full)/m(empty) on the 768 circuit pairs). The owner decides on scoring the children on symbol after seeing it. A box agent rents the 2× 3090 and runs step S while the other agents build the definite KL parts in parallel; the box is kept for the main run (destroyed if the build is not pushed within 2 h). Not started; fresh session.
- 2026-10-08: owner. Symbol form is ELICIT ROUTE ONLY; the teach route never trained on symbols, so it stays on NL add/sub (the default for `fmt_parent`). The `elicit_parent` plan is pending the owner: no `elicit_parent` symbol job until decided. Step S now measures `elicit_child` (symbol add 50k + NL add/sub 100k), `fmt_parent` and `teach_child` (NL add/sub 100k each), with strict and |a−b|-lenient EM for NL subtraction. The handoff's box uses one shared GPU job queue.
- 2026-10-08: step S done (box 54866627, ~$0.45; results `results/eapig_perf_eval/`). Wave B launched with symbol off, then stopped by the owner at 15:44 for a new order: edge/node change for each metric first, then the KL circuit tests at 2%, then 5% (LD tests at 2/5% reused from `results_large/`, same frozen code). Owner then asked for `elicit_parent` on the op form vs `elicit_child` on word, and the functional test; both run (`functional_reuse.py`). Symbol subtraction dropped: its prompts are not token-prefixes. Everything pushed and verified from the laptop (`missing: none`), box destroyed; session spend ~$1.56. 0.1/0.2/1% KL tests NOT run (owner's go needed). Results block above.
- 2026-10-21: translate-then-solve step 0 failed (results block above). Owner: Path B skipped. Signed add/sub over all answer tokens considered, then dropped after a literature check (MIB, Nikankin et al. drop negative and multi-token answers; no standard for scoring the sign). Owner: keep the original 4x4 addition only. Its KL already covers every answer token (4x4 sums are always 2 tokens), so the 2026-10-08 op-form result is the reference and nothing new runs.
- 2026-10-21: owner asked for addition at smaller sizes (overlap + circuit tests). Overlap at 0.1/0.2/0.5/1% computed on CPU from the HF `results/eapig_kl_symbol/` scores (table in `HANDOFF-small-sizes.md`). Why tests are slow: ~304 patched passes per size per child (3 tests × 1 real + 100 random, + copy), 3–7 s each. Speedup planned, not built (owner: plan first); owner OK'd a 4-GPU box. Found: `--parent-tag elicit_parent` would load the word-form parent, so elicit_child runs without it and the op-form check comes from `functional_reuse.py`.
- 2026-10-21: small sizes BUILT and LAUNCHED (3f3c6b9). Patched-forward speedup in `geode/circuits/eapig.py` (sparse exceptions for on/off masks, shared input for receivers without exceptions, `last_n` logits; old path via `GEODE_EAPIG_DENSE=1`), `evaluate --task symbol` for `elicit_parent`, `run_box_small.sh` + `box_check_speed.py`. Box check on the real 1B (RTX 3090, 256 pairs): 0.92 vs 3.40 s/mask at 0.1% (×3.7), ~1.75 vs ~3.6 s/mask at 5% (×2.0), max mean |ΔKL| 3.8e-6 nats → every job used the new path. Box 57127284 (4× RTX 3090); edge_node overlap + nulls done in 49 s; tests running. Results go to HF `results/eapig_kl_tests_small/`; results block pending.
- 2026-10-21: small sizes DONE. All jobs exit 0 in ~25 min (elicit_parent 832 s, children 1,210–1,233 s, fmt_parent 438 s + functional 111/116 s); box pushed 16 files to HF `results/eapig_kl_tests_small/`, laptop watchdog verified them and destroyed box 57127284 (0 instances on team-tj). Overlap Jaccards equal the step-0 table at every size. Read-only reviews (sparse forward vs dense: 132 tests + 696-case CPU stress, worst |ΔKL| 1.3e-6; `last_n=2` checked on the real data files; live run + watchdog) found no bug. Results block above.
- 2026-10-22: CMAP child → parent BUILT + RUN (393455c … 47a6305). Owner: include (d), push, add a deck slide. Parallel build (tests / box script / math review by subagents). Box 57225900 (2× 3090); 12 per-size jobs over both GPUs. First pass showed the type-matched null shared 40–71% of edges with (b); added a disjoint null and reran everything (identical numbers otherwise). Results on HF `results/eapig_cmap/`, verified from the laptop; box destroyed; ~$1.5 total.
