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
`results_large/circuit_change.json` and `results_large/figures/cc_*.png`
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
- 2026-10-01: first run done (box 53715523, results on HF `mhieuuu/geode-internals:results/eapig_check/`). No model passes all five tests at 0.1–1% (max 2/5). Sanity EM decode bug (post-EOS text glued on) fixed in 24b3d48 and sanity rerun before the push.
- 2026-10-07: owner asked for the pre-registered "no circuit at ≤ 1%" branch at 2%, 5% and 10% (10% added by owner), same plan otherwise: reuse first-run `scores.pt` + `sanity.json` (same draw counts), outputs to `results_large/` / HF `results/eapig_check_large/`. `--sizes` / `SIZES`, `RES_NAME`, `HF_SUBDIR`, `BATCH_SIZE` added. Runbook for the executing agent: `HANDOFF-sizes-2-5-10.md`.
- 2026-10-08 01:32 UTC: 2/5/10% rerun launched on box 54750972 (RTX 3090 24 GB, 8 cores, Spain; 3-box race, the China box was destroyed for repeated pip download drops). Seeded `results_large/` from first-run `scores.pt` + `sanity.json` (EM 0.953125; `sec_per_circuit_eval` ≈ 4.4 s for all four, so draw counts unchanged); all four models logged `skip ... sanity/score`. Batch-size probe on `elicit_child` sanity (scratch dir): bs 16/32/64 → eval 4.81/4.52/4.39 s, peak VRAM 8.3/11.3/17.2 GB of 24.6, m_full/m_empty equal to 1e-5. Chose **bs 32**: bs 64 is only 3% faster than 32 and its 70% peak leaves little headroom for evaluate; an OOM costs a ~40-min model restart. TinyStories path uses a fixed bs 8 regardless.
- 2026-10-08 01:36 UTC: utilization check during `elicit_parent` evaluate: GPU 100%, VRAM 11.2/24.6 GB, run.py ~100% of one core, load 1.5/8, RAM 3.7/32 GB, swap 942 MB (pre-existing before the run, not growing). Temp peaked 85 °C then settled 81–83 °C with throttle flags alternating SW power cap (0x4) / SW thermal (0x20) at the 300 W limit, SM clock ~1.4 GHz (= 3090 base), fan 62–66%, no HW thermal flag. Treated as normal sustained 3090 load (a smaller batch would not cut power draw at 100% util); no change.
- 2026-10-08 01:53 UTC: checked whether "100% GPU util" is real saturation (nvidia-smi util only means a kernel was resident each sample). `nvidia-smi dmon`: power 298–299/300 W (at cap), memory-controller busy 78–100%, clock ~1.4 GHz power/thermal-limited. The job is memory-bandwidth bound and the card is genuinely saturated; consistent with the bs probe (4× batch → only 9% faster). No side-by-side process (it would split the same power/bandwidth).
- 2026-10-08 04:39 UTC: rerun DONE (`EXIT=0`, box log `verify: missing on hub: none`). Evaluate times at bs 32: `elicit_parent` 1866 s, `fmt_parent` 1849 s, `elicit_child` 3720 s, `teach_child` 3767 s; launch → push 3 h 07 min (vs ~2–2.5 h estimated; children run 100 draws). Verified from the laptop: 24 files under HF `results/eapig_check_large/`, every `evaluate.json` has k = 3917/9793/19586, draws 50 (parents) / 100 (children), TinyStories 10, gate pass, selected size none; seeded `sanity.json` files untouched (01:29 timestamps). Box 54750972 destroyed. `compare.py` ran on the box (summary.md + figures pushed); not re-run on the laptop since it needs the four 200 MB `scores.pt` and gives the same output. Results above.
- 2026-10-08: owner: treat the consistency and specificity failures as a test-design question for later, and for now only examine how the 10% circuit changes from parent to child on each route (CPU only, from saved `scores.pt`; only final checkpoints exist, so "over time" = the two endpoints). Two design notes from the saved numbers, not yet acted on: consistency coverage is capped at n_shared/k (0.26 `elicit_child`, 0.24 `teach_child` at 10%; at or below 0.70 at all 7 sizes, best 0.699 for `elicit_child` at 0.1%), and removing even the top 0.1% floors both addition and copy (relative drop 2.0 at all 7 sizes), so specificity cannot separate the tasks as built. Runbook: `HANDOFF-circuit-change.md`.
- 2026-10-08: parent → child circuit change at 10% (`HANDOFF-circuit-change.md`), CPU only on the laptop from the HF `results/eapig_check_large/` `scores.pt` (pulled single-threaded into `results_large/`). New script `circuit_change.py`; edge Jaccards reproduce `compare.json` (asserted). Added beyond the runbook: a cross-half sign check (top-n by |score| in one half, sign read in the other; null 0.5) and |score| versions of the overlap stats, because the child's top edges turned out to sit at the parent's very top or very bottom in about equal shares. Finding: parent edge signs are at or near chance (0.52 `fmt_parent`, 0.63 `elicit_parent` vs ≥ 0.99 children), so the signed parent → child comparison mostly measures parent sign noise. No elicit-vs-teach separation. Results block above.
- 2026-10-08: owner picked three follow-ups from Wang et al.'s fine-tuning circuit paper: #1 KL-metric scoring (gives the parents a defined circuit), #2 symbol-form circuits (`elicit_parent` performs there), #4 edge-vs-node change rates with a random-set null. GPU: one box with 2× RTX 3090 (owner); ~1 h, ~$0.35–0.45. Not started; the owner starts it in a fresh session. Runbook: `HANDOFF-kl-symbol-nodeedge.md`. Two proposed choices still need a one-line confirm: score children on symbol too, 20/10 random draws. (Owner, same day: the sign-stable gate was dropped from the plan as not important.)
