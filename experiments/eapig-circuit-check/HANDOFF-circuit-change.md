# Handoff: how the 10% circuit changes from parent to child

Branch `eapig-circuit-check`. CPU only, on the laptop. **No GPU, no box, no
money.** Plan only until the owner says go. If you are reading this in a fresh
session, the owner has said go.

## Why

Two runs are done (PLAN.md Results). At 0.1–10% of the 195,865 edges, no model
passes all five frozen tests. The owner (2026-10-08) treats the consistency and
specificity failures as a test-design question for later. For now the only
question is:

> **How does the 10% circuit change from the parent (before fine-tuning) to
> the child (after fine-tuning), on the elicit route and on the teach route?**

"Over time" means the two endpoints. The teammate's HF repos
(`podhajskimarcin/<run_id>`) hold only the final checkpoint of each run, so
there is no trajectory within fine-tuning. Getting one would need
intermediate checkpoints plus GPU scoring, which is out of scope here.

## Labels (define once, use everywhere)

- **circuit**: the top 10% of edges by signed mean EAP-IG score,
  k = 19,586, `et.topk_edges(scores["mean"], k)`, exactly as `run.py` does.
- **elicit route**: `elicit_parent` → `elicit_child`.
  **teach route**: `fmt_parent` → `teach_child`.
- **kept / added / dropped**: edges in both circuits / only the child's /
  only the parent's.
- **score mass**: the sum of positive mean scores over a set of edges.
- **ceiling**: split-half agreement within one model, computed by comparing
  `mean_a` with `mean_b`. It is the most agreement noise allows. A
  parent→child number at the parent's ceiling means "no change we can detect".

## Inputs (already on HF, nothing to compute on GPU)

Pull these from `mhieuuu/geode-internals`, under `results/eapig_check_large/<tag>/`,
for the 4 tags. Run on the laptop with `env -u HF_TOKEN`, because the `.env`
token is stale. Save into `experiments/eapig-circuit-check/results_large/`,
which is gitignored. Download single-threaded, one file at a time: parallel
HF transfers freeze the laptop.

- `scores.pt` (~200 MB each). It holds `mean`, `mean_a`, `mean_b` (each
  (195,865,) fp32) and `per_example` (512 × 195,865 fp16, not needed here).
- `evaluate.json`, `sanity.json`, and also `../compare.json`.

The graph is model-free: `EdgeGraph(n_layers=16, n_heads=32, n_kv_heads=8)`,
as in `compare.build_graph_ctx()`. It gives `edge_names()`,
`edge_upstream_node()`, `edge_receiver_node()` and `edge_receiving_layer()`
(logits = layer 16).

## What to compute (one script: `circuit_change.py`)

Pairs: elicit route, teach route, plus two references.
`elicit_child` vs `teach_child` shows how different two fine-tuned models are.
`elicit_parent` vs `fmt_parent` shows how different two parents are.

1. **Reliability first.** For each model, compute the 10% ceiling (Jaccard
   of the top-k of `mean_a` vs `mean_b`) and the Spearman correlation of
   `mean_a` vs `mean_b` over all edges. The parents score weakly: EM 0,
   m_full 0.21 and 0.03. If a parent's halves barely agree, the parent's side
   of every comparison is mostly noise. Say so before reading anything else.
2. **How much changed.** Report these per pair:
   - edge Jaccard. It must reproduce `compare.json`: 0.179 elicit, 0.162
     teach, 0.231 children at 10%. That match is the correctness check.
   - Spearman correlation of the full score vectors.
   - the share of the child's score mass on kept edges.
   Read all three against chance (0.053 Jaccard at 10%) and against both
   models' ceilings.
3. **Where it changed.** Count kept, added and dropped edges by sender type
   (embed / attention head / MLP) and by receiver type (q / k / v / MLP /
   logits). Break them down by sending layer × receiving layer as well. Give
   each as counts and as score mass. Make one heatmap per route of added
   minus dropped, by layer × layer.
4. **What became important.**
   - The child's top 50 added edges, each with its rank in the parent.
   - The parent's top 50 dropped edges, each with its rank in the child.
   - For the child's top 1,000 edges, the distribution of rank shifts.
5. **Shape of the circuit.** Per model, report total positive score mass,
   and how many top edges hold 50% and 90% of it. This shows whether
   fine-tuning made the circuit more concentrated.
6. **Elicit vs teach.** Put the two routes' tables side by side, with the
   references next to them. No verdict beyond what the numbers support.

Outputs:
- `results_large/circuit_change.json`, plus PNGs in
  `results_large/figures/`. PNGs are gitignored; ship the script.
- A "2026-10 circuit change at 10% (parent → child)" block in PLAN.md
  Results.
- A dated Log entry in PLAN.md.

## Guardrails

- Do not touch the frozen tests, the thresholds, `evaluate.json`, or the HF
  `results/eapig_check*/` folders. Everything here is exploratory and is
  labelled as such.
- Caveats to carry into the write-up:
  - Parents can't do the task (EM 0), so their circuits are weakly defined.
  - `fmt_parent`'s m_full − m_empty is 0.002.
  - Accuracy gap: `elicit_child` EM 0.953 vs `teach_child` 0.145.
  - One training seed per child. Every elicit-vs-teach difference is
    single-run.
- Do not drift into the test-design questions (consistency coverage
  ceiling, the specificity bottleneck). They are parked; see memory.
- Light compute only: numpy over 4 vectors of 195,865 edges.
  `per_example` is not needed.
- Commit the script and the PLAN.md block to `eapig-circuit-check` and push.
  **No PR.** Then update memory `project-eapig-circuit-check-2026-10-01.md`.
- Report in the owner's style: plain, short, anchored to the run, with one or
  two numbers per point.
