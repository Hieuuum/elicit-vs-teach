# specs/04-crosscoder-adaptation.md — Scoping Note (plan, not code)

Status: **not scheduled for implementation yet.** The planning session
produces an *adaptation plan* for this, no code. Kept here so PLAN.md has a
spec anchor.

## Goal

Adapt the Minder et al. 2025 pipeline (`reference/sparsity-artifacts-
crosscoders`: BatchTopK crosscoder training + latent scaling; built on
their dictionary_learning fork) from its native setting (7B+ chat models,
~4TB activation cache per model) to ours (1B models, narrow arithmetic
finetunes, activations cached per spec 00 §6).

## What the adaptation plan must decide

1. Which components are consumed as-is (crosscoder module, BatchTopK,
   latent-scaler closed form) vs. re-wired (activation loading — replace
   their $DATASTORE cache with our spec 00 §6 loaders).
2. Scale-down of dictionary size / L0 / training tokens for 1B models and
   narrow tasks, with an explicit smoke-test config.
3. Output mapping into our results schema (spec 00 §7): per-latent Δnorm,
   latent-scaling diagnostics (νε, νr), exclusive-latent lists.
4. Whether Delta-Crosscoder (arXiv 2603.04426) has usable public code; if
   yes, an equivalent integration sketch, and the criterion for choosing
   between the two (integration cost, smoke-test behavior on a planted
   synthetic diff).
5. A synthetic acceptance test mirroring specs/03 V3.2–V3.3: on a toy pair
   with a planted model-exclusive direction, the pipeline's exclusive-latent
   detection finds it and does not flag pure re-weighting.

## Attribution reminder

Minder et al. 2025 (arXiv 2504.02922) = BatchTopK + latent scaling (this
reference repo). Jiralerspong & Bricken 2026 (arXiv 2602.11729) = DFC,
cross-architecture diffing — a different method, planned for
different-model comparisons (endpoint-vs-endpoint), not adapted here.
