# specs/02-steering-library.md — Steering & Sufficiency Library (`geode.steering`)

Implements the Block A headline experiment: how much of a finetune's effect
is recovered by a minimal-structure intervention applied to the **base**
model. Template: Wang et al. 2025 (OOCR finetuning ≈ constant
steering-vector shift). The pointer hypothesis (H1) predicts near-full
recovery in the elicitation regime and near-zero in the teaching regime.

## 1. Direction extraction

Two extractor families, both returning a `Direction` object
(`layer/hook, vector, method, provenance`):

- **E1 — Activation-diff extractor.** Run base and finetuned models on
  identical matched inputs (spec 00 §6 enforces matching). At a chosen hook
  and position policy, `v_ℓ = mean over samples/positions of
  (h_ft − h_base)`. Report per-layer vector norms so the analysis can pick
  concentration layers.
- **E2 — Weight-diff extractor.** For each adapted module, ΔW = scaled LoRA
  product (α/r · B@A) or (merged − base). Return top-k singular
  triples (per module) and, for rank-r patches, the truncated-SVD ΔW_r.

## 2. Interventions

- **I1 — Constant shift ("rank-0").** Forward hook on the base model adding
  `α · v` at hook ℓ (position policy configurable: all positions or
  generated positions only). α swept over a small grid; report best-α and
  the full sweep.
- **I2 — Low-rank weight patch.** Apply ΔW_r (rank-r truncation from E2) to
  the base model's corresponding modules. r sweeps the ladder
  {1, 2, 4, ..., full}.
- **I3 — Re-fit ladder (optional, flag-gated).** Fit a fresh rank-r map on
  frozen base activations to mimic the finetuned model's activations at ℓ
  (least squares). Distinguishes "the diff is low-rank" from "a low-rank
  object suffices."

## 3. Evaluation

- Primary metric: **PGR** (spec 01) of the intervened base model on the
  task's held-out set: `(Perf_intervened − Perf_base) / (Perf_ft − Perf_base)`.
- Output row per (run_id, extractor, intervention, layer, rank, α) into the
  results table (spec 00 §7): metric_name ∈ {pgr, best_alpha, vector_norm,
  sv_spectrum_k}.
- **Minimal sufficient rank**: smallest r with PGR ≥ 0.5 (threshold
  configurable). This scalar is a candidate internal quantity for the
  EDL bridge (H3).

## 4. Controls (implemented, not optional)

- **C1 — Random-direction control.** I1 with a random unit vector at the
  same layer and best-α protocol; PGR should be ≈ 0. Any headline PGR is
  reported alongside its random control.
- **C2 — Shuffled-pairing control.** E1 with mismatched input pairing must
  not produce a high-PGR direction.
- **C3 — Norm-matched teaching comparison.** When comparing regimes, match
  intervention norm budgets so "teaching recovers less" is not an artifact
  of scale.

## 5. Public API

```python
def extract_activation_diff(base, tuned, dataset_key, hook, positions) -> Direction
def extract_weight_diff(base, adapter_or_merged) -> ModuleSVDs
def apply_constant_shift(model, direction, alpha) -> ContextManager
def apply_lowrank_patch(model, module_svds, rank) -> ContextManager
def refit_lowrank(base_acts, tuned_acts, rank) -> Direction        # I3
def sufficiency_ladder(run_id, ranks, alphas, eval_fn) -> DataFrame
```

Interventions are context managers that guarantee restoration of original
weights/hooks on exit (including on exception).

## 6. Validation properties (tiny CPU models)

- **V2.1 — Planted constant shift.** Take a tiny random model; create a
  "finetuned" copy by permanently adding a known bias u at hook ℓ. E1 must
  recover v with |cos(v, u)| ≥ 0.99, and I1 with the recovered v at α=1
  must reproduce the tuned model's outputs to numerical tolerance.
- **V2.2 — Planted low-rank weight diff.** Create the tuned copy by adding
  a known rank-2 ΔW to one module. E2's top-2 singular subspace matches the
  planted subspace (principal angles ≈ 0); I2 at r=2 reproduces tuned
  outputs; r=1 does not.
- **V2.3 — Restoration.** After every intervention context exits (normally
  or via raised exception), base model outputs are bit-identical to
  pre-intervention outputs.
- **V2.4 — Random control sanity.** On the planted-shift model, C1 yields
  PGR ≈ 0 while the recovered direction yields PGR ≈ 1.
- **V2.5 — Matched-input enforcement.** E1 raises when activation metadata
  (dataset_key / positions / tokenizer hash) differs between models.
- **V2.6 — SVD correctness.** E2 on a LoRA adapter equals SVD of the
  explicitly materialized α/r·B@A to tolerance, including the scaling.

## 7. Non-goals

- Choosing which layers/positions matter (analysis decision, in notebooks).
- SAE-basis interpretation of directions (lives in `geode.saediff`).
- Crosscoder training.
