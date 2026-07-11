# specs/03-base-sae-pipeline.md — Base-SAE Analysis Pipeline (`geode.saediff`)

Implements Block B: quantify how much of a finetuned model's activation
structure is expressible in the **base model's** SAE dictionary. H1 predicts
elicitation ≈ re-weighting of existing latents with stable reconstruction;
H2 predicts teaching ⇒ growing residual "dark matter" the base dictionary
cannot express; H3 predicts an internal quantity from this module tracks EDL.

## 1. Inputs

- An SAE in SAELens format for (base model, hook): encoder/decoder weights,
  activation function config. Loaded read-only; training SAEs is out of
  scope here (SAELens handles it; configs live elsewhere).
- Cached activations per spec 00 §6 for base and finetuned models on
  identical matched inputs.

## 2. Metrics (exact definitions; tests derive from these)

For activation batch X at hook ℓ and SAE s with reconstruction x̂:

- **FVU** (fraction of variance unexplained):
  `FVU(X) = Σ‖x − x̂‖² / Σ‖x − mean(X)‖²`.
- **Baseline FVU**: FVU of *base-model* activations under the base SAE —
  the dictionary's own imperfection.
- **Novel-feature mass (NFM)** for a finetuned model at checkpoint t:
  `NFM = FVU(X_ft_t) − FVU(X_base)` on matched inputs. Reported per
  (layer, checkpoint_step, dataset_size). This is the headline scalar.
- **Latent re-weighting stats**, per latent j: firing frequency
  `f_j = P(a_j > 0)` and mean active magnitude `m_j = E[a_j | a_j > 0]`,
  computed for base and finetuned activations; report per-latent deltas,
  ranked movers, and a summary distance (L1 over frequency deltas).
- **Residual direction analysis**: top principal components of the SAE
  residual `(x − x̂)` for finetuned activations, with their variance
  fractions — the candidate "new features" handed to crosscoder/steering
  follow-ups.

## 3. Outputs

Long-format rows into `results/` (spec 00 §7):
`metric_name ∈ {fvu, nfm, latent_freq_delta_l1, residual_pc_varfrac_k, ...}`
keyed by run_id, layer, checkpoint_step, dataset_size. The EDL bridge plot
(H3) is then a join of this table with the EDL table on run_id — no special
code path.

## 4. Public API

```python
def load_sae(model_key, hook) -> SAE                     # SAELens read-only
def fvu(sae, acts) -> float
def nfm(sae, acts_base, acts_ft) -> float
def latent_stats(sae, acts) -> DataFrame                 # f_j, m_j
def reweighting_report(sae, acts_base, acts_ft, top_k) -> DataFrame
def residual_pcs(sae, acts, k) -> tuple[Tensor, Tensor]  # dirs, varfracs
def run_saediff(run_id, hooks, checkpoints) -> DataFrame # orchestrator
```

## 5. Validation properties (tiny CPU models + synthetic SAEs)

Tests use a **synthetic SAE with a known dictionary** (orthogonal decoder
directions, ReLU/TopK encoder built to invert exactly on in-span inputs).

- **V3.1 — In-span identity.** Activations synthesized as sparse positive
  combinations of dictionary directions reconstruct with FVU ≈ 0; NFM of
  such data against itself is ≈ 0.
- **V3.2 — Planted novel direction.** Add energy fraction ε along a
  direction orthogonal to the dictionary span; NFM increases monotonically
  with ε and ≈ ε for small ε. This is the calibration test for the
  headline metric.
- **V3.3 — Pure re-weighting.** Rescale the coefficients of existing
  dictionary directions (no new directions): NFM stays ≈ 0 while the
  re-weighting stats (f_j, m_j deltas) change as constructed. This is the
  H1-vs-H2 discriminator working as designed.
- **V3.4 — Residual recovery.** In V3.2's setup, the top residual PC
  matches the planted orthogonal direction with |cos| ≥ 0.99.
- **V3.5 — Matched-input enforcement.** `nfm` raises on metadata mismatch
  (delegates to spec 00 V0.4; asserted here at this API surface too).
- **V3.6 — Streaming equivalence.** Metrics computed in streaming batches
  equal the in-memory computation to tolerance (activations won't fit in
  RAM at real scale).

## 6. Non-goals

- SAE training (SAELens configs, run scripts outside the library).
- Crosscoder training or adaptation of the Minder et al. code (that gets
  its own spec once the adaptation plan exists — see specs/04).
- Choosing hooks/layers (analysis decision).
