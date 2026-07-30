# specs/01-edl-harness.md — Prequential MDL / EDL Harness (`geode.edl`)

Implements the central metric of Donoway et al. (ICML 2026 "Bits That
Count"; arXiv 2601.04728 theory). This module must be independently
trustworthy: when access to the authors' repo arrives it becomes a
cross-check, not a dependency.

## 1. Definitions (source of truth for tests)

Setup: pretrained parameters θ₀, dataset D = {(xᵢ, yᵢ)}ⁿ presented in a
fixed order, training algorithm A (optimizer + hyperparameters + seed).
ℓ(θ; x, y) = cross-entropy of the model on the **label tokens of y only**,
summed over those tokens, in nats.

- **Prequential MDL** (first epoch only):
  `MDL(D; θ₀, A) = Σᵢ ℓ(θᵢ₋₁; xᵢ, yᵢ)`
  where θᵢ₋₁ is the parameter state before the update on example/batch i.
  Operationally: sum `loss_sum_nats` over all epoch-1 records of
  `prequential.jsonl` (spec 00 §3).
- **Test loss**: `L_test(θ_T)` = per-label-token loss of the final model on
  held-out data, same masking config (spec 00 §5).
- **EDL**: `EDL = MDL − N_label · L_test(θ_T)` where `N_label` is the total
  label-token count of epoch 1. (Equivalently MDL minus what the final
  model would need to encode the same stream.)
- **Normalizations**: `EDL/D` = EDL / N_label (per label token; scaling
  analysis). `EDL/P` = EDL / trainable_param_count (capacity analysis).
- **Capacity**: `EDL(P, D) / EDL_ref(D)`, reference = rank-512 LoRA (or
  full FT).
- **PGR**: `(Perf_LoRA − Perf_base) / (Perf_FullFT − Perf_base)`.
- Units: nats internally; report in bits (÷ ln 2) at boundaries only.

## 2. Masking rules (the two footguns, mechanically enforced)

- **M1 — label-only loss.** Loss is computed on label tokens only — never
  prompt or formatting tokens. The token-mask construction is a single
  shared function used by both the training loop and the test-loss
  evaluator; its configuration is hashed into `masking_config_hash` on both
  sides (spec 00 §5), and the harness refuses to compute EDL when hashes
  differ.
- **M2 — first epoch only.** MDL accumulation stops at the end of epoch 1
  even when training continues for many epochs (as it does in elicitation
  runs). The accumulator is structurally incapable of adding epoch>1 records.

## 3. Public API (module `geode.edl`)

```python
def label_mask(batch, task_format) -> BoolTensor          # M1, shared
def prequential_step(model, batch, mask) -> StepLoss      # pre-update loss
class PrequentialAccumulator:                              # writes spec00 §3
def mdl_nats(run_id) -> float
def edl_nats(run_id) -> float
def edl_per_label_token(run_id) -> float
def edl_per_param(run_id) -> float
def pgr(perf_tuned, perf_base, perf_fullft) -> float
def training_curve(run_id) -> DataFrame   # loss vs example index, epoch 1
```

Plus a thin training-loop wrapper (`train_prequential`) around the pinned
LoRA adapter (`geode.train.apply_lora`, spec 02 §6: scaling α/(2r) —
deliberately not PEFT's α/r — seeded A init ±1/√d_in, B zero, so θ₀
computes the pretrained function exactly) that: evaluates the pre-update
loss, steps the optimizer (SGD or AdamW per the manifest, with optional
grad clipping), writes prequential + gradstat records, saves snapshots per
the manifest's `snapshot_steps`, and exposes an optional per-update
`step_callback` (per-step LR / train-loss / label-accuracy scalars; may
request an early stop — the runs-5/6 stopping rule lives with the launch
script, spec 02 §6).

## 4. Validation properties (each maps to ≥1 named test; tiny CPU models)

- **V1.1 — Random labels ⇒ EDL ≈ 0.** Train a tiny random-init model on
  labels drawn uniformly at random (i.i.d., no learnable structure). Theory:
  no generalizable information exists, so E[EDL] ≈ 0; the final test loss
  stays ≈ ln(vocab) on the label distribution. Assert |EDL/D| below a small
  tolerance and materially smaller than EDL/D from V1.2's structured task.
- **V1.2 — Learnable structure ⇒ EDL > 0.** A trivially learnable synthetic
  task (e.g., copy a marked input token) yields EDL/D clearly positive and
  a training curve that descends toward L_test.
- **V1.3 — Pre-update evaluation.** The loss recorded for batch i must be
  computed before updating on batch i. Test: on a task learnable in one
  step, recording post-update losses produces systematically lower MDL;
  assert the implementation matches an explicit two-pass reference
  (evaluate-all-then-train) on a fixed seed, and differs from the corrupted
  variant.
- **V1.4 — M1 corruption signature.** Deliberately unmask prompt tokens:
  MDL inflates by approximately the prompt-token loss mass; EDL changes.
  The test asserts (a) the harness's guard raises on hash mismatch, and
  (b) with the guard bypassed, MDL strictly increases.
- **V1.5 — M2 corruption signature.** Accumulating epochs 1–2 strictly
  increases MDL relative to epoch 1 alone (redundant information). Assert
  the public accumulator cannot produce the corrupted value.
- **V1.6 — Bookkeeping exactness.** `Σ label_token_count` over epoch 1
  equals the tokenizer-derived label-token count of the dataset; MDL from
  the accumulator equals recomputation from the JSONL to float tolerance.
- **V1.7 — Determinism.** Same seed + config twice ⇒ identical
  prequential logs (CPU, fixed threads).
- **V1.8 — Units.** Reported bits = nats / ln 2 (checked to tolerance);
  no public reporting function returns unlabeled units.
- **V1.9 — Pinned adapter wiring.** The adapter `train_prequential` builds
  is `geode.train.apply_lora`'s: adapter tensors are named
  `<module>.A/.B.weight` (and the frozen `<module>.base.weight` twins live
  in the once-per-run base file) on the target modules, every A factor is
  bit-identical to the seeded init at the loop's explicit `seed`, B is
  zero — and θ₀ therefore computes the pretrained function bit-exactly
  (asserted on logits, not approximately).
- **V1.10 — step_callback.** Called once per completed update with (step,
  epoch, lr, train_loss_nats, train_accuracy); returning True ends training
  after that update with all four artifacts still written and
  `eval/test_loss.json` evaluated at the stopped θ_T; a stop inside epoch 1
  truncates the MDL stream to the seen prefix; a None callback reproduces
  the prior behavior exactly.
- **V1.11 — Adapter-only snapshots** (2026-07-22, supersedes the
  2026-07-18 self-contained format). (a) `snapshots/step_{k}/
  adapter.safetensors` holds exactly the trainable tensors — no more, no
  less — and `snapshots/base/model.safetensors` (written once, before any
  step file) holds the frozen complement; together they cover the full
  state dict. (b) Reassembly via `geode.edl.load_snapshot` is bit-exact
  tensor-by-tensor, including under tied embeddings (the base save stores
  a shared-storage pair once; load restores the alias from its twin).
  (c) Legacy full `model.safetensors` snapshots still strict-load.

## 5. Non-goals

- Reproducing the paper's hyperparameters (that is Phase 3 science, driven
  by configs, not library logic).
- Bayesian/Hyperband search (thin external wrapper later).
- Multi-GPU training.
