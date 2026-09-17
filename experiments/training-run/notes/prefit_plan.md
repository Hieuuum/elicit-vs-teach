# Pre-fine-tuning predictors (opened 2026-09-17)

**Question.** Can the parent alone — its weights, its activations on the target
prompts, its gradients on the target loss — say whether the coming fine-tune will
elicit a latent capability or teach an absent one? Every metric in
`analysis/prefit_metrics.py` reads the parent only; no fine-tuned checkpoint is
involved. This is the practitioner's question: what to measure before spending a
fine-tuning budget.

**Ground truth.** Five parents with known learning-curve shapes on the NL target:

| tag | parent | known curve |
|---|---|---|
| blank | `evt-ts1b-base` | teach: hump, EM 0.093 at 1M (LoRA) |
| fmt | `evt-ts1b-fig2ts-installer` | teach: hump with the format prepaid, EM 0.079 |
| engine | `evt-ts1b-op-install` | symbol engine, no word binding: op-parent >> blank at n<=1000; bridging index ~0 |
| latent | `evt-ts1b-op-bridge-mix` | elicit: monotone, EM 0.54 at 3,162 |
| llama | `meta-llama/Llama-3.2-1B` | elicit: monotone (Fig-2 noinst sweep) |

A predictor is reported only if it orders these five the way their curves do;
the `engine` parent is the informative middle case (arithmetic present, words
not bound).

**Metrics** (parent + labelled task data; bare NL surface, rows 60,000.. of
D_algo_eval; sign appended for negative answers, first digit chunk scored):

| metric | what it asks | already in the paper? |
|---|---|---|
| `pref` | hidden preference: logit(correct) − logit(mismatched), rank, top-1 | yes (M8 lens read-out row) |
| `geometry` | PC1 / cos-to-mean of the parent's own answer-position states | yes (M7 second check) |
| `probe` | ridge R² on helix features of the ANSWER vs the OPERANDS per layer; first-digit logistic acc | new (agreed) |
| `das` | causal answer subspace: k-dim orthonormal patch at layer L flips the answer preference; vs random / full | new |
| `dcm` | operand roles in the frozen parent, logit-diff flip criterion | new (tool exists) |
| `attn` | attention mass from the answer position onto the operand digits, per head | new |
| `grad` | per-example gradient coherence at step 0: pairwise cos, ‖mean g‖/mean‖g‖, Gram erank, kernel-target alignment | new |
| `hessian` | top Hessian eigenvalue, Hutchinson trace, gᵀHg/‖g‖², one-step gain ‖g‖⁴/(2gᵀHg) | new |
| `llc` | local learning coefficient (SGLD estimator) at the task loss | new, exploratory |

Repeatable task circuit (P1 in the paper) is already measured for latent (0.600)
and fmt (noise); the remaining three parents get it through the circuit tool.

**Run.** `bash scripts/launch_prefit.sh --confirm-cost` (cluster, GPU, ~1–2 h);
outputs `analysis/prefit_<tag>.json`, log `analysis/prefit.log`, table from
`prefit_metrics.py compare latent fmt blank engine llama`.

**Predictions (written before running).** latent and llama: pref > 0, PC1 low,
answer R² > operand R², DAS flips at small k, compact operand head sets, high
operand attention, coherent gradients (low erank), a few sharp Hessian
directions. blank and fmt: pref ≈ 0, PC1 ≈ 1, answer R² ≈ operand R² ≈ 0 at
the answer position, DAS no better than random and the full-patch ceiling itself
low, no compact head set, low operand attention, scattered gradients. engine:
arithmetic geometry present but the NL words do not reach it — the split
between "engine present" metrics (probe on the state? no: state is NL-driven)
and "interface present" metrics (attn, dcm) is what this parent should expose.
