# CLAUDE.md — geode

Analysis infrastructure for the MARS V project "Mechanistic Understanding of
Elicitation vs. Teaching" (Donoway et al.'s EDL framework + internals methods).
A geode looks the same outside whether hollow or crystal-lined; so do models
with and without latent capabilities. This codebase cracks them open.

## What lives here

- `specs/` — schema and math ground truth for the tested core. The property
  lists in specs 01 and 02 are what tests derive from; new script work needs
  no spec. Holds exactly three files: `00-interfaces.md`, `01-edl-harness.md`,
  `02-training-run.md`. Note: pre-cut references to `specs/02`–`specs/05` in
  git history mean different files (02 was the deleted steering spec; today's
  02 is the renamed elicit-vs-teach spec).
- `geode/` — the library: `edl` (prequential/EDL harness), `train` (corpus
  packing + full-FT/SFT trainers), `zoo` (checkpoint manifest + run
  registry), `arith` (task data + evals), `probe` (extraction + metrics,
  planned).
- `tests/` — pytest suite. See testing policy below.
- `reference/` — cloned third-party repos. **READ-ONLY. Never modify, never
  import from at runtime.** Kept read-only in case the crosscoder track
  revives.
  - `reference/sparsity-artifacts-crosscoders` — Minder et al. 2025
    (arXiv 2504.02922), BatchTopK crosscoders + latent scaling.
  - `reference/quantifying-elicitation` — Donoway et al. NeurIPS 2025 code
    (added when access is granted).
- `EXPERIMENTS.md` — the live plan: runs, DAG, gate status, and remaining
  work. All work executes against it. (It replaced the 2026-07-17 cut plan;
  PLAN.md and docs/CUT-PLAN.md are git history.)

## Attribution (do not confuse these)

- **Minder et al. 2025 (arXiv 2504.02922)** — sparsity artifacts, BatchTopK
  crosscoders, latent scaling. The `reference/` crosscoder repo is theirs.
- **Jiralerspong & Bricken 2026 (arXiv 2602.11729)** — Dedicated Feature
  Crosscoders (DFC), cross-architecture diffing. Different paper, different
  method, used for different-model comparisons.
- **Donoway et al.** — EDL (ICML 2026 "Bits That Count"; NeurIPS 2025
  elicitation; arXiv 2601.04728 theory).
- **Wang et al. 2025** — OOCR ≈ constant steering-vector shift; template for
  direction-extraction analysis under `experiments/…/analysis/`.
  (`geode.steering` was planned but never built; deleted 2026-07-17.)

## Conventions

- Python ≥ 3.11. Type hints on all public functions. `ruff` for lint/format.
- Losses are computed and stored in **nats**; convert to bits (`/ ln 2`) only
  at reporting boundaries. Every stored loss field name ends in `_nats`.
- All library code is **device-agnostic**: accept a `device` argument, never
  hardcode `cuda`, never assume a GPU exists.
- Randomness is always seeded through an explicit `seed` argument.
- Artifacts (runs, activations, results) follow the schemas in
  `specs/00-interfaces.md` exactly. Schema changes require editing the spec
  first, in the same PR.

## Testing policy (non-negotiable)

- The full suite runs on **CPU only**, in **under ~2 minutes**, with **no
  network access**. CI and subagents must be able to run it freely.
- Fixture models are **randomly initialized tiny configs** built in-process
  (e.g., a 2–4 layer, d_model≈64 Llama-style config via `transformers`).
  Never download pretrained weights inside a test.
- Tests validate *code correctness via mathematical properties from the
  specs* (e.g., random labels ⇒ EDL≈0; planted direction ⇒ recovered). They
  do NOT validate scientific claims — that happens in budgeted GPU runs,
  outside this suite.
- Every module's spec has a "Validation properties" section; each property
  maps to at least one named test.

## Workflow

The four-stage spec-first protocol was retired 2026-07-17 (see
`EXPERIMENTS.md` §5). It cost ~60–70% of each task in ceremony and mostly
protected clerical code. Replacement:

**Tested core** (`geode/`): code and its property tests are written
together in one pass. A change to core math updates its property tests in
the same commit. Property lists live in `specs/01-edl-harness.md` §4 and
`specs/02-training-run.md` (V-numbers); name tests after the property
they check (e.g. `test_v5_1_no_probe_leakage`).

**Scripts** (`experiments/`): single pass, self-reviewed. Smoke test only
where cheap. No spec edits, no stage agents.

**What earns a property test:** code whose *silent* failure would waste
GPU budget or invalidate the elicit-vs-teach comparison — EDL/MDL math,
data integrity, matched-input guards, analysis metrics. Code that fails
*loudly* (config validators, schema type-checks, serialization plumbing)
gets one round-trip smoke test at most.

**Promotion rule:** logic used by two or more scripts, or whose silent
failure would corrupt results, moves into `geode/` and gains property
tests. Nothing else does.

## Documentation

Per-task implementation logs are retired (2026-07-17 cut);
`docs/impl-logs/` is frozen history. Record decisions where they belong:

- Experiment decisions, pilot outcomes, closed OPEN(n) items →
  `experiments/training-run/notes/decisions.md`.
- Structure / plan changes → `EXPERIMENTS.md`.
- Spec changes → the spec itself, in the same commit as the code.

## Budget rule

Nothing in this repo launches a GPU job implicitly. Any script that could
incur compute cost requires an explicit `--confirm-cost` flag and prints its
estimated cost first. The compute budget (~$2k total) lives in an external
sheet; this repo never spends it silently.
