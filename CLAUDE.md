# CLAUDE.md — geode

Analysis infrastructure for the MARS V project "Mechanistic Understanding of
Elicitation vs. Teaching" (Donoway et al.'s EDL framework + internals methods).
A geode looks the same outside whether hollow or crystal-lined; so do models
with and without latent capabilities. This codebase cracks them open.

## What lives here

- `specs/` — the source of truth. Every module is specified before it is
  implemented. Tests are derived from specs, never from implementations.
- `geode/` — the library: `edl` (prequential/EDL harness), `steering`
  (direction extraction + sufficiency tests), `saediff` (base-SAE analysis),
  `zoo` (checkpoint manifest + run registry).
- `tests/` — pytest suite. See testing policy below.
- `reference/` — cloned third-party repos. **READ-ONLY. Never modify, never
  import from at runtime.** Used only to inform adaptation plans.
  - `reference/sparsity-artifacts-crosscoders` — Minder et al. 2025
    (arXiv 2504.02922), BatchTopK crosscoders + latent scaling.
  - `reference/quantifying-elicitation` — Donoway et al. NeurIPS 2025 code
    (added when access is granted).
- `PLAN.md` — the approved build plan. All work executes against it.

## Attribution (do not confuse these)

- **Minder et al. 2025 (arXiv 2504.02922)** — sparsity artifacts, BatchTopK
  crosscoders, latent scaling. The `reference/` crosscoder repo is theirs.
- **Jiralerspong & Bricken 2026 (arXiv 2602.11729)** — Dedicated Feature
  Crosscoders (DFC), cross-architecture diffing. Different paper, different
  method, used for different-model comparisons.
- **Donoway et al.** — EDL (ICML 2026 "Bits That Count"; NeurIPS 2025
  elicitation; arXiv 2601.04728 theory).
- **Wang et al. 2025** — OOCR ≈ constant steering-vector shift; template for
  `geode.steering`.

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

## Execution protocol (per PLAN.md task)

Each task runs as four sequential subagent stages, each in a fresh context:

1. **TEST-WRITER** — writes failing tests from the referenced spec sections
   only. Must not read any implementation code.
2. **TEST-AUDITOR** — reviews tests against the spec; flags missing
   properties, mismatches, or tests that overfit to an assumed
   implementation. Tests revised until clean.
3. **IMPLEMENTER** — makes the tests pass. Must not modify tests.
4. **CONFORMANCE-REVIEWER** — reviews the diff against spec + tests; reports
   gaps with file/line references.

A task is done only when stage 4 reports no findings and the full suite
passes. Commit after each completed task with the task ID in the message.

## Budget rule

Nothing in this repo launches a GPU job implicitly. Any script that could
incur compute cost requires an explicit `--confirm-cost` flag and prints its
estimated cost first. The compute budget (~$2k total) lives in an external
sheet; this repo never spends it silently.
