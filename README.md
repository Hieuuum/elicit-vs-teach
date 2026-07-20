# geode

Mechanistic analysis infrastructure for the MARS V project on elicitation
vs. teaching (Donoway et al.'s EDL framework). A geode looks identical from
the outside whether hollow or crystal-lined — like a model with or without a
latent capability. This code cracks them open.

Modules (specified in `specs/`; current plan: `EXPERIMENTS.md`):

- `geode.zoo` — checkpoint-zoo manifests, run registry, storage schemas
- `geode.edl` — prequential MDL / EDL harness (label-masked, first-epoch)
- `geode.train` — corpus packing + full-FT/pretrain trainer
- `geode.arith` — arithmetic task data + evals (planned)
- `geode.probe` — snapshot schedule, activation/grad extraction, metrics (planned)

Start here: `CLAUDE.md` (conventions + workflow), then `EXPERIMENTS.md`.
Version pins in `pyproject.toml` are minimums — tighten them to the exact
versions of your environment on first install.

Clone references (read-only):

    git clone https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders reference/sparsity-artifacts-crosscoders
