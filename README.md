# geode

Mechanistic analysis infrastructure for the MARS V project on elicitation
vs. teaching (Donoway et al.'s EDL framework). A geode looks identical from
the outside whether hollow or crystal-lined — like a model with or without a
latent capability. This code cracks them open.

Modules (specified in `specs/`, implemented via `PLAN.md`):

- `geode.zoo` — checkpoint-zoo manifests, run registry, storage schemas
- `geode.edl` — prequential MDL / EDL harness (label-masked, first-epoch)
- `geode.steering` — direction extraction + rank-sufficiency interventions
- `geode.saediff` — base-SAE reconstruction / novel-feature-mass analysis

Start here: `CLAUDE.md` (conventions + agent protocol), then
`PLANNING_PROMPT.md`. Version pins in `pyproject.toml` are minimums —
tighten them to the exact versions of your environment on first install.

Clone references (read-only):

    git clone https://github.com/science-of-finetuning/sparsity-artifacts-crosscoders reference/sparsity-artifacts-crosscoders
