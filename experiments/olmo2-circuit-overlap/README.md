# OLMo 2 circuit overlap

Evaluates fixed OLMo 2 1B phase endpoints on native ETHICS, GSM-Symbolic,
and CRUXEval tasks, then compares attribution, causal interventions and
held-out linear diagnostics. The pilot is deliberately separate from a full run.

## Local validation and data preparation

```bash
python -m pip install -e '.[circuits,dev]'
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/lib/circuits -q
python -m geode.circuits.runner prepare --data geode-store/olmo2-data
```

Tests construct tiny OLMo2 models in process and never download weights. The data
preparation command explicitly downloads public, pinned sources and checks hashes.
For CPU reporting use NumPy, SciPy, pandas, matplotlib and PyArrow; inference also
requires PyTorch and Transformers with OLMo2 support. Exact runtime versions are
recorded in each run. All tokenizers and model weights use immutable revisions.

Before renting, also check all seven checkpoint configurations, tokenizers,
advertised weight-file hashes, dataset structure, label tokens, and sampled
context lengths on CPU:

```bash
python -m geode.circuits.preflight \
  --data geode-store/olmo2-data --output geode-store/olmo2-sanity-preflight \
  --max-context 4096 --max-new-tokens 2048 --sample-groups 128 --seed 0
```

This downloads public metadata and tokenizer assets only. It does not download
model weights, instantiate a model, or rent compute. Inspect `preflight.json`;
advertised weight hashes are provenance metadata, not local verification of the
weight bytes. `--metadata-only` skips dataset/context checks and is therefore
not a replacement for the complete pre-rental check above.

## Bounded pilot

Rent only after local tests and data checks pass. Run two released endpoints,
Stage 2 and RLVR-2, with 16 source groups per behavioral task (two instances per
GSM template), eight circuit source pairs per task, all eight ETHICS label
controls, twenty target probe groups, and four independent intervention pairs.
Insufficient matched groups are explicitly reported rather than filled by
unmatched examples. The output-token cap is 512 for this throughput pilot;
truncated outputs remain scored and reported as truncated.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m geode.circuits.runner run \
  --mode pilot --data geode-store/olmo2-data --output geode-store/olmo2-pilot \
  --stages stage2 rlvr2 --batch-size 8 --max-new-tokens 512 \
  --max-wall-seconds 7200 --hourly-rate 0.66 --confirm-cost
```

The wall-time check runs between batches; it is not an account-level spending
cap and does not stop instance billing. Destroy the rental after backing up the
pilot artifacts. No command automatically advances from pilot to full execution.

The recorded two-checkpoint pilot is documented in [PILOT.md](PILOT.md).

## Seven-checkpoint sanity check with a frozen CPU plan

The next bounded check uses all seven phase endpoints with the pilot sample
sizes and a 2,048-token generation cap. This is a technical sanity check, not
the full experiment. The full experiment has not been launched.

After implementation and local checks are complete, freeze source selection,
clean/corrupt pairs, held-out interventions, and probe candidates/encodings on
CPU. Preparing a plan loads a tokenizer only; Stage 2 can supply the tokenizer
even when the eventual run covers every checkpoint:

```bash
python -m geode.circuits.runner prepare-plan \
  --mode pilot --stages stage2 \
  --data geode-store/olmo2-data --plan geode-store/olmo2-sanity-plan.json \
  --groups 16 --instances-per-group 2 --pairs 8 --pair-pool-groups 128 \
  --probe-groups 20 --interventions 4 --batch-size 8 \
  --max-context 4096 --max-new-tokens 2048 --seed 0
```

The immutable plan fingerprints its payload, dataset manifest, tokenizer,
planning source files, and sampling/context settings. Changes to those inputs
require a newly prepared plan at a new path; mismatches fail before model
weights load. Checkpoint scope, device, output path, and pricing can differ
without changing the samples. Reuse the same source files, dataset cache,
preflight report, and plan on the rental. Matching and probe-candidate
verification then run once
locally, rather than once per checkpoint on the rented GPU.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python -m geode.circuits.runner run \
  --mode pilot --stages init stage1 stage2 sft dpo rlvr1 rlvr2 \
  --data geode-store/olmo2-data --output geode-store/olmo2-sanity \
  --plan geode-store/olmo2-sanity-plan.json --skip-report \
  --groups 16 --instances-per-group 2 --pairs 8 --pair-pool-groups 128 \
  --probe-groups 20 --interventions 4 --batch-size 8 \
  --max-context 4096 --max-new-tokens 2048 --seed 0 \
  --max-wall-seconds 7200 --hourly-rate 0.66 --confirm-cost
```

Set `--hourly-rate` to the actual rental rate. `--skip-report` saves the model
results and defers report/plot generation; probe fitting still occurs during
the run. After inference, audit the downloaded checkpoint bytes and loader
results on CPU **before destroying the rental**, while its HF cache still
exists:

```bash
PYTHONPATH=. python experiments/olmo2-circuit-overlap/audit_loads.py \
  --preflight geode-store/olmo2-sanity-preflight/preflight.json \
  --output geode-store/olmo2-sanity/load_audit.json
```

Use `--cache-dir PATH` if inference used a nondefault HF cache. This helper
disables CUDA and networking, hashes the cached weight files, and reloads each
pinned checkpoint on CPU to check missing, unexpected, or mismatched keys.
It neither downloads weights nor performs inference. Inspect `load_audit.json`
and resolve any failures before discarding the cache.

Back up the complete artifact directory, including the load audit, by
transferring it to the local machine. For an archive transfer, create these on
the rental and transfer both files over the existing SSH connection:

```bash
tar -czf olmo2-sanity.tar.gz -C geode-store olmo2-sanity
sha256sum olmo2-sanity.tar.gz > olmo2-sanity.tar.gz.sha256
```

On the local machine, verify the transferred archive before extraction:

```bash
sha256sum -c olmo2-sanity.tar.gz.sha256
```

Continue only if SHA256 verification succeeds, then extract:

```bash
mkdir -p geode-store
tar -xzf olmo2-sanity.tar.gz -C geode-store
```

Destroy the rental after the local backup is verified; upload from the local
machine using its HF credentials. Credentials stay on the local machine; the
rented host receives no HF token:

```bash
python experiments/olmo2-circuit-overlap/upload.py geode-store/olmo2-sanity \
  --repo mhieuuu/olmo2-circuit-overlap --prefix sanity/REPLACE_WITH_RUN_ID
```

Then audit the local copy of the saved run. `audit_run` is a Python API, not a
standalone CLI; it checks all seven pinned stages by default, matched examples,
finite arrays, grouped probe
splits, and ordinary plus type-matched random interventions:

```bash
python - <<'PY'
from pathlib import Path
from geode.circuits.artifacts import write_json
from geode.circuits.sanity import audit_run

root = Path("geode-store/olmo2-sanity")
audit = audit_run(root)
write_json(root / "sanity_audit.json", audit)
print(audit["status"], audit["failures"], audit["warnings"])
raise SystemExit(0 if audit["status"] == "passed" else 1)
PY
```

Review any failures or warnings, then generate the pilot-scale report locally:

```bash
python -m geode.circuits.runner report \
  --mode pilot --output geode-store/olmo2-sanity --seed 0
```

A passing audit establishes technical integrity. Weak accuracy, truncation,
limited independent groups, and unstable circuit rankings remain scientific
limitations. Upload the local audit/report artifacts to the same run prefix
after review; neither reporting nor audit starts a full GPU experiment.

## Full evaluation (not authorized for execution yet)

The command below is a planning reference; the full run has not been launched.
A full run needs its own full-mode frozen plan with matching settings. A pilot
plan cannot be reused as a full-mode plan.

```bash
python -m geode.circuits.runner prepare-plan \
  --mode full --stages init stage1 stage2 sft dpo rlvr1 rlvr2 \
  --data geode-store/olmo2-data --plan geode-store/olmo2-full-plan.json \
  --pairs 512 --probe-groups 128 --interventions 128 --max-new-tokens 2048

python -m geode.circuits.runner run \
  --mode full --stages init stage1 stage2 sft dpo rlvr1 rlvr2 \
  --data geode-store/olmo2-data --output geode-store/olmo2-full \
  --plan geode-store/olmo2-full-plan.json --skip-report \
  --pairs 512 --probe-groups 128 --interventions 128 --max-new-tokens 2048 \
  --max-wall-seconds 259200 --hourly-rate 0.66 --confirm-cost
```

Full behavior uses all standard test items. The ETHICS attribution budget is
512 source pairs across five tasks, each rendered under all eight controls.
Math repeats remain within fixed source-template partnerships; bootstrap groups
are template pairs, never their independent-looking generated instances.
CRUXEval has only 800 functions: matching and held-out intervention requirements
reduce its available source-pair count. Full mode reserves 20% of source groups
for independent interventions. Every realized/excluded count is saved.

Behavior uses published benchmark generation formats, while auxiliary math/code
likelihoods explicitly request the complete unknown answer without target gold
reasoning or known assertion boilerplate. Probes exchange matched candidates
between problems, so every candidate appears equally often as correct and
incorrect. The four reciprocal rows share a joint source-group split. Candidate
length/type, answer-only, and shuffled-label controls diagnose shortcuts; these
probes are not proofs of latent generation capability.

## Artifacts and reporting

The output directory includes metadata, source/data hashes, selected prompts,
per-example predictions and scores, compact per-node arrays, extracted probe
positions, group assignments, wall times, memory use, plots and a Markdown report.
Node scores use the post-normalization MLP residual write for OLMo2. Inference
is batched; attribution projects only scored answer positions. Statistics reuse
saved scores and run on CPU.

```bash
python -m geode.circuits.runner report --output geode-store/olmo2-pilot
python experiments/olmo2-circuit-overlap/upload.py geode-store/olmo2-pilot \
  --repo mhieuuu/olmo2-circuit-overlap --prefix pilots/REPLACE_WITH_RUN_ID
```

Uploads use local HF credentials and an explicit private destination. Only
allowlisted artifacts are uploaded; model caches and credentials are excluded.
Vast.ai network egress still applies to downloads/uploads, including Hugging Face.
