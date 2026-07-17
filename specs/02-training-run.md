# 02 — Training Runs: elicit-vs-teach experiment organization & requirements

Status: **design, pre-implementation** (2026-07-16). Defines folder/run
organization and per-part requirements for the mechanistic
elicitation-vs-teaching comparison. EDL/D signature analysis is dropped;
regime classification is assumed from Bits That Count. `OPEN(n)` markers
denote unresolved items collected in §12. Nothing here launches GPU work
without `--confirm-cost` (CLAUDE.md budget rule).

Decisions locked 2026-07-16 (owner):

- **Code home:** thin scripts/configs in `experiments/elicit-vs-teach/`;
  reusable logic as geode library modules (`geode.arith`, `geode.probe`).
- **Rigor:** library modules (`geode.arith`, `geode.probe`) are written
  with their property tests in one pass (CLAUDE.md → "Workflow");
  experiment scripts, configs, and plotting are single-pass +
  self-review.
- **Run tracking:** geode.zoo is the local registry for all six runs; the HF
  dataset repo holds bulk tensors; `manifest.parquet` is an *export* of zoo
  records, never a second source of truth.
- **Naming:** `armA_elicit` / `armB_teach`. "Pre-elicit" in the paper means
  the E.1.1 single-example procedure; Arm A here is E.2-style pre-teaching
  of the algorithm (1M correct-label NL examples). Procedures are always
  spelled out; the paper's vocabulary is never reused for a different
  procedure.

## 1. Runs, arms, DAG

| # | run_id (proposed)      | Role            | Init      | Method  | Data |
|---|------------------------|-----------------|-----------|---------|------|
| 1 | `evt-run1-base`        | pretrain        | random    | full FT | TinyStories-v2 (~2.6M stories), Llama-3.2-1B arch |
| 2 | `evt-run2-armA-algo`   | pre-teach       | run 1     | full FT | NL add/sub, correct labels, 1M unique, 1 epoch |
| 3 | `evt-run3-armA-inst`   | format install  | run 2     | full FT | operator-notation mult, random labels, OPEN(1) count |
| 4 | `evt-run4-armB-inst`   | format install  | run 1     | full FT | identical dataset + count as run 3 |
| 5 | `evt-run5-armA-target` | target          | run 3     | LoRA    | operator-notation add/sub, OPEN(2) count |
| 6 | `evt-run6-armB-target` | target          | run 4     | LoRA    | identical data + identical order as run 5 |

DAG: `1 → 2 → 3 → 5` (Arm A) and `1 → 4 → 6` (Arm B). Arms differ **only**
in run 2's presence. No Arm C (generic-transfer confound assumed away per
paper Table 6). Single seed — recorded limitation (§13). Run 1 may be
replaced by an external TinyStories checkpoint pending mentor — OPEN(8);
only `bases/` contents change downstream.

Every run is registered in zoo before training starts and marked complete
only after its gates (§8) pass. A run refuses to launch if its parent run
is missing, incomplete, or has failing gates.

## 2. Repository layout (code)

```
experiments/elicit-vs-teach/
  README.md            # experiment card: goal, arm definitions, DAG, gate status
  configs/
    common.yaml        # shared blocks: optimizer, precision, batch, LoRA
    run1_pretrain.yaml … run6_armB_target.yaml     # one per run, §6 contract
    pilot/             # same six files, pilot-sized overrides (§11)
  scripts/             # thin CLIs; GPU-cost paths gated by --confirm-cost
    make_data.py       # geode.arith → datasets + probe set + hashes (CPU)
    train.py           # dispatch per config: full-FT trainer | train_prequential
    extract.py         # offline probe pass over snapshots (§7)
    gates.py           # run verification gates, write results into zoo manifest
    export_hf.py       # build hf-staging layout, export manifest.parquet, upload
  analysis/
    alignment.py  drift.py  adapters.py  matching.py   # drivers → zoo results/
    figures/           # gitignored
  notes/
    decisions.md       # running log; pilot outcomes close OPEN items here first

geode/
  arith/               # library (§5): generator, formats, evals
  train/               # library (§6.1): packing, full-FT/pretrain loop
  probe/               # library (§7): schedule, extraction, metrics
```

Library modules are written together with their property tests against
§5/§7 of this spec. Everything under `experiments/` is script-land:
single-pass + self-review.

## 3. Store layout (artifacts)

Local, on the training/extraction rental (`$GEODE_STORE`):

```
$GEODE_STORE/
  runs/<run_id>/...                    # spec 00 §1, unchanged: manifest.json,
                                       # prequential.jsonl, gradstats.jsonl,
                                       # snapshots/step_k/, eval/
  experiments/elicit-vs-teach/
    data/                              # generated datasets, probe set, hash files
    hf-staging/                        # exact mirror of the HF repo below
```

HF dataset repo (HF PRO public, ≤10 TB; pilot uses a **separate** repo
suffixed `-pilot` so the production manifest stays clean):

```
elicit-vs-teach/
  manifest.parquet                     # exported from zoo (§4); one row per snapshot:
                                       # run_id, arm, step, probe/train metrics,
                                       # digit metadata, file paths
  probe/                               # probe_set.safetensors + probe_set.json (hashes, strata)
  bases/                               # run1 final + run2/3/4 finals (~5 GB)
  armA_elicit/
    adapters/{000,001,002}/step_XXXXXXX.safetensors
    acts/{000,...}/step_XXXXXXX.safetensors      # 17 named tensors each (§7)
    grads/{000,...}/step_XXXXXXX.safetensors
  armB_teach/                          # same shape
  optimizer/                           # optional, ~10 ckpts, 3.6 GB — OPEN(10)
```

Chunking: subdir index = snapshot_index // 500 (keeps every dir ≤ 1000
files). One file per (snapshot, quantity). Uploads in commits of 50–100
files, resumable, verified against local hashes after push. Budget:
~2.14 GB/snapshot × 1024 × 2 arms ≈ 4.38 TB tensors + 92 GB adapters +
5 GB bases (+3.6 GB optimizer) + logs < 1 GB.

## 4. Zoo schema additions (spec 00 edit, same PR as implementation)

One optional `experiment` object on the run manifest, validated when
present:

```json
"experiment": {
  "name": "elicit-vs-teach",
  "arm": "armA_elicit" | "armB_teach" | "shared",
  "role": "pretrain" | "preteach" | "installer" | "target",
  "parent_run_id": "<run_id>" | null,
  "data_order_hash": "<sha256 of example-index sequence>" | null,
  "probe_set_hash": "<sha256>" | null,
  "gates": { "<gate_id>": {"pass": bool, "value": float, "threshold": float} }
}
```

`manifest.parquet` for HF is generated by joining zoo manifests +
per-snapshot metrics + staging file paths; it carries no information that
zoo lacks. The schema rule (CLAUDE.md) applies: spec 00 is edited in the
same PR that implements validation.

## 5. `geode.arith` — task data + evals

**Generator requirements.** Procedural (own generator; not DMM splits —
those are ~87% decimals, mixed formats). Operands 1–4 digits; ops add/sub
(mult for the installer). Both formats share a two-line `Question: <body>` / `Answer: <answer>`
scaffold (exact template frozen 2026-07-17, closing OPEN(9); padded length
OPEN(5)): operator body `a op b` (e.g. `Question: 23 + 45` then `Answer: 68`),
NL body `What is the sum of a and b?` / `What is the difference between a and
b?` (add/sub only). Label modes:
correct | random. Random-label sampling distribution OPEN(6) (default:
uniform over answers with digit-count distribution matched to true
answers). Subtraction negatives OPEN(7) (default: allowed). Datasets are
generated **once** by `scripts/make_data.py` and frozen to files (not
regenerated at train time); `geode.arith` supplies only rendering, the
random-label rule, evals, the water-fill allocation, and the validators.
Every emitted example carries the answer **character** span (tokenizer-
agnostic); token-level label spans are derived at load once the tokenizer is
fixed (OPEN(11)), and masking then goes through `geode.edl.masking.label_mask`
— the single mask path.

**Integrity rules.** Every training row is a distinct question `(a, op, b)`
— no repeats. Probe **questions** (the triple `(a, op, b)`, format-independent)
are excluded from the target set **and every pre-teach/installer set**; the
same operand pair under a different op, or commuted, is *not* excluded.
(Softened from the original pair rule: both arms train on the identical target
set, so probe overlap inflates both equally and never biases the A-vs-B
comparison.) Both target runs consume the identical dataset in identical order;
the index-sequence hash is stored as `data_order_hash` in both manifests.

**Probe set.** 1024 held-out examples, target format, fixed across arms
and snapshots. Stratified 64 per `(x_digits, y_digits)` cell (16 cells,
x, y ∈ {1,2,3,4}), ops balanced within cell (uniform sampling would give
~98% 4-digit). Training sets use the same 16-cell grid with capacity-capped
water-fill (`geode.arith.stratify`). Serialized once with `probe_set_hash`;
every consumer records that hash.

**Evals.** Answer parser + exact-match accuracy (negatives included),
format-validity check (output parses as a number in the expected slot),
zero/16-shot prompt builder for G5.

**Validation properties (tests derive from these at task cut):**

- V5.1 no probe **question** `(a, op, b)` appears in any generated train set;
  the same operand pair under a different op, or commuted, is allowed.
- V5.2 every question unique: all `(a, op, b)` distinct in each set; requested
  n produced exactly (1,000,000 per training set).
- V5.3 stratification: 16 `(x_digits, y_digits)` cells filled by capacity-capped
  water-fill — small cells taken whole, the remainder split as evenly as
  capacities allow; ops balanced where capacity permits.
- V5.4 determinism: same seed ⇒ byte-identical datasets.
- V5.5 label spans cover exactly the answer characters under the
  `Question:/Answer:` scaffold (both formats), negatives included.
- V5.6 random-label mode: labels statistically independent of operands.
- V5.7 parser/exact-match correct on constructed outputs incl. negatives
  and malformed strings.

## 6. Training runs — per-run needs

**Config contract (one YAML per run):** run_id, experiment/arm/role,
init (parent run_id | external | random), model (arch config), data
(task, format, label mode, n_examples, seed), train (optimizer, lr,
schedule, clip, precision, batch, stopping), lora (target runs only),
snapshots, logging. Fixed values from the paper's tables:

- AdamW β₁ 0.9, β₂ 0.999, wd 0.01; constant LR; grad clip 1.0; bfloat16.
- LR 2e-5 full FT / 3.53e-4 LoRA (TinyStories-1B row). Batch 128
  (paper: eff. 1024 on 8×H100) — step count and snapshot schedule follow
  from OPEN(2)+OPEN(4).
- LoRA (runs 5–6 only): r=64; Q,K,V,O,G,U,D all layers; α=32; scaling
  α/2r; dropout 0; A Kaiming 1/√d_in, B zero. 45.09M params,
  ~90 MB/adapter bf16.
- Loss on label tokens only, identical masking train/test (masking hash
  guard from spec 00 §5 applies as usual).
- Stopping (runs 1–4): validation-loss convergence with pinned ε, k —
  OPEN(3).

**Runs 1–4 (full FT):** need a small full-FT trainer with validation-loss
stopping; snapshots = final checkpoint only (plus the base). **Decided
2026-07-16 (task TRAIN-1):** a separate thin module `geode.train` (§6.1),
leaving the validated prequential loop untouched. Run 1 needs only the
pretrain mode (loss over all next-token positions); the label-masked SFT
mode for runs 2–4 routes through `geode.edl.masking.label_mask` and is
added when those runs are cut, not before.

### 6.1 `geode.train` — corpus packing + full-FT trainer (task TRAIN-1)

Files: `geode/train/{__init__,packing,stopping,loop}.py`. All CLAUDE.md
conventions bind (device-agnostic, explicit seeds, `_nats` suffixes,
CPU-only tests on tiny in-process models). The module never touches the
zoo registry — registration is the launch script's job (§6.2) — and never
imports `datasets`/network loaders; it consumes in-memory token tensors.

```python
# geode/train/packing.py
def pack_corpus(texts: Iterable[str], tokenizer, seq_len: int) -> torch.LongTensor
    # Tokenize each document (no special tokens added), append exactly one
    # eos_token_id after every document, concatenate in input order, slice
    # the stream into consecutive rows of length seq_len, drop the short
    # tail. Raises ValueError if tokenizer.eos_token_id is None or
    # seq_len < 2. Deterministic: output is a pure function of inputs.
def train_val_split(seqs: torch.LongTensor, val_fraction: float, seed: int
                    ) -> tuple[torch.LongTensor, torch.LongTensor]
    # Seeded permutation, then split. n_val = round(val_fraction * n)
    # clamped to [1, n-1]; requires 0 < val_fraction < 1 and n >= 2, else
    # ValueError. Rows are preserved exactly (a partition, no mutation).

# geode/train/stopping.py
@dataclass(frozen=True)
class StoppingRule:
    eps_nats: float   # minimum improvement that counts
    k: int            # consecutive non-improving evals that trigger a stop
class ConvergenceTracker:
    def __init__(self, rule: StoppingRule): ...
    def update(self, val_loss_nats: float) -> bool
    # An eval improves iff (best_so_far - val_loss_nats) > eps_nats
    # (strict; equality does NOT improve). Improvement updates best and
    # resets the stale counter; the k-th consecutive non-improving eval
    # returns True (and keeps returning True). NaN input raises ValueError.
    best_nats: float          # +inf before first update
    stale_evals: int

# geode/train/loop.py
@dataclass(frozen=True)
class TrainResult:
    final_step: int
    best_val_nats: float
    stop_reason: Literal["converged", "max_steps"]
    checkpoint_dir: Path
def evaluate_nll_nats(model, seqs: torch.LongTensor, *, batch_size: int,
                      device: str) -> float
    # Mean next-token cross-entropy in nats over ALL predicted positions
    # (positions 0..L-2 predict 1..L-1), under no_grad. Batch size must
    # not change the value beyond float tolerance.
def train_full(model, train_seqs: torch.LongTensor,
               val_seqs: torch.LongTensor, *, lr: float, batch_size: int,
               stopping: StoppingRule, eval_every: int,
               max_steps: int | None, grad_clip: float,
               weight_decay: float, betas: tuple[float, float],
               device: str, seed: int, out_dir: Path,
               precision: Literal["fp32", "bf16"] = "fp32") -> TrainResult
```

`train_full` contract:

- Optimizer AdamW(lr, betas, weight_decay); **constant** LR; global-norm
  grad clipping at `grad_clip`.
- Data order: a seeded permutation of `train_seqs` per epoch, derived
  deterministically from `seed` and the epoch index; fixed-size batches,
  drop-last. Epochs repeat until a stop condition fires (multi-epoch is
  fine here — this is pretraining, not prequential MDL; `geode.edl` guards
  are not in play).
- Step = one optimizer update. Loss = mean next-token CE per token (nats)
  over the batch.
- Eval: `evaluate_nll_nats(val_seqs)` at every step where
  `step % eval_every == 0`, and additionally at the final step. Every eval
  updates one `ConvergenceTracker`; a True return stops training with
  `stop_reason="converged"`. Reaching `max_steps` stops with
  `stop_reason="max_steps"`. `max_steps=None` means no cap. If both
  conditions fire on the same final-step eval, `converged` wins (the run
  really did converge; labeling it `max_steps` would misreport run health
  in persisted artifacts). `train_full` raises `ValueError` upfront if
  `train_seqs` has fewer rows than `batch_size` — drop-last would
  otherwise yield zero batches per epoch and the loop could never reach
  any stop condition (added 2026-07-16 from CONFORMANCE-REVIEWER finding:
  silent infinite busy-loop, unreachable by timing-safe tests).
- Logs, written under `out_dir`: `train_log.jsonl` with
  `{"step", "train_loss_nats", "lr", "grad_norm"}` per step (`grad_norm`
  is the pre-clip global norm); `eval_log.jsonl` with
  `{"step", "val_loss_nats"}` per eval.
- Checkpoint: final model saved to `out_dir/model/` via `save_pretrained`,
  plus `out_dir/training_meta.json` recording `stop_reason`, `final_step`,
  `best_val_nats`, and a `config` object echoing exactly the call
  arguments {lr, batch_size, eval_every, max_steps, grad_clip,
  weight_decay, betas, seed, precision, stopping: {eps_nats, k}}.
  (Echo keys pinned 2026-07-16 after TEST-AUDITOR flagged the phrase as
  untestable; the echo itself stays untested — asserting it would overfit
  — but CONFORMANCE-REVIEWER checks it by inspection.)
- Determinism: identical seed + inputs on CPU ⇒ byte-identical log files.
- `precision="bf16"` wraps forward/loss in autocast; tests exercise fp32
  only (CPU) and treat bf16 as config plumbing.

**Validation properties (tests derive from these):**

- V5.17 packing: every emitted row has length exactly `seq_len`; the
  flattened rows equal the concatenated per-document token streams with
  one EOS after each document, in input order, up to the dropped tail
  (tail strictly shorter than `seq_len`); missing EOS token raises;
  deterministic across calls.
- V5.18 split: train/val are an exact partition of the input rows;
  n_val matches the clamped-round formula; same seed ⇒ identical split;
  different seed ⇒ different permutation (on non-trivial input).
- V5.19 `evaluate_nll_nats` equals an explicit hand-computed per-position
  reference on a tiny model, and is invariant to `batch_size`.
- V5.20 stopping semantics: a plateau (improvements ≤ eps) stops after
  exactly k evals; any improvement > eps resets the counter; improvement
  of exactly eps does not count; NaN raises.
- V5.21 on a trivially overfittable tiny corpus, training terminates via
  `stop_reason="converged"` before a generous `max_steps`, and final
  train loss < initial train loss.
- V5.22 same seed ⇒ byte-identical `train_log.jsonl` and
  `eval_log.jsonl` across two runs (CPU, fixed threads).
- V5.23 log schema: every train record carries exactly
  {step, train_loss_nats, lr, grad_norm} with finite values and constant
  lr; every eval record carries {step, val_loss_nats}; eval steps are
  exactly the multiples of `eval_every` plus the final step.
- V5.24 checkpoint roundtrip: reloading `out_dir/model/` reproduces the
  saved model's `evaluate_nll_nats` on the val set exactly;
  `training_meta.json` fields match the returned `TrainResult`.
- V5.25 with stopping effectively disabled (huge k), training stops at
  exactly `max_steps` with `stop_reason="max_steps"`.

### 6.2 Run-1 launch surface (scripts — single-pass)

`experiments/elicit-vs-teach/scripts/train.py` + `configs/`: parses the
run YAML, builds the Llama-3.2-1B-shaped `LlamaConfig` from the config
block, loads + packs TinyStories-v2 (the only place `datasets` is
imported), registers the run in zoo (spec 00 §2 required fields;
`experiment` block rides as preserved extra fields until its validation
task lands), prints a cost estimate and refuses to run without
`--confirm-cost` (CLAUDE.md budget rule), then calls
`geode.train.train_full`. Pretrain hyperparameter values in
`run1_pretrain.yaml` are placeholders — OPEN(11) — and say so inline.

**Runs 5–6 (LoRA target):** use `train_prequential` as-is — pre-update
losses, gradstats (per-module grad norms already covered), PEFT adapter
snapshots at `manifest.snapshot_steps`. Additions needed: LR + train-acc
scalars per step (small logging extension). Probe loss/acc are **not**
computed in-training: the extraction pass (§7) yields per-example probe
loss at every snapshot, and early snapshots are per-step anyway, so probe
curves at snapshot resolution come free. Snapshot schedule: 1024 steps,
log-spaced early (every step through ~30, then stretching), uniform later
— produced by `geode.probe.schedule` and written into the manifest before
launch.

**Optimizer state:** ~10 checkpoints, optional — OPEN(10).

## 7. `geode.probe` — schedule, extraction, analysis metrics

**Snapshot scheduler.** `snapshot_steps(total_steps, n=1024, dense_until≈30)`
→ strictly increasing, includes first and final step, dense unit-stride
prefix, log-then-uniform tail. Exact parameters OPEN(4).

**Extraction pass** (offline, separate rental). For each snapshot: load
base + adapter; run the probe set forward + backward (loss on label
tokens, **sum** reduction); capture activations and activation gradients
at all 17 residual points (embedding output + 16 post-block residuals for
this arch — hook count is n_layers+1, never hardcoded), per example,
bf16, padded + attention mask stored alongside. Write one safetensors
file per (snapshot, quantity ∈ {acts, grads}) containing 17 named
tensors; sidecar meta: run_id, arm, step, probe_set_hash, tokenizer_hash,
base_model_key, dtype. Per-example probe loss saved per snapshot (late
gradients are numerically degenerate — analyses condition on nonzero
loss).

**Matched-load guard** (V0.4 pattern re-asserted at this surface): the
pairwise loader for cross-arm comparison refuses unless probe_set_hash,
tokenizer_hash, and template/format metadata all match, with a clear
error.

**Analysis metrics** (pure functions over dumps; results written through
the ZOO-4 writer as spec 00 §7 long-format rows, `regime` column = arm):

- Cross-example activation-gradient alignment per (snapshot, layer):
  pairwise-cosine summary and top-PC explained-variance fraction of the
  per-example gradient matrix. Expectation: elicitation ⇒ near-parallel;
  teaching ⇒ diverse.
- Representation drift from the init snapshot, per layer, per digit class.
- Adapter diffs: cumulative ‖ΔW‖, effective rank, per-layer allocation.
  Needs a small weight-diff helper written fresh in
  `analysis/adapters.py` (LoRA ΔW = B@A × α/2r per module, ~20 lines).
  (`geode.steering` was planned but never built and was deleted in the
  2026-07-17 cut; its V2.6 was a spec property, never a test.)
- Performance-aligned matching: map snapshots across arms at equal probe
  accuracy (primary comparison axis); step-aligned secondary.

**Validation properties:**

- V5.8  schedule: exact count, strictly increasing, dense prefix, includes
  first + final step.
- V5.9  extraction captures n_layers+1 hook points with correct shapes and
  stored mask on a tiny fixture model.
- V5.10 per-example gradients match an explicit one-example-at-a-time
  backward reference (sum reduction ⇒ equality), and are nonzero iff the
  example's loss is nonzero.
- V5.11 dump ↔ load roundtrip preserves values (bf16), names, metadata.
- V5.12 matched-load guard raises on probe_set_hash / tokenizer_hash /
  template mismatch with a clear error.
- V5.13 alignment metric: planted parallel gradients ⇒ ≈1; random
  gradients ⇒ ≈0 (with the n≪d caveat pinned numerically).
- V5.14 drift: zero at the init snapshot; planted per-class shift
  recovered per class.
- V5.15 effective rank: planted rank-r adapter delta ⇒ r.
- V5.16 performance-aligned matching: monotone in accuracy; planted
  curves ⇒ known pairing; ties broken deterministically.

## 8. Verification gates

Recorded per run in `experiment.gates` (§4); a child run refuses to start
while a parent gate fails. Thresholds frozen at pilot where marked.

| Gate | After | Check |
|------|-------|-------|
| G1 | run 2 | Arm A near ceiling on NL add/sub (threshold ~≥95%, frozen at pilot) |
| G2 | run 3 | Arm A still near ceiling on NL add/sub (installer didn't corrupt; δ frozen at pilot) |
| G3 | run 4 | Arm B ≈ 0% on real add/sub (random labels didn't leak; ≤ chance + margin) |
| G4 | runs 3, 4 | Format validity on operator-notation prompts (both arms; ~≥99%) |
| G5 | runs 3, 4 | Zero/16-shot operator add/sub. Expectation: A ~2%/12%, B 0%/0% — the only remaining independent regime evidence |
| G6 | data gen | V5.1/V5.2 integrity checks run against the *real* generated sets; hashes recorded |
| G7 | before run 6 | `data_order_hash`(run 5) == `data_order_hash`(run 6), enforced at launch |

## 9. Analysis deliverables

One driver script per analysis under `experiments/.../analysis/`, each
writing `results/{analysis}.parquet` via ZOO-4 (join key: run_id,
checkpoint_step, layer). Primary axis: performance-aligned snapshot
pairs; secondary: step-aligned. Late-training gradient analyses condition
on per-example probe loss > 0 (§7). Figures land in `analysis/figures/`
(gitignored); the pilot must produce at least one real
gradient-alignment plot end-to-end (§11).

## 10. HF publication

`export_hf.py`: builds `hf-staging/` (§3 layout), exports
`manifest.parquet` from zoo, uploads in 50–100-file commits with retry +
resume, verifies remote file listing + sizes/hashes against staging, and
never deletes remote content without an explicit flag. Pilot repo is
separate (`-pilot` suffix). Script-level code (scripts — single-pass) but must
have a `--dry-run` mode that prints the commit plan without network.

## 11. Pilot protocol (runs before any production run)

End-to-end de-risk at toy scale: ~10K target examples, 20 snapshots, 64
probe examples, through training → extraction → upload (pilot repo) → one
real gradient-alignment plot. Then parameter pilots close open items:

- OPEN(1): installer convergence sweep on random-label mult ⇒ example
  count (identical for both arms).
- OPEN(2): train Arm B to convergence at 10K/50K/200K/500K ⇒ smallest n
  where B lands within a few points of A. (Paper's teaching peak for this
  task ≈300K; 10K likely far too small — Fig 1(b)'s "60K" is a cartoon,
  not a measurement.)
- OPEN(3): ε, k frozen from pilot validation curves.
- OPEN(4): batch → step count → snapshot-schedule parameters (needs
  OPEN(2)).
- OPEN(5): tokenize the frozen `Question:/Answer:` templates ⇒ padded max
  seq_len (OPEN(9) template string decided 2026-07-17).

Pilot outcomes are logged in `notes/decisions.md`, then the `OPEN(n)`
markers in this spec are replaced with pinned values in the same PR.

## 12. Open items

| ID | Item | Closed by |
|----|------|-----------|
| OPEN(1) | Format-installer example count | pilot (installer convergence) |
| OPEN(2) | Target dataset size | pilot (B convergence sweep) |
| OPEN(3) | Stopping-rule ε, k | pilot validation curves |
| OPEN(4) | Batch → step count → snapshot schedule params | after OPEN(2) |
| OPEN(5) | Padded max seq_len | tokenizer check on frozen template |
| OPEN(6) | Random-label sampling distribution (installer) | decision before pilot; default digit-count-matched uniform |
| OPEN(7) | Subtraction negatives allowed | decision before pilot; default allowed |
| OPEN(8) | Run 1: pretrain from scratch vs external TinyStories checkpoint | mentor (cost flag: run 1 is the most expensive item and pure reproduction; no existing checkpoint matches Llama-3.2-1B arch exactly) |
| OPEN(9) | Exact template string (both formats) | **decided 2026-07-17**: two-line `Question: <body>` / `Answer: <answer>` scaffold; padded length still OPEN(5) |
| OPEN(10) | Keep optimizer-state snapshots (3.6 GB) | decision before production run 5 |
| OPEN(11) | Run-1 pretrain hyperparameters (LR + schedule/warmup, seq len, batch, epochs/tokens, val-split size, eval cadence) and tokenizer choice (vocab size drives embedding params; Llama-3.2 tokenizer is license-gated — mirror or alternative needed) | paper tables / mentor, before any `--confirm-cost` spend; config ships documented placeholders |

## 13. Limitations / notes

- Single seed (paper uses 3) — stated limitation in any write-up.
- No Arm C; generic-transfer confound assumed away per paper Table 6.
- G5 is the only independent regime evidence retained after dropping
  EDL/D.
- Losses in nats with `_nats` suffixes, device-agnostic code, explicit
  seeds, CPU-only tests on tiny fixture models — all CLAUDE.md
  conventions bind on `geode.arith` / `geode.probe` exactly as on
  existing modules.

Next step: implement `geode.arith` (§5) with property tests V5.1–V5.7,
then `geode.probe` (§7) with V5.8–V5.16, then pilot (§11). Order and
rationale: `EXPERIMENTS.md` §4.
