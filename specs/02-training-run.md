# 02 — Training Runs: experiment organization & requirements

Status: **design, pre-implementation** (2026-07-16). Defines folder/run
organization and per-part requirements for the mechanistic
elicitation-vs-teaching comparison. EDL/D signature analysis is dropped;
regime classification is assumed from Bits That Count. `OPEN(n)` markers
denote unresolved items collected in §12. Nothing here launches GPU work
without `--confirm-cost` (CLAUDE.md budget rule).

Decisions locked 2026-07-16 (owner):

- **Code home:** thin scripts/configs in `experiments/training-run/`;
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

Decision locked 2026-07-18 (owner) — **architecture downscale**: the
Llama-3.2-1B arch is replaced by a custom small Llama-style config —
hidden 512, 8 layers, intermediate 2048 (4×d), 8 heads (head_dim 64),
num_key_value_heads 8 (plain MHA; GQA is an inference optimization,
unneeded at this scale), RoPE/RMSNorm/SwiGLU/pre-norm kept, tied
embeddings, vocab ~10K from a **custom BPE tokenizer** trained on
TinyStories-v2 with digits 0–9 forced as single tokens (plus `+`, `-`,
`*` and the template literals) — single-digit tokenization makes
arithmetic easier to learn and removes Llama's 3-digit chunking.
~25–34M non-embedding params; ~77 MB full checkpoint (bf16).
Consequences threaded through this spec: 9 residual points (§7), LoRA
r=128 (§6), adapter-only snapshots + a once-per-run base file (spec 00
§1, 2026-07-22), and all
1B-derived numbers (paper Table-3 LRs, the ~300K teaching peak,
capacity thresholds) are void — the pilot re-establishes them (§11).

Contents: §1 Runs, arms, DAG · §2 Repository layout (code) · §3 Store layout
(artifacts) · §4 Zoo schema additions · §5 `geode.arith` · §6 Training runs
(§6.1 `geode.train`, §6.2 run-1 launch surface) · §7 `geode.probe` · §8
Verification gates · §9 Analysis deliverables · §10 HF publication · §11
Pilot protocol · §12 Open items · §13 Limitations / notes

## 1. Runs, arms, DAG

| # | run_id (proposed)      | Role            | Init      | Method  | Data |
|---|------------------------|-----------------|-----------|---------|------|
| 1 | `evt-run1-base`        | pretrain        | random    | full FT | TinyStories-v2 (~2.6M stories), custom small arch (2026-07-18) |
| 2 | `evt-run2-armA-algo`   | pre-teach       | run 1     | full FT | NL add/sub, correct labels, 1M unique, to convergence |
| 3 | `evt-run3-armA-inst`   | format install  | run 2     | full FT | operator-notation mult, random labels, to behavioral stop (§6) |
| 4 | `evt-run4-armB-inst`   | format install  | run 1     | full FT | identical dataset + order as run 3, same behavioral stop (§6) |
| 5 | `evt-run5-armA-target` | target          | run 3     | LoRA    | operator-notation add/sub, 500K prefix (OPEN(2) closed 2026-07-22) |
| 6 | `evt-run6-armB-target` | target          | run 4     | LoRA    | identical data + identical order as run 5 |

DAG: `1 → 2 → 3 → 5` (Arm A) and `1 → 4 → 6` (Arm B). Arms differ **only**
in run 2's presence. No Arm C (generic-transfer confound assumed away per
paper Table 6). Single seed — recorded limitation (§13). Run 1 is
pretrained from scratch (OPEN(8) closed 2026-07-18: the custom arch +
tokenizer match no external checkpoint, and at ~30M params the run is
single-GPU, <30h territory).

Every run is registered in zoo before training starts and marked complete
only after its gates (§8) pass. A run refuses to launch if its parent run
is missing, incomplete, or has failing gates.

## 2. Repository layout (code)

```
experiments/training-run/
  README.md            # experiment card: goal, arm definitions, DAG, gate status
  configs/
    common.yaml        # shared blocks: optimizer, precision, batch, LoRA
    run1_pretrain.yaml … run6_armB_target.yaml     # one per run, §6 contract
    pilot/             # same six files, pilot-sized overrides (§11)
  datagen/             # one-time dataset/tokenizer generation (CPU, outputs frozen)
    make_data.py       # geode.arith → datasets + probe set + hashes
    make_tokenizer.py  # frozen custom BPE → tokenizer/
  scripts/             # GPU/box operations; cost paths gated by --confirm-cost
    train.py           # dispatch per config: full-FT trainer | train_prequential
    extract.py         # offline probe pass over snapshots (§7)
    gates.py           # run verification gates, write results into zoo manifest
    export_hf.py       # build hf-staging layout, export manifest.parquet, upload
  analysis/            # CPU post-hoc drivers → zoo results/, plus plotting
    alignment.py  drift.py  adapters.py  matching.py
    plot_losses.py  sample_stories.py
    figures/           # gitignored — ALL figures land here
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
  experiments/training-run/
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
  bases/                               # run1 final + run2/3/4 finals (~0.3 GB)
  armA_elicit/
    snapshots/{000,001,002}/step_XXXXXXX.safetensors   # full model state (spec 00 §1)
    acts/{000,...}/step_XXXXXXX.safetensors      # 9 named tensors each (§7)
    grads/{000,...}/step_XXXXXXX.safetensors
  armB_teach/                          # same shape
  optimizer/                           # optional, ~10 ckpts — OPEN(10)
```

Chunking: subdir index = snapshot_index // 500 (keeps every dir ≤ 1000
files). One file per (snapshot, quantity). Uploads in commits of 50–100
files, resumable, verified against local hashes after push. Budget
(2026-07-18 arch, rough): ~0.4 GB/snapshot × 1024 × 2 arms ≈ ~0.9 TB
tensors + ~158 GB full-model snapshots (~77 MB each) + <1 GB bases +
logs < 1 GB; re-estimate at pilot (padded seq_len closed 2026-07-18:
per-example max 33 tokens).

## 4. Zoo schema additions (spec 00 edit, same PR as implementation)

One optional `experiment` object on the run manifest, validated when
present:

```json
"experiment": {
  "name": "training-run",
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
closed 2026-07-18: max 33 tokens, see OPEN(5)): operator body `a op b`
(e.g. `Question: 23 + 45` then `Answer: 68`),
NL body `What is the sum of a and b?` / `What is the difference between a and
b?` (add/sub only). A third format, `bare_nl` (ADDITIVE, 2026-08-12,
fig2nl3/EXPERIMENTS §6.13), deliberately drops the scaffold: the bare NL body,
one newline, the answer as plain continuation (`What is the sum of 23 and
45?` newline `68`) — the fig2nl2 outcome measured that the scaffold alone
pre-installs the output convention in both sweep arms (base Llama ~0.31
zero-shot EM, ~0.83 format validity untrained), voiding the paper's Figure-2
pre-elicit transient; `bare_nl` restores the paper's regime. It reuses the
frozen `_NL_PHRASE` bodies byte-identically and leaves both frozen formats
untouched. A fourth format, `bare_op` (ADDITIVE, 2026-08-28, ts1b op-install
/ paper App. I.2.1), is the paper's pre-training-intervention surface: bare
operator notation, no scaffold, answer after `" = "` (`23 + 45 = 68`; the
paper's literal example `2 * 3 = 6`). It exists so the intervention task
(op-form add/sub install) and the `bare_nl` target share no surface form
beyond digits — the op→NL latency claim is exactly that transfer. Its
prompt-side trailing space is the same whitespace-overhang boundary as the
frozen `Answer: ` scaffold (V5.38); both frozen formats and `bare_nl` remain
untouched. Label modes:
correct | random | permuted (2026-07-26, new-phase teach installer: the true
answers shuffled across examples via `geode.arith.permute_labels` — each label
individually wrong up to chance collisions while the marginal label
distribution is exact by construction, so the answer-shape prior installs and
the mapping carries no signal; §6 new-phase block, V5.64). Random-label
sampling distribution OPEN(6) (default:
uniform over answers with digit-count distribution matched to true
answers). Subtraction negatives OPEN(7) (default: allowed). Datasets are
generated **once** by `datagen/make_data.py` and frozen to files (not
regenerated at train time); `geode.arith` supplies only rendering, the
random-label rule, evals, the water-fill allocation, and the validators.
Every emitted example carries the answer **character** span (tokenizer-
agnostic); token-level label spans are derived at load against the frozen
tokenizer (`experiments/training-run/tokenizer/`, built 2026-07-18), and
masking then goes through `geode.edl.masking.label_mask`
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

**Eval set** (`D_target_eval`, owner 2026-07-22 — supersedes the 900k
eval-reserved tail of the same date, which capped training at 900k rows).
100,000 fresh operator add/sub examples generated by `make_data.py
--eval-set` with the **union of D_target ∪ D_algo ∪ probe questions as the
exclusion** (D_algo included because Arm A's pre-teach trained on those
exact questions in NL notation — overlap would advantage A asymmetrically;
D_inst is multiplication, disjoint by op). Every eval question is provably
never-trained while the full 1M D_target stays trainable. The six cells
with x_digits + y_digits ≤ 4 have their add/sub question space fully
consumed by the frozen sets (verified 2026-07-22) and contribute zero rows
(≈5.2% of the training distribution, water-fill gives them 0 naturally);
the other ten cells carry 10,000 each. Frozen, hash-pinned, and uploaded
like the training sets (`report.json` records `disjoint_from` pins).
Structure: rows 0–2047 = the ε/k **stopping block** (in-loop stopping
evals, identical for every run and every n); rows 2048+ = the **reporting
block** — the harness's final θ_T test loss (`eval/test_loss.json`, hence
EDL = MDL − N·L_test) and G5's fixed shots/questions. No reported number
touches the rows that drove the stop decision. The launcher pins it as
`data.eval_file`/`data.eval_order_hash` and refuses on hash mismatch;
`data.val_fraction` is retired for target runs — the training prefix
trains whole.

**New-phase installer sets (owner 2026-07-26).** `make_data.py
--installer-set` generates the role-matched installer artifacts against the
frozen files: `D_inst_perm` — 200K add/sub operator-notation examples with
**permuted** labels, question-disjoint from D_target ∪ D_algo ∪ probe ∪
D_target_eval (a target question ever seen with a wrong label would
contaminate the prequential first-sight assumption) — and `D_dose_mult` —
16 correct-label mult examples, one per `(x_digits, y_digits)` cell,
disjoint from D_inst; an elicit dose of size n is a prefix of this file's
frozen order. Both are hash-pinned in report.json; the permuted set records
its `label_coincidence` (0.0145% at the frozen seed 20260717 — below even
D_inst's accepted 0.07%).

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
- V5.64 permuted-label mode (2026-07-26): `permute_labels` returns exactly
  the input multiset (marginal preservation — duplicates and negatives
  included), deterministically in `seed`; different seeds give different
  permutations; on all-distinct inputs the fixed-point count stays at
  chance level (the mapping is destroyed).
- V5.67 translation bridge (phase 3, 2026-07-27): every sampled positive
  addition pair emits exactly one NL→operator and one operator→NL row under
  the shared `Question:/Answer:` scaffold; the answer span covers the rewritten
  question body exactly, and no rendered row contains `-` or the computed sum.
  Train and held-out translation pairs are disjoint, probe questions and exact
  target pairs are excluded, and allocation uses the same capacity-capped
  water-fill over phase 3's 64 operand-length cells.
- V5.68 input-embedding warm-start (phase 3, 2026-07-28): untying the language-
  model head is a bitwise logit no-op before training; a row-masked optimizer
  changes only the declared input-embedding rows and leaves the frozen output
  head and every other parameter bit-identical. A prompt that contains none of
  the changed rows therefore keeps bit-identical logits. The practical
  warm-start corpus is deterministic, correct-label NL addition, and excludes
  both direct questions and answer-identical commuted twins from the frozen
  phase-3 parent, target, eval, and probe sets.
- V5.69 warm-start selection (phase 3, revised 2026-07-28): all persisted LR
  candidates are ranked by held-out NL exact match, held-out masked NLL, and
  smaller LR in that order. Operator-addition retention is recorded evidence but
  never filters or ranks candidates. The target stream, D_p3_nl_eval, and target
  EDL are never selection inputs.
- V5.70 frozen-parquet load (2026-07-29): `geode.arith.load_frozen_parquet(d,
  root=...)` recomputes the content-and-order hash (V5.40) over the fetched file
  and refuses (`ValueError`) unless it equals `d["order_hash"]`; rows come back
  in the frozen order; a relative `data.local_path` resolves under the injected
  `root`, an absolute one as-is, and neither path skips the hash check. The SFT
  and target launchers keep thin adapters that bind `root=REPO_ROOT` and preserve
  their existing `(cfg)` / `(d)` call contracts, so every importer is unchanged.

## 6. Training runs — per-run needs

**Config contract (one YAML per run):** run_id, experiment/arm/role,
init (parent run_id | external | random), model (arch config), data
(task, format, label mode, n_examples, seed), train (optimizer, lr,
schedule, clip, precision, batch, stopping), lora (target runs only),
snapshots, logging. Fixed values (paper conventions where they survive
the 2026-07-18 downscale):

- AdamW β₁ 0.9, β₂ 0.999, wd 0.01; constant LR; grad clip 1.0; bfloat16.
- LR: the paper's Table-3 values (2e-5 full FT / 3.53e-4 LoRA) were
  tuned for 1B and are void at this scale — LR is pilot-determined
  (starting points ~3e-4 pretrain, ~1e-4 full-FT pre-teach, ~3e-4 LoRA;
  brief sweep; executed runs record theirs in their manifests). LoRA
  target LR pinned 1e-3 for runs 5–6 (sweep 2026-07-22 — grid extended
  upward to bracket the monotone first decade; decisions.md). Batch 128;
  n=500K pinned (OPEN(2) closed 2026-07-22, §12); snapshot schedule
  pinned over max_steps 23442 (OPEN(4) closed 2026-07-22, §12).
- LoRA (runs 5–6 only): r=128; Q,K,V,O,G,U,D all layers; α=32; scaling
  α/2r; dropout 0; A Kaiming 1/√d_in, B zero. 12.1M params,
  ~24 MB/adapter bf16.
- Loss on label tokens only, identical masking train/test (masking hash
  guard from spec 00 §5 applies as usual).
- Stopping (runs 1–2): validation-loss convergence with **ε=0.002 nats,
  k=5, min_steps=5000 grace (V5.42), eval_every 1000** — one canonical
  rule for the loss-stopped runs (owner 2026-07-21; supersedes the
  0.005/3 close of OPEN(3) after v3 hit its ceiling still descending at
  2.5–3.0 mnat/1k — 0.005/3 abandons descent below 1.7, the new rule
  holds on until 0.4). Runs already executed keep the rule their
  manifest records. Every run trains until the rule fires; `max_steps`
  is a generous cost ceiling (~10–15 epochs; run 2's raised to 15 at
  the LR pin, owner 2026-07-21), never a planned stop —
  `stop_reason="max_steps"` means "did not converge": investigate,
  don't ship. This matches the paper: it also trains to convergence,
  and its "1 epoch" figure (formerly echoed in the §1 row-2 cell and
  misread as a training cap) is **information accounting** — bits per
  example are counted on first exposure (first epoch) only, so
  multi-epoch training changes no measured quantity.
- Stopping (runs 3–4, format installers — owner 2026-07-21):
  **behavior-matched, not loss-based.** Each arm stops at the k-th
  consecutive in-loop format-validity eval scoring ≥99% — the G4 metric
  (greedy decode on 512 held-out operator-notation prompts, output
  parses as a number in the answer slot) — with **k=3, eval every 250
  steps**. Both arms consume the identical frozen `D_inst` in the
  identical order, so the earlier-stopping arm's exposure is a strict
  prefix of the other's; per-arm step counts are **emergent, not
  matched**, recorded in each manifest (closes OPEN(1)). Rationale: the
  installer's goal is a behavior (the answer scaffold), not a loss
  minimum — random labels floor the val loss near ln 10 per answer
  digit, and the ε/k rule's slow tail would spend steps absorbing the
  one learnable signal left (the digit-count leak, accepted won't-fix
  2026-07-19), the anti-goal. Matching step counts instead would pin
  both arms to the slower learner (Arm B, format-naive init) and
  concentrate maximal post-saturation surplus on Arm A — the arm whose
  arithmetic G2 must certify intact. An identical pre-registered rule
  makes any duration difference a *mediator* of run 2's presence (the
  treatment), not a confound. Val loss is still logged every eval for
  the record. `max_steps` stays a pure ceiling at ~2 epochs (repeat
  epochs over random labels invite label memorization); a ceiling exit
  means the format never installed — investigate, don't ship. Rule
  parameters are frozen before either installer launches; the in-loop
  format-validity eval is new trainer tooling (§6.1), lands with the
  run-3 task, and is property-tested (its silent failure would break
  the matched-arms design). Target runs 5–6 are the EDL measurement
  itself — their training schedule is part of the metric; the ε/k rule
  is ratified below (2026-07-22); n=500K pinned (OPEN(2) closed
  2026-07-22, §12), snapshot schedule pinned (OPEN(4) closed same day).
- Stopping (new-phase installers, owner 2026-07-26 — the dose phase;
  executed runs 3/4/9 keep the rule their manifests record): installers
  are **role-matched, not identical** — each arm gets the installer that
  is non-destructive for its state, replacing the identical-installer
  design. Motivation (step-0 measurement 2026-07-25, decisions.md
  2026-07-26): Arm A's G4 criterion was saturated before training began
  (1.0000 at step 0), so the "shared" rule only ever measured Arm B, and
  the mult-shaped random labels corrupt the add/sub length prior by
  construction. **Teach**: `D_inst_perm` (permuted add/sub — correct
  marginals install the true answer-shape prior), stop at the k-th
  consecutive G4 eval ≥ **0.90** with **k=3, eval every step** (batch 128
  ⇒ a 3-step floor in place of the old 750-step one; a tiny-dose config
  at batch 1 makes the cadence literally per-example); the step-0 value
  is recorded always. **Elicit**: the dose — a prefix of `D_dose_mult`,
  1 real correct-label mult example at the smallest dose — stops on the
  ε/k plateau of the **full-dose training loss**
  (`stopping_metric: train_loss`, batch = dose; V5.65/V5.66): G4 is
  saturated at step 0 for this arm and can never be its stop, and "the
  dose is absorbed" is the convergence-policy-consistent stop. LR 3e-6
  inherited from the installer retention sweep for both arms, and **G2
  retention gates it** — scope re-validation by gate, not assumption
  (the run-9 lesson). **EDL accounting, same decision:** the installer
  now deliberately installs format **and answer-shape**, so the new
  phase's target EDL is **mapping-only** — conditional on format+shape,
  billing only the question→answer mapping. This deliberately reverses
  the digit-count-leak anti-goal above *for the new phase*: matched,
  correct shape priors across arms replace the shape-naive start, and
  the change shrinks teach's EDL — conservative for the elicit-vs-teach
  ratio. `max_steps` stays a pure cost ceiling; a ceiling exit means
  investigate, don't ship.
  **Built out 2026-07-26** (configs `p2_*`, decisions.md "new phase built
  out"): the dose grid is **n ∈ {1, 2, 4, 8, 16}**, each dose a prefix of
  the frozen `D_dose_mult` order (so the doses nest), and **each installer
  gets its own target run** — five dose targets plus one teach target, all
  on the identical frozen 1M `D_target` order, rule and ceiling as runs
  7/8, with `snapshots.n: 0`. The dose rule's **ε/k is pinned from
  calibration pilots** run at both ends of the grid with `eps_nats: 0.0`
  and replayed through `ConvergenceTracker`
  (`analysis/dose_stop_calibration.py`), never inherited from the target
  stage: a coarse rule was measured to fire at 99.08% of descent at n=1
  but 93.70% at n=16, which would have made the dose-response curve partly
  a measurement of its own stopping rule. Until the pin is set the config
  carries a null ε and both the launcher and `launch_phase2.sh` refuse.
  The phase runs **fp32 in both arms and both stages** (the target harness
  already is), and **step 0 is recorded for every run**
  (`experiment.step0`) — the phase-0 defect fixed at its source.
  **The dose grid RAN and its result retired it — see the next bullet
  before acting on any of the elicit-dose text above.**
- Installers, owner revision (2026-07-26, after the dose grid ran):
  **the elicit arm gets no installer at all.** The dose grid measured the
  mult dose to be monotone damage at every size (target zero-shot 0.1016
  → 0.0068, test loss 5.1935 → 6.8277, G2 retention breaking at n=16);
  there is no dose at which the intervention is neutral, so n=0 is the
  elicit arm's installer. What the two arms must share is the **state**
  at the start of the target stage — format-valid and holding the true
  answer-shape prior, with no target mapping — not the *procedure* of
  having had an installer stage. Arm A's parent (`evt-run2-armA-algo`)
  measures that state directly: G4 1.0000 and mean answer digits 3.726
  against a true 3.746 (phase 0b), so it satisfies decision 5's
  mapping-only precondition without training. Arm B's parent does not
  (G4 0.0039), which is exactly why it still takes `D_inst_perm`.
  Consequences: the phase is **two runs**, one target per arm, sharing
  the frozen 1M `D_target` order (G7 anchor = the Arm A target); the
  elicit target inits from `evt-run2-armA-algo` with `G1` as its only
  parent gate; the teach installer's rule, data and LR are unchanged.
  The residual asymmetry is now **exposure**, and it runs the other way:
  Arm B sees its installer's examples unbilled under mapping-only EDL
  while Arm A sees none, which favours teach and so cannot manufacture
  the elicit result. It is bounded by the teach installer's own stop
  (G4 ≥ 0.90, k=3, per step) and **reported as examples seen**
  (`final_step` × batch), not assumed small.
- Stopping (runs 5–6, target runs — owner 2026-07-22): the canonical
  loss rule at short-run cadence — **ε=0.002 nats, k=5, eval_every 500,
  min_steps 0**. The canonical min_steps=5000 grace and eval_every 1000
  were scaled for ~100k-step runs; at ~390-step epochs (50K prefix,
  batch 128) the k=5 patience is the grace. Ratified before the OPEN(2)
  grid so the grid runs under the production rule; the target-LR sweep
  arms ran under the superseded 0.005/3 their manifests record (LR
  ranking unaffected — the winner's margin was ~3×). The rule is part
  of the metric: it sets θ_T, hence L_test, hence EDL = MDL −
  N·L_test. Stopping evals run on the fixed **stopping block** — the
  first 2048 rows of the frozen eval set (§5, owner 2026-07-22), the
  identical data for every run and every n; the OPEN(2) pilots
  predate this and stopped on per-run 0.5% val carves their manifests
  record. **Curve evals** (owner 2026-07-22): the launcher additionally
  evaluates the stopping block at a dense-then-log-spaced set of steps
  (`snapshot_steps(max_steps, n=64, dense_until=16)` — the same tested
  scheduler as snapshots) so the val curve has resolution on a log step
  axis. These rows land in `eval_log.jsonl` with `stopping_eval: false`
  and are **never fed to the ε/k tracker** — the ratified rule consumes
  exactly the eval_every cadence (+ the ceiling eval). Logging-only:
  they change no stop decision and no reported number.

**Runs 1–4 (full FT):** need a small full-FT trainer with validation-loss
stopping; snapshots = final checkpoint only (plus the base). **Decided
2026-07-16 (task TRAIN-1):** a separate thin module `geode.train` (§6.1),
leaving the validated prequential loop untouched. Run 1 needs only the
pretrain mode (loss over all next-token positions); the label-masked SFT
mode for runs 2–4 routes through `geode.edl.masking.label_mask` and landed
2026-07-19 as `geode/train/sft.py` (§6.1).

### 6.1 `geode.train` — corpus packing + full-FT trainer (task TRAIN-1)

Files: `geode/train/{__init__,packing,stopping,loop,sft}.py`. All CLAUDE.md
conventions bind (device-agnostic, explicit seeds, `_nats` suffixes,
CPU-only tests on tiny in-process models). The module never touches the
zoo registry — registration is the launch script's job (§6.2) — and never
imports `datasets`/network loaders; it consumes in-memory text/token streams.

```python
# geode/train/packing.py
def split_documents(lines: Iterable[str], delimiter: str = "<|endoftext|>") -> Iterator[str]
    # Cut a raw-text line stream into documents on delimiter-only lines
    # (the TinyStories txt convention). Documents are stripped; empties
    # dropped; the delimiter never appears in output — a mid-line
    # delimiter raises ValueError (silent corpus corruption otherwise).
    # Lazy: multi-GB files stream.
def pack_corpus(texts: Iterable[str], tokenizer, seq_len: int,
                *, chunk_tokens: int = 1 << 20,
                batch_docs: int = 1024) -> torch.LongTensor
    # Tokenize each document (no special tokens added), append exactly one
    # eos_token_id after every document, concatenate in input order, slice
    # the stream into consecutive rows of length seq_len, drop the short
    # tail. Streaming: documents are consumed batch_docs at a time and
    # tokenized as one batch (the fast tokenizer parallelizes encode_batch
    # across CPU cores); completed rows move to tensor storage whenever the
    # token buffer reaches chunk_tokens (the buffer never exceeds
    # chunk_tokens + one batch of documents), so full-corpus packing costs
    # the output tensor's memory, not a full-stream Python list. Neither
    # chunk_tokens nor batch_docs changes the result. Raises ValueError if
    # tokenizer.eos_token_id is None, seq_len < 2, chunk_tokens < seq_len,
    # or batch_docs < 1. Deterministic: a pure function of
    # (texts, tokenizer, seq_len).
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
    min_steps: int = 0  # grace: evals before this step never count
class ConvergenceTracker:
    def __init__(self, rule: StoppingRule): ...
    def update(self, val_loss_nats: float, step: int | None = None) -> bool
    # An eval improves iff (best_so_far - val_loss_nats) > eps_nats
    # (strict; equality does NOT improve). Improvement updates best and
    # resets the stale counter; the k-th consecutive non-improving eval
    # returns True (and keeps returning True). NaN input raises ValueError.
    # Grace (2026-07-21): an eval at step < min_steps updates min_nats
    # ONLY — best stays frozen, the stale counter does not move, the
    # tracker cannot stop; the first counted eval is the first at
    # step >= min_steps (so the earliest possible stop is the k-th eval
    # after that). min_steps > 0 with no step passed raises ValueError.
    best_nats: float          # +inf before first update; eps-gated (freezes
                              # while improvements stay <= eps_nats)
    min_nats: float           # +inf before first update; exact min over
                              # EVERY value passed to update, no eps gate
                              # (added 2026-07-20: run-1 v2/v2-ext recorded
                              # a stale first-eval "best")
    stale_evals: int

@dataclass(frozen=True)
class BehavioralStoppingRule:  # runs 3-4 format installers (§6, 2026-07-21)
    threshold: float  # metric value that counts as a hit — INCLUSIVE (>=)
    k: int            # consecutive hits that trigger the stop
class BehaviorTracker:
    def __init__(self, rule: BehavioralStoppingRule): ...
    def update(self, rate: float) -> bool
    # The k-th CONSECUTIVE eval with rate >= threshold returns True; a
    # sub-threshold eval resets the count. Latched like ConvergenceTracker
    # (True forever after the stop). NaN raises ValueError. best_rate is
    # the exact running max over every update (-inf before the first),
    # the behavioral analogue of min_nats.

# geode/train/loop.py
@dataclass(frozen=True)
class TrainResult:
    final_step: int
    best_val_nats: float
    min_val_nats: float
    stop_reason: Literal["converged", "max_steps", "behavior"]
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
               precision: Literal["fp32", "bf16"] = "fp32",
               micro_batch_size: int | None = None,
               lr_schedule: Literal["constant", "cosine"] = "constant",
               min_lr: float | None = None) -> TrainResult
```

`train_full` contract:

- Optimizer AdamW(lr, betas, weight_decay); **constant** LR by default;
  global-norm grad clipping at `grad_clip`.
- LR schedule (added 2026-07-19 after gate G0 failed at the constant-LR
  loss floor; decisions.md): `lr_schedule="cosine"` sets the LR applied at
  1-indexed optimizer step `t` to
  `min_lr + 0.5*(lr - min_lr)*(1 + cos(pi*(t-1)/(max_steps-1)))` — exactly
  `lr` at step 1, exactly `min_lr` at step `max_steps`, non-increasing in
  between (`max_steps=1` degenerates to a single step at `lr`). The value
  is set on the optimizer before every update and is what the train log's
  `lr` field records. Guards (all `ValueError` upfront, before any disk
  write): unknown schedule name; cosine with `max_steps=None` (the
  schedule needs a fixed horizon); cosine with `min_lr` missing or outside
  `[0, lr]`; `min_lr` supplied with a constant schedule (likely config
  typo). Under cosine the plateau rule is **inert**: decay shrinks
  late-run improvements below any sensible `eps_nats` by design, so
  honoring it would cut the schedule short — the run always ends at
  exactly `max_steps` with `stop_reason="max_steps"`, while the tracker
  still records `best_val_nats` and `min_val_nats`. The SFT mode
  (`geode.train.sft`) is deliberately unchanged.
- Data order: a seeded permutation of `train_seqs` per epoch, derived
  deterministically from `seed` and the epoch index; fixed-size batches,
  drop-last. Epochs repeat until a stop condition fires (multi-epoch is
  fine here — this is pretraining, not prequential MDL; `geode.edl` guards
  are not in play).
- Step = one optimizer update. Loss = mean next-token CE per token (nats)
  over the batch.
- Gradient accumulation (added 2026-07-19: a full 128-row fwd+loss OOMs
  the 24 GB 4090): `micro_batch_size` (default `batch_size`) runs each
  step as `batch_size // micro_batch_size` sequential micro-batches whose
  `1/n_micro`-scaled losses accumulate gradients before the single
  clip + update. Must divide `batch_size` exactly, else `ValueError`
  upfront (equal micro-batches keep mean-of-means equal to the full-batch
  mean — in pretrain mode every position counts, so this is exact, not
  approximate). Effective batch, logged `train_loss_nats`, and all
  step/eval/stopping semantics are unchanged. In-loop evals run
  `evaluate_nll_nats` at `micro_batch_size` rows (value-safe by V5.19).
- Eval: `evaluate_nll_nats(val_seqs)` at every step where
  `step % eval_every == 0`, and additionally at the final step. Every eval
  updates one `ConvergenceTracker` (with the step, so `min_steps` grace
  applies); a True return stops training with
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
  `{"step", "train_loss_nats", "lr", "grad_norm", "time_unix"}` per step
  (`grad_norm` is the pre-clip global norm; `lr` is the per-step scheduled
  value — constant `lr` in constant mode); `eval_log.jsonl` with
  `{"step", "val_loss_nats", "time_unix"}` per eval. `time_unix` is the
  wall-clock `time.time()` at record write (added 2026-07-19 after the
  run-1 v2 launch: throughput questions were unanswerable from
  timestamp-free logs).
- Checkpoint: final model saved to `out_dir/model/` via `save_pretrained`,
  plus `out_dir/training_meta.json` recording `stop_reason`, `final_step`,
  `best_val_nats`, `min_val_nats`, and a `config` object echoing exactly the call
  arguments {lr, lr_schedule, min_lr, batch_size, micro_batch_size
  (resolved: equals batch_size when accumulation is unused), eval_every,
  max_steps, grad_clip, weight_decay, betas, seed, precision,
  stopping: {eps_nats, k}}.
  (Echo keys pinned 2026-07-16 after TEST-AUDITOR flagged the phrase as
  untestable; the echo itself stays untested — asserting it would overfit
  — but CONFORMANCE-REVIEWER checks it by inspection.)
- Determinism: identical seed + inputs on CPU ⇒ identical log files after
  deleting the `time_unix` field from every record (wall-clock is the one
  deliberately nondeterministic field).
- `precision="bf16"` wraps forward/loss in autocast; tests exercise fp32
  only (CPU) and treat bf16 as config plumbing.

**Label-masked SFT mode** (`geode/train/sft.py`, added 2026-07-19; runs
2–4):

```python
# geode/train/sft.py
def evaluate_sft_nll_nats(model, examples: Sequence[SpanExample],
                          task_format: TaskFormat, *, batch_size: int,
                          device: str) -> float
def train_sft(model, train_examples: Sequence[SpanExample],
              val_examples: Sequence[SpanExample], task_format: TaskFormat,
              *, lr, batch_size,
              stopping,  # StoppingRule | BehavioralStoppingRule
              eval_every, max_steps,
              grad_clip, weight_decay, betas, device, seed, out_dir,
              precision="fp32", behavioral_eval=None,
              stopping_metric="val_loss") -> TrainResult
```

- Examples are span-carrying (`input_ids` + half-open `label_span`, spec 00
  OQ-8); masks are built by `geode.edl.masking.label_mask` — the single
  mask path (§5) — and applied via the standard label=-100 convention.
  Loss = mean CE (nats) over label positions only; question/format tokens
  and padding never contribute to loss or gradients.
- Batches right-pad `input_ids` to the set max length (pad id 0), no
  attention mask — the `geode.edl.loop` convention: under causal attention
  a right-pad position cannot influence logits at earlier (label)
  positions.
- Span guard (upfront `ValueError`, before any training or disk write):
  every span must satisfy `1 <= start < end <= len(input_ids)` — a label
  at position 0 has no predecessor under the causal shift, an empty span
  contributes no loss, and a span past the sequence end would silently
  mark padding as labels.
- Everything else inherits the `train_full` contract verbatim except
  `micro_batch_size`, which is pretrain-mode only for now (SFT sequences
  are short Q/A pairs; add on demonstrated need) (optimizer,
  seeded per-epoch data order, step/eval/stopping semantics and tie-break,
  log schemas — `train_loss_nats` is the masked mean — final checkpoint +
  `training_meta.json`); the config echo additionally records
  `task_format: {name, format_version, span_source}`.
- `evaluate_sft_nll_nats` sums loss and label-token count over the whole
  set before dividing once (batch-size invariant, as `evaluate_nll_nats`).
- `masking_config_hash` recording (spec 00 §5) stays with the launch
  script (§6.2 pattern): the module has no tokenizer hash and never
  touches zoo.
- **Behavioral stopping mode** (2026-07-21, runs 3–4, §6): passing a
  `BehavioralStoppingRule` as `stopping` — with the paired
  `behavioral_eval: Callable[[], float]` closure, both-or-neither
  (upfront `ValueError` otherwise, before any disk write) — switches the
  stop decision to the in-loop behavioral metric. Every eval still
  computes and logs `val_loss_nats` (§6: "val loss is still logged for
  the record"), then calls `behavioral_eval()` and logs its value as
  `format_valid_rate` in the same eval record; the run stops at the k-th
  consecutive rate `>= threshold` with `stop_reason="behavior"`, or at
  the ceiling with `"max_steps"`. The loss plateau rule is deliberately
  never consulted in this mode. The module stays tokenizer-free: the
  closure owns the decode (the launch script builds it from
  `geode.arith.greedy_completions` + `format_valid`); the trainer
  restores `model.train()` after each call. `best_val_nats` and
  `min_val_nats` both report the exact running min (no eps gate is in
  play); the meta's `stopping` echo is `{threshold, k}`.
- **Train-loss stopping mode** (2026-07-26, new-phase dose installers,
  §6): `stopping_metric="train_loss"` makes the ε/k `ConvergenceTracker`
  consume the step's own training loss — exact, because the mode
  requires `batch_size == n_train` (every step consumes the whole dose;
  upfront `ValueError` otherwise) and a plain `StoppingRule` (upfront
  `ValueError` with a behavioral rule). The only mode that runs with
  empty `val_examples` — the other modes now refuse an empty val upfront
  instead of dividing by zero at the first eval; when a val set is
  provided it is still evaluated and logged for the record. Eval records
  carry `train_loss_nats` (plus `val_loss_nats` when val is present);
  `best_val_nats`/`min_val_nats` carry the stopping metric. The config
  echo gains `stopping_metric` (recorded in every mode).

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
- V5.22 same seed ⇒ identical `train_log.jsonl` and `eval_log.jsonl`
  across two runs (CPU, fixed threads) after deleting the `time_unix`
  field from every record.
- V5.23 log schema: every train record carries exactly
  {step, train_loss_nats, lr, grad_norm, time_unix} with finite values
  and constant lr; every eval record carries
  {step, val_loss_nats, time_unix}; eval steps are exactly the multiples
  of `eval_every` plus the final step.
- V5.24 checkpoint roundtrip: reloading `out_dir/model/` reproduces the
  saved model's `evaluate_nll_nats` on the val set exactly;
  `training_meta.json` fields match the returned `TrainResult`.
- V5.25 with stopping effectively disabled (huge k), training stops at
  exactly `max_steps` with `stop_reason="max_steps"`.
- V5.26 document splitting: documents are exactly the stripped text
  between delimiter-only lines, in order, empties dropped, delimiter
  never emitted; a mid-line delimiter raises; consumption is lazy.
- V5.27 packing streams: the packed tensor is identical for every valid
  `chunk_tokens` × `batch_docs` combination and equals a
  tokenize-all-then-slice reference; consumption is incremental — at most
  `batch_docs` documents are pulled before tokenization runs (the
  iterable is never drained first); `chunk_tokens < seq_len` or
  `batch_docs < 1` raises.
- V5.28 SFT loss reference: `evaluate_sft_nll_nats` and the step-1
  training loss equal a hand-computed mean CE over exactly the label-span
  positions on a tiny fixture model; the all-positions mean differs
  (the exclusion has teeth).
- V5.29 full-coverage reduction: with spans covering every predictable
  position (`(1, L)`) on an equal-length pad-free batch, the SFT loss
  equals `evaluate_nll_nats` on the same rows.
- V5.30 padding/question inertness: on a ragged batch the set loss equals
  the label-count-weighted combination of per-example pad-free losses and
  is batch-size invariant; parameter gradients of the training objective
  equal those of a pad-free, label-positions-only reference — padding and
  question tokens contribute zero loss and zero gradient.
- V5.31 span guard: a label span touching position 0, empty, reversed, or
  extending past its sequence raises `ValueError` before any training or
  disk write.
- V5.32 SFT determinism: identical seed + inputs on CPU (fixed threads) ⇒
  identical `train_log.jsonl` and `eval_log.jsonl` across two
  `train_sft` runs after deleting the `time_unix` field from every
  record.
- V5.33 SFT convergence: on a memorizable question→answer task,
  `train_sft` stops with `stop_reason="converged"` before a generous
  `max_steps`, with final train loss below initial.
- V5.34 gradient accumulation: `train_full` with
  `micro_batch_size < batch_size` reproduces the full-batch run — same
  logged per-step train losses and same final parameters within float
  tolerance (same seed, same init, fp32/CPU); a `micro_batch_size` that
  is 0, negative, larger than `batch_size`, or not a divisor of it
  raises `ValueError` before any training or disk write.
- V5.35 cosine schedule values: with `lr_schedule="cosine"` the logged
  per-step `lr` equals exactly `lr` at step 1 and exactly `min_lr` at
  step `max_steps`, is non-increasing throughout, and matches the
  closed-form half cosine at an interior point; constant mode still logs
  a constant `lr` (V5.23 unchanged).
- V5.36 cosine horizon semantics + guards: a stopping rule that halts an
  otherwise-identical constant-LR run early cannot end a cosine run —
  the cosine run reaches exactly `max_steps` with
  `stop_reason="max_steps"` and echoes `lr_schedule`/`min_lr` in
  `training_meta.json`; cosine with `max_steps=None`, `min_lr` missing
  or outside `[0, lr]`, an unknown schedule name, or `min_lr` supplied
  with a constant schedule raises `ValueError` before any disk write.
- V5.37 log timestamps: every train and eval record (both `train_full`
  and `train_sft`) carries `time_unix`; within each log file the values
  are nondecreasing and lie between wall-clock readings taken
  immediately before and after the run.
- V5.38 span conversion (`geode.arith.spans`, 2026-07-20): the frozen
  files' answer **char** spans convert to **token** spans exactly — the
  overlapping tokens form one contiguous, gapless run ending exactly at
  the span end, with overhang permitted only at the left edge and only
  over whitespace (the frozen byte-level BPE merges the space after
  `Answer:` into the sign token of negative answers, measured
  2026-07-20). Any inexact alignment raises; verified against the real
  frozen tokenizer artifact across both formats, all ops, negatives.
- V5.39 example split (`geode.train.split_indices`): the index-list form
  of the frozen `train_val_split` partition — same permutation, same
  clamping, byte-exact against V5.18 when used to index rows — shared by
  the SFT launch path and `gates.py` so a gate can never score trained
  examples.
- V5.40 order hash (`geode.arith.order_hash`, promoted from
  `make_data.py` 2026-07-20): deterministic, order- and
  content-sensitive, and invariant to parquet round-trip scalar types;
  reproduces the hashes frozen in `report.json`. The SFT launch path and
  `gates.py` refuse on mismatch against the config-pinned value.
- V5.41 true-min val tracking (2026-07-20): `ConvergenceTracker.min_nats`
  equals the exact minimum of every value passed to `update` — including
  sub-eps improvements that leave `best_nats` frozen, and values seen
  after the stop latches; +inf before the first update; NaN still raises
  without corrupting the min. `TrainResult.min_val_nats` and
  `training_meta.json` propagate it (motivation: run-1 v2/v2-ext, where
  the eps-gated `best_val_nats` overstated the loss actually reached —
  1.1140-recorded vs 1.1125-true on v2, 1.1110 vs 1.1066 on v2-ext).
- V5.42 stopping grace (2026-07-21): with `min_steps > 0`, evals at
  `step < min_steps` can never stop training, consume patience, or set
  `best_nats` — a k-long plateau entirely below `min_steps` does not
  fire, and the same value sequence re-fed at steps >= `min_steps` stops
  at exactly the k-th counted eval measured from there. `min_nats` still
  tracks every value including grace-period evals. `min_steps=0` (the
  default) reproduces the pre-grace behavior exactly; `min_steps > 0`
  with no step passed to `update` raises ValueError (motivation: run-1
  v3-ext — `--init-from` resets AdamW moments and the val transient must
  not trip or distort the plateau rule).
- V5.43 EOS termination (2026-07-21): `tokenize_with_spans(...,
  append_eos=True)` appends exactly one `tokenizer.eos_token_id` per
  example and extends the label span to cover it, so "stop after the
  answer" is trained behavior. An answer-span-only mask supervises no
  post-answer position, and the run-2 sweep's greedy decodes ran past
  correct answers into drifted junk — G1 read 0.0000 on all four
  converged arms (diagnosed 2026-07-21, decisions.md). Matches the
  pretrain convention of one EOS per document (§6.1), so the warm start
  already knows the token. Requires the label span to end at the
  sequence end and an EOS-bearing tokenizer; raises otherwise.
  `append_eos=False` (default) is byte-identical to the pre-change
  output.
- V5.44 behavioral rule (2026-07-21, §6): `BehaviorTracker` stops at
  exactly the k-th consecutive eval with rate `>= threshold` (inclusive
  — "scoring ≥99%" counts 0.99 itself), a sub-threshold eval resets the
  consecutive count, the stop latches forever, NaN raises without
  corrupting state, and `best_rate` is the exact running max over every
  update (-inf before the first).
- V5.45 behavioral trainer wiring (2026-07-21, §6): with a
  `BehavioralStoppingRule`, `train_sft` calls `behavioral_eval` at every
  eval, logs its value as `format_valid_rate` alongside the still-logged
  `val_loss_nats`, stops at the k-th consecutive above-threshold eval
  with `stop_reason="behavior"` (`final_step` = that eval's step), exits
  sub-threshold runs only at the ceiling with `"max_steps"`, reports the
  exact running-min val loss as both `best_val_nats` and `min_val_nats`,
  echoes `{threshold, k}` as the meta's `stopping`, and raises upfront
  (before any disk write) on an unpaired rule/callback. Silent failure
  here breaks the matched-arms design — the pre-registered rule is the
  treatment-mediator argument's foundation.
- V5.46 greedy decode (`geode.arith.decode.greedy_completions`, promoted
  from `gates.py` 2026-07-21 — now backs the G1/G2 gates AND the
  installers' in-loop eval): completions over ragged token-prefix
  prompts are identical one-at-a-time vs. one maximal left-padded batch
  (the property a wrong pad id, missing attention mask, or right-side
  padding silently breaks), greedy decode is deterministic across calls,
  and generation stops at the tokenizer's EOS with the EOS excluded from
  the decoded completion.
- V5.47 LoRA identity + scaling (2026-07-21, §6): `geode.train.lora.apply_lora`
  wraps every target linear as `base(x) + (α/2r)·B(A(x))` — the 2r is the
  pin, deliberately not PEFT's α/r — and with B zero-initialised the wrapped
  model's logits are bit-identical to the base model's (θ_0 is the
  pretrained state, specs/01 §1).
- V5.48 adapter-only training (2026-07-21, §6): after `apply_lora`,
  `requires_grad` is False on every base parameter and True exactly on the
  A/B factors; optimizer steps leave every base tensor bit-identical while
  both A and B move (B on step 1; A only once B ≠ 0 — ∂L/∂A is exactly zero
  at B = 0).
- V5.49 count + coverage (2026-07-21, §6): trainable parameters equal
  Σ rank·(d_in+d_out) over wrapped linears — the formula behind the 12.1M
  figure at the real arch — every listed target module on every layer is
  wrapped (`lm_head` is not), and `apply_lora` refuses non-zero dropout and
  any target name matching no `nn.Linear` (a typo would silently train a
  smaller adapter).
- V5.50 seeded A init (2026-07-21, §6): drawn from a dedicated CPU generator
  seeded by the explicit `seed`, in module registration order, then copied to
  the model's device/dtype — same seed ⇒ bit-identical A factors on any
  device, different seed ⇒ all differ; B is always zero.
- V5.51 self-contained round-trip (2026-07-21, §6; spec 00 §1): the wrapped
  model's `state_dict()` carries base + adapter tensors together
  (`<name>.base/.A/.B.weight`); fresh base model (any weights) →
  `reapply_lora` (same rank/α/targets) → strict `load_state_dict` reproduces
  outputs bit-exactly — no separate base checkpoint, no merge.
- V5.52 merge for parent handoff (2026-07-23, §6): `geode.train.lora.merge_lora`
  folds each `LoRALinear` into its base `nn.Linear`
  (`W_base + scaling·B.weight@A.weight`, in place) — cross-stage parent
  handoff only (a LoRA install run's result warm-starting a plain
  `from_pretrained` child run); snapshots are never merged, V5.51 stands.
  Merged logits match the wrapped model's up to float-rounding tolerance; a
  freshly wrapped model (B=0) merges to the base weights bit-exactly (the
  update is exactly zero); merged `state_dict()` keys equal a never-wrapped
  model's, so `save_pretrained` → plain `from_pretrained` round-trips.
- V5.62 bf16 touches only the update path (2026-07-24, §6, the Llama-1B
  chain): with `manifest.training.precision == "bf16"`, `train_prequential`
  autocasts the grad-enabled update forward and nothing else — the
  prequential stream and θ_T test loss are always measured fp32 on the fp32
  master weights (losses are reported quantities, the §7 principle). The
  step-0 record is taken at θ_0 before any update, so it is bit-identical to
  a same-seed fp32 run's; all artifacts stay complete and finite. An unknown
  precision string raises. Runs 5–8 ship fp32; only the Llama chain sets
  `train.precision: bf16`.
- V5.65 full-dose batch guard (2026-07-26, §6): `stopping_metric=
  "train_loss"` with `batch_size != n_train`, or with a
  `BehavioralStoppingRule`, raises `ValueError` before any training or
  disk write — a subsampled per-step loss would plateau on batch noise,
  not on dose absorption.
- V5.66 train-loss plateau (2026-07-26, §6): in train-loss mode the run
  stops `converged` at exactly the k-th stale eval of the logged
  per-step training loss with no val split anywhere; the eval log's
  `train_loss_nats` equals the train log's value at the same step
  exactly; `min_val_nats` propagates the metric's true min; empty
  `val_examples` in any other mode raises upfront; a 1-example dose
  memorizes and stops `converged` before a generous ceiling.
- V5.71 LR scope guard (2026-07-29, §6): `geode.train.assert_lr_scope(cfg, pin,
  stage=...)` refuses a `train.lr` that is mispinned or cross-scoped for its role.
  `stage="target"` requires the pinned target LR; a full-FT stage
  (`installer`/`bridge`/`parent`) that pins the target LR raises the run-9 scope
  leak, and `installer`/`bridge` additionally require the pinned installer LR
  (`parent` inherits its rate by role and pins no positive value).
  `phase3_guards.py` is a thin CLI shim over it (its `ValueError` surfaces as the
  guard's `SystemExit`); the four `launch_*.sh` heredoc copies retire by archival.
- V5.73 prefix EDL curve (2026-07-29, §6): `geode.edl.prefix_edl_curve(run_id,
  floor=..., store=...)` returns the running `EDL(e) = MDL(e) − tokens(e)·floor(e)`
  at each in-loop eval step from the epoch-1 `prequential.jsonl` prefix and
  `eval_log.jsonl`, unifying the three hand-rolled cumsum sites. `floor` is a
  REQUIRED keyword (omitting it is a `TypeError`): `"val"` is the moving per-step
  val floor, `"test"` the run's one constant `eval/test_loss.json` per-token floor
  (masking-parity guarded, D-1) — the choice sets the curve's shape (the
  floor-artifact, decisions.md 2026-07-27). `analysis/plot_edl_per_token.py` keeps
  its `--floor {val,test}` flag and `notebooks/key_figures.py` its accumulation
  figure, both delegating here.
- V5.74 eval-protocol constants + G5 leak bar (2026-07-29, §5/§8): `geode.edl`
  exposes `EVAL_STOP_ROWS = 2048` (the frozen eval file's stopping/reporting split)
  and `G5_N_SHOTS = 16` (the G5 few-shot count, protocol not a knob), and
  `g5_leak_ok(zero_shot)` — the leak bar, inclusive at 0.02: a zero-shot exact-match
  rate ≤ 0.02 has not leaked the target mapping. `gates.py` and `analysis/steering.py`
  import the constants from `geode.edl`; `train_target.py` re-exports `EVAL_STOP_ROWS`
  so `from train_target import EVAL_STOP_ROWS` still resolves. The two leak-bar shell
  heredocs retire by archival.

### 6.2 Run-1 launch surface (scripts — single-pass)

`experiments/training-run/scripts/train.py` + `configs/`: parses the
run YAML, builds the `LlamaConfig` from the config's model block
(custom small arch, 2026-07-18), loads + packs TinyStories-v2 — which
ships **only** as the delimiter-separated `TinyStoriesV2-GPT4-train.txt`
inside `roneneldan/TinyStories` (verified 2026-07-18; the repo's parquet
config is v1 data): downloaded via `huggingface_hub`, split with
`geode.train.split_documents`, the script's only network data access.
Registers the run in zoo (spec 00 §2 required fields;
`experiment` block rides as preserved extra fields until its validation
task lands), prints a cost estimate and refuses to run without
`--confirm-cost` (CLAUDE.md budget rule), then calls
`geode.train.train_full`. All hyperparameter values in
`run1_pretrain.yaml` are now pinned: seq_len + tokenizer 2026-07-18;
LR, batch, eval cadence, val fraction, stopping ε/k closed 2026-07-19
from the run-1 LR sweep (§12 OPEN(11)/OPEN(3)).

**Runs 2–4 launch surface** (`scripts/train_sft.py`, 2026-07-20): same
shape — parses run YAML (`run2_algo.yaml` + `configs/pilot/run2_*`
overlays), downloads the frozen parquet from
`mhieuuu/elicit-vs-teach-arith` and refuses on `order_hash` mismatch
against the config-pinned value (V5.40), converts char→token spans and
appends a label-covered EOS via `geode.arith.spans` (V5.38, V5.43),
splits with `split_indices` (V5.39),
enforces the parent DAG rule via `require_parent_ready` with the
config's `parent_required_gates` (spec 00 V0.6), records
`masking_config_hash` + `data_order_hash` in the manifest's experiment
block, and calls `geode.train.train_sft` behind the same
`--confirm-cost` gate. Gate verdicts land in `experiment.gates` via
`scripts/gates.py` (§8). Behavioral runs (3–4, 2026-07-21): a
`train.stopping` block with `metric: format_validity` selects the
behavioral mode — the launcher seeds `random.Random(prompt_seed)`,
samples `n_prompts` val examples, takes their token-prefix prompts
(`input_ids[:label_span.start]`, the G1 protocol), and builds the
`behavioral_eval` closure from `geode.arith.greedy_completions` +
`format_valid("Answer:" + completion)`; the manifest's
`training.stopping` records `{metric, threshold, k, n_prompts,
prompt_seed}` (spec 00 §2). The launcher also refuses `train.lr: null`
upfront (the canonical run-3 config ships null until the installer
sweep pins the winner — a placeholder-lr launch is a silent redo, the
run-2 2026-07-21 incident class).

**Runs 5–6 (LoRA target):** use `train_prequential` as-is — pre-update
losses, gradstats (per-module grad norms already covered), full-model
snapshots (spec 00 §1) at `manifest.snapshot_steps`. Additions needed: LR + train-acc
scalars per step (small logging extension). The harness reads
`training.precision` from the manifest (default fp32): `"bf16"` autocasts
only the update forward — every recorded loss stays fp32 (V5.62; the
Llama-1B chain is the only user, 2026-07-24). Probe loss/acc are **not**
computed in-training: the extraction pass (§7) yields per-example probe
loss at every snapshot, and early snapshots are per-step anyway, so probe
curves at snapshot resolution come free. Snapshot schedule: 1024 steps,
log-spaced early (every step through ~30, then stretching), uniform later
— produced by `geode.probe.schedule` and written into the manifest before
launch.

**Optimizer state:** never saved — OPEN(10) closed **no** (owner
2026-07-22, §12).

## 7. `geode.probe` — schedule, extraction, analysis metrics

**Snapshot scheduler.** `snapshot_steps(total_steps, n=1024, dense_until≈30)`
→ strictly increasing, includes first and final step, dense unit-stride
prefix, log-then-uniform tail. Exact parameters pinned by OPEN(4)'s close
(2026-07-22, §12): the defaults over max_steps 23442. Scheme implemented
2026-07-21 (`geode/probe/schedule.py`, V5.55–V5.61; V5.8 is the umbrella).

**Extraction pass** (offline, separate rental). For each snapshot: load
θ_k via `geode.edl.load_snapshot` (spec 00 §1 — once-per-run base file +
the step's adapter tensors; legacy full snapshots still strict-load);
run the probe set forward + backward (loss on label
tokens, **sum** reduction, through the shared M1 `label_mask`); capture
activations and activation gradients at all 9 residual points (embedding
output + 8 post-block residuals for this arch — hook count is
n_layers+1, never hardcoded; names follow the TransformerLens convention
of spec 00 §6: `hook_embed`, `blocks.{i}.hook_resid_post`), per example,
bf16. One dump per snapshot at `runs/{run_id}/probe/step_{k}/`
(spec 00 §1): one safetensors file per quantity ∈ {acts, grads}
containing the 9 named tensors, plus `probe_data.safetensors` holding
the padded `input_ids`, attention mask, label mask, and the per-example
probe loss in **fp32** (losses are reported quantities, never
down-cast; late gradients are numerically degenerate — analyses
condition on nonzero loss). Sidecar `meta.json`: run_id, arm, step,
probe_set_hash, tokenizer_hash, base_model_key, dtype, plus the
template/format identity `task_name` + `format_version`.

**Matched-load guard** (V0.4 pattern re-asserted at this surface): the
pairwise loader for cross-arm comparison refuses unless probe_set_hash,
tokenizer_hash, and the template/format metadata (`task_name`,
`format_version`) all match, with a clear error naming the field. The
guard deliberately ignores run_id, arm, step, and base_model_key —
those legitimately differ between the two dumps being compared.

**Analysis metrics** (pure functions over dumps; results written through
the ZOO-4 writer as spec 00 §7 long-format rows, `regime` column = arm):

- Cross-example activation-gradient alignment per (snapshot, layer):
  pairwise-cosine summary and top-PC explained-variance fraction of the
  per-example gradient matrix. Expectation: elicitation ⇒ near-parallel;
  teaching ⇒ diverse.
- Representation drift from the init snapshot, per layer, per digit class.
- Adapter diffs: cumulative ‖ΔW‖, effective rank, per-layer allocation.
  Needs a small weight-diff helper written fresh in
  `analysis/adapters.py` (LoRA ΔW = B@A × α/2r per module, ~20 lines;
  B/A are read straight from the full-model snapshot tensors).
  (`geode.steering` was planned but never built and was deleted in the
  2026-07-17 cut; its V2.6 was a spec property, never a test.)
- Performance-aligned matching: map snapshots across arms at equal probe
  accuracy (primary comparison axis); step-aligned secondary.

**Validation properties:**

- V5.8  schedule: exact count, strictly increasing, dense prefix, includes
  first + final step.
- V5.55 (2026-07-21) schedule is purely deterministic: repeated calls, under
  perturbed global RNG, return identical plain-int lists; no seed argument
  exists.
- V5.56 schedule steps are strictly increasing ints in [1, total_steps]
  across production-shaped and degenerate parameter grids.
- V5.57 schedule length is exactly min(n, total_steps), including
  n == total_steps and n > total_steps (⇒ every step).
- V5.58 dense unit-stride prefix 1..dense_until present whenever budget
  allows; clamps: total_steps <= dense_until ⇒ every step; n <= dense_until
  ⇒ prefix truncated to n-1 with the final step kept.
- V5.59 first and final step always present; at n == 1 the final snapshot
  outranks step 1.
- V5.60 schedule inputs validated loudly: total_steps < 1, n < 1, or
  dense_until < 0 raises ValueError.
- V5.61 spacing shape: gaps never shrink by more than 1 (rounding jitter);
  unit stride continues just past the dense prefix; the largest gap lives
  in the late tail; last-tenth gaps ~uniform (max <= 1.6 x min) —
  discriminating log-then-uniform from log-to-the-end.
- V5.9  extraction captures n_layers+1 hook points with correct shapes and
  stored mask on a tiny fixture model.
- V5.10 per-example gradients match an explicit one-example-at-a-time
  backward reference (sum reduction ⇒ equality up to kernel-order float
  noise, measured ≤1e-6 vs gradient scale ≥1; padding positions and
  zero-loss example rows are EXACTLY zero), and are nonzero iff the
  example's loss is nonzero.
- V5.11 dump ↔ load roundtrip preserves values (bf16), names, metadata.
- V5.12 matched-load guard raises on probe_set_hash / tokenizer_hash /
  template mismatch with a clear error.
- V5.13 alignment metric: planted parallel gradients ⇒ ≈1; random
  gradients ⇒ pairwise cosine ≈0 while top-PC explained variance ⇒
  ≈1/n, not 0 (the n≪d caveat, pinned numerically).
- V5.14 drift: zero at the init snapshot; planted per-class shift
  recovered per class.
- V5.15 effective rank: planted rank-r adapter delta ⇒ r.
- V5.16 performance-aligned matching: monotone in accuracy; planted
  curves ⇒ known pairing; ties broken deterministically.
- V5.63 linear CKA: a representation against itself, against any
  orthogonal rotation + isotropic rescaling of itself, and against a
  translated copy ⇒ 1; independent gaussians ⇒ ≈0 (at the d/n
  finite-sample floor, not 0); planted shared structure y = xM ⇒ high;
  degenerate inputs (non-matrix, row-count mismatch, n<2, non-finite,
  zero-variance) refuse.
- V5.72 dump iterator + alignment guard (2026-07-29): `load_probe_dumps(root,
  marker=...)` returns the ascending step indices of the `step_*` dump
  directories under `root` that carry `marker` (the `meta.json` sidecar for
  probe dumps by default; a tuple = "any present", e.g. snapshot dirs carrying
  `adapter.safetensors` OR `model.safetensors`), replacing the hand-rolled glob
  the analysis drivers and `extract.py` duplicated; empties return `[]` so the
  caller raises its own contextual error. `assert_probe_alignment(dump_hash,
  probe_set_hash, run_id=..., step=...)` is the row-alignment guard (spec 00 §6):
  a dump aligns with the frozen probe parquet only when the two hashes agree,
  else parquet row i and dump row i are not the same probe example.

## 8. Verification gates

Recorded per run in `experiment.gates` (§4); a child run refuses to start
while a parent gate fails. Thresholds frozen at pilot where marked.

**G0 removed 2026-07-20 (owner).** The run-1 coherence gate (20 seeded
samples, ≥16/20 coherent) is no longer part of the protocol: run 1 must
train with the paper's exact recipe (constant LR, stop on validation-loss
convergence), and whatever that recipe converges to *is* floor 1 —
gating it on sample quality would license off-protocol fixes (the v2
cosine retrain was exactly that). `analysis/sample_stories.py` remains as
an ungated qualitative inspection tool. Historical verdicts (v1 fail,
v2-ext pass) stay recorded in decisions.md and the v2-ext manifest.

| Gate | After | Check |
|------|-------|-------|
| G1 | run 2 | Arm A near ceiling on NL add/sub, threshold ≥95%. Protocol (owner 2026-07-20, revised 2026-07-21 after the sweep G1=0 incident, `scripts/gates.py g1`): 1,024 examples seeded-sampled (seed 316) from D_algo's held-out val split — re-derived via `split_indices` (V5.39), so never trained on. Prompts are token-level prefixes of the training tokenization (`input_ids[:label_span.start]`, V5.38) — re-tokenizing the char-sliced prompt yields a standalone trailing-space token never seen in training and makes the merged ` -` sign token unreachable (measured sign-drop on negatives). Greedy decode stopped at the trained EOS (V5.43), first line, `exact_match` on `"Answer:" + completion`; verdict + accuracy recorded in `experiment.gates.G1` |
| G2 | run 3 | Arm A still ≥95% exact match on NL add/sub after the installer — same protocol and bar as G1, no separate δ (owner 2026-07-21: δ is a tolerance choice, not pilot-measurable — a base-init pilot has no arithmetic to lose; 0.95 is already the committed definition of "capability present", which is what the elicitation claim needs at target time; the actual drop from G1 is reported in the write-up) |
| G3 | run 4 | Arm B ≈ 0% on real add/sub (random labels didn't leak; ≤ chance + margin) |
| G4 | runs 3, 4 | Format validity on operator-notation prompts (both arms; ~≥99%). The same metric is the installers' in-loop stopping signal (§6). Same decode protocol as G1: token-prefix prompts, greedy, EOS-stopped (the 2026-07-21 fix predates this tooling) |
| G5 | runs 3–6 | Zero/16-shot operator add/sub + shared-set test loss. Expectation: A ~2%/12%, B 0%/0% — the only remaining independent regime evidence. Protocol (owner 2026-07-22, second revision — supersedes the same-day eval-reserved-tail draw): data is the frozen `D_target_eval` file (§5), question-disjoint from D_target ∪ D_algo ∪ probe by construction; shots = reporting-block rows [2048, 2064), questions = the next `--n` (default 1024) rows — **fixed slices, the identical set for every run, no sampling**. Also records `test_loss_nats`: masked NLL over the full reporting block (rows 2048+), the same data as the runs-5/6 harness θ_T test loss, so every run's loss lands on identical data. `gates.py g5` refuses a run whose manifest records training on the eval file itself (belt and suspenders — disjointness is a generation-time property). History: the original full-file draw overlapped pilot training prefixes (1.2%–50.6% of eval questions at n10k–n500k; measured accuracy inflation ≤ 0.4 points); the intermediate reserved-tail protocol fixed contamination but capped training at 900k rows |
| G6 | phase-3 bridge | Bidirectional translation exact match on the entire frozen held-out `D_p3_bridge_eval`: token-prefix prompts from the training tokenization (V5.38), greedy EOS-stopped first-line decode (V5.43), and exact text comparison of the answer slot after stripping surrounding whitespace only. The aggregate, NL→operator, and operator→NL rates must each be ≥95%; verdict, rates, row count, checkpoint, file, and `order_hash` are recorded. G4/G5 refuse `task.name: arith_translate` before model/data loading because their integer-answer protocol is inapplicable. |
| G7 | before matched target | `data_order_hash` and `n_examples` match the designated target anchor, enforced at launch |

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

- OPEN(1): **closed 2026-07-21** — see §12; installer stopping is
  behavioral (§6), so there is no count-finding pilot. What remains
  before runs 3–4 launch is an LR sweep only.
- OPEN(2): **closed 2026-07-22** — see §12; the B-convergence grid ran
  at 10K/50K/200K/500K under the fixed shared eval protocol (§5) and
  500K is the smallest grid n where B lands within a few points of A.
- OPEN(3): **closed 2026-07-19** — see §12.
- OPEN(4): **closed 2026-07-22** — see §12; resolved mechanically by
  the OPEN(2) pin (batch 128 → max_steps 23442 → the §6 schedule over
  it).
- OPEN(5): **closed 2026-07-18** — see §12.

Pilot outcomes are logged in `notes/decisions.md`, then the `OPEN(n)`
markers in this spec are replaced with pinned values in the same PR.

## 12. Open items

| ID | Item | Closed by |
|----|------|-----------|
| OPEN(1) | Format-installer example count | **closed 2026-07-21** (owner): runs 3–4 stop on behavior, not a pinned count — identical pre-registered rule (§6: G4 metric ≥99% on 512 held-out prompts, k=3 consecutive evals, eval_every 250), identical data order; per-arm step counts are emergent, recorded in each manifest |
| OPEN(2) | Target dataset size | **closed 2026-07-22** (owner, B-convergence grid on the fixed shared eval file): **n=500K** — smallest grid n where B lands within a few points of A (G5 zero-shot / shared 98K-row test loss: B@500K 0.9805 / 0.0140 nats vs A@50K 0.9941 / 0.0059; B@50K 0.9414 / 0.0428, 5.3 points short). The n200K dip (0.8740 / 0.1109) is a stopping-rule artifact — ε/k fired at step 5500 (~3.5 epochs) mid-descent — not a property of the data size. Ceiling raised to max_steps=23442 (6 epochs of 500K): the B@500K pilot's ε/k stop at 15,500 sat 126 steps under the old 2-epochs-of-1M ceiling |
| OPEN(3) | Stopping-rule ε, k | **closed 2026-07-19** (run-1 LR sweep): ε=0.005 nats, k=3 at eval_every=500. Sweep val curves at the pinned LR are monotone at 100-step spacing (eval noise ≪ 0.005 with 5225 val seqs) and end-of-sweep improvement is ~0.06 nats/500 steps (12× ε), so ε separates signal from noise with wide margin. Runs 2–4 inherit; revisit only if their (arithmetic-val) curves misbehave |
| OPEN(4) | Batch → step count → snapshot schedule params | **closed 2026-07-22** (mechanical from OPEN(2)): batch 128, max_steps 23442, schedule = `snapshot_steps(23442, n=1024, dense_until=30)` — unit stride through the dense prefix and beyond (geometric ratio ≈ 1.007), stretching to ~57-step gaps at the tail. Snapshots are adapter-only fp32 (~48 MB each, 12.1M params; the frozen base ~155 MB written once per run — format 2026-07-22, spec 00 §1/V1.11); expected materialization ~880 for a B-like stop (~15.5K steps, ~43 GB) / ~630 for an A-like stop (~3.5K, ~31 GB) — steps past the stop never materialize |
| OPEN(5) | Padded max seq_len | **closed 2026-07-18**: per-example max 33 tokens over all four frozen full-scale files (longest: 4-digit NL sum, 5-digit answer); G5 16-shot worst case 593 tokens ⇒ model `max_position_embeddings: 1024` (free with RoPE), packing stays at `data.seq_len` 512 |
| OPEN(6) | Random-label sampling distribution (installer) | decision before pilot; default digit-count-matched uniform |
| OPEN(7) | Subtraction negatives allowed | decision before pilot; default allowed |
| OPEN(8) | Run 1: pretrain from scratch vs external TinyStories checkpoint | **closed 2026-07-18**: from scratch — the custom arch + tokenizer match no external checkpoint, and the run is single-GPU small |
| OPEN(9) | Exact template string (both formats) | **decided 2026-07-17**: two-line `Question: <body>` / `Answer: <answer>` scaffold; padded length still OPEN(5) |
| OPEN(10) | Keep optimizer-state snapshots (sizes TBD at 2026-07-18 scale) | **closed 2026-07-22** (owner): **no** — snapshots stay model-only (`_save_snapshot` already saves only the model state_dict; zero code change). AdamW moments would have added ~2× params (~400 MB/snapshot) for an analysis (optimizer-trajectory) nothing in the plan consumes; mid-run resume is not needed at ≲30-min run lengths |
| OPEN(11) | Run-1 pretrain hyperparameters (LR + schedule/warmup, seq len, batch, epochs/tokens, val-split size, eval cadence) | tokenizer **frozen 2026-07-18** at `experiments/training-run/tokenizer/`: 10K byte-level BPE on TinyStories-v2, digits 0–9 single-token forced, `Question:`/`Answer:` plain BPE (owner decision), EOS `<|endoftext|>` + PAD `<|pad|>`, provenance in `meta.json`. Dataset id verified 2026-07-18 (v2 = txt file in `roneneldan/TinyStories`; no v2 repo exists) and seq_len pinned at 512 (story p90 = 265 > 256; 1.6% of stories exceed 512). Remainder **closed 2026-07-19** by the 4-point LR sweep (docs/run1-guide.md phase 3, guide deleted 2026-07-24 — git history; 2000 steps each, production batch 128 via grad-accum 4×32, full data): **LR=1e-3** — best val 1.4389 nats vs 1.4552 @ 3e-4 with a consistent lead from step ~600, monotone descent, grad-norm max 6.4 / last 0.19; 3e-3 unstable (grad spike 109, val plateau ~3.15, self-stopped at 1700); 1e-4 far behind (1.7241). Constant LR, no schedule/warmup (structural — no scheduler exists). Batch 128, val_fraction 0.005, eval_every 500 as swept; epochs uncapped, ended by the stopping rule (ε/k → OPEN(3)). **Amended 2026-07-19 (gate G0 FAIL):** the constant-LR run plateaued at its gradient-noise floor (1.146 nats) with ~5/20 coherent samples; §6.1 gained a cosine schedule and the run-1 retrain uses cosine 1e-3→1e-4 over `max_steps=17000` (decisions.md). Constant LR remains the default elsewhere |

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
