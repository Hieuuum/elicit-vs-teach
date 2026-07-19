# EXPERIMENTS.md — post-cut structure & experiment specs

Status: **plan, pre-cut** (2026-07-17). This file defines (1) the geode
cut — exactly what is kept, trimmed, or deleted and why — and (2) the
distilled spec of each experiment part, from
`specs/02-training-run.md` (which remains the detailed source).
The cut executes after owner review of this file; the CLAUDE.md rewrite
(§6) lands in the same change.

Guiding rule for everything below: **test what can lie to you silently,
not what fails loudly.** Code whose silent failure wastes GPU budget or
invalidates the elicit-vs-teach comparison keeps property tests.
Clerical code gets a smoke test at most. Process ceremony goes away.

---

## 1. The cut

### 1.1 Keep unchanged (code + tests)

| Path | Why it stays |
|---|---|
| `geode/edl/` (prequential, metrics, masking, loop) | The measurement instrument. Silent math bug ⇒ wrong science. Tests check math properties and caught real bugs. |
| `geode/train/` (packing, stopping, loop) | The "teach" arm's trainer (runs 1–4). Property tests V5.17–V5.25 already landed with TRAIN-1. |
| `geode/zoo/` code (manifest, records, activations, checks, results, store) | Spec 02 locks zoo as the local registry for all six runs. Code is small (652 lines) and load-bearing. |
| `tests/edl/`, `tests/train/`, `tests/conftest.py`, `tests/test_fixtures_smoke.py` | Math-property tests for the kept core. |
| `tests/zoo/test_activations.py` | Matched-input enforcement — a silent mismatch invalidates the cross-arm comparison and nothing crashes. |
| `specs/00-interfaces.md` | Documents live schemas the kept zoo code implements; spec 02 §4 extends it (experiment block). |
| `specs/01-edl-harness.md` | Ground truth the kept EDL tests are checked against. |
| `specs/02-training-run.md` | The experiment design itself — science, not ceremony. |

### 1.2 Trim (optional, default skip)

| Path | Scope |
|---|---|
| `tests/zoo/test_manifest.py` | Optionally drop the wrong-type / null / enum cluster (~lines 200–348) — pure schema validation that fails loudly. Everything else in the file stays. Default: **skip**; see `docs/CUT-PLAN.md` §9. |

**`tests/zoo/test_records.py` and `tests/zoo/test_results.py` are kept in
full.** An earlier draft of this file said to gut them as "clerical
bookkeeping". That was wrong, and reading the test bodies proved it:
despite the `zoo/` path, they hold the epoch-1 exact-cover checks
(`:129`, `:146`, `:157`), the masking-hash guard (`:182`), and the
cross-arm join contract (`test_results.py:158`) — three of the four
guarantees the experiment's numbers depend on, all of which fail
*silently*. Deleting existing tests also saves nothing: the whole suite
is 242 tests in ~13s.

### 1.3 Delete

| Path | Why |
|---|---|
| `geode/steering/`, `geode/saediff/` (+ `tests/steering/`, `tests/saediff/`) | Empty 1-line stubs for modules never built. |
| `specs/02-steering-library.md` | Spec for an unbuilt module. Wang-et-al direction-extraction ideas move to analysis scripts if ever needed. |
| `specs/03-base-sae-pipeline.md`, `specs/04-crosscoder-adaptation.md` | SAE/crosscoder track is out of scope for elicit-vs-teach; `reference/` clones stay (read-only) in case it revives. |
| `.claude/agents/*` (all six) + `.claude/hooks/agent-guard.sh` | The four stage agents implement the retired protocol; `adapt-author`/`adapt-reviewer` serve the deleted specs/04 task; the hook only enforces per-stage path limits. Dead weight that would otherwise be advertised to every future session. `.claude/settings.json` keeps its `reference/**` deny rules. |
| `PLAN.md` | The task-based build plan for the retired protocol. **Its live content migrates first** — see §1.4. |
| `sae-lens`, `transformer-lens` deps in `pyproject.toml` | Present only for the deleted `geode.saediff`; nothing in the kept core imports either. Heavy installs for dead code. |
| Four-stage protocol + impl-log policy (process, not files) | The dominant per-task time cost (~60–70%). Replaced by §5 workflow. `docs/impl-logs/` stays as frozen history, banner-marked superseded. |

### 1.4 PLAN.md: note the live decisions, then delete

PLAN.md is the only place the owner-approved decisions **OQ-3 … OQ-14**
are written down. Some still explain how the kept core behaves — OQ-4 is
the rationale for the EDL epoch-1 coverage invariant, OQ-7/8 for the
masking-hash guard. Those are the *why* behind the two guarantees the
measurement rests on, so they get copied into `specs/00` §9 as a plain
reference note before the file goes. OQ-1/2 (ops) and OQ-9/10/11/12/13
(steering + saediff) die with their modules.

**Noted, not cited.** The `OQ-n` labels stay in existing docstrings only
because they are already typed there — stripping them means ~20 manual
rewordings for no benefit. There is no citation contract, no mapping to
maintain, and nothing verifies that a cited ID exists. If one dangles
later, nothing breaks.

### 1.5 Spec renaming

Survivors are renamed to read sequentially: `00-interfaces` and
`01-edl-harness` keep their names; `05-elicit-vs-teach` →
**`02-training-run`**.

This is safe *because* PLAN.md is deleted — it was the only live file
citing `specs/02` as the steering spec, so nothing surviving resolves
`02` to the old meaning. Git history still does; accepted.

The `experiments/` directory was later renamed `elicit-vs-teach` →
`training-run` (2026-07-17) so it matches the spec file `02-training-run`;
`elicit-vs-teach` is now the project/repo name. Property IDs stay `V5.x`: renumbering means
editing 25 IDs plus the shipped `tests/train/*.py` docstrings that cite
V5.17–V5.25, and IDs are labels, not paths.

### 1.6 Factual fixes required at cut time

**Spec-02 §7** claims adapter diffs "reuse
`geode.steering.extract_weight_diff` (validated by V2.6)". Steering was
never implemented; V2.6 never existed as a test. The weight-diff helper
must be written fresh in `analysis/adapters.py` (it is ~20 lines: LoRA
ΔW = B@A × α/2r per module).

**Spec-02 rigor statements** (lines 14, 69–71, 74, 138, 333, 358, 446, 497) mandate the four-stage
protocol and instruct cutting §5/§7 into PLAN.md tasks. Both are retired
and PLAN.md is deleted; the statements are rewritten to the §5 workflow.
Exact replacement text: `docs/CUT-PLAN.md` §7.4.

---

## 2. Post-cut repository structure

```
geode/                       # tested core — property tests required
  edl/                       #   prequential EDL harness (kept as-is)
  train/                     #   packing + full-FT trainer (kept as-is)
  zoo/                       #   run registry + activation store (kept as-is)
  arith/                     #   NEW — task data + evals        (§3.2)
  probe/                     #   NEW — schedule + extraction + metrics (§3.4–3.5)
tests/                       # CPU-only, < 2 min, no network (unchanged policy)
  edl/  train/  zoo/         #   kept (zoo manifest cluster optionally trimmed, §1.2)
  arith/  probe/             #   NEW — property tests from spec 02 V5.x lists
specs/                       # only the used specs, renamed sequentially (§1.5)
  00-interfaces.md           # live schemas + §9 resolved decisions (OQ-n, §1.4)
  01-edl-harness.md          # EDL math ground truth
  02-training-run.md         # experiment design (detailed source for §3)
experiments/training-run/    # scripts — smoke tests at most, no ceremony
  README.md                  #   experiment card: goal, arms, DAG, gate status
  configs/                   #   common.yaml, run1…run6 yamls, pilot/ overrides
  scripts/                   #   make_data.py train.py extract.py gates.py export_hf.py
  analysis/                  #   alignment.py drift.py adapters.py matching.py, figures/ (gitignored)
  notes/decisions.md         #   running log; pilot outcomes close OPEN items
reference/                   # read-only third-party clones (unchanged)
docs/impl-logs/              # frozen history, banner-marked superseded
docs/CUT-PLAN.md             # how the cut was executed (delete once done, or keep as record)
EXPERIMENTS.md               # this file — the plan: structure + experiment specs
```

`PLAN.md` and `.claude/agents/` are gone (§1.3–1.4). `EXPERIMENTS.md` is
now the only plan document.

`geode.arith` and `geode.probe` are the only new *library* code. They
qualify as tested core because both can lie silently: a leaked probe
operand pair or a wrong alignment metric invalidates the headline
comparison with no crash. They are written directly with their property
tests (single pass — no stage cycle, no new spec documents; spec 02
§5/§7 + the V5.x property lists are their spec).

---

## 3. Experiment specs (distilled from specs/02)

**Question:** does eliciting a latent capability differ mechanistically
from teaching it? Two arms differ *only* in whether the model was
pre-taught the algorithm in another format.

### 3.1 Runs and DAG

| # | run_id | Role | Init | Method | Data |
|---|---|---|---|---|---|
| 1 | `evt-run1-base` | pretrain | random | full FT | TinyStories-v2, custom small arch (2026-07-18, spec 02) |
| 2 | `evt-run2-armA-algo` | pre-teach | run 1 | full FT | NL add/sub, correct labels, 1M, 1 epoch |
| 3 | `evt-run3-armA-inst` | format install | run 2 | full FT | operator-notation mult, random labels |
| 4 | `evt-run4-armB-inst` | format install | run 1 | full FT | identical dataset + count as run 3 |
| 5 | `evt-run5-armA-target` | target | run 3 | LoRA | operator-notation add/sub |
| 6 | `evt-run6-armB-target` | target | run 4 | LoRA | identical data, identical order as run 5 |

DAG: `1→2→3→5` (Arm A, elicit) and `1→4→6` (Arm B, teach). Arms differ
only in run 2. Single seed (stated limitation). Every run registers in
zoo before launch; a child run refuses to start if its parent is
missing, incomplete, or has failing gates.

### 3.2 Data — `geode.arith` (tested core)

- **Goal:** procedural arithmetic datasets + probe set + evals.
- **Method:** own generator; operands 1–4 digits; add/sub (+ mult for
  installers); NL and operator-notation formats; correct | random label
  modes; every example carries label-token spans consumed by
  `geode.edl.masking.label_mask` (the single mask path).
- **Integrity:** dedup on (a,b,op); probe operand pairs excluded from
  *every* training set regardless of op/format; runs 5–6 consume
  byte-identical data in identical order (`data_order_hash` recorded in
  both manifests).
- **Probe set:** 1024 held-out target-format examples, fixed across arms
  and snapshots; stratified 256 per digit class, ops balanced;
  serialized once with `probe_set_hash`.
- **Uses core:** `geode.edl.masking`; zoo manifest for hashes.
- **Tests:** properties V5.1–V5.7 (no leakage, dedup, stratification,
  determinism, label-span correctness, random-label independence,
  parser correctness).

### 3.3 Training — runs 1–4 (`geode.train`) and 5–6 (`geode.edl`)

- **Runs 1–4 (full FT):** `geode.train.train_full` — AdamW, constant
  LR (pilot-determined; the paper's 1B values are void at the
  2026-07-18 scale), clip 1.0, bf16, batch 128, loss on label tokens only
  (run 1: all positions, pretrain mode), validation-convergence
  stopping (ε, k from pilot). Snapshot = final checkpoint only.
- **Runs 5–6 (LoRA target):** `train_prequential` as-is — pre-update
  losses, gradstats, full-model snapshots at manifest steps (spec 00
  §1). LoRA r=128 on Q,K,V,O,G,U,D all layers, α=32, LR
  pilot-determined. Small logging extension
  needed: LR + train-acc scalars per step.
- **Snapshots:** 1024, dense unit-stride through ~step 30 then
  log/uniform, from `geode.probe.schedule`, written into the manifest
  before launch.
- **Launch surface:** `scripts/train.py` + per-run YAML (script land).
  Only place that imports `datasets`. Registers run in zoo, prints cost
  estimate, refuses without `--confirm-cost`.

### 3.4 Extraction — `geode.probe` (tested core)

Offline pass per snapshot: load the self-contained snapshot (spec 00
§1), forward + backward on
the probe set (label-token loss, sum reduction), capture activations
and activation-gradients at all n_layers+1 residual points (9 for this
arch, never hardcoded), per example, bf16, mask stored alongside. One
safetensors file per (snapshot, quantity); sidecar metadata: run_id,
arm, step, probe_set_hash, tokenizer_hash, base_model_key, dtype.
Per-example probe loss saved per snapshot. **Matched-load guard:** the
pairwise cross-arm loader refuses on probe_set_hash / tokenizer_hash /
template mismatch. Tests: V5.8–V5.12.

### 3.5 Analyses (metrics in `geode.probe`, drivers in `analysis/`)

Each driver writes long-format rows through the ZOO-4 results writer
(join key: run_id, checkpoint_step, layer; `regime` column = arm).
Primary comparison axis: performance-aligned snapshot pairs (equal
probe accuracy); step-aligned secondary. Late-training gradient
analyses condition on per-example probe loss > 0.

| Analysis | Metric | Expectation |
|---|---|---|
| Gradient alignment | pairwise cosine + top-PC explained variance of per-example activation-gradient matrix, per (snapshot, layer) | elicit ⇒ near-parallel; teach ⇒ diverse |
| Representation drift | distance from init snapshot, per layer, per digit class | regime-dependent trajectories |
| Adapter diffs | cumulative ‖ΔW‖, effective rank, per-layer allocation (weight-diff helper written here — see §1.6 spec-02 fix) | elicit ⇒ low rank / concentrated |
| Snapshot matching | map snapshots across arms at equal probe accuracy | defines the primary axis |

Metric tests: V5.13–V5.16 (planted parallel gradients ⇒ ≈1, random ⇒
≈0; planted drift recovered; planted rank-r ⇒ r; matching monotone).

### 3.6 Verification gates (`scripts/gates.py` → zoo manifest)

| Gate | After | Check |
|---|---|---|
| G1 | run 2 | Arm A near ceiling on NL add/sub (~≥95%) |
| G2 | run 3 | Arm A still near ceiling (installer didn't corrupt) |
| G3 | run 4 | Arm B ≈ 0% on real add/sub (random labels didn't leak) |
| G4 | runs 3–4 | operator-notation format validity ~≥99%, both arms |
| G5 | runs 3–4 | zero/16-shot operator add/sub (expected A ~2%/12%, B 0%/0%) |
| G6 | data gen | V5.1/V5.2 integrity re-run on the *real* generated sets |
| G7 | before run 6 | `data_order_hash`(run 5) == (run 6), enforced at launch |

### 3.7 Pilot (before any production spend)

End-to-end de-risk at toy scale (~10K examples, 20 snapshots, 64 probe
examples) through train → extract → upload (separate `-pilot` HF repo)
→ one real gradient-alignment plot. Parameter pilots then close the
OPEN items (installer count, target dataset size, stopping ε/k,
snapshot schedule, template + seq_len). Outcomes logged in
`notes/decisions.md`; OPEN markers in spec 02 replaced with pinned
values in the same PR. Full OPEN table: spec 02 §12.

### 3.8 Publication

`export_hf.py` builds `hf-staging/`, exports `manifest.parquet` from
zoo (export, never a second source of truth), uploads in 50–100-file
commits with resume + hash verification, never deletes remote content
without an explicit flag, and has a `--dry-run` printing the commit
plan without network. Budget ≈ ~1 TB on HF PRO (≤10 TB; 2026-07-18
arch — re-estimate at pilot).

---

## 4. Order of work

1. Execute the cut (§1) + CLAUDE.md rewrite (§6).
2. `geode.arith` + tests (V5.1–V5.7) — blocks everything.
3. `geode.probe.schedule` + `extract` + tests (V5.8–V5.12).
4. Scripts: `make_data.py`, `train.py`, `gates.py` (smoke-level).
5. **Pilot** (§3.7) — closes OPEN items, freezes thresholds.
6. Analysis metrics + tests (V5.13–V5.16); drivers.
7. Production runs per DAG, gates enforced; extraction; analyses;
   `export_hf.py`.

## 5. Workflow going forward (replaces four-stage protocol)

- **Tested core** (`geode/*`): code + property tests written together,
  single pass. A change to core math means updating its property tests
  in the same commit. Property lists live in specs 01 and 02 — name
  tests after the property they check (e.g. `test_v5_1_no_probe_leakage`).
- **Scripts** (`experiments/*`): single-pass + self-review. Smoke test
  only where cheap. `--confirm-cost` on any GPU path (unchanged).
- **Promotion rule:** logic used by two or more scripts, or whose
  silent failure would corrupt results, moves into `geode/` and gains
  property tests. Nothing else does.
- Suite stays CPU-only, < 2 minutes, no network, tiny in-process
  fixture models (unchanged, non-negotiable).
- No stage subagents, no impl logs, no spec edits for script work.
  Decisions worth recording go in `notes/decisions.md` (experiment) or
  this file (structure).

## 6. CLAUDE.md changes (land with the cut)

Exact replacement text: `docs/CUT-PLAN.md` §8.

- Replace "Execution protocol" (four-stage) with §5 above.
- Replace "Documentation policy" (impl logs) with the
  `notes/decisions.md` convention.
- Update "What lives here": drop steering/saediff from the module list,
  add train/arith/probe, point planning at EXPERIMENTS.md, and note the
  spec numbering (00, 01, 02).
- Rewrite the `specs/` line — "every module is specified before it is
  implemented; tests are derived from specs" contradicts the new
  workflow, under which `geode.arith`/`geode.probe` get **no new spec
  documents**.
- Attribution: one edit only — the Wang et al. bullet points at
  `geode.steering`, which this cut deletes.
- Testing policy and budget rule: unchanged.
