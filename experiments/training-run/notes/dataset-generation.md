# Dataset generation — implementation brief

Status: **design finalized 2026-07-17, ready to implement.** Audience: an agent
with no prior context. Everything needed is here. The decision log is
`notes/decisions.md` (2026-07-17 entry); the experiment plan is the repo-root
`EXPERIMENTS.md`; the detailed source is `specs/02-training-run.md` §5. Where
this file and the spec disagree, **this file wins** (the spec still describes the
pre-pivot runtime generator; see §8 for the reconciliation edits owed).

## 0. What this is

Generate the frozen arithmetic datasets for the training-run experiment,
**once**, as files, then upload to HuggingFace. They are not regenerated from a
seed at train time. This replaces the original `geode.arith` runtime generator.

Guiding rule (CLAUDE.md → "Workflow"): test what can lie to you silently. The
tested core is the small pieces whose *silent* failure would waste GPU budget or
invalidate the arm comparison. The sampler is a single-pass script.

## 1. What already exists on disk (uncommitted working tree)

The CUT is committed (`bc07f69` on branch `cut-to-core`). The dataset work below
is **uncommitted WIP** — implement against it, then commit.

Tested core, **built and passing (25 tests)** — mostly final:
- `geode/arith/formats.py` — `render(a,b,op,shown_answer,fmt) -> (text, char_span)`,
  `true_answer`, `digits`. **NEEDS CHANGE:** NL wording (§2).
- `geode/arith/labels.py` — `random_label(true_answer, seed, index)`. **FINAL.**
- `geode/arith/evals.py` — `parse_answer`, `exact_match`, `format_valid`,
  `few_shot_prompt`. **FINAL** (parser keys off the trailing `?`/`=`, so the new
  NL wording still parses — verify with a test).
- `geode/arith/validate.py` — `cell_counts`, `uniqueness_by_cell` (final);
  `probe_leakage` is **pair-level + commutative** and must be replaced by a
  **question-level (triple)** check (§3, §4).
- `geode/arith/__init__.py` — re-exports; update when validators change.
- `tests/arith/test_{formats,evals,labels,validate}.py` — update the two that
  encode the old NL wording / old leakage rule.

Script, **pilot-only, superseded design — rewrite**:
- `experiments/training-run/datagen/make_data.py` — currently does
  commutative pair-level exclusion and a *hybrid with-replacement* fill (repeats
  in small cells). The finalized design forbids repeated questions and uses
  question-level exclusion. Keep its structure (build_probe / build_dataset /
  validate / parquet write / `--scale`/`--dry-run`, `[evt]` prints) and replace
  the sampler + exclusion (§5).

## 2. Formats (final)

Digit bands: 1→`1..9`, 2→`10..99`, 3→`100..999`, 4→`1000..9999`. A cell is
`(digits(a), digits(b))`; 16 cells for x,y ∈ {1,2,3,4}.

- **Operator** (`D_inst`, `D_target`, probe): `"{a} + {b} = {answer}"`,
  `"{a} - {b} = {answer}"`, `"{a} * {b} = {answer}"`.
- **NL** (`D_algo`, add/sub only) — **change from the current `plus`/`minus`:**
  - add: `"What is the sum of {a} and {b}? {answer}"`
  - sub: `"What is the difference between {a} and {b}? {answer}"`  (= a − b)

Subtraction negatives are allowed (OPEN(7) default); the answer char span
includes the leading `-`. The prompt (everything before the span) is identical
whether the shown answer is correct or random.

## 3. Datasets (final)

Three training files (1,000,000 rows each) + one probe (1024):

| file      | runs   | op(s) | format   | labels  |
|-----------|--------|-------|----------|---------|
| `D_algo`  | 2      | + −   | nl       | correct |
| `D_inst`  | 3, 4   | *     | operator | random  |
| `D_target`| 5, 6   | + −   | operator | correct |
| `probe`   | eval   | + −   | operator | correct |

Runs 3/4 share `D_inst` and runs 5/6 share `D_target` byte-for-byte, so their
`data_order_hash` match by construction. Training sets may overlap each other in
arithmetic instances freely (both are training) — the only exclusion is
probe∉train.

**Uniqueness = the question.** Every training example has a unique rendered
question, i.e. a unique ordered triple `(a, op, b)` (so `(3,5,+) ≠ (5,3,+)` and
`(a,+,b) ≠ (a,-,b)`). Answers may repeat; for `D_inst` the answer is a random
wrong label. Zero repeated questions in any dataset.

**Probe exclusion = question-level, format-independent.** A probe example blocks
exactly its triple `(a, op, b)` from training — NOT the operand pair across other
ops, NOT the commuted twin. Probe is add/sub only, so it constrains
`D_algo`/`D_target`; `D_inst` (mult) is unaffected. Probe questions are carved
out first; training draws from the remaining unique triples. (Rationale: both
arms train on the identical `D_target`, so probe overlap inflates both arms'
absolute accuracy equally and never biases the A-vs-B comparison.)

## 4. Stratification + redistribution (final)

Aim for even 62,500 per cell. Small cells cannot reach it — their unique-question
capacity is too low:

| cell | capacity add/sub (×2 ops) | capacity mult (×1 op) |
|---|---|---|
| 1×1 | 162 | 81 |
| 1×2, 2×1 | 1,620 | 810 |
| 2×2, 1×3, 3×1 | 16,200 | 8,100 |
| 1×4, 4×1, 2×3, 3×2 | 162,000 | 81,000 |
| ≥ 2×4 | ≥ 1,620,000 | ≥ 810,000 |

(capacity = `size(dx) * size(dy) * |ops|`, minus probe triples in that cell for
add/sub sets.)

Allocation per dataset (`N = 1_000_000`, `target = N // 16 = 62_500`):

```
for each cell: cap = unique-question capacity (after removing probe triples)
constrained = cells with cap <= target       # take ALL of them: alloc = cap
free        = cells with cap  > target        # start at alloc = target
deficit     = sum(target - cap for constrained cells)   # ≈323k add/sub, ≈349k mult
distribute `deficit` across free cells, weighted toward bigger numbers
  weight(cell) = x_digits + y_digits          # simple bias; a knob (see below)
  add proportional share, capped at (cap - alloc); iterate until deficit == 0
assert sum(alloc) == N                          # big cells have ample capacity
```

Feasible: the free cells' spare capacity vastly exceeds the deficit. The exact
redistribution weight is a **knob** — `x+y` is the default; the pilot must print
the resulting per-cell counts for owner review before the full run.

Sampling per cell (no replacement, deterministic):
- If `alloc[cell] == cap` and the cell is small (≤ ~1.6M triples): enumerate all
  eligible triples (sorted), take them all.
- Else (big cell, `alloc < cap`): rejection-sample distinct eligible triples
  (`a∈band_x, b∈band_y, op∈ops`; skip probe triples and already-used) until
  `alloc[cell]`. Collision rate is low because `alloc ≪ cap`.
- Balance ops ~50/50 within add/sub cells where capacity allows; tiny cells take
  all of both.
- Seed each cell's RNG from `(master_seed, dataset_name, dx, dy)`; final shuffle
  the whole dataset from `(master_seed, dataset_name, "order")`, then reindex.

## 5. Sampler rewrite checklist (`make_data.py`)

1. `build_probe(seed)`: operator add/sub, 64/cell, all-unique triples, correct
   labels. Return `(records, probe_triples: set[(a,op,b)])` and `probe_set_hash`.
2. `capacity(cell, ops, probe_triples)` and `allocate(N, capacities, weights)` per §4.
3. `build_dataset(spec, N, probe_triples, seed)`: allocate, sample distinct
   triples per cell (enumerate|reject), assign answers (`true_answer` |
   `random_label`), render, shuffle, reindex, `data_order_hash`.
4. Drop the old with-replacement fill and the commutative `_canon` exclusion.
5. `validate()` (call the `geode.arith` validators; **raise** on any violation):
   - question-level leakage == 0 for add/sub datasets (triple check),
   - every question unique (`uniqueness_by_cell`: n_rows == n_distinct in every
     cell),
   - `cell_counts` == the planned allocation (NOT uniform — assert against the
     `allocate()` output, print the distribution).
6. Write parquet + `report.json`; keep `--scale {pilot,full}` (`SIZES` =
   `{pilot: 10_000, full: 1_000_000}`), `--dry-run`, `--seed`, `--out`.
7. Determinism: two runs at `--scale pilot` must produce byte-identical output
   (this is the V5.4 check; it is a script self-check, not a pytest test).

## 6. Schema (parquet columns, tokenizer-agnostic — final)

`idx, dataset, a, b, op, x_digits, y_digits, cell, format, label_mode,
true_answer, shown_answer, prompt_text, answer_text, full_text,
answer_char_start, answer_char_end`.

Token-level label spans are derived from `answer_char_start/end` at load, once
the tokenizer (OPEN(11), Llama-3.2 tokenizer is license-gated) is fixed. Do not
bake a tokenizer into the frozen files.

## 7. Property tests (write with the code, CPU fixtures, name after the property)

- **V5.5** char spans exact for both formats incl. new NL wording and negatives
  (`test_formats.py`).
- **V5.7** parser/exact-match/format-valid incl. negatives + malformed; few-shot
  builder (`test_evals.py`, already passing — add a new-NL-wording parse case).
- **V5.6** random label matches true-answer digit-count+sign and is invariant to
  operands given answer shape (`test_labels.py`, passing).
- **V5.1** question-level leakage: a train triple equal to a probe triple is
  flagged; a train example sharing only the operand *pair* (different op, or
  commuted) is NOT flagged (`test_validate.py`, rewrite the commutative cases).
- **V5.2** every question unique: `uniqueness_by_cell` reports `n == n_distinct`
  for a clean fixture; a planted duplicate is caught.
- **V5.3** allocation: `cell_counts` matches the `allocate()` plan for a small
  synthetic capacity map; small cells capped at capacity, deficit lands in big
  cells weighted toward bigger.

The real 1M artifact is validated by `make_data.py` calling these same
validators and failing loudly — it is not a pytest test (too big for the
CPU/<2-min suite).

## 8. Reconcile the spec (same commit as the code)

Edit `specs/02-training-run.md` §5 to match (CLAUDE.md: spec changes land with
the code). Changes: V5.1 pair→triple exclusion; V5.2 "every question unique,
exactly 1M"; V5.3 16 (x,y) cells with capacity-aware redistribution (delete the
"256 per max-digit class" wording); V5.5 new NL wording; note the generator now
lives in `scripts/make_data.py`, not `geode.arith`.

## 9. HF upload (separate script — owner runs the real push)

New `scripts/upload_hf.py` (or extend `export_hf.py` per spec §3.8): `--dry-run`
default prints the commit plan and writes nothing to the network; a `--push`
flag performs the upload in 50–100-file commits with hash verification, resumable,
never deleting remote content without an explicit flag. **Do not push** — build
and dry-run it; the owner runs `--push`. Dataset repo is
`Hieuuum/elicit-vs-teach-arith` (owner decision 2026-07-17; visibility still
owner's call). This is not GPU compute, so `--confirm-cost` does not
apply, but the network write must stay behind `--push`.

## 10. Verify

```
python -m pytest -p no:cacheprovider              # 242 (cut) + arith tests, 0 failed
ruff check . && ruff format --check .
python experiments/training-run/datagen/make_data.py --scale pilot --out <scratch> --seed 20260717
# then a second run to a second dir; report.json must be byte-identical (V5.4)
# inspect report.json: leakage 0, all-unique, per-cell distribution matches the plan
```

Scale to production only after the owner approves the pilot's printed per-cell
distribution: `--scale full` (a few minutes CPU; parquet is large). Then build
and dry-run the uploader.

## 11. Guardrails

- CPU-only, no network in the test suite; tiny in-process fixtures only.
- Nothing here spends GPU budget; the HF push stays behind `--push` and is the
  owner's to run.
- Commit when the suite is green, with the spec §5 edits in the same commit. No
  stage agents, no impl log (both retired in the 2026-07-17 cut).
