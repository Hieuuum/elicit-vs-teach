# CUT-PLAN.md — execute the geode cut

Status: **approved plan, not yet executed** (written 2026-07-17;
revised after adversarial review). Audience: an agent with **no prior
context**. Everything needed is here. Read `EXPERIMENTS.md` (repo root)
first — it is the post-cut target state; this file is the mechanical how.
**Where the two disagree, this file wins.**

Execute the phases **in order**. Phase 2 moves content out of PLAN.md
before deleting it; Phase 3 renames the spec that Phase 4's CLAUDE.md
text refers to. Do not reorder.

## 0. Context in five lines

geode was built spec-first under a four-stage subagent protocol. Eight
tasks landed (SETUP-0, ZOO-1..4, EDL-1..3, TRAIN-1): 1,692 lines of
library, 3,779 lines of tests, 242 tests passing in ~13s. The owner is
cutting the parts that never paid off — unbuilt modules, their specs, and
the per-task ceremony — and keeping the tested core that guards the
science. The remaining work is the elicit-vs-teach experiment, not more
infrastructure.

**Guiding rule, applied throughout:** test what can lie to you silently
(wrong numbers, no crash); do not test what fails loudly (a crash, or an
obviously broken file, on the first run, before any GPU spend).

## 1. Preconditions

```bash
cd /home/mhieuuu/Github/geode
git status --porcelain    # expect only: EXPERIMENTS.md, docs/CUT-PLAN.md
python -m pytest -p no:cacheprovider -q 2>&1 | tail -2   # expect: 242 passed
```

Baseline is **242 passed in ~13s**. Every phase must leave it at 242
except the optional Phase 5, which states its own number.

Work on a branch: `git checkout -b cut-to-core`.

## 2. Owner decisions this plan encodes

Recorded because they overrode earlier drafts — do not "fix" them back:

1. **PLAN.md is deleted, not archived.** Its still-live content is noted
   into `specs/00` first (§6.2); git history keeps the task blocks.
2. **Keep only the used specs** — survivors are 00, 01, and the
   elicit-vs-teach spec (§5.2).
3. **The elicit-vs-teach spec is renamed** `05-elicit-vs-teach.md` →
   **`02-training-run.md`** (§7). Specs then read 00, 01, 02. The owner
   knows the new name is narrower than the spec's contents (which also
   cover data generation, extraction, analysis, gates, and publication).
4. **Decisions are noted, not cited.** No citation contract, no mapping
   to verify, no ceremony — see §6.1.

## 3. Shell gotchas — read before running any command here

`grep -rl` in this environment emits paths **without** a `./` prefix.
Filters anchored as `grep -v '^./reference/'` therefore **never match**.
Every filter in this plan uses the un-anchored form (`grep -v
'reference/'`). If you "improve" one back to `^./`, you will silently
rewrite files this plan protects.

Every `xargs sed` here uses `xargs -r`. Without it, a grep that matches
nothing leaves `sed -i` with no file operands, so it reads stdin and
**hangs forever**. This bites on any re-run after the seds have done
their work.

## 4. What must NOT be touched

- `geode/edl/`, `geode/train/`, `geode/zoo/` — implementation code stays
  behaviourally identical. This cut deletes no library code except two
  empty stub packages (§5.1); the only other changes are docstring text.
- `tests/edl/`, `tests/train/`, `tests/conftest.py`,
  `tests/test_fixtures_smoke.py`, `tests/zoo/test_activations.py` — the
  math + matched-input tests. Docstring reference updates only.
- **`tests/zoo/test_records.py` and `tests/zoo/test_results.py` — kept in
  full.** Despite the "zoo" path they guard EDL math and the cross-arm
  comparison; see §9. An earlier draft of EXPERIMENTS.md said to gut
  them. That was wrong. Do not.
- `experiments/elicit-vs-teach/` — the **directory keeps its name**. Only
  the spec file is renamed. The §7.2 sed is written so it cannot touch
  this path (no expression matches bare `elicit-vs-teach`); verify after.
- `reference/` — read-only third-party clones. Never modify.
- `docs/impl-logs/TRAIN-1.md` + `assets/` — frozen history. Its `PLAN.md`
  and `specs/05` references are left dangling **on purpose**: they record
  what was true on 2026-07-16, and git has the files. §7.2 has a check
  that fails if the sed touches TRAIN-1.md. (`docs/impl-logs/README.md`
  *does* get a superseded banner — §6.6. That is the only permitted edit
  under this path.)

---

## 5. Phase 1 — deletions and dead-reference fixes

### 5.1 Delete the stub packages

`geode/steering/__init__.py` and `geode/saediff/__init__.py` are 1-line
stubs. `tests/steering/` and `tests/saediff/` hold only empty
`__init__.py`. **Verified: nothing imports either package.**

```bash
git rm -r geode/steering geode/saediff tests/steering tests/saediff
```

### 5.2 Delete the unused specs

```bash
git rm specs/02-steering-library.md specs/03-base-sae-pipeline.md specs/04-crosscoder-adaptation.md
```

02 specifies `geode.steering` (never built); 03 specifies `geode.saediff`
(never built); 04 is a crosscoder scoping note for a track out of scope
for elicit-vs-teach. `reference/` clones stay in case that track revives.
Survivors: `00-interfaces.md`, `01-edl-harness.md`,
`05-elicit-vs-teach.md` — the last is renamed in Phase 3.

### 5.3 Retire the protocol enforcement in `.claude/`

```bash
git rm .claude/agents/test-writer.md .claude/agents/test-auditor.md \
       .claude/agents/implementer.md .claude/agents/conformance-reviewer.md \
       .claude/agents/adapt-author.md .claude/agents/adapt-reviewer.md \
       .claude/hooks/agent-guard.sh
```

- The first four implement the retired protocol.
- `adapt-author` / `adapt-reviewer` are stage agents for the ADAPT-1
  crosscoder task (specs/04) — deleted in §5.2, so the task is gone.
- `agent-guard.sh` is a PreToolUse hook enforcing per-stage path limits.
  With no stage agents it is dead code (exits 0 for unknown agents).

Then edit `.claude/settings.json`: **delete the entire `"hooks"` block**
(it invokes the deleted script); **keep `permissions.deny` exactly as
is** — that is CLAUDE.md's `reference/**` read-only policy, not part of
the cut. Result:

```json
{
  "permissions": {
    "deny": [
      "Edit(/reference/**)",
      "Write(/reference/**)",
      "NotebookEdit(/reference/**)"
    ]
  }
}
```

### 5.4 Fix every live reference to the deleted modules

Each of these keeps a pointer to a package §5.1 just deleted. All are
prose; none is an import.

| File | Line | Current → Replacement |
|---|---|---|
| `pyproject.toml` | 4 | `description = "Mechanistic analysis infrastructure for elicitation vs. teaching (EDL, steering sufficiency, base-SAE diffing)"` → `description = "Mechanistic analysis infrastructure for elicitation vs. teaching (EDL, prequential MDL, training runs)"` |
| `geode/zoo/results.py` | 6 | `…and internal quantities (from steering/saediff) land in the same` → `…and internal quantities (from analysis drivers) land in the same` |
| `specs/00-interfaces.md` | 3 | `Everything downstream (EDL harness, steering library, base-SAE pipeline, external repo adapters) reads and writes through these schemas.` → `Everything downstream (EDL harness, training runs, probe extraction, analysis drivers) reads and writes through these schemas.` |
| `specs/00-interfaces.md` | 127 | `values (from the harness) and internal quantities (from steering/saediff)` → `values (from the harness) and internal quantities (from analysis drivers)` |
| `CLAUDE.md` | 33–34 | `- **Wang et al. 2025** — OOCR ≈ constant steering-vector shift; template for `geode.steering`.` → `- **Wang et al. 2025** — OOCR ≈ constant steering-vector shift; template for direction-extraction analysis under `experiments/…/analysis/`. (`geode.steering` was planned but never built; deleted 2026-07-17.)` |

**`tests/zoo/test_results.py` (~lines 171–194) uses the string
`"steering_sufficiency"` as a metric-name fixture. Leave it alone** — it
is test data, not an import, and that test is a keeper (§9). The §10
verification grep excludes it by name.

### 5.5 Drop the dead dependencies

`pyproject.toml` lines 12–13: `"sae-lens>=4.0",` and
`"transformer-lens>=2.0",`. Both exist solely for `geode.saediff`
(PLAN.md OQ-12: "thin wrapper over `sae_lens.SAE.load_from_disk`").
Nothing in the kept core imports either. Delete both lines — they are
heavy installs for deleted code, and git has them if the crosscoder track
revives.

### 5.6 Gitignore the figures directory

`EXPERIMENTS.md` §2 and the spec both annotate `analysis/figures/` as
gitignored, but `.gitignore` has no such entry (it has `results/`).
Append `figures/` to `.gitignore` so the first plot run does not dirty
the tree.

---

## 6. Phase 2 — note PLAN.md's live decisions, then delete it

### 6.1 What this is, and what it is deliberately not

PLAN.md is the only place the owner-approved decisions **OQ-3 … OQ-14**
(PLAN.md § "Resolved decisions", lines ~11–44) are written down. Some
still describe how the kept core behaves — OQ-4 is the rationale for the
EDL epoch-1 coverage invariant, OQ-7/8 for the masking-hash guard.
Losing them to git archaeology loses the *why* behind the two guarantees
the measurement rests on.

**They are noted, not cited.** Copy the live ones into `specs/00` as a
plain reference note and move on. This plan deliberately does **not**:

- strip the existing `OQ-n` labels from docstrings — they are already
  typed into ~20 sites, embedded in prose ("Validation follows resolved
  decision OQ-3: every key listed in spec 00 §2 is…"). Removing them
  means ~20 manual rewordings for no benefit. Leaving them costs nothing;
  after this phase they resolve to spec 00 instead of a deleted file.
- maintain any citation contract, mapping table, or verification that
  every cited ID exists. The labels are labels. If one dangles later,
  nothing breaks.

### 6.2 Note them into `specs/00-interfaces.md`

Add a section **`## 9. Decisions (owner-approved 2026-07-11)`**
immediately **before** the existing
`## Validation properties (tests derive from these)` section (currently
line 142), so §1–§9 stay contiguous and the properties section remains
the appendix. Lead it with:

```markdown
Noted here from the deleted PLAN.md (2026-07-17 cut) because they explain
why the shipped code behaves as it does. `OQ-n` labels are kept only
because existing docstrings already use them — they are a reference note,
not a contract.
```

Then copy these bullets **verbatim** from PLAN.md lines 16–44 — OQ-3,
OQ-4, OQ-5, OQ-6, OQ-7, OQ-8, OQ-14 — with one edit: **OQ-5**, drop the
trailing clause `; spec 02 wording updated in same PR` (that spec is
deleted). Do not reword the rest; they are the recorded rationale for
shipped behaviour.

Skip the rest — they died with their modules and nothing surviving
mentions them: OQ-1/OQ-2 (repo init + reference clone, ops, closed),
OQ-9 (steering `LowRankMap` refit type), OQ-10 (steering defaults `k=8`,
`alphas=…`), OQ-11 (capacity metric = notebook ratio, no library
function), OQ-12 (`load_sae` / saediff), OQ-13 (`residual_pcs` /
saediff).

### 6.3 Delete PLAN.md

Only after §6.2 is written and saved:

```bash
git rm PLAN.md
```

### 6.4 Fix references to the deleted PLAN.md

Pass (a) — point resolved-decision mentions at their new home. **The
plural expression must come first**, or the singular consumes its prefix
and produces `decision (specs/00 §9)s`:

```bash
grep -rl "PLAN.md resolved decision" --include='*.py' tests/ geode/ \
  | xargs -r sed -i \
      -e 's|PLAN\.md resolved decisions|resolved decisions (specs/00 §9)|g' \
      -e 's|PLAN\.md resolved decision|resolved decision (specs/00 §9)|g'
```

Pass (b) — remaining provenance mentions; PLAN.md no longer exists:

```bash
grep -rl "PLAN\.md" --include='*.py' tests/ geode/ \
  | xargs -r sed -i 's|PLAN\.md|the retired build plan (git history)|g'
```

**Four sites need a manual reflow afterwards** — sed is line-based and
blunt. Review `git diff` and fix each by hand:

1. `tests/zoo/test_records.py:5-6` — the phrase is **line-wrapped**
   (`…and PLAN.md resolved` / `decisions OQ-4 (…`), so pass (a) never
   matches it and pass (b) mangles it instead, losing the `specs/00 §9`
   pointer on the file that documents the epoch-1 invariant. Reflow to
   `…validation properties V0.3 and V0.5, and resolved decisions
   (specs/00 §9) OQ-4 (…`.
2. `tests/zoo/test_activations.py:4` — plural form; confirm pass (a)'s
   plural expression caught it and the line still reads well.
3. `tests/zoo/test_records.py:13` — `PLAN.md "### ZOO-2" and are expected
   to fail until ZOO-2 is implemented.` Stale regardless (ZOO-2 shipped).
   Rewrite or drop the sentence.
4. `tests/edl/test_masking.py:26` — quotes a PLAN.md fragment
   (`"# + span rule params (OQ-8)"`). Reword to cite `specs/00 §9`.

Prose references outside `tests/`/`geode/`:

| File | Line | Change |
|---|---|---|
| `README.md` | 8–17 | Replace wholesale — see §6.5 |
| `docs/impl-logs/README.md` | 1–16 | Prepend the superseded banner (§6.6) |
| `CLAUDE.md` | 22, 62, 80, 92 | Handled wholesale by Phase 4 (§8) |
| `docs/impl-logs/TRAIN-1.md` | 5 | **Leave** — frozen history (§4) |

### 6.5 Rewrite README.md lines 8–17

Currently advertises both deleted modules and points at
`PLANNING_PROMPT.md`, **which does not exist** (pre-existing rot). Also
"agent protocol" names what Phase 4 retires. Replace lines 8–17 with:

```markdown
Modules (specified in `specs/`; current plan: `EXPERIMENTS.md`):

- `geode.zoo` — checkpoint-zoo manifests, run registry, storage schemas
- `geode.edl` — prequential MDL / EDL harness (label-masked, first-epoch)
- `geode.train` — corpus packing + full-FT/pretrain trainer
- `geode.arith` — arithmetic task data + evals (planned)
- `geode.probe` — snapshot schedule, activation/grad extraction, metrics (planned)

Start here: `CLAUDE.md` (conventions + workflow), then `EXPERIMENTS.md`.
Version pins in `pyproject.toml` are minimums — tighten them to the exact
versions of your environment on first install.
```

### 6.6 Banner for `docs/impl-logs/README.md`

Its opening reads as a *binding, live* policy ("every implementation run
… ends with a log written here"). Prepend:

```markdown
> **⚠ SUPERSEDED 2026-07-17.** The per-task implementation-log policy was
> retired with the four-stage protocol; see `CLAUDE.md` → "Documentation".
> This directory is frozen history. Do not add logs. Record decisions in
> `experiments/elicit-vs-teach/notes/decisions.md` or `EXPERIMENTS.md`.
> The text below describes the retired process and references the deleted
> `PLAN.md` and the pre-rename `specs/05`; both are preserved as-written
> so the existing logs read in context.
```

Leave the rest of that file, and TRAIN-1.md, untouched.

---

## 7. Phase 3 — rename the spec to `02-training-run.md` and fix its stale content

Specs become **00, 01, 02**. Safe *because* PLAN.md is deleted: it was
the only live file citing `specs/02` as the steering spec, so nothing
surviving resolves `02` to the old meaning. (Git history and old commit
messages still do; accepted and unavoidable.)

### 7.1 Move the file and fix its title

```bash
git mv specs/05-elicit-vs-teach.md specs/02-training-run.md
```

Edit line 1 of the moved file:

```markdown
# 02 — Training Runs: elicit-vs-teach experiment organization & requirements
```

### 7.2 Update every live reference

**Expression order matters**, and no expression may match bare
`elicit-vs-teach` — that is what keeps the `experiments/elicit-vs-teach/`
*directory* safe. Filters are un-anchored per §3. `EXPERIMENTS.md` is
excluded: it is already written in post-rename terms, and its one
mention of the old name is the rename sentence itself, which the sed
would garble into `02-training-run → 02-training-run`.

```bash
grep -rl "specs/05\|spec 05\|spec-05\|05-elicit" --include='*.py' --include='*.md' --include='*.yaml' . \
  | grep -v 'reference/' | grep -v 'docs/impl-logs/' | grep -v 'CUT-PLAN' | grep -v 'EXPERIMENTS.md' \
  | xargs -r sed -i \
      -e 's|05-elicit-vs-teach|02-training-run|g' \
      -e 's|specs/05|specs/02|g' \
      -e 's|spec 05|spec 02|g' \
      -e 's|spec-05|spec-02|g'
```

Expected files touched: `geode/train/{__init__,packing,loop,stopping}.py`
(docstrings), `tests/train/test_{loop,packing,stopping}.py` (docstrings),
`experiments/elicit-vs-teach/{README.md, notes/decisions.md,
scripts/train.py, configs/common.yaml, configs/run1_pretrain.yaml,
configs/pilot/run1_pretrain.yaml}`.

Then check the two things the sed must not have done:

```bash
ls -d experiments/elicit-vs-teach          # must still exist
git diff --name-only docs/impl-logs/TRAIN-1.md   # MUST be empty (frozen history)
```

### 7.3 Property IDs stay `V5.x`

`specs/02-training-run.md` keeps its `V5.1`–`V5.25` IDs even though the
file is now `02`. Renumbering means editing 25 IDs in the spec plus the
shipped `tests/train/*.py` docstrings citing `V5.17`–`V5.25`, and `V2.x`
was the deleted steering spec's range. IDs are labels, not paths.

### 7.4 Fix the spec's stale protocol and steering claims

The rename does not touch content — §7.2's patterns match nothing in this
file, so its line numbers are unshifted. Nine statements still mandate the
retired protocol or point at a deleted package.

**Line numbers below are locators only.** (a)–(d) each quote a unique
string: **match on the text, not the line number** — the list is in
importance order, not line order, and each edit shifts the ones below it
(by the time you reach "line 497" it sits at ~500). (e) is pattern-based
and order-independent.

**(a) The false steering claim — §7 Analysis metrics, lines ~389–391.**
`geode.steering` was never implemented, and `V2.6` was a *spec property*
in the deleted `specs/02-steering-library.md:90`, never a test. Replace:

```markdown
- Adapter diffs: cumulative ‖ΔW‖, effective rank, per-layer allocation —
  reuses `geode.steering.extract_weight_diff` (E2, α-scaling already
  validated by V2.6); only small summary helpers are new.
```

with:

```markdown
- Adapter diffs: cumulative ‖ΔW‖, effective rank, per-layer allocation.
  Needs a small weight-diff helper written fresh in
  `analysis/adapters.py` (LoRA ΔW = B@A × α/2r per module, ~20 lines).
  (`geode.steering` was planned but never built and was deleted in the
  2026-07-17 cut; its V2.6 was a spec property, never a test.)
```

**(b) Rigor decision — line 14.** Replace:

```markdown
- **Rigor:** four-stage spec-first protocol applies to the library modules
  only. Experiment scripts, configs, and plotting are single-pass + review
  (SETUP-0-style planned deviation).
```

with:

```markdown
- **Rigor:** library modules (`geode.arith`, `geode.probe`) are written
  with their property tests in one pass (CLAUDE.md → "Workflow");
  experiment scripts, configs, and plotting are single-pass +
  self-review.
```

**(c) Lines 74–76.** Replace:

```markdown
Library modules go through the four-stage protocol against §5/§7 of this
spec (interfaces finalized at task-cut time). Everything under
`experiments/` is exempt (single-pass + review).
```

with:

```markdown
Library modules are written together with their property tests against
§5/§7 of this spec. Everything under `experiments/` is script-land:
single-pass + self-review.
```

**(d) Line 497 — the closing "Next step".** It instructs cutting §5/§7
into PLAN.md tasks; PLAN.md is deleted. Replace:

```markdown
Next step (not this session): cut §5/§7 into PLAN.md tasks (ARITH-*,
PROBE-*, plus the spec 00 `experiment`-block edit) with named tests per
validation property, then pilot.
```

with:

```markdown
Next step: implement `geode.arith` (§5) with property tests V5.1–V5.7,
then `geode.probe` (§7) with V5.8–V5.16, then pilot (§11). Order and
rationale: `EXPERIMENTS.md` §4.
```

**(e) The remaining "spec-first" / "protocol-exempt" annotations** —
lines 69, 70, 71, 138, 333, 358, 446. Mechanical:

```bash
sed -i -e 's| (library, spec-first)||g' \
       -e 's|# library, spec-first (§\([0-9.]*\)):|# library (§\1):|g' \
       -e 's|(protocol-exempt)|(scripts — single-pass)|g' \
       specs/02-training-run.md
```

Leave §11 "Pilot protocol" alone — unrelated use of the word.

---

## 8. Phase 4 — rewrite CLAUDE.md

Currently mandates the four-stage protocol and impl logs as
"non-negotiable", and points at PLAN.md. Both retired, PLAN.md gone.
**"Conventions", "Testing policy", and "Budget rule" stay exactly as they
are.** "Attribution" stays except its one `geode.steering` pointer, fixed
in §5.4.

Line numbers below are pre-§5.4 approximations — that phase rewrites
CLAUDE.md:33–34 and may add a line, shifting everything under it. **Match
on the section heading, not the line number.**

**(a) Replace the whole "Execution protocol (per PLAN.md task)" section**
(starts line 62) with:

```markdown
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
```

**(b) Replace the whole "Documentation policy (non-negotiable)" section**
(starts line 78) with:

```markdown
## Documentation

Per-task implementation logs are retired (2026-07-17 cut);
`docs/impl-logs/` is frozen history. Record decisions where they belong:

- Experiment decisions, pilot outcomes, closed OPEN(n) items →
  `experiments/elicit-vs-teach/notes/decisions.md`.
- Structure / plan changes → `EXPERIMENTS.md`.
- Spec changes → the spec itself, in the same commit as the code.
```

**(c) In "What lives here"** (starts line 8):

- **Lines 10–11** currently read: `` - `specs/` — the source of truth.
  Every module is specified before it is implemented. Tests are derived
  from specs, never from implementations. `` This now contradicts the
  workflow (`geode.arith`/`geode.probe` get **no new spec documents**).
  Replace with: `` - `specs/` — schema and math ground truth for the
  tested core. The property lists in specs 01 and 02 are what tests
  derive from; new script work needs no spec. ``
- Drop `steering` and `saediff` from the module list; add `train`
  (corpus packing + full-FT trainer), `arith` (task data + evals,
  planned), `probe` (extraction + metrics, planned).
- Replace line 22 (``- `PLAN.md` — the approved build plan. All work
  executes against it.``) with ``- `EXPERIMENTS.md` — the approved plan:
  post-cut structure + per-experiment specs. All work executes against
  it. (PLAN.md was deleted in the 2026-07-17 cut; see git history.)``
- Note that `specs/` holds 00, 01, 02 — and that pre-cut references to
  `specs/02`–`specs/05` in git history mean different files (02 was the
  deleted steering spec; today's 02 is the renamed elicit-vs-teach spec).
- **Keep both `reference/` sub-bullets** — the crosscoder clone is
  explicitly retained. Reword line 17's `Used only to inform adaptation
  plans.` → `Kept read-only in case the crosscoder track revives.`

---

## 9. Phase 5 — trim manifest type-validation tests (OPTIONAL)

**Default: SKIP. Execute only on explicit owner instruction.**

Honest accounting, so no future agent re-litigates this: the suite is 242
tests in ~13s, far inside the CPU/2-minute budget. Deleting
already-written tests saves no runtime and no authoring time — that cost
is sunk. Deletion is *churn with a small risk* of removing a guard
someone later wants.

The one real argument: `specs/02` §4 adds an `experiment` block to the
manifest schema, and `test_wrong_type_map_covers_every_schema_field`
(test_manifest.py:278) asserts the type-case map covers **every schema
field** — so that cluster forces busywork when the block lands. Genuine
forward cost, unlike the rest.

**Scope is `tests/zoo/test_manifest.py` only.** Delete just the "wrong
type" / "null" / "enum" sections (~lines 200–348):

- `test_null_allowed_only_where_schema_says` (:209)
- `test_wrong_type_map_covers_every_schema_field` (:278)
- `test_wrong_primitive_type_rejected` (:284)
- `test_bool_rejected_for_int_typed_field` (:303)
- `test_invalid_enum_value_rejected` (:333)
- `test_valid_enum_values_accepted` (:342)

Then delete constants that become unused — `NULLABLE_FIELDS` (:107),
`NON_NULLABLE_FIELDS` (:119), `ENUM_FIELDS` (:227), `WRONG_TYPE_CASES`
(:271), `BOOL_FOR_INT_CASES` (:295), `INVALID_ENUM_CASES` (:315),
`VALID_ENUM_CASES` (:321) — and helpers `_walk_to_parent` (:122),
`_set_field` (:130), `_delete_field` (:135), `_load_and_validate` (:140),
`REQUIRED_FIELDS` (:66) **only if** grep shows no remaining references.
Ruff does not flag unused module-level names; check by hand:

```bash
grep -n "NULLABLE_FIELDS\|WRONG_TYPE_CASES\|_set_field\|_load_and_validate\|REQUIRED_FIELDS" tests/zoo/test_manifest.py
```

**Keep** in that file: `test_valid_manifest_passes_validation`,
`test_missing_required_field_error_names_field`,
`test_lora_fields_required_even_for_full_ft`,
`test_roundtrip_preserves_unknown_extra_fields` (**load-bearing** — spec
§4 parks the whole `experiment` block in the manifest as *unvalidated
extra fields*; if the round-trip silently drops unknown fields you lose
arm / role / parent_run_id / data_order_hash / probe_set_hash / gates
provenance), `test_register_then_load_identity`,
`test_iter_runs_filters_regime_task_status`,
`test_store_root_comes_from_env`.

**Do NOT touch `tests/zoo/test_records.py`.** Despite living under "zoo",
`test_epoch1_ids_skip_rejected` / `_repeat_rejected` / `_exact_cover_accepted`
(:129, :146, :157) and `test_masking_hash_mismatch_raises` (:182) guard
**EDL math** — prequential MDL is valid only if every example is seen
exactly once in epoch 1 (OQ-4), and a train/test masking mismatch makes
the EDL number meaningless (OQ-7). Both fail silently in production.
Among the most load-bearing tests in the repo.

**Do NOT touch `tests/zoo/test_results.py`.**
`test_two_tables_join_on_run_id_dataset_size` (:158) guards the join
contract; a misaligned join silently pairs arm A's numbers with arm B's
steps. (A known row-alignment concern from the 2026-07-12 review.)

Expected: 242 → ~200 passed (the deleted tests are parametrized, so the
drop exceeds 6).

---

## 10. Verification

```bash
python -m pytest -p no:cacheprovider -q 2>&1 | tail -2
# Phases 1-4 only: expect 242 passed, 0 failed
# With Phase 5:    expect ~200 passed, 0 failed

ruff check . && ruff format --check .
python -c "import geode.edl, geode.train, geode.zoo; print('core imports ok')"

ls specs/     # expect exactly: 00-interfaces.md  01-edl-harness.md  02-training-run.md
ls PLAN.md    # expect: No such file or directory
ls -d experiments/elicit-vs-teach     # must STILL EXIST (dir is not renamed)
ls .claude/agents 2>/dev/null         # expect: no such directory (or empty)
git diff --name-only docs/impl-logs/TRAIN-1.md  # MUST be empty (frozen history)

# no live refs to deleted modules. NOTE the .md include — an earlier draft
# omitted it and missed dangling refs in README/specs. test_results.py's
# "steering_sufficiency" metric-name string is test data: the one allowed hit.
grep -rn "steering\|saediff" --include='*.py' --include='*.toml' --include='*.md' . \
  | grep -v 'reference/' | grep -v 'docs/impl-logs/' | grep -v 'test_results.py' \
  | grep -v 'CUT-PLAN' | grep -v 'EXPERIMENTS.md'
# expect: only the intentional "never built / deleted" mentions in
# CLAUDE.md Attribution and specs/02 §7.

# no live refs to deleted/renamed files:
grep -rn "PLAN\.md\|specs/05\|05-elicit\|02-steering\|03-base-sae\|04-crosscoder\|PLANNING_PROMPT" \
  --include='*.md' --include='*.py' --include='*.yaml' . \
  | grep -v 'reference/' | grep -v 'docs/impl-logs/' | grep -v 'CUT-PLAN' | grep -v 'EXPERIMENTS.md'
# expect: only the intentional git-history pointers this plan's own replacement
# text creates — CLAUDE.md's "PLAN.md was deleted in the 2026-07-17 cut" line
# and its specs/02–specs/05 numbering note (§8c), and specs/00 §9's "Noted here
# from the deleted PLAN.md" lead-in (§6.2). Anything else is a real dangle.

# dead deps gone:
grep -n "sae-lens\|transformer-lens" pyproject.toml   # expect: no output
```

Done when: the suite passes at the expected count, ruff is clean, the
greps are clean, `specs/` holds exactly the three kept files, the
`experiments/elicit-vs-teach/` directory still exists, `docs/impl-logs/`
is untouched, and `EXPERIMENTS.md` §2's tree matches reality.

## 11. Commit

One commit.

```
CUT: retire spec-first protocol, delete unbuilt modules and PLAN.md

Keep the tested core (edl, train, zoo) and its property tests. Delete
geode.steering / geode.saediff stubs and specs 02-04 (modules never
built), the four-stage protocol's stage agents and enforcement hook, and
PLAN.md. Rename the surviving experiment spec 05-elicit-vs-teach.md ->
02-training-run.md; specs now read 00, 01, 02.

PLAN.md's live decisions (OQ-3/4/5/6/7/8/14) are noted into specs/00 §9
before deletion -- they explain why the shipped core behaves as it does,
including the EDL epoch-1 coverage invariant and the masking-hash guard.
They are a reference note, not a citation contract. OQ-1/2 and OQ-9..13
died with the modules they served.

Also fixes stale content in the experiment spec: the four-stage rigor
mandates, and a false claim that adapter diffs reuse
geode.steering.extract_weight_diff "validated by V2.6" -- that module was
never built and V2.6 was a spec property, never a test.

Drops sae-lens / transformer-lens: dependencies of the deleted saediff.

Plan: EXPERIMENTS.md. Execution record: docs/CUT-PLAN.md.
Suite: 242 passed, unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

## 12. After the cut — next task

Per `EXPERIMENTS.md` §4, next is `geode.arith` (task data + evals) with
property tests V5.1–V5.7 from `specs/02-training-run.md` §5. It blocks
everything else. Write code and tests together in one pass per the new
workflow; do **not** spawn stage agents or write an impl log.
