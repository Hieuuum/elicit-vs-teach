# docs/impl-logs — implementation logs

> **⚠ SUPERSEDED 2026-07-17.** The per-task implementation-log policy was
> retired with the four-stage protocol; see `CLAUDE.md` → "Documentation".
> This directory is frozen history. Do not add logs. Record decisions in
> `experiments/elicit-vs-teach/notes/decisions.md` or `EXPERIMENTS.md`.
> The text below describes the retired process and references the deleted
> `PLAN.md` and the pre-rename `specs/05`; both are preserved as-written
> so the existing logs read in context.

**Policy (binding, see CLAUDE.md "Documentation policy"):** every
implementation run — one PLAN.md task through its full stage cycle — ends
with a log written here, `<TASK-ID>.md`, before the task's commit. The
log is the durable account of *why* the diff looks the way it does; specs
say what must be true, logs say what happened.

## Index

| Task | Date | Module | Outcome |
|------|------|--------|---------|
| [TRAIN-1](TRAIN-1.md) | 2026-07-16 | `geode.train` | done — 4 stages + 1 finding-fix loop, 38 tests |

(Tasks completed before this policy existed — SETUP-0 … EDL-3 — are
documented in PLAN.md task blocks and commit messages only.)

## Template (copy for each new task)

```markdown
# <TASK-ID> — <one-line title>
Date · task scope · spec sections · stage-model roster (+ why any deviation)

## 1. Decisions made (numbered; each with WHY and alternatives rejected)

## 2. Stage-by-stage account
Per stage: agent/model, inputs given, what it produced, findings raised
(verbatim severity + disposition), anything it flagged for later stages.

## 3. Tests
Table: test name → V-property → what breakage it catches. Rationale for
extras beyond the PLAN list.

## 4. Diagrams
Mermaid where structure helps: module graph, stage flow, state machines.
Generated plots (matplotlib) go in docs/impl-logs/assets/<TASK-ID>/ and
are embedded — REQUIRED whenever the task produced runnable numeric
behavior worth eyeballing (training curves, schedules, distributions).

## 5. Verification evidence
Exact commands + outcomes (suite count, wall time, ruff, greps).

## 6. Loose ends
Deferred items, OPEN(n) markers touched, flags for future tasks.
```

Rules of thumb: write for someone reading six months from now with no
session context; quote findings rather than paraphrasing severity down;
every decision gets its *why*; link spec lines, don't restate them.
