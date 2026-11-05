# eapig-circuit-check

EAP-IG circuit analysis of the ts1b elicit-vs-teach models (Oct–Nov 2026).
**`PLAN.md` is the log.** Each stage's outcome is a dated block under
`## Results`. The repo-wide state is in `/STATUS.md`.

## Stages and their handoffs (all closed)

Every `HANDOFF-*.md` here is the plan written *before* its stage ran. Several
still say "PLAN ONLY" at the top, because they were never edited afterwards.
The outcome is always in `PLAN.md`.

| Handoff | Outcome (PLAN.md block) | Result |
|---|---|---|
| `HANDOFF-sizes-2-5-10.md` | 2026-10 rerun at 2/5/10% | elicit reuses the parent circuit |
| `HANDOFF-circuit-change.md` | 2026-10 circuit change at 10% | parent → child change |
| `HANDOFF-kl-symbol-nodeedge.md` | 2026-10-08 KL metric, op-form parent circuit | Jaccard 0.66 ≈ ceiling; teach 0.37 |
| `HANDOFF-translate-circuit.md` | 2026-10-21 translate-then-solve | gate FAILED, no circuit scored |
| `HANDOFF-small-sizes.md` | 2026-10-21 small sizes 0.1–5% | Jaccard 0.72–0.77; teach ~1/3 |
| `HANDOFF-cmap.md` | 2026-10-22 edge-level CMAP | elicit recovery 0.77–0.98; teach negative |
| `HANDOFF-base-control.md` | 2026-10-22 base TS-1B overlap control | floor ~0.2; elicit far above |
| (no handoff) | 2026-10-28 tiny sizes 0.01–0.05% | sufficiency fails at 0.01–0.02% |
| `HANDOFF-folds.md` | 2026-11-04 10-fold robustness | 10/10 folds hold; hidden preference fails |

## Code map

- Core: `run.py` (circuits), `data.py`, `compare.py`, `overlap_boot.py`,
  `functional_reuse.py`, `cmap.py`, `perf_eval.py`, `edge_node.py`,
  `circuit_change.py`.
- Folds (2026-11-04): `build_folds.py`, `fold_runner.py`, `fold_overlap.py`,
  `fold_report.py`, `hidden_pref.py`.
- Box scripts: `run_box*.sh`, one per stage.
- Results: `results_*` dirs are gitignored. The canonical copies are on HF
  `mhieuuu/geode-internals` under `results/`.
- Summary page: `ts1b_summary_page.html`. Slide decks: see memory
  `project-eapig-10fold-proposal-2026-11-03.md`.
