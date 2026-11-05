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
- Results: see below.

## Local results (`results/`, gitignored)

One folder per stage, named `<date>_<stage>` (date = the stage's `PLAN.md` block),
listed in run order. The canonical copies are on HF `mhieuuu/geode-internals`;
the HF column gives the path under `results/`. Reorganized 2026-11-05 from the old
`results_*` dirs (old → new map at the top of `PLAN.md`); no file was deleted.

| Folder | Stage | What's inside | Size | HF |
|---|---|---|---|---|
| `2026-10-08_ld_check_large/` | 2026-10 rerun at 2/5/10% (LD, word). **Also holds** the circuit-change outputs and an LD-word `edge_node/` | 4 tags × `scores.pt`/`sanity.json`/`evaluate.json` (scores from the 2026-10-01 first run), `compare.json`, `circuit_change_<frac>.json`, `figures/cc_*`, `edge_node/` | 777 MB | `eapig_check_large/` (circuit change + edge_node local only) |
| `2026-10-08_perf_eval/` | KL stage, step S (behavior) | `perf_eval.json` per model / surface | 60 KB | `eapig_perf_eval/` |
| `2026-10-08_kl_word/` | KL stage: KL-word score sets | `sanity.json` (+ `probe.json` for parents); `scores.pt` not kept here. Later box scripts fetch scores into it (base adds `base/`, 2026-10-22) | 92 KB | `eapig_kl_symbol/klword/` |
| `2026-10-08_kl_symbol/` | KL stage: KL op-form score sets (elicit route) | same layout | 48 KB | `eapig_kl_symbol/klsym/` |
| `2026-10-08_ld_symbol/` | KL stage: LD op-form score sets (elicit route) | same layout | 48 KB | `eapig_kl_symbol/ldsym/` |
| `2026-10-08_edge_node/` | KL stage: edge-vs-node change | `kl/`, `ld/` (word), `sym_kl/`, `sym_ld/` (op-form parent): `edge_node.json`, `cc_0.02/`, `cc_0.05/`, figures | 7.2 MB | `eapig_kl_symbol/` |
| `2026-10-08_functional/` | KL stage: functional reuse | `kl/`, `ld/` `<child>.json`, plus the box job scripts/logs `plan*.sh`, `plan*.log` | 92 KB | `eapig_kl_symbol/` |
| `2026-10-08_kl_tests_02/`, `2026-10-08_kl_tests_05/` | KL stage: five tests with KL at 2% / 5% (`plan2.sh`; PLAN cites them as `results_kltests_{02,05}`) | `<tag>/evaluate.json`, `per_example.npz` | 136 KB each | `eapig_kl_tests/frac_0.02/`, `frac_0.05/` |
| `2026-10-22_cmap/` | edge-level CMAP | `{elicit,teach}.json/.npz`, `summary.md`, `run_log.txt` (no shards) | 160 KB | `eapig_cmap/` |
| `2026-10-22_base_control/` | base TS-1B overlap control | `edge_node/`, `boot/overlap_boot.json`, `base_control_table.md`, figure, box logs | 1.2 MB | `eapig_base_control/` (scores: `eapig_kl_symbol/klword/base/`) |
| `2026-11-04_folds/` | 10-fold robustness | `fold_00`–`fold_09/`, `overlap.json`, `fold_report*.{md,json}`, `repro/`, `old_split/`, `paper_pref/`, `figures/` | 141 MB | `eapig_folds/` |

On HF only (no local copy); the scripts write these paths if rerun:
`2026-10-01_ld_check/` (first run, 0.1–1%; default of `run.py`, `compare.py`,
`run_box.sh`; HF `eapig_check/`), `2026-10-21_translate/` (`eapig_translate/`),
`2026-10-21_small_sizes/{tests,functional,overlap}/` (`eapig_kl_tests_small/`),
`2026-10-28_tiny/` (`eapig_tiny/`). `2026-10-08_kl_tests/` is `run_box_2gpu.sh`'s
step-6 dir; the actual 2%/5% runs went to `_02`/`_05` above.
- Summary page: `ts1b_summary_page.html`. Slide decks: see memory
  `project-eapig-10fold-proposal-2026-11-03.md`.
