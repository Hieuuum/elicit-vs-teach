# Artifact manifest — figures

Figures are regenerable from `geode-store/results/` parquets + the pinned
driver script; this manifest records provenance and the exact bytes that
existed on 2026-07-29. Both figure directories are **gitignored**
(`.gitignore` line `figures/`) — this manifest is the tracked record.
`notebooks/data/pilot/` (the superseded 10K-row pilot) is deliberately
excluded from this manifest.

Producer scripts identified by grepping for each output stem under
`analysis/` and `notebooks/`; two filenames (see below) are not hardcoded
anywhere in the tracked repo — they are custom `--out` values from ad hoc
invocations of `plot_edl_per_token.py`, confirmed by the script's `--floor`
and `--per` flag semantics, not by a literal string match.

## `analysis/figures/` — 8 PNGs

| file | sha256 | bytes | produced by |
|---|---|---|---|
| `activation_rank.png` | `933aa0baa117cfd071b73279c1d41c5adb10249570829de783587f63d0ec2719` | 231,202 | `act_rank.py` |
| `cka_matched.png` | `44543a767d4e1f3fd258f395ecb95845d3a47d6432cc7621e602f8b958549193` | 244,757 | `cka.py` |
| `direction_emergence.png` | `9120889ade2d42a79778cc099f13585692f04757ec12dc758992fe4ecc52d465` | 353,892 | `emergence.py` |
| `edl_per_example_n1.png` | `bbbe695ad54785c26cd70be7abfde686a51b41b3bcaa5be243b34e89b8163f5a` | 107,224 | `plot_edl_per_token.py` (`--per example`; custom `--out`, not the script's default stem — see note below) |
| `edl_per_example_n1_logy.png` | `fe6ffc4f6702fdef852d904329c5f98c1c8d9ef7a822b7d64f6e1fcda7d8f41a` | 126,759 | `plot_edl_per_token.py` (`--per example --logy`; custom `--out`) |
| `edl_per_token.png` | `b3d16e19d3809f9ab9f348c10f5647d0383f24d2a603f839de4145a49022a253` | 194,927 | `plot_edl_per_token.py` (default `--out`, `--floor val`) |
| `phase3_teach_vs_elicit_edl_per_token_moving_validation_floor_from_first_batch.png` | `57e5d31ea7970492755d56ae23557ce07a8276b7977ca2c6b114d44c816b60dd` | 136,853 | `plot_edl_per_token.py` (`--floor val`; custom `--out`) |
| `phase3_teach_vs_elicit_edl_per_token_test_floor_from_first_batch.png` | `fc6efa60209b497fa759b954bcc85bc8e66f87325833822c075d00c83391cae5` | 158,012 | `plot_edl_per_token.py` (`--floor test`; custom `--out`) |
| `run8_teach_edl_per_token_logy.png` | `837813b343f09146b61b99b1cb034758370eab674ed9717d6f047d2d86937788` | 114,050 | `plot_edl_per_token.py --floor val --logy --smooth 7 --plain-ticks --run-id evt-run8-armB-target-1m` |

**2026-07-30 — the two `phase3_teach_vs_elicit_*` figures' "teach" curve has NO
verified source.** The actual phase-3 teaching arm (`evt-p3-teach-inst` ->
`evt-p3-teach-target`) was built but never launched (`docs/runbooks/
phase3-runbook.md`, decisions.md "Phase-3 teaching arm built (unrun)"; no
`evt-p3-teach-*` directory exists under `geode-store/runs/`). Reproducing
`--floor val` against every real phase-3 run that has training logs
(`evt-p3-elicit-recover-target`, `evt-p3-elicit-target-bridge`, the three
embedding-warmstart targets) matches none of the shown "teach" curve's shape
(it rises from near-zero and spikes late, unlike any of those runs, which all
peak early and decline under the moving-val floor per this script's own
docstring). The two PNGs' title/axis-label/legend text also doesn't match
this script's current output format (legend shows "Phase 3 elicit"/"Phase 3
teach", not a run-id), so they were produced by a different, uncommitted
script. Treat both `phase3_teach_vs_elicit_*` files as **unverified** until
their source is found; `run8_teach_edl_per_token_logy.png` above is a
verified substitute (real `evt-run8-armB-target-1m` data, the main project's
armB_teach 1M-scale target run — cross-phase, not phase-3 data, labeled
plainly by run-id rather than an "elicit"/"teach" gloss).

**Deviation from the original plan draft:** the plan guessed
`edl_per_example_n1*.png` came from `learning_curves.py`. Grepping both
scripts shows `learning_curves.py`'s only output is `learning_curves.png`
(per-cell acquisition figure); `plot_edl_per_token.py` is the only script
with a `--per example` mode and a configurable `--out`, matching the
`edl_per_example_*` filename pattern. Corrected here per rule 0 (verify
before recommending). The `_n1` suffix's exact meaning (likely an n=1
example-count or run-selection label chosen at generation time) could not
be reconstructed — `figures/` is gitignored, so there is no git history for
the originating command.

Also present in this directory, **not figures** (excluded from this
manifest by design — CSV/JSON evidence tables, not PNGs):
`g5_predictions_n0_n1.csv`, `g5_predictions_n0_n1.html`,
`unlock_forward.csv`, `unlock_forward.meta.json`, `unlock_mirror.csv`,
`unlock_mirror.meta.json`, `unlock_provenance.csv`.

## `notebooks/figures/` — 5 PNGs

All five produced by `notebooks/key_figures.py`.

| file | sha256 | bytes | produced by |
|---|---|---|---|
| `1_prequential_curve.png` | `7925a43da3fbb8f9205bee2b73012b47bb831f28ecf2df053e171d579b3c4685` | 101,145 | `notebooks/key_figures.py` |
| `2_edl_accumulation.png` | `420003aaafcf87570667e16fbe690c967e60425b9d05f9a70e8687fc4c0d5bd5` | 74,395 | `notebooks/key_figures.py` |
| `3_val_loss.png` | `e07d34d597863b52ae14eba565722d3dbadc0e7c29bf83654463eb7ad84ec35e` | 86,493 | `notebooks/key_figures.py` |
| `4_mdl_edl_summary.png` | `851687eed306f8f2ae2ca8b9915cea407c26835c9724769e8764cc4e666735d9` | 63,585 | `notebooks/key_figures.py` |
| `5_g5_evidence.png` | `6fa8c024999f7d546e01efda2d7a5bbed39f36bda12c01c015bfca74a8359d53` | 59,463 | `notebooks/key_figures.py` |
