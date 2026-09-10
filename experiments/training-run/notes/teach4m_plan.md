# Teach-4M replication program (opened 2026-09-10)

**Why.** The teach twin's 1M endpoint (`evt-ts1b-fig2ts-noinst-n1000000`) solves
0.093 of held-out problems. It stopped on the validation-convergence rule after
3.2 passes over 1M examples, mid-hump — where the paper's Fig. 2 also puts the
TinyStories base at 1M. Every teach-side mechanistic number so far is measured
on a model that has not finished learning. The paper's remedy is more *unique*
data (its pre-teach: one epoch over 4M unique examples), not more epochs.

**Fix.** `evt-ts1b-fig2ts-noinst-n4000000`: blank base → LoRA r=512 on
4,000,000 unique NL add/sub problems (`D_algo_bare_4m`; first 1M rows byte-
identical to `D_algo_bare`, next 3M unique and eval-disjoint), same protocol
as the 1M run (lr 3.53e-4, batch 128, seed 316, eps/k stop), min one full
epoch, endpoint snapshots. Then rerun every teach-side result on the new
endpoint and update decisions.md / results_ts.tex where it does not replicate.

## Checklist — teach-side claims to re-measure (current number → 4M number)

| result | claim (teach side, 1M endpoint) | current | 4M | verdict |
|---|---|---|---|---|
| R1 behaviour | teaching hump; EM at endpoint | EDL 1.40 nats/tok, EM 0.093 | | |
| R2/R4 machinery | taught circuit ≠ engine: J@32 vs elicited, Spearman | 0.231 / −0.11 | | |
| R2/R4 | taught circuit layer profile (layer-0 army) | layer-0 heads dominate | | |
| R4 validation | true patching: top-8 necessity / top-32 | 0.986 / ~0.999 | | |
| R3 DCM | roles vs elicited (J), compact set recoverable? | J 0.085–0.125; cf-flip 0.02–0.04 vs ceiling 0.15 | | |
| R5 formation | circuit built late (window) | 1.3K–5.4K steps, noise floor early | | |
| R6 gradients | late-run norm grows; cumulative mass; VO share | 6.80 (×40); 116,877 (150×); .87 | | |
| R7 weights | relative write; travel; EM per unit travel | 0.212; 457; 2.0e-4 | | |
| R8 residual shift | final-layer shift; generic shift; PC1 at read-out | 14.1×; 0.156; 0.351 | | |
| R8 v2 | KL task/generic; generic ΔNLL; content-removed PC1 | (pending v2 rerun) | | |
| R10 lens | J-rank L12/L14/read-out; first layer ld>1; settled | 41,677 / 19,933 / 8; L12; L14 | | |
| R11 ladder | taught donor into blank twin (per-prompt states) | 0.000 every condition | | |
| non-discr. | erank(PR) of ΔW | 5.2 | | |

Elicit-side numbers are unchanged by this program; comparisons that involve
the taught child are re-done against the 4M endpoint.

## Stages — `scripts/launch_teach4m.sh --confirm-cost [--stage N] [--no-stream] [--no-prune]`

Everything is skip-if-done; rerun after a crash to resume. Run from `scripts/`
with the `geode` conda env, `GEODE_STORE` exported, and (unless `--no-stream`)
`HF_WRITE_TOKEN` set.

| stage | what | cost |
|---|---|---|
| 0 data | `datagen/make_algo_4m.py --out ../data/full` → `D_algo_bare_4m.parquet` (4,000,000 rows, 174 MB), pin `b05a65df…98ff` verified against the overlay | ~5 min CPU, ~6 GB RAM |
| 1 train | `train_target.py --config ts1b_fig2ts_noinst.yaml --override sweeps/ts1b_fig2ts/ts1b_fig2ts_noinst_n4000000.yaml --init-from runs/evt-ts1b-base/model`; snapshots streamed to `podhajskimarcin/evt-ts1b-fig2ts-noinst-n4000000`; G5; push; sha verify; prune | 31,250–62,500 steps ≈ 1.3–2.5× the 1M run |
| 2 behaviour | `dataset_size_sweep.py --family ts` with the 19 noinst points + the 4M point | CPU |
| 3 battery | circuit map + split-half + compares (vs elicited, vs taught-1M, vs base), faithfulness (suff/nec), DCM roles + compares, steering donor (mean, per-prompt), weight shift, gradient strength, residual shift v2 + compare, lens depth + compare + breakdown, formation curve, weight travel — all with the flags used for the 1M endpoint; log in `analysis/teach4m_battery.log` | ~2–3 h GPU |

| 4 train-ft | `train_sft.py --config ts1b_teach_ft.yaml --init-from runs/evt-ts1b-base/model` → `evt-ts1b-teach-ft-n4000000`: the blank twin FULLY fine-tuned (lr 2e-5, the symbol-install recipe) on the same 4M file, one pass minimum, two-pass ceiling; G5. Never pruned. Not concurrent with stage 3 on a 40 GB card (~25 GB). | ~10–19 h GPU |
| 5 battery-ft | the same battery on the full-FT endpoint; gradient strength, formation curve and weight travel are skipped (train_sft.py logs no gradstats and takes no snapshots); weight shift from the checkpoint diff; extra compare: FT circuit vs the LoRA-4M circuit | ~2–3 h GPU |

**Stage 1 result (2026-09-10):** `evt-ts1b-fig2ts-noinst-n4000000` converged at step 39,500 (1.3 passes), EDL/tok 0.930 (1M: 1.40), best val 1.42 nats, G5 EM 0.139 0-shot / 0.000 16-shot. More unique data did NOT produce a performing taught model under the LoRA protocol — the paper's Fig. 2 shape (TS base still ~1 nat/token at 4M). Hence stage 4: the one teach recipe that has worked here is full FT (symbol install 0.67–0.73).

Comparator artifacts (the elicited child's circuit map, the taught-1M map,
the base 16-shot map, the DCM JSONs) are auto-discovered from the JSON
sidecars in `analysis/` by their recorded `model` (and `surface`); override
with `MAP_ELICITED`, `MAP_TAUGHT1M`, `MAP_BASE16`, `DCM_ELICITED_NL`,
`DCM_PARENT_OP` if discovery picks the wrong stem.

## Data deviation (recorded)

The exclusion `D_algo ∪ D_algo_eval ∪ probe` (1,101,024 triples) exhausts six
operand-length cells (1x1, 1x2, 1x3, 2x1, 2x2, 3x1) and leaves ~57K in four
more (1x4, 2x3, 3x2, 4x1). The 3M extension is therefore water-filled as
57,097–57,098 in those four and 461,935 in each of the six largest cells
(2x4, 3x3, 3x4, 4x2, 4x3, 4x4) — ~92% from the six largest cells, versus 57%
in the 1M prefix. Task, evaluation set and protocol are unchanged.

## Stage 4/5 interim (2026-09-10)

`evt-ts1b-teach-ft-n4000000` (full FT): max_steps at 62,500 (2 passes, still
improving), val 0.244, test loss 0.204, **G5 EM 0.771** (16-shot 0.001). The
first performing taught model. Valid stage-5 outputs so far (map-independent):
DCM roles (operand_a/b 37 heads, cf-flip at ceiling; J vs elicited 0.587 /
0.763, vs parent-op 0.763 / 0.684), residual shift (final-layer 1.42x, generic
0.464, KL task 22.2 vs generic 1.15 nats/token, generic NLL +1.13, PC1 0.52
with content removed 0.52), lens depth (settled L15, q25 L15; J-rank
4,196 / 1,916 / 79 at L12 / 13 / 14; onset ld>1 at L8), weight shift (rel
0.094, erank 571 of 2048), steering nulls (0.000 all conditions). Circuit map,
compares, faithfulness and steering-with-own-map are being redone: the stage-5
map stem collided with the LoRA-4M map (stale guard added).

## Verdicts (2026-09-10) — see decisions.md entry "teach-4M program COMPLETE"

Replicates with performing taught models: R1, R2, R4 (graded), R6, R7
(magnitude), R8 (direction), R10, R11, faithfulness. Revised: R3 (roles are
substrate-level; graded 0.8–0.9 vs 0.6–0.76), R4 wording ("partial"), R8
magnitude ("loud write" = LoRA-teach artefact; generic displacement
method-dependent). Pending: LoRA-4M formation curve + weight travel (stage 3
resume after the trajectory-loader fix).
