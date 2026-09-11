Implementation and the Stage-2/RLVR-2 pilot are complete; the GPU instance is deleted.
Math improved on this small sample, while controlled ETHICS and circuit reuse remain inconclusive.
The pilot took 10.2 minutes; the entire rental cost approximately $0.23, including setup and backup.

The [plot report](../../geode-store/olmo2-pilot-20260910/report.md) gives a takeaway,
caption, identified data example and caveats for each figure. Artifacts are saved
privately at [Hugging Face](https://huggingface.co/datasets/mhieuuu/olmo2-circuit-overlap).
This pilot compares two endpoints, so it cannot localize changes to individual
post-training phases. No full evaluation was launched.

The pilot used 16 independent behavioral groups per task, with two GSM instances
per template; eight attribution source pairs and four held-out intervention pairs
per task; eight ETHICS label/position controls; and twenty target probe groups.
Math/code probes contain reciprocal candidate pairs. Confidence intervals resample
source groups, keeping controls and related instances together.

| Measurement | Stage 2 | RLVR-2 | Interpretation |
|---|---:|---:|---|
| GSM-Symbolic accuracy | 11/32 (34.4%) | 20/32 (62.5%) | Paired gain 28.1 points; grouped 95% CI 6.2–46.9 points |
| CRUXEval input accuracy | 5/16 (31.3%) | 2/16 (12.5%) | Small sample; paired interval includes zero |
| CRUXEval output accuracy | 4/16 (25.0%) | 3/16 (18.8%) | Small sample; paired interval includes zero |
| ETHICS balanced-control macro | 20.6% | 21.1% | Native group scores averaged across eight label/position variants |
| Top-16 overlap | Across endpoints: 0.185–0.391 | Nine task/domain comparisons | All inside the MLP/head-preserving null's 95% reference range |

The uniform-node random baseline is retained, but whole MLP nodes occupy 7–12
of each top-16 set despite being only 16 of 272 available nodes. A conditional
null preserves that concentration. Probe test sets have only four independent
groups; high accuracy must be compared with majority and shuffled-label controls.
One Stage-2 math generation reaches the token cap even at 2,048 output tokens.

| Work | Measured pilot, both checkpoints | Full-run projection, seven checkpoints | Why |
|---|---:|---:|---|
| Download/load checkpoints | 1.6 min | 4–7 min | One checkpoint loaded at a time |
| Native behavior and answer scores | 1.1 min | 6.9–17.3 h | Autoregressive math/code generation dominates |
| Attribution and interventions | 1.8 min | 0.8–1.7 h | Gradients plus top/random activation patches |
| Feature extraction and probes | 5.3 min | About 1.8 h before scaling overhead | Per-layer activations and held-out classifiers |

The **9.5–21.0-hour projection ($5.86–$12.88)** uses exact realized full-workload
counts and measured batch-8 throughput, with the 2,048-token math calibration.
It is a sensitivity range, not a confidence interval or a guaranteed upper bound.
Initialization, other unmeasured checkpoints, long generation tails and larger
probe fits can increase it. CRUX throughput still uses the 512-token pilot cap.
The full workload has 186,312 behavioral rows per checkpoint. Available source
groups and matching restrict GSM to eight intervention pairs and 47 probe groups;
CRUX-I/O yield 203/298 attribution pairs and 68/81 intervention pairs.

Batch calibration on the same 32 math items at a 2,048-token cap:

| Batch | Stage-2 seconds / correct | RLVR-2 seconds / correct |
|---|---:|---:|
| 8 | 49.6 / 11 | 15.7 / 20 |
| 16 | 52.6 / 10 | 9.1 / 21 |
| 32 | 77.9 / 10 | 5.5 / 19 |

Larger batches helped RLVR-2 but hurt Stage 2; BF16 rounding changed some answers.
Keep batch 8 fixed for comparison. The tested GPU was an **RTX 6000 Ada, 48 GB
VRAM**, with 12 allocated vCPUs, about 128 GB advertised host RAM and 100 GB disk,
at $0.61444/hour including disk. Smaller GPUs were not benchmarked. Future savings
can come from reusing checkpoint/data caches and computing pair plans and report
statistics locally; no additional GPU rental is needed for analysis of saved arrays.

All 103 original artifacts (429.6 MB) passed SHA256 verification before instance
50537723 was destroyed. Rental duration through deletion verification was 22.3
minutes; the $0.23 figure is an estimate, not an invoice. Its outbound rate was
$0.004/TB. Uploading to HF still counts as Vast outbound traffic, but this host's
transfer charge is negligible. HF account access and private-repository write
access were verified; cloud GPU credentials were not used for HF uploads.

All **240 new tests pass** and Ruff is clean. The full repository has one
confirmed pre-existing test failure: the
`ts1b_fig2ts` sweep-count guard expects 38 YAMLs, while `fig2nl2` already has 41.
Detailed new-suite validation, immutable checkpoint revisions, dataset/source
hashes, exact runtime source, predictions, arrays, controls and timings are saved
with the pilot artifacts. The branch is `olmo2-circuit-overlap`, created from
`fig2nl2` at `08ba917bee6459ec19c87739b58a69a7e2f0ca02`.
