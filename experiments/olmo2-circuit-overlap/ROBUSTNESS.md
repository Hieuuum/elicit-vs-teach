# Robustness of node overlap — exploratory review, 2026-09-21

Replace headline Jaccard@16 with continuous attribution-profile comparisons,
separately for attention heads and MLP blocks. Keep the cutoff curve as a
sensitivity diagnostic. A higher within-stage overlap is not itself evidence
that a metric is better: selecting every node gives perfect stability.

## What the saved data show

The existing ranking is `abs(mean(signed attribution))`, with equal source-group
weighting. This measures the magnitude of a **net directional effect**. Averaging
absolute attribution first measures **involvement regardless of direction**.
These are different scientific targets, not interchangeable implementations.

A small CPU-only audit used all eight saved tasks and seven checkpoints, with
200 disjoint group splits and 500 paired group bootstrap draws (seed 20260921).
The node universe contains 256 individual attention heads and 16 entire MLP
blocks. No model inference, GPU rental, or existing-report regeneration occurred.

Example results at SFT, averaged over the same 200 group splits:

| Task | Current top-16 Jaccard, net effect | Top-16 Jaccard, average magnitude | MLPs among full-data magnitude top 16 | Heads-only magnitude top-16 Jaccard | Heads-only magnitude weighted overlap |
|---|---:|---:|---:|---:|---:|
| ETHICS commonsense | 0.274 | 1.000 | 16/16 | 0.629 | 0.922 |
| GSM-Symbolic | 0.432 | 0.874 | 16/16 | 0.606 | 0.831 |
| CRUXEval output | 0.512 | 0.900 | 16/16 | 0.708 | 0.908 |

These are descriptive split averages, not accuracy scores or comparable scales
of statistical reliability. Higher values for weighted overlap cannot be read
as a measured improvement over Jaccard by subtracting the two numbers.

For ETHICS commonsense at SFT, total absolute net attribution is just 2.1% of
total average absolute attribution. Signed effects largely cancel, making the
remaining net ranking sensitive to which examples enter each half. Its mean
signed cosine between halves is -0.121. Changing only the cutoff cannot resolve
that instability. This does not establish whether cancellation is due to
task heterogeneity, prompt/control design, or weak directional signal.

The perfect magnitude top-16 stability here is also misleading as evidence of
task specificity: it selects all 16 MLP blocks. Heads-only results demonstrate
why node-type separation is necessary. Large average absolute effects can also
reflect noisy or generic sensitivity, so magnitude alone is not causal evidence.

## Recommended primary measurements

For source group g and node i, define `m[g,i]` as the mean absolute per-example
attribution within that group. Define `w[i] = mean_g(m[g,i])` and normalize
`p[i] = w[i] / sum_j(w[j])`. Do this separately within heads and within MLPs.
Keep ETHICS categories separate; any pooled result should weight categories
explicitly rather than weighting their row counts.

Compare two stages' profiles using normalized weighted Jaccard:

`W(p,q) = sum_i min(p[i],q[i]) / sum_i max(p[i],q[i])`.

It uses every node's weight and has no rank cutoff. Near-tied nodes exchanging
ranks cause a small change rather than abrupt entry/exit. For normalized
profiles it equals `(1-TV)/(1+TV)`, where `TV = 0.5 * sum_i abs(p[i]-q[i])`.
Report zero-total-mass profiles as undefined. Also report total attribution
mass: normalization intentionally discards global changes in effect magnitude.
The many weak nodes can collectively matter; inspecting concentration and a
cosine sensitivity check helps reveal whether they drive the answer.

Retain the signed mean profile and its signed cosine as a companion. Magnitude
overlap cannot distinguish a positive effect from an equally large negative
one. Use `abs(mean(signed attribution))` as an explicit sensitivity analysis,
rather than silently replacing the original question. Do not choose the
aggregation order based on which produces the highest stability.

## Compare stage changes to sampling variability

Use identical disjoint source-group halves L and R across checkpoints. Compute:

- Within-stage similarity: `S(A_L,A_R)` and `S(B_L,B_R)`.
- Cross-stage similarity on independent halves:
  `(S(A_L,B_R) + S(A_R,B_L))/2`.
- A descriptive gap: average within-stage similarity minus cross-stage
  similarity. Positive values indicate additional between-stage disagreement
  under that sampling scheme; values near zero do not prove equivalence.

For the original net-effect profile and weighted overlap on GSM-Symbolic, the
Stage 2 to SFT gap averages 0.139; its central 95% split range is 0.105–0.178.
The SFT to DPO gap averages 0.001; its range is -0.011–0.014. This illustrates
how to distinguish a stage difference from split sensitivity without a top-k
cutoff. These ranges are descriptive, **not confidence intervals**, significance
tests, or formal noise ceilings. Overlapping random partitions are not 200
independent experiments. These numbers do not validate the proposed magnitude
metric or establish circuit acquisition/reuse.

Report paired source-group bootstrap 95% intervals for full-data similarities.
All examples, renderings, and template variants from a source group must travel
together. Stage IDs, group IDs, node order, protocol and tokenizer hashes were
checked before the audit; no source was shared between its resampling groups.
GSM has only 38 independent pair groups despite 512 attribution rows; more
bootstrap repetitions do not create more independent data. Audit intervals are
exploratory percentile intervals, not calibrated coverage guarantees. Near-zero
net profiles and nonlinear normalization can yield substantial bootstrap bias.

An advanced companion is a cross-validated squared distance on **unnormalized
signed group-mean profiles**:

`D_cv = (mean(A_L)-mean(B_L)) dot (mean(A_R)-mean(B_R))`.

With independent, identically sampled source groups, it estimates the squared
distance between population means without the usual additive noise bias.
Estimates can be negative and should not be clipped. This is an adaptation of
cross-validated dissimilarity methods, not an estimator validated specifically
for this experiment. It measures absolute effect changes, including scale;
normalizing separately inside each split changes the estimand and loses the
simple unbiasedness argument. It has not been computed in this audit.

## Additional safeguards and alternatives

1. Show Jaccard for k = 4, 8, 16, 32, 64, 128 where k is smaller than the node
   universe, with its own within-stage and random-reference curves at every k.
   Increasing k also raises chance overlap and eventually forces overlap to 1;
   observed overlap need not increase monotonically. Do not select k to maximize
   stability or a preferred stage story. An area under this curve still depends
   on the selected range and weighting.
2. Compare continuous overlaps to node-type-preserving identity permutations.
   Treat them as structural reference distributions, not automatic p-values:
   nodes at different layers are not necessarily exchangeable. Test attention
   identity within layers when the question concerns head identity beyond layer
   position, and compare with other-task profiles to detect generic importance.
   There is only one MLP block per layer, so a within-layer MLP shuffle is
   degenerate. These references were not computed in this audit.
3. Plot stability versus number of independent source groups. If stability
   remains poor, collect more independent problems/templates or stratify by
   meaningful task categories. Additional renderings of the same source do not
   substitute for independent sources. Bootstrap node-selection frequencies
   can show uncertain membership, but still depend on k and are not posterior
   probabilities that nodes belong to a true circuit.
4. Rank-biased overlap averages prefix agreement with smooth depth weights.
   It avoids one abrupt cutoff, but its persistence parameter sets the emphasis
   on rank depth and it discards attribution magnitudes. Use it as a ranking
   sensitivity check, not a parameter-free solution. Spearman correlation is
   another cutoff-free rank comparison, but can be dominated by the numerous
   weak nodes. Neither resolves signed cancellation.
5. Ultimately validate rankings with held-out interventions across several
   circuit sizes and type-matched controls. Ideally check whether a node set
   selected at A is effective when intervened on within B, and compare with
   B-selected sets under the same budget. A stable attribution map alone cannot
   establish that SFT teaches and DPO elicits, or that a new circuit was built.

## Sources and reproducibility

- [Webber, Moffat and Zobel (2010), A Similarity Measure for Indefinite Rankings](https://www.codalism.com/research/papers/wmz10_tois.pdf): rank-biased overlap and its depth parameter.
- [Schütt et al. (2023), Statistical inference on representational geometries](https://elifesciences.org/articles/82566): cross-validated distances and uncertainty methodology; applying the idea to attribution maps above is our proposed adaptation.
- [Hanna, Pezzelle and Belinkov (2024), Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806): high circuit overlap can coexist with poor causal faithfulness.

Local audit script and complete numerical results:
`geode-store/olmo2-authorized-20260911/robustness-review-20260921/audit.py`
and `audit.json`. Run the script from the repository root with `.venv/bin/python`.
It processes one task at a time, uses one numerical thread, and prints one line
per task. The result records NumPy version, seed, resampling counts, and SHA-256
hashes of score files, metadata, and analysis source. It uses the pre-existing
untracked `geode/circuits/robustness.py`; simple identity, disjoint-support, scale,
and sign-reversal sanity checks passed. Existing source edits were preserved.
