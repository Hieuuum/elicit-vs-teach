# Finding and validating task circuits in OLMo 2

**Objective:** identify pathways that causally support behavior on a dataset, validate them on unseen problems, then compare their function across training stages. Start with ETHICS; extend to coding and math after the method passes validation. This is a proposed protocol, not completed experimentation.

## 1. Why the current circuit-overlap evidence is not robust enough

Our analysis ranks **estimated answer-score contributions**, not activation strength. It selects the top 16 of 272 components: 256 attention heads and 16 whole MLP blocks.

- **The cutoff changes membership.** Nearly tied nodes can swap across rank 16 with little change in their contributions. Larger sets also have higher chance overlap; choosing a cutoff that improves reliability can be misleading.
- **Signed averaging can hide involvement.** We rank `abs(mean(signed attribution))`. Opposing effects cancel across questions. In the saved SFT ETHICS-commonsense audit, split-half Jaccard@16 was 0.274; ranking mean absolute effects raised it to 1.000—but selected all 16 MLP blocks. Neither result alone establishes a task-specific circuit.
- **Components have unequal granularity.** A whole MLP block and one attention head count equally toward membership. Generic MLP importance can dominate overlap. Uniform random-node sets do not control for this.
- **Node membership does not specify a mechanism.** The same nodes can exchange different information or compute different functions after training. Node-set interventions can establish relevance without validating a complete circuit or its edges.
- **Attribution is an approximation.** A gradient estimate can disagree with actual intervention effects, and individually important components need not form a sufficient subgraph.
- **Repeatability is not identity.** Split-half overlap is a sampling-stability reference, not a strict ceiling. Dividing cross-stage overlap by it does not measure the fraction of a circuit retained.
- **An unvalidated parent map does not establish acquisition.** Low accuracy or a negative answer margin does not make a map random. Failure to identify a parent circuit cannot establish that the child built a new one.

The defensible current claim is **similarity or change in attribution patterns**, not proof of elicitation or teaching. Numerical audit details: [ROBUSTNESS.md](ROBUSTNESS.md).

## 2. Proposed high-level workflow

1. **Define the behavior:** specify the task, answer score, prompts, and contrasting inputs.
2. **Separate data:** use source-disjoint discovery, validation, and final-test sets.
3. **Discover candidate graphs:** score edges with EAP-IG and construct circuits at several sizes; verify attribution estimates with actual patches.
4. **Validate causally:** preserve or disrupt candidate pathways on held-out questions, comparing with matched controls.
5. **Check generalization and robustness:** vary question subsets, wording, label mappings, circuit sizes, and intervention choices.
6. **Compare checkpoints:** test earlier and later circuit selections within both models; distinguish structural overlap from functional reuse.

**Deliverable:** a validated circuit-performance curve and uncertainty estimates for each task/checkpoint, followed by evidence of retained or changed function.

## 3. Papers supporting each step

| Workflow step | Paper | What to use |
|---|---|---|
| 1–3: score contributions and verify them | [Kramár et al., 2024 — AtP*](https://arxiv.org/abs/2403.00745) | Compare attribution estimates with actual activation patches; diagnose missed components. Equation 5 motivates averaging absolute attribution to reduce cross-example cancellation. |
| 3: discover edge circuits | [Hanna et al., 2024 — Have Faith in Faithfulness](https://arxiv.org/abs/2403.17806) | EAP-IG and evidence that high membership overlap can coexist with poor causal faithfulness. |
| 3: alternative discovery method | [Bhaskar et al., 2024 — Finding Transformer Circuits with Edge Pruning](https://arxiv.org/abs/2406.16778) | Optimize a sparse edge mask over a dataset while preserving model behavior. Use as a comparison method if EAP-IG candidates validate poorly. |
| 4–5: evaluate across circuit sizes | [Mueller et al., 2025 — MIB](https://arxiv.org/abs/2504.13151) | Faithfulness curves and integrated CPR/CMD metrics, reducing dependence on one cutoff. |
| 4–5: statistical validation | [Shi et al., 2024 — Hypothesis Testing the Circuit Hypothesis](https://arxiv.org/abs/2410.13032) | Explicit tests of behavior preservation, localization, and minimality. |
| 5: intervention sensitivity | [Miller et al., 2024 — Transformer Circuit Faithfulness Metrics Are Not Robust](https://arxiv.org/abs/2407.08734) | Check whether conclusions depend on the ablation or replacement procedure. |
| 6: track structural changes | [Wang et al., 2025 — Towards Understanding Fine-Tuning Mechanisms via Circuit Analysis](https://arxiv.org/abs/2502.11812) | Compare edges across checkpoints and test discovery stability under dataset perturbations. |
| 6: investigate functional reuse | [Prakash et al., 2024 — Fine-Tuning Enhances Existing Mechanisms](https://arxiv.org/abs/2402.14811) | Test circuit function before and after fine-tuning, including cross-model interventions. |

These papers motivate the protocol; the ETHICS-specific design below is our proposed adaptation.

## 4. Specific experimental protocol

### A. Pilot scope and data

- **First task:** ETHICS commonsense. **First checkpoints:** Stage 2, SFT, DPO. Then expand to all ETHICS categories and all seven training stages: initialization, Stage 1, Stage 2, SFT, DPO, RLVR-1, RLVR-2.
- **Claim being tested:** pathways supporting ethical judgments on the evaluated distribution—not a universal “alignment circuit.” Analyze categories separately before testing a shared circuit across categories.
- **Primary score:** correct-label logit minus incorrect-label logit. Also report benchmark-native accuracy under the same prompt format.
- **Contrasts:** matched clean/corrupted scenarios that change the relevant judgment while minimizing unrelated wording and formatting differences. Audit pair validity before discovery. Balance the existing eight label/prompt controls.
- **Split:** 60% discovery, 20% validation, 20% final test, seed 0. Keep each original scenario, clean/corrupt pair, and all its variants together. Use identical splits across checkpoints. Previously inspected examples remain exploratory; reserve fresh source groups for the final confirmation test. Determine the required total sample size from pilot variance and a prespecified precision target.

### B. Discovery and attribution verification

1. Implement EAP-IG for OLMo 2's actual computation graph, respecting its normalization and residual pathways. Define edge endpoints and token-position aggregation explicitly; verify the implementation on a small tractable case before scaling.
2. Rank edges by mean absolute attribution, averaging variants within source groups first. Preserve signed scores and repeat with absolute mean signed attribution as a sensitivity analysis.
3. Build candidate graphs at fixed edge fractions: **0.1%, 0.2%, 0.5%, 1%, 2%, 5%, 10%, 20%, 50%, 100%**. Inspect attention and MLP contributions separately.
4. Verify a preregistered sample of **30 edges per task/checkpoint** with exact patches: 10 highly ranked, 10 middle-ranked, and 10 low-ranked. Report effect-size agreement and important misses, not just rank correlation.
5. Use validation data to choose numerical settings and representative circuit sizes. Freeze choices before final testing. If attribution errors remain large, improve the estimator or compare with Edge Pruning before interpreting the graph.

### C. Held-out causal tests

| Test | Procedure and control |
|---|---|
| **Preservation** | Keep candidate pathways and replace excluded contributions using matched corrupted-input activations. Compare raw answer margins and accuracy with the full model and fully corrupted baseline. |
| **Disruption** | Patch candidate pathways while preserving the remaining computation. Compare target-score loss with random graphs matched for edge count, endpoint types, and layer distribution. Use 20 control draws per candidate size. |
| **Specificity** | Repeat disruption on matched nonethical classification and label-format controls. A broad language or output-format failure is not evidence of an ethical-judgment mechanism. |
| **Minimality** | Remove candidate edges or coherent edge groups conditionally on the remaining circuit. Report removals that preserve behavior; do not claim the graph is uniquely minimal. |

Plot preservation and disruption against circuit size. Add MIB-style summaries where normalization is well defined. If the full-model and corrupted-baseline scores are insufficiently separated, report raw differences instead of unstable normalized ratios.

### D. Robustness and decision rules

- Rediscover circuits on **five 80% source-group subsamples** of the discovery set. Report selection stability and held-out causal performance; these overlapping subsamples are not independent experiments.
- Test unseen paraphrases and label mappings, then transfer across ETHICS categories. Check a second defensible corruption/replacement procedure.
- Report **95% paired source-group bootstrap intervals with 2,000 draws** for test-set score differences. These quantify example-sampling uncertainty, not variation across training seeds. Treat multiple task/stage comparisons as exploratory unless a primary contrast or multiplicity procedure was specified in advance.
- Before final testing, specify a practically acceptable preservation loss and a meaningful disruption effect. Claim support when preservation meets that tolerance and interventions show target-relevant causal effects beyond matched controls, with conclusions stable across the planned checks.
- Report weak or contradictory results as **inconclusive**. Weak disruption can reflect backup pathways; failure to discover a circuit is not proof that none exists.

### E. Training-stage comparison and interpretation

Evaluate four combinations at matched circuit budgets: parent-selected pathways in the parent and child, and child-selected pathways in the parent and child. Keep each model's own weights. This measures **transfer of circuit structure**, not identity of the underlying computation.

For pathways implicated in the improvement, follow up with targeted functional and, where valid, cross-model interventions. Reuse requires evidence that an earlier computation remains causally responsible. Acquisition requires evidence of newly learned task-relevant computation; low overlap alone is insufficient, and failed cross-model patches can reflect representation incompatibility.

**What can happen now:** cutoff sweeps, continuous node comparisons, and sampling diagnostics from saved scores. **What requires new model runs:** edge discovery, exact edge patches, graph-level causal tests, and fresh held-out evaluation. This document does not launch those runs.
