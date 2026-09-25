# Unlearning: is the knowledge gone or latent? (elicit vs teach on unlearned models)

Status 2026-09-25: code, launcher and CPU smoke complete on branch
`unlearn-elicit-teach`; nothing has run on a GPU yet. Owner runs
`launch_unlearn.sh` (§10) and pastes `out/unlearn.log`.

## 1. Question

An "unlearned" model no longer says what it was trained to forget. Is it a
**pre-teach** parent (the knowledge is absent: relearning it costs what teaching
costs and builds a new circuit) or a **pre-elicit** parent (the knowledge is
latent: a small fine-tune, a patch, or a hidden-preference read-out brings it
back through the original circuit)? We answer with the paper's sixteen
instruments (results_ts.tex M1–M16) plus one behavioural check (M17), each
read against two anchors: the model that knows (elicit-like) and a model that
never learned the material (teach-like).

## 2. Choice of models and data (checked on the Hub 2026-09-25)

**TOFU on Llama-3.2-1B-Instruct, forget10**, all checkpoints from
open-unlearning (HF org `open-unlearning`, ungated, bf16 safetensors,
2.47 GB each). `models.py` pins these revisions and `launch_unlearn.sh` stage 0
downloads them on the cluster.

| tag | role | repo (`open-unlearning/…`) | revision |
|---|---|---|---|
| `orig` | elicit anchor: trained on all 200 authors | `tofu_Llama-3.2-1B-Instruct_full` | `88e31200b97e` |
| `retain` | teach anchor: trained on the 180 retain authors, never saw forget10 | `tofu_Llama-3.2-1B-Instruct_retain90` | `7114300c0049` |
| `npo` | unlearned (NPO) | `unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.05_alpha1_epoch10` | `faeb45f30e0a` |
| `graddiff` | unlearned (GradDiff) | `…_forget10_GradDiff_lr1e-05_alpha5_epoch10` | `8b2217e8daec` |
| `rmu` | unlearned (RMU, layer 10) | `…_forget10_RMU_lr1e-05_layer10_scoeff100_epoch10` | `6fbd37715710` |
| `simnpo` | unlearned (SimNPO) | `…_forget10_SimNPO_lr1e-05_b3.5_a1_d0_g0.125_ep10` | `77224bb05582` |
| `idkdpo`, `undial` | optional extra unlearned models (`--tags`) | `…_IdkDPO_lr1e-05_beta0.05_alpha1_epoch10`, `…_UNDIAL_lr0.0001_beta10_alpha1_epoch10` | `47d05b6f396a`, `a73180cb43d5` |
| `base` | optional, no TOFU at all (gated meta-llama repo) | `meta-llama/Llama-3.2-1B-Instruct` | main |

Full shas are in `models.py`.

- **Dataset:** `locuslab/TOFU` at revision `324592d84ae4f482ac7249b9285c2ecdb53e3a68` (MIT, ungated).
- **Files used:** `full`, `forget10`, `forget10_perturbed`, `retain_perturbed`, `holdout10`. `data/prepare.py` pins each file's sha256.
- **Split identity:** `forget10 == full[3600:4000]`, the last 20 authors (checked by `prepare.py`).

Why this choice:

- **It is the only candidate with a true never-learned control.** `retain90` shares the recipe and the format with `full`; it simply never saw the forget authors. That is exactly the paper's "format-installed teach parent".
- **It matches our architecture.** Llama-3.2-1B-Instruct has 16 layers, 32 heads (8 KV) and width 2048, the same as the paper's second elicit parent. Every tool runs unmodified at the size it was built for, and a full battery is about 10 GPU-h.
- **Unlearned checkpoints exist only for this base and forget10.** 398 repos across 8 methods (GradDiff, IdkNLL, NPO, IdkDPO, AltPO, UNDIAL, RMU, SimNPO).
- **Alternatives rejected:**
  - The locuslab unlearned repos (`locuslab/phi_*`, `llama2-7b_*`) contain no weights.
  - WMDP/RMU (`cais/Zephyr_RMU` vs `HuggingFaceH4/zephyr-7b-beta`, MIT, 7B Mistral) and Who-is-Harry-Potter (`microsoft/Llama2-7b-WhoIsHarryPotter`, research licence, 27 GB fp16 .bin) have no never-learned model, and WMDP answers are MCQ letters. They are follow-ups (§12).
- **Unlearning hyperparameters.** One mid-grid checkpoint per method (lr 1e-5, 10 epochs, a central beta/alpha). The per-method best settings of the OpenUnlearning paper (arXiv 2506.12618) may differ: open question 1.

## 3. Data, prompt format, answer-token protocol

**Prompt.** The Llama-3.2-Instruct chat prompt, exactly as open-unlearning renders it (`configs/model/Llama-3.2-1B-Instruct.yaml`: tokenizer chat template, `add_generation_prompt`, `date_string: 10 Apr 2025`):

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 10 Apr 2025\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
```

The template is rendered as a string and tokenized with `add_special_tokens=False`; the special-token strings map to their ids. Caveat: the `full`/`retain90` models were trained before the date was pinned, so they saw the wall-clock date of their run (unknown). Open question 2.

**Probe item: the fact slot.** TOFU answers are sentences, and its perturbed answers are perturbations of the *paraphrased* answer (swapped facts, same sentence). `prepare.py` word-aligns the paraphrase with each perturbed answer (difflib) and takes the first replaced word that meets all of these conditions:

- it is a content word (not a stopword);
- it is absent from the question (an echo would be copyable);
- it occurs in the ORIGINAL answer, which is the text the models were trained on.

Given that word:

- **Probe prompt** = chat(question) + the original answer up to the fact word.
  Example: `…The author's full name is` → ` Hsiao` (distractors ` Chen`, ` Lin`, …).
- **Scored token** = the first token of the continuation, tokenized **in context**. `geode.adapt.first_answer_token` requires the prompt's ids to be an exact prefix of the joint ids.
- **Distractors** = the perturbed words at the same slot, aligned the same way. Distractors whose first token equals the target are dropped, and so are items left with no distractor.

This replaces the arithmetic "first digit chunk + sign trick".

On the real TOFU files (run locally on the 3 MB of JSON; the cluster rebuilds the same output):

| split | items | source |
|---|---|---|
| `forget_A` | 144 | 10 forget authors, relearned in stage 2 |
| `forget_B` | 148 | the other 10 forget authors, held out from relearning |
| `retain` | 296 | first 20 retain authors, known to all models (positive control) |
| `null` | = forget | author name swapped for an invented one: nobody knows the answer (noise floor) |

- **Dropped items:** 111 of 400 forget and 125 of 400 retain rows, the open-ended "how has X influenced…" questions that have no single fact slot.
- **A/B assignment:** the author split uses a seeded permutation (seed 316). The author names come from the most frequent capitalised n-gram in each 20-question block, and all 20 were extracted correctly (listed in `prepare_report.json`).

**Relearning sets** (SFT schema: `full_text`, answer char span, EOS appended by `tokenize_with_spans`):

- `relearn_forgetA`: 200 QA about the forget_A authors.
- `relearn_forgetA_val`: the same facts asked with TOFU's paraphrased question. It is the val loss for the eps/k stopping rule, and it measures learning the fact rather than the string.
- `relearn_holdoutA` / `_val`: 10 of the 20 holdout10 authors, never seen by any model (the teach-in-every-model control).

## 4. Counterfactual pairs and guards

- **Name-swap pairs** (default for maps, faithfulness, edges and DCM).
  - Clean = the probe prompt. Corrupt = the same prompt with every mention of the author replaced by an invented name, chosen from a pool of about 1,600 made-up names that share no word with TOFU.
  - The swap must keep the token length and change token ids only inside the name spans (V5.77). Invented names are used instead of another forget author, so the corrupt run has no knowledge to express.
  - Metric M = logit(fact token) − logit(same-slot perturbed token). This is the analogue of "same format, different operands": only the identity that the recall keys on changes.
- **Item pairs** (DAS): two different probes of equal token length with different target tokens, the contrast being the other item's answer. This is the arithmetic "different problem" design, rotated through length buckets to get enough pairs from a few hundred items.
- **Unlabelled DCM counterfactuals.** The name-swapped author's answer is unknown, so the DCM target is the model's OWN next-token distribution on the counterfactual (`dcm_roles.learn_role(cf_target="cf_dist")`). A role set counts only on pairs whose preference can move (clean − cf ≥ 1 nat), with the copy-every-head ceiling reported.
- **Performing guard** (unchanged):
  - node maps need a mean logit-diff > 1 nat, edge maps > 2 nats;
  - `verdict.py` marks any metric resting on a non-performing map NOISE, except M11, where "the map is noise" is itself the teach reading.

## 5. Three-way design and children

**Parents:** orig (knows), retain (never learned), and each unlearned model.

**Children.** Every parent gets the same relearning fine-tune on the forget_A authors (`relearn.py`, `configs/relearn_forgetA.yaml`):

- **Recipe:** full FT, AdamW, lr 1e-5 (open-unlearning's TOFU fine-tune lr), wd 0.01, batch 8, constant lr, bf16 autocast over fp32 master weights. Stops by the repo's eps/k rule on the paraphrase val loss (eps 0.002, k 5, after 50 steps); ceiling 500 steps (20 passes).
- **The three children:**
  - `orig-rl` relearns facts it already knows: the elicit anchor child.
  - `retain-rl` is taught them: the teach anchor child.
  - `<unlearned>-rl` is the object under test.
- **Logged per step:** loss before the update (prequential code), pre-clip gradient norm, and exact fp32 weight travel ‖θt−θ0‖ and speed.
- **Logged every 10 steps:** val loss, plus hidden preference and top-1 on forget_A and **forget_B**. forget_B is the held-out-author recovery curve.
- **Saved:** 10 log-spaced bf16 snapshots (formation curve) and the final model in fp32 (exact ΔW).
- **Optional (`--holdout`):** the same fine-tune on holdout authors, which calibrates each parent's own teach cost.

## 6. The seventeen checks

Position score s = (value(unlearned) − value(teach anchor)) / (value(elicit anchor) − value(teach anchor)).

- **Pre-registered cut:** s ≥ 0.5 → PRE-ELICIT, s < 0.5 → PRE-TEACH.
- **UNDETERMINED:** the two anchors differ by less than the metric's minimum separation (listed in `verdict.py`), so the instrument does not transfer to this task.
- **Anchors:** parent-only (★) metrics are read on the parents themselves. Child metrics are read on the children, with `orig-rl` as the elicit anchor and `retain-rl` as the teach anchor.
- **Split:** "forget" = forget_A ∪ forget_B; circuit and child metrics use forget_A (both children perform there after relearning).

| # | metric | applies? how (tool, split) | expected PRE-ELICIT reading | expected PRE-TEACH reading |
|---|---|---|---|---|
| M1 | Circuit overlap | yes: `circuit_nodes` maps of the child vs orig on forget_A. Heads-only J@32 over the split-half ceilings; functional check: orig's top-32 heads corrupted in the child vs 100 type-matched random head sets (`circuit_faithfulness --heads-only --nodes-from`) | child's heads ≈ orig's (near ceiling); orig's heads destroy most of the child's behaviour | child builds its own heads; low overlap; orig's heads carry little |
| M2 | Wiring | yes: `circuit_edges` J@256 child vs orig over the edge ceilings (15–25-token chat prompts; the 0-shot rule holds) | ≈ ceiling | lower |
| M3 | Head roles | yes, adapted: DCM **subject-reading** heads (name-swap cf, own-output target), child vs orig | same role set | different or larger set |
| M4 | Formation time | yes: node map at each snapshot vs the final child map | present from the first snapshot | at chance early, built mid-run |
| M5 | Gradient pressure | yes: pre-clip norm, first 1 % vs last 10 % of steps (`grad_strength.py` on train_log) | fades (ratio > 1) | grows or stays |
| M6 | Weight write | yes: ‖ΔW‖/‖W‖ (exact travel; `weight_shift.py` per module) | small | larger |
| M7 | State change | yes: `resid_shift` parent→child on forget_A, shared-direction score (PC1) at the last layer, also with answer directions removed | per-item changes (low PC1) | one shared push (high PC1) |
| M8★ | Latent reach | adapted: bridging index between the **question and TOFU's paraphrased question** (`cross_format_probe`), plus the parent's own state PC1 | item-specific states, reached by paraphrase (orig-like) | no item-specific state, or one state for all items (a refusal mode would also show here) |
| M9★ | Answer depth | yes: `lens_depth` logit + J lens at the answer position, max logit-diff over layers (fact vs perturbed token); after training, the child's settled depth | the fact is present at intermediate layers even when the output suppresses it | ≈ 0 at every layer |
| M10 | Switch-on by patching | yes, **heads-only** as the headline (MLP outputs at the answer position write answers directly: the 2026-09-24 lesson): the child's per-prompt head states at its top-32 heads written into the frozen parent, fact-token top-1. **M10★ variant (no training):** orig's top-32 heads written into each parent | the parent answers: its MLPs still store the facts, only the routing changed | the patch cannot help: no stored facts to route |
| M11★ | Repeatable task circuit | yes: parent map on forget, performing guard + split-half J@32 | defined, repeatable map | map is noise |
| M12★ | Hidden preference | yes: logit(fact) − logit(perturbed) at the fact slot (also on retain = positive control, null = noise floor) | clearly > 0 while top-1 is low | ≈ null floor |
| M13★ | Accelerating descent | yes: Hessian of the SFT loss on the forget answers, negative-curvature share (the answer is well formed in every model, so the arithmetic caveat about format-dominated loss is absent) | high share | ≈ 0 |
| M14★ | Answer carried by the state | yes: DAS / full-state swap between two items at the answer position, flip gain over no swap at the best layer | positive gain | ≈ 0 |
| M15★ | Number-reading → **subject-reading heads** | adapted (M3's DCM on the frozen parent) | a compact set moves the preference | none (no performing pairs) |
| M16★ | Operands present → **fact present at the subject** | adapted: logit lens of the fact vs distractor at the last token of the author's name (Geva et al. 2023 attribute extraction), best layer; the J-lens at the subject position is in `lens_<p>.json` | enriched subject: fact readable | nothing readable |
| **M17** | **Relearning: held-out recovery (new)** | fine-tune on forget_A authors, read forget_B (authors never trained in the child) at the end; plus the prequential first-pass loss on forget_A | forget_B comes back although never trained (Deeb & Roger 2024) | forget_B stays at the null level; first pass costs like retain |

**Why a 17th.** The sixteen are mechanistic, and the paper checks them against its behavioural ground truth: the EDL learning curve (elicitation is cheap and monotone; teaching costs a hump). M17 is that ground truth in the form that applies to facts. Relearning some authors cannot *teach* other, independent fictitious authors, so any recovery on the held-out authors is elicitation by definition. The prequential first-pass loss is the online code length of relearning, the paper's own learning cost; it is compared with retain's.

Not applicable, and refused by the tools with a message:

- the arithmetic probe v3 (answer chunk helix) and the operand helix probe (replaced by M16 as above);
- the attention-mass interface;
- cliff depth, and the LLC (invalid anyway).

## 7. Pre-registered predictions (to be checked against the paste)

1. **Anchors.**
   - orig: every map performing, hidden preference several nats, top-1 high.
   - retain on forget: the guard fails (noise), hidden preference ≈ null floor, no subject-reading heads. On retain, retain performs like orig (positive control).
   - Anchors not separating on M1–M3 (both use a generic recall circuit) would mean that set-overlap metrics do not discriminate for facts. It would be recorded, not read as a verdict.
2. **NPO, GradDiff, SimNPO: majority PRE-ELICIT**, carried by M9, M12, M14, M16 and M17. These methods suppress the output while subject enrichment and mid-layer states survive, and relearning other authors brings the held-out ones back.
   - M11 (circuit repeatable) may be NOISE or PRE-TEACH if the output-level preference falls below 1 nat. That is the instrument reading the suppression, and it is the most likely split.
3. **RMU: split by depth.**
   - M16 at layers ≤ 10 is PRE-ELICIT: RMU perturbs activations from layer 10 on, so the early enrichment survives.
   - M9 and M12 are PRE-TEACH: the lens is scrambled after layer 10, and the output preference is gone.
   - M10★ is PRE-TEACH: orig's heads cannot route through the scrambled residual.
   - M17 is PRE-ELICIT: fine-tuning removes the injected noise direction.
4. **IdkDPO (if run):** M8 state PC1 near 1 (one refusal state), while the M12 hidden preference remains positive.
5. **If an unlearned model is PRE-TEACH on M17** (no held-out recovery) **but PRE-ELICIT on the ★ metrics**, the knowledge is readable but not cheaply re-installable. Record it as a disagreement between mechanism and behaviour; do not force a verdict.

## 8. Cost per stage (one 80 GB GPU; ~$2/h assumed, `USD_PER_H`)

| stage | what | per unit | default six parents |
|---|---|---|---|
| 0 | data (3 MB) + model snapshots (6 × 2.47 GB) | — | CPU, network only |
| 1 ★ | parent-only battery (prefit ×8 blocks, lens with J-lens, xfmt, 6 node maps, 3 edge maps, DCM, M10★) | ~40 min / parent | ~4 GPU-h |
| 2 | relearning child (≤ 500 steps, 10 snapshots × 2.5 GB) | ~10 min / child | ~1 GPU-h (+~0.8 h with `--holdout`) |
| 3 | child battery (3 maps, 3 edge maps, DCM, 3 × heads-only necessity with 100 random sets, ~10 snapshot maps, resid, lens, 2 patchings, ΔW, gradients, recovery pref) | ~50 min / child | ~5 GPU-h |
| 4 | verdict | seconds | CPU |
| | **total** | | **~10 GPU-h ≈ $20** |

**Disk:** models 15 GB; snapshots 25 GB per child (150 GB for six; delete after stage 3's formation maps if needed).

## 9. Lessons from decisions.md, and how they are honoured

- **Performing-regime guard.** A map is noise below 1 nat, an edge map below 2; `verdict.py` marks dependent metrics NOISE.
- **Circuit comparisons** are read against split-half ceilings and type-matched chance, heads-only first (MLP-heavy top-k).
- **Restore-sufficiency is not evidence of circuit specificity.** Only heads-only necessity against 100 type-matched random head sets is used as the functional test, and all-position "maintain" is not used.
- **Aggregation:** signed-sum only (`--agg sum`); per-pair-absolute makes noise maps repeatable.
- **Edges** are mapped at 0-shot only (few-shot kills the parent's signal).
- **No sign-token artefact:** the scored token is a fact word; nothing is appended to the prompt.
- **Failed pre-fine-tuning predictors** (cliff depth, gradient coherence, attention mass, LLC) are not used as metrics. The curvature *share* is kept.

## 10. What the owner runs (cluster, conda env `geode`)

```bash
cd experiments/unlearning
export GEODE_STORE=/path/to/store          # models + runs land under $GEODE_STORE
bash launch_unlearn.sh --smoke             # optional: ~8 min CPU code-path check (tiny random models)
bash launch_unlearn.sh --stage 0           # CPU: TOFU + pinned model snapshots
bash launch_unlearn.sh --confirm-cost --gpu --stage 1    # parent-only ★ battery (~4 h)
bash launch_unlearn.sh --stage 4           # verdict on the ★ metrics alone (child rows MISSING)
bash launch_unlearn.sh --confirm-cost --gpu --stage 2    # relearning children (~1 h)
bash launch_unlearn.sh --confirm-cost --gpu --stage 3    # child battery (~5 h)
bash launch_unlearn.sh --stage 4           # full 17-check verdict
# paste experiments/unlearning/out/unlearn.log
```

- **Environment:** `TS_VALID=<TinyStoriesV2-GPT4-valid.txt>` avoids the hub fetch of the generic text used by the J-lens and the residual shift.
- **Parents:** `--tags "orig retain npo"` restricts them; `orig` always runs first because its circuit is the reference.

## 11. Differences from the arithmetic setting

1. **The capability is stored facts, not an algorithm.** The recall machinery (reading the subject, answering in the QA format) exists in all three models, since retain knows 180 other authors.
   - Set-overlap metrics (M1–M3, M11) may therefore show overlap for both arms.
   - The discriminating weight moves to fact-specific instruments (M9, M12, M14, M16, M17) and to the heads-only functional tests.
   - The retain split is the positive control that the recall circuit is defined in every model.
2. **No generalisation across items.** A child taught forget_A cannot answer forget_B. Child circuits are therefore mapped on forget_A (both children perform there), and forget_B carries the elicitation test proper (M17).
3. **The elicit anchor expresses the knowledge; it is not latent.** A latent model should fall between the anchors, hence the position score with a 0.5 cut rather than "matches the elicit parent".
4. **Answer:** the first token of a fact word after a teacher-forced prefix of the trained answer (top-1 = the exact-match analogue). **Contrast:** TOFU's perturbed word in the same slot, not a mismatched problem's answer. **Corruption:** an author-name swap, not different operands.
5. **Surfaces:** question vs paraphrased question replaces words vs symbols (M8).
6. **M15/M16 are re-targeted:** subject-reading heads, and the fact at the subject position. The operand helix probe has no analogue.
7. **Relearning is short** (tens to hundreds of steps, not 10⁴). Formation curves have ~10 snapshots, and gradient pressure uses 1 % / 10 % of a few hundred steps: the first-1 % window is one or two steps, so read it with the curve.
8. **The chat template carries a date line** (open question 2), and the prompts are 60–110 tokens instead of 15. Edge maps get longer contexts than the TinyStories word task.

## 12. Open questions for the owner

1. **Unlearned checkpoints.** One mid-grid checkpoint per method is pinned. Should the per-method best of the OpenUnlearning leaderboard be used instead? A cheap screen is possible: stage-1 `pref` only over a grid, about 1 min per checkpoint plus a 2.5 GB download each.
2. **Date line.** "10 Apr 2025" is used for every model. The full and retain models saw an unknown training date. A robustness check would rebuild with `prepare.py --date "<other>" --out-dir …` and rerun `pref`.
3. **Verdict rule.** Linear anchor interpolation with a 0.5 cut, and min-separation tolerances set by hand. An alternative is bootstrap intervals over items, so a verdict can be "inside the teach anchor's interval".
4. **Relearning recipe.** Constant lr 1e-5, batch 8, full FT, ≤ 20 passes. open-unlearning warms up for one epoch. A LoRA variant (the paper's robustness row) is not implemented.
5. **Base model.** Should `base` (Llama-3.2-1B-Instruct, no TOFU, gated) be added as a fourth parent? It anchors "format absent" as well as "facts absent".
6. **Snapshots and budget.** Keep or delete snapshots after stage 3 (150 GB)? Is ~10 GPU-h in budget?
7. **Follow-ups without a never-learned control.**
   - WMDP/RMU (`cais/Zephyr_RMU` @`70c55b3b` vs `HuggingFaceH4/zephyr-7b-beta` @`892b3d7a`; `cais/wmdp` @`7125571f`; MCQ letter answers → a letter-token task adapter; teach anchor = none, use held-out recovery only).
   - WHP (`microsoft/Llama2-7b-WhoIsHarryPotter` @`e4347ee6` vs Llama-2-7b-chat; research licence; completion probes).
   Both are 7B (~4× cost).
8. **The pytest suite in this environment.** Pre-existing failures that do not come from this branch: the frozen tokenizer needs transformers 5, and one fig2ts overlay-count test fails. Is the suite still maintained or run in CI?
