# Bits That Count — Experiment Reference

Source: Donoway, Joren, DeWeese, Perez, Schulman, Roger, Leike. "Bits That
Count: Quantifying and Predicting Capabilities of Language Models." ICML
2026 (PMLR 306). Companion papers: Donoway et al. NeurIPS 2025 ("Quantifying
elicitation of latent capabilities in language models"); Donoway et al.,
arXiv:2601.04728 ("Excess description length of learning generalizable
predictors"); Donoway, ISIT 2026.

This document was built from a raw-text extraction of the PDF (2172
content lines). Section/figure/table numbers cited below are the paper's
own numbers, not raw-dump line numbers. Where the extraction is internally
ambiguous or contradicts itself, that is flagged explicitly rather than
resolved by inference — see §7.

## 0. Purpose and one-paragraph overview

This document is a replication-grade reference for every experiment,
protocol detail, hyperparameter, and numeric result reported in "Bits That
Count," for researchers checking their own runs against the paper's exact
setup. The paper's central claim: standard evaluation metrics (zero-shot
accuracy, final loss) cannot distinguish a model *eliciting* a latent,
pre-existing capability from a model being *taught* a new one, because both
processes can start and end at the same performance. The paper introduces
**excess description length (EDL)** — the gap between the total bits a
model needs to encode training labels during learning (prequential MDL)
and the bits it needs after training (final test loss × n) — as an
operational, information-theoretic metric that separates these two
regimes. Empirically, EDL per token *decreases monotonically* with dataset
size under elicitation and *increases before decreasing* under teaching
(§4); causal pre-training interventions that install a capability shift a
task's signature from teaching-like to elicitation-like (§5); and EDL per
trainable parameter predicts a capacity threshold beyond which LoRA
underperforms full fine-tuning, with elicitation saturating around
0.01–0.1 bits/parameter versus ~1+ bit/parameter for teaching (§6). The
paper reports this pattern across Llama 3.2 1B/3B, Llama 3.1 8B,
TinyStories–1B, Qwen2.5 1.5B/14B/32B, and a randomly-initialized Llama 3.2
1B, across arithmetic, reasoning-distillation, language modeling,
instruction-following, reading comprehension, and science-QA tasks
(§7, Table 5).

## 1. Measurement protocol (applies to every experiment)

### 1.1 Setup and loss convention (§2.1, App. F)

Fine-tuning dataset D = {(xᵢ, yᵢ)}ⁿᵢ₌₁, xᵢ an input (question/prompt), yᵢ
the corresponding label (answer). θ₀ = pretrained parameters, A = the
training algorithm (optimizer, LR, hyperparameters). **Cross-entropy loss
is computed only on designated label tokens** — typically the answer
portion of the output, excluding the prompt and formatting tokens (§2.1).
This isolates capability-relevant learning from format acquisition, "which
can otherwise dominate early training dynamics" (§2.1).

App. F (Token Accounting and Output Scoring) makes this concrete per task
family:
- Arithmetic tasks: only the numerical answer tokens are scored. The
  prompt (e.g., "What is the sum of 23 and 45?") and any formatting tokens
  are excluded from both MDL computation and test-loss evaluation.
- Reasoning tasks (Qwen): the **full response is scored**, including
  chain-of-thought reasoning and the final answer — "this captures the
  information required to elicit the full reasoning behavior."
- All losses are computed in **nats**. EDL, MDL, and all other
  information measures are converted to **bits** by dividing by ln 2
  (bits = nats / ln 2 ≈ nats / 0.693) — conversion happens only at
  reporting time.

### 1.2 Prequential Minimum Description Length (§2.2 + footnote 1, App. A.2)

Prequential MDL accumulates the cross-entropy loss incurred on each
training example **before** the model updates on it:

```
MDL(D; θ0, A) = Σ_{i=1}^{n} ℓ(θ_{i-1}; x_i, y_i)
```

where ℓ(θ; x, y) = −log p_θ(y | x) is cross-entropy on label tokens only
(eq. 1, §2.2). In practice, training proceeds by batches: "For each batch,
we accumulate the total log-loss (summed over all labels in the batch)
before the gradient update. The sum across all labels in the dataset
(i.e., in the first epoch) gives the MDL" (§2.2).

**Footnote 1 (critical):** "Prequential MDL accumulates loss over the
first epoch only, as all labels have been encoded a single time at the
end of the first epoch. Accumulation over additional epochs would
correspond to encoding redundant information, which would make the
description length of the data no longer minimal." App. A.2's footnote 4
repeats this: counting losses from subsequent epochs "would correspond to
transmitting redundant data."

App. A.1 frames MDL via a communication analogy (Alice/Bob): MDL is the
minimum number of bits Alice needs to send Bob (who has the base model and
the inputs, but not the labels) so Bob can train an identical copy of
Alice's fine-tuned model — "the minimal information required to elicit
from or teach the base model a set of capabilities identical to those of
Alice's model" (App. A.1). Prequential coding is described as computing a
**true upper bound** on the (uncomputable) true MDL, using the model
itself as the compression algorithm (App. A.2).

**Fig. 6** illustrates this with single-epoch vs. multi-epoch training
diagrams: in panel (a) (single epoch), EDL (blue hatched region) reflects
structure learned from the data S so far, up through |S|; in panel (b)
(multi-epoch), additional training on the same S improves generalization
(final population loss L(θ_T) = ε* < ε, the one-epoch generalization
error), yielding a **larger** EDL, since "additional training extracts
more of the learnable structure in S without adding new information." The
total area under the first-epoch curve (up to |S|) is MDL (gray); EDL is
the portion of that area above the final population loss L(θ_T).

### 1.3 Excess Description Length (§2.3 eq. 3, App. A.3)

After training to some termination condition (convergence, a fixed number
of epochs, or a fixed compute budget), obtain final parameters θ*.
Evaluate test loss on held-out data from the same distribution:

```
L_test(θ*) = (1/n_test) Σ_{j=1}^{n_test} ℓ(θ*; x_j, y_j)     (eq. 2)
```

Excess description length:

```
EDL(D; θ0, A) = MDL(D; θ0, A) − n · L_test(θ*)     (eq. 3)
```

"EDL measures the gap between the bits spent encoding training labels
during learning and the bits expected when using the final model. This
gap represents predictive information compressed into the parameters —
structure extracted from the train set that improves generalization on
unseen data" (§2.3). By construction, EDL "(i) separates data (information
source) from computation (extraction mechanism) and (ii) distinguishes
generalization from memorization by excluding the generalization gap
(Figure 6). EDL remains negligible when no generalizable structure
exists, regardless of training compute and data memorization (Appendix
I.10)" (§2.3).

**Footnote 5** (App. E.1.1, referenced throughout for "convergence"):
"'Convergence' is determined by maximal performance on the chosen
validation metric (either maximization of the validation accuracy or
minimization of the validation loss); for a single example, this usually
occurs after a single training step."

App. A.3 restates EDL as "the predictive information the model
compresses from the data over the entire learning process — the
information learned from the data that generalizes beyond the train set
to the actual task," visualized as the area under the first-epoch
training curve above the test loss (Figs. 6 and 7, hatched regions). It
notes that even if a model overfits and perfectly predicts the train set,
"the test loss reveals the task-related information that the model was
unable to compress further" — i.e., EDL is bounded by what generalizes,
not by what is memorized.

**Fig. 7** is a conceptual (non-empirical) diagram with three columns
(eliciting a *demonstrated* capability, eliciting a *latent* capability,
*teaching* an absent capability) crossed with small-n and large-n rows,
plus an EDL-scaling row and a parameter-capacity row. Two details not
found elsewhere in the paper:
- The EDL-scaling row (third row) normalizes dataset size n "by the total
  number of constituent concepts K that must be known for task mastery" —
  this K-normalization is **never defined quantitatively or used in any
  reported figure/table**; it appears only in this conceptual caption.
- The capacity row (bottom row) gives approximate P* magnitudes per
  column: "P* ≲ 0.01 bits" (demonstrated elicitation), "P* ∼ 0.01 bits"
  (latent elicitation), "P* ≳ 1 bit" (teaching) — consistent with, but
  more granular than, the headline 0.01–0.1 vs ~1+ bits/parameter split
  used elsewhere.

**Fig. 1** (introductory, conceptual) gives the paper's first quoted
numbers: elicitation converges in "~100 examples," teaching in "~60K
examples" (panel a/b, illustrative training curves with L_test marked as
a horizontal dashed line); panel (c) labels parameter-capacity example
points "P*_elicit 0.01" and "P*_teach 1" (bits/parameter).

### 1.4 Conventions and normalizations (§2.4)

- **EDL per token (EDL/D)**, D = total label token count: "Used for
  scaling analysis. This measures the absorbed information per unit of
  supervision."
- **EDL per parameter (EDL/P)**, P = trainable parameter count: "Used for
  capacity analysis. This measures how densely information is packed into
  the adapter."
- **Capacity, or EDL ratio**: EDL(P, D) / EDL_ref(D). "Used for comparing
  LoRA to full fine-tuning. The reference EDL is obtained from full
  fine-tuning or high-rank LoRA on the same data."

**Footnote 2**: "Since LoRA can have different training dynamics than
full fine-tuning, we use rank 512 LoRA as a reference, EDL_ref =
EDL_{r=512}, which obtains the same performance as full fine-tuning in
all of our experiments."

**Performance Gap Recovery (PGR)** (§3.4, eq. 4):

```
PGR = (Perf_LoRA − Perf_base) / (Perf_FullFT − Perf_base)     (eq. 4)
```

"where performance (Perf) is measured by accuracy or loss improvement
relative to the model's zero-shot performance. PGR = 1 indicates LoRA
matches full fine-tuning; PGR = 0 indicates no improvement over the base
rate (zero-shot)."

### 1.5 Practical considerations (§2.5)

1. Token-masking procedures for loss computation "should be consistent
   between the train and test sets."
2. For multi-epoch training: MDL is computed **only on the first epoch**
   (first exposure to data); test loss is computed **using the final
   trained model** (arbitrary number of epochs). Rationale: "MDL describes
   the total information content of the source (data), whereas training
   (computation) serves as a mechanism to extract this information."
3. Cross-entropy is computed in nats; figures report EDL in bits (÷ ln 2).

### 1.6 Compression ratio / space saving (App. A.4)

```
Compression ratio = MDL / (D · L(θ_T))
Space saving       = EDL / MDL                                (eq. 5)
```

D = number of labels in the train set. "A high compression ratio
indicates that a model can compress most of the information it initially
required to encode the training data by learning the general, underlying
features of the task, rather than by memorizing specific examples. In
contrast, a low compression ratio implies that the model was unable to
compress most of the information in the training data, either because it
was unable to learn the generalizable features or because there are few
predictive patterns left."

### 1.7 Formal definitions (App. J)

**J.1 Learning processes.** Elicitation (process): "characterized by
decreasing learning efficiency with additional data. Elicitation surfaces
existing knowledge by accessing information that has already been learned
and stored in the model parameters." Formally, elicitation predominates
when

```
∂/∂n [ EDL(n) / n ] < 0
```

Teaching (process): "characterized by increasing learning efficiency with
additional data (up to a saturation point)... builds new capability by
direct instruction when the capability cannot be deduced from existing
knowledge." Formally predominates when

```
∂/∂n [ EDL(n) / n ] > 0
```

**J.2 Learning regimes.** Elicitation (regime): "limited information
required to achieve maximum capability. The parameter capacity threshold
is low (P* ≈ 0.01–0.1 bits/parameter)." Teaching (regime): "substantial
information required because minimal relevant capability exists
originally... P* ≈ 1 bit/parameter."

**J.3 Relationship between processes and regimes.** Process (what happens
at each n) and regime (overall information requirement) are related but
distinct:
- A model in the elicitation regime predominantly shows elicitation
  process signatures across most dataset sizes.
- A model in the teaching regime initially shows teaching process
  signatures (increasing EDL/token), then elicitation process signatures
  once capability is learned or parameter capacity saturates.
- The teaching→elicitation crossover marks the point where the model has
  acquired enough capability that additional data becomes primarily
  redundant.

**J.4 Capacity thresholds under finite compute** (the "finite-compute
paradox"): "Capacity thresholds P* mark where learning efficiency drops
(i.e., significantly slows or degrades relative to full fine-tuning) —
not where learning stops entirely. With infinite compute, less-capable
models eventually absorb more information (higher EDL) and learn more.
However, with finite compute budgets, this relationship can reverse. When
a model lacks prerequisite knowledge, learning may stall before
discovering efficient algorithms, causing it to memorize inefficiently and
hit capacity limits at low accuracy. Pre-teaching can enable the model to
learn compressive algorithms early, freeing parameter capacity for
additional learning. The pre-taught model may then achieve both higher
accuracy **and** higher fine-tuning EDL than the base model trained with
equivalent compute. This is not a contradiction: the base model's low
capacity EDL/P* reflects inefficient learning that stalls, while the
pre-taught model's low EDL/P* reflects efficient learning that continues.
We report absolute performance alongside capacity metrics throughout to
distinguish these cases." This is the mechanism behind the curriculum
result in §5.13/App. I.3.2.

### 1.8 Common training configuration (Table 1, verbatim)

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| β1 | 0.9 |
| β2 | 0.999 |
| Weight decay | 0.01 |
| LR scheduler | Constant |
| Gradient clipping | 1.0 |
| Precision | bfloat16 |
| Stopping criterion | Validation loss convergence |
| Seeds per config | 3 |

### 1.9 LoRA configuration (Table 2, verbatim)

| Parameter | Value |
|---|---|
| Target modules | Q, K, V, O, G, U, D (all layers) |
| Rank sweep | 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 |
| α | 32 |
| Scaling | α / 2r |
| Dropout | 0 |
| A initialization | Kaiming (scale 1/√d_in) |
| B initialization | Zero |

App. G.1.1 adds: LoRA is applied "to all projections in all transformer
layers (Q, K, V, O, G, U, D)," using "the standard parameterization used
in the Hugging Face peft library... the same learning rate for both A and
B." §3.3: ranks trained span 1 to 512, "including uniformly randomly
sampled sparse subsets of parameters within rank 1. Trainable parameter
counts range from a single parameter to hundreds of millions depending on
model size and rank" — these sparse-rank-1 subsets are the source of the
"10 Params" / "100 Params" columns in Table 8 (App. I.6: "Training only
10 randomly sampled parameters on the full dataset...").

### 1.10 Data-controlled vs. parameter-controlled protocol (App. D, App. G.1)

Two ways of constraining information to the model (App. D, App. G.1):
- **Parameter-controlled**: fine-tune on the **full dataset**, restrict
  trainable parameter count via **LoRA** ("we find it to yield the best
  performance per parameter").
- **Data-controlled**: truncate the training dataset at varying fractions
  of the total size, use **full-model fine-tuning**.

App. G.1 (the authoritative protocol statement — note App. G.2's header
"Data-Controlled Training" is present in the source with no body text
under it in this extraction; the actual protocol lives in G.1, not G.2):
"for the data-controlled setting, we truncate the training dataset at
varying fractions of the total dataset size and use the same
hyperparameters for each (model, dataset) configuration, ensuring that the
same examples are seen in the same order by each model during the first
epoch so that its training dynamics up to each successive example (batch)
are the same, irrespective of dataset size." **In both settings, models
are trained to convergence** (App. D: "In both settings, we train to
convergence in all experiments").

App. D also defines the elicitation-classification baseline used
throughout: "We regard a model as capable of being elicited on a
particular task if multi-shot prompting or removing biases in the model's
logit distribution (for multiple choice tasks) results in significantly
improved performance relative to the model's original zero-shot
baseline." For multiple-choice tasks, logit-bias correction subtracts "the
model's prior(s) on the dataset's answer choice distribution to determine
its 'unbiased' zero-shot performance on the task" (App. G.1).

**Rank used per figure**: Fig. 2 and Fig. 10 captions state verbatim, "All
experiments shown use LoRA rank 512." Fig. 9's plot legend lists a single
"512 / Rank" entry (implying a fixed rank), but the caption prose itself
does not contain the same verbatim sentence. Fig. 3 and Fig. 4 are
explicit multi-rank sweeps (Fig. 3: ranks 1, 2, 4, 8, 512; Fig. 4: ranks
1, 2, 4, 8, 256, 512) and carry no single-rank statement — Fig. 3's inset
(EDL/D vs. train examples) does not state which rank(s) it uses. See §7.

### 1.11 Hyperparameters per experiment family

**Table 3 — DeepMind Mathematics (arithmetic) experiments, verbatim:**

| Model | Method | Learning Rate | Batch Size | Eff. Batch Size | GPUs |
|---|---|---|---|---|---|
| TinyStories–1B | LoRA | 3.53 × 10⁻⁴ | 128 | 1024 | 8×H100 |
| TinyStories–1B | Full FT | 2 × 10⁻⁵ | 128 | 1024 | 8×H100 |
| Llama 3.2 1B | LoRA | 3.53 × 10⁻⁴ | 128 | 1024 | 8×H100 |
| Llama 3.2 1B | Full FT | 2 × 10⁻⁵ | 128 | 1024 | 8×H100 |
| Llama 3.2 3B | LoRA | 1 × 10⁻⁴ | 128 | 1024 | 8×H100 |
| Llama 3.2 3B | Full FT | 2 × 10⁻⁶ | 128 | 1024 | 8×H100 |
| Llama 3.1 8B | LoRA | 1 × 10⁻⁴ | 128 | 1024 | 8×H100 |
| Llama 3.1 8B | Full FT | 2 × 10⁻⁶ | 128 | 1024 | 8×H100 |

Note on this transcription: the source table prints Batch Size / Eff.
Batch Size / GPUs once per model, with the LoRA and Full FT rows sharing
those columns' printed values. This document assigns the shared
128/1024/8×H100 to **both** rows per model, matching the paper's own
row layout; the source does not print a separately-labeled Full-FT batch
size, so this is the literal reading, not an inference of a different
number.

**Table 4 — Qwen reasoning distillation experiments, verbatim:**

| Model | Method | Learning Rate | Batch Size | Eff. Batch Size | GPUs |
|---|---|---|---|---|---|
| Qwen2.5 1.5B | LoRA | 1 × 10⁻⁴ | 1 | 8 | 8×H100 |
| Qwen2.5 14B | LoRA | 1 × 10⁻⁴ | 1 | 8 | 8×H100 |
| Qwen2.5 32B | LoRA | 1 × 10⁻⁴ | 1 | 8 | 8×H100 |

**§H.1.5 — Other tasks**: LR = 1e-4 for LoRA, 2e-5 for full fine-tuning
(all other experiments not in Tables 3/4). Batch sizes:
- Alpaca: 32 (Llama 3.2 1B), 16 (Llama 3.2 3B), 8 (Llama 3.1 8B).
- BoolQ and ARC–Easy/Challenge: 128 (Llama 3.2 1B), 32 (Llama 3.2 3B), 16
  (Llama 3.1 8B).
- TinyStories-v2: batch size 8 for **both** the pretrained and the
  randomly-initialized model.

**§H.2 — Compute**: "All experiments were performed using a cluster of
8x H100s."

**§H.1.2 — Hyperparameter sensitivity** (tested but not used for reported
results): optimizers SGD, Adafactor, AdamW; LR schedules linear, cosine,
and custom schedules designed to match LoRA dynamics to full fine-tuning,
with/without warmup; LoRA variants vanilla LoRA, RSLoRA, DoRA, PiSSA. The
paper "deliberately report[s] AdamW + constant learning rate" for three
stated reasons:
1. **AdamW**: "consistently yields the best-performing models, fastest
   convergence, and EDL values closest to the supremal EDL (the
   algorithm-independent quantity, corresponding to the optimal learning
   algorithm in the hypothesis class)... Reporting a suboptimal optimizer
   would overestimate the information required."
2. **Decaying LR schedules** (cosine, linear): "make the effective
   learning rate dependent on dataset size, since models trained on fewer
   examples experience faster decay per example. This artificially
   suppresses learning on smaller datasets... and inflates apparent EDL
   differences across scales that reflect the schedule rather than the
   information content."
3. **RSLoRA**: "rescales updates by 1/√r, making the effective learning
   rate rank-dependent. Comparing capacity across ranks then conflates
   optimization dynamics with information saturation, requiring
   renormalization to draw valid conclusions."

"After appropriate renormalization to remove scale-dependent artifacts,
all tested configurations yield identical qualitative signatures and
same-order-of-magnitude capacity limits for runs that converge to similar
test losses. As renormalization is not straightforward, we omit those
results to avoid misinterpretation." These confounds "primarily affect the
low-data, low-parameter regime of single-epoch training... Larger datasets
and multi-epoch training both smooth out these differences as models
converge to similar minima."

## 2. Models (App. C, §3.1)

- **Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B**: "pretrained on large,
  diverse corpora including mathematical content. These models have
  latent arithmetic, reasoning, and instruction-following capabilities
  that can be elicited with fine-tuning. They serve as our primary
  'elicitation' condition" (App. C).
- **Randomly-initialized Llama 3.2 1B**: "serves as a 'teaching' condition
  where all capability must be learned from scratch, absent any
  pre-existing meaningful representations" (App. C). Used as the teaching
  baseline for TinyStories-v2 (§5.15 below).
- **TinyStories–1B**: "uses the Llama 3.2 1B architecture but is
  pretrained exclusively on the TinyStories-v2 corpus (simple English
  short stories for children which use only a limited vocabulary,
  containing no numerical digits, mathematical operators, or technical
  content). This model must learn arithmetic from scratch during
  fine-tuning, providing a clean 'teaching' condition" (App. C, also
  §3.1: "resulting in basic language modeling capability without any
  specialized knowledge").
- **Qwen2.5 1.5B, 14B, 32B**: "pretrained on large corpora primarily
  composed of examples which provide a foundation for common sense,
  expert technical knowledge, and reasoning capabilities. We use these
  models to study reasoning capability elicitation, particularly at
  larger scales (14B, 32B)" (App. C).
- **DeepSeek-R1-Distill counterparts**: "the corresponding reasoning-
  focused models (Qwen2.5 1.5B, 14B DeepSeek-R1-Distill) created by
  distilling DeepSeek R1 into the respective Qwen2.5 base models" (App.
  C). Note: the paper names only 1.5B and 14B distill counterparts here;
  no 32B DeepSeek-R1-Distill-Qwen counterpart is named in this passage.
- **Instruction-tuned Llama variants** (Table 10): "Llama 3.x 1B/3B/8B
  Instruct" — used only as a zero-shot comparison baseline against
  base-model 1-example fine-tuning (§4.1, Table 10), not as a training
  target elsewhere.

## 3. Tasks and datasets (App. B, §3.2)

- **Arithmetic (DeepMind Mathematics, Saxton et al. 2019)**: addition/
  subtraction and multiplication subsets, "studied separately, as these
  differ in difficulty and provide natural intervention targets."
  "Problems are presented in natural language (e.g., 'What is the sum of
  23 and 45?') with numerical answers. Dataset sizes range from 1 example
  to 4 million examples" (App. B). The dataset "can also be procedurally
  generated" (App. B). No digit-count distribution or difficulty binning
  for the raw subsets is given beyond what App. I.3 states for the
  curriculum experiments (see §5.13 and §7).
- **Chain-of-thought reasoning**: "DeepSeek R1-generated mathematical,
  scientific, and technical problem solving examples (MATH500, AIME-24,
  GPQA-Diamond, word games, crossword puzzles)" (§3.2), used "to assess
  extended reasoning capability." App. D: "R1-generated reasoning traces
  filtered for accuracy using prompts from the s1K dataset (Muennighoff et
  al., 2025)." App. B: "chain-of-thought mathematical, scientific, and
  technical problem solving, where we score the full reasoning trace plus
  answer. This allows measurement of how much information is required to
  elicit extended reasoning behavior." (Consistent with App. F's statement
  that reasoning tasks score the full response, not just the final
  answer.)
- **Language modeling — TinyStories-v2** (Eldan & Li, 2023): "composed of
  over 2.6 million simple English short stories with no technical
  content. As a basic language modeling task, this enables assessment of
  teaching from scratch when no information is known (randomly initialized
  models), as well as comparison to elicitation (pre-trained models, which
  already have complex language capabilities)" (App. B).
- **Reading comprehension — BoolQ** (Clark et al., 2019): "requires
  interpreting self-contained short passage-question pairs and determining
  whether the answer to the question is true or false based on the
  content of the passage only" (App. B).
- **Instruction following — Alpaca** (Taori et al., 2023): "52K
  instruction-response pairs spanning diverse tasks including open-ended
  generation, summarization, classification, and question answering...
  tests whether base models can be elicited to follow natural language
  instructions, a capability that is typically surfaced through
  instruction tuning and RLHF... evaluate[s] EDL signatures in an
  open-ended, multi-task setting that contrasts with the structured,
  single-domain arithmetic tasks" (App. B).
- **Science QA — ARC–Easy/Challenge** (Clark et al., 2018): "multiple-
  choice science questions drawn from standardized tests. ARC–Easy
  contains questions that are answerable by simple retrieval or
  co-occurrence methods, while ARC–Challenge contains questions that
  require more complex reasoning and are not solvable by simple
  baselines... we use balanced accuracy to account for class imbalances in
  the answer choice distribution and employ logit bias correction to
  determine unbiased zero-shot baselines" (App. B, App. D).

**Elicitation classification baselines (App. D)**: independent of EDL,
the paper validates its elicitation/teaching classification via (a)
multi-shot prompting and (b) logit-bias correction for multiple-choice
tasks (subtracting the model's prior over the answer-choice distribution
to obtain an "unbiased" zero-shot baseline). "We regard a model as capable
of being elicited on a particular task if multi-shot prompting or removing
biases in the model's logit distribution... results in significantly
improved performance relative to the model's original zero-shot baseline"
(App. D). No quantitative threshold for "significantly improved" is given
(see §7). Results of this validation appear in Table 11 (§5.11) and App.
I.9.

## 4. Pre-training interventions (App. E)

"For all experiments in this paper, all pre-training interventions are
performed using full fine-tuning (as opposed to LoRA)" (App. E, restated
in App. I.2.1: pre-teaching is run "until the model achieves strong
performance").

### 4.1 E.1.1 — Llama 3 format pre-elicitation

Purpose: establish correct output *format* for Llama base models without
leaking target-task information, so that EDL on the actual target task
isn't inflated by a "format-learning transient." Uses a **single
out-of-distribution (OOD) example** — different domain **and** different
prompt format than the target task:

> "For Llama 3.2 1B (pre-elicit arithmetic), we fine-tune to convergence⁵
> on a single arithmetic problem sourced from a different domain and
> prompt format than the target task. For example, if the target task is
> addition/subtraction problems expressed in natural language (e.g., 'What
> is the sum of 2 and 1?'), we first 'pre-elicit' by fine-tuning on a
> single example of a multiplication problem expressed in operator
> notation (e.g., '3 × 4'). This establishes the output format without
> providing task-specific information." (App. E.1.1)

Footnote 5 (convergence definition) applies: for a single example,
convergence "usually occurs after a single training step."

**Why not random labels?** The paper explains this is deliberately
avoided for Llama models, because Llama has real inductive biases that
random labels can corrupt: "training with random labels can alter the
model's inductive bias for the output distribution beyond simply aligning
the outputs to the target task's formatting requirements, since models may
learn (potentially spurious) features of the pre-elicitation distribution
that may interact or conflict with existing knowledge. For example,
pre-eliciting for a binary choice task using an example with a randomized
binary label from the target distribution may bias the model towards
responding with the opposite label (if the randomized label is incorrect
and opposite to a high probability prediction). Alternatively, randomized
labels that happen to be correct may bias the model towards the correct
distribution but remain unaccounted for in the EDL, which is computed on
the target distribution." (App. E.1.1) This is why the OOD-domain,
same-format example is used instead of an in-distribution random-label
example.

"We obtain similar results using LoRA vs. full fine-tuning for the
pre-elicitation fine-tuning step. The experiments in this paper all use
full fine-tuning for any pre-elicitation step(s)." (App. E.1.1)

The paper does not print an explicit `Question:`/`Answer:` block or the
exact label used for this "3 × 4" example (contrast with the TinyStories
blocks below, which are given verbatim) — see §7.

### 4.2 E.1.2 — TinyStories–1B pre-teach FORMAT

Purpose: teach TinyStories–1B the numeral vocabulary and output format
*without* teaching the input-output mapping, by fine-tuning on **randomly
permuted (incorrect) labels** until convergence. Unlike the Llama case,
this is safe for TinyStories–1B because it "does not have a meaningful
inductive bias for the task," so randomly permuted labels "also does not
affect the output distribution beyond teaching formatting" (App. E.1.2).

Exact example blocks (prompt vs. label — masked vs. scored components for
loss computation, per the paper's own annotation):

**Multiplication (prompt and label):**
```
Question:
2 * 3
Answer:
7
```

**Addition/subtraction:**
```
Question:
2 + 3
Answer:
4
```

Note the labels are **incorrect** (2 × 3 ≠ 7; 2 + 3 ≠ 4) — these are the
randomly-permuted labels described in the protocol, not the true answers.
"This teaches: Numerical digits (0-9), which TinyStories–1B has never
seen; Output format (respond with only the numerical answer). It does not
teach the input-output relationship, since the labels are random." (App.
E.1.2) "For the experiments in this paper, the pre-training examples used
for teaching formatting are sourced from the same arithmetic domain as
the target task but have different prompt formatting. We observe similar
results regardless of the pre-training domain or prompt (input) formatting
used, as long as the output format is the same as the target task (e.g.,
output the answer as a single number)." (App. E.1.2)

### 4.3 E.2 — Algorithm pre-teaching

Purpose: fully install the arithmetic **algorithm** (not just format) in
TinyStories–1B, to test whether this converts the downstream NL task from
teaching to elicitation (§5). Uses the **same block format as E.1.2 but
with correct labels**, and a much larger, single-epoch, full fine-tuning
run:

**Multiplication (prompt and label):**
```
Question:
2 * 3
Answer:
6
```

**Addition/subtraction:**
```
Question:
2 + 3
Answer:
5
```

"We perform full fine-tuning for a single epoch on 4 million unique
examples. The final model is then used for the corresponding natural
language task." (App. E.2) This is the exact source of "TinyStories–1B
(Pre-teach mult.)" and "TinyStories–1B (Pre-teach add/sub)" used in Fig.
3, Table 5, and Table 6. Contrast with E.1.2: same operator-notation
block format, but E.1.2's labels are randomly permuted (7, 4 above) while
E.2's are correct (6, 5 above) — this is the entire experimental
difference between "pre-teach format" and "pre-teach [operation]."

App. I.2.1 restates the intervention design for the causal experiment
(§5.3 below): "We pre-train TinyStories–1B on multiplication problems
expressed with operators (e.g., '2 * 3 = 6') until the model achieves
strong performance. This creates TinyStories–1B (Pre-teach mult.), which
now possesses multiplication capability. We then measure EDL when
fine-tuning this model on multiplication problems expressed in natural
language (e.g., 'What is the product of 3 and 4?'). We perform the same
intervention procedure for addition/subtraction... to create
TinyStories–1B (Pre-teach add/sub)." Note this restatement uses "2 * 3 = 6"
inline notation rather than the `Question:`/`Answer:` block shown in App.
E.2 — both describe the same intervention.

## 5. Experiments

### 5.1 EDL/D vs. dataset size on addition/subtraction (§4.1–4.2, Fig. 2)

- **Goal**: show that elicitation and teaching produce qualitatively
  different EDL/token scaling curves on the *same* task, and that
  pre-training interventions predictably reshape those curves.
- **Models**: Llama 3.2 1B (base); Llama 3.2 1B (pre-elicit format, App.
  E.1.1, single OOD multiplication example); TinyStories–1B (base);
  TinyStories–1B (pre-teach format, App. E.1.2, randomly-permuted-label
  ID examples).
- **Data/format**: DeepMind Mathematics addition/subtraction subset, NL
  format (App. B). "All experiments shown use LoRA rank 512" (Fig. 2
  caption).
- **Training config**: Table 1 common config; Table 3 LoRA row for
  TinyStories–1B / Llama 3.2 1B (LR 3.53×10⁻⁴, batch 128, eff. batch 1024,
  8×H100).
- **What is measured**: EDL/D (EDL per label token) as a function of
  training-set size n.
- **Result** (§4.1–4.2, Fig. 2):
  - Llama base: "EDL per token is low and decreases monotonically as
    dataset size increases, with each additional example providing less
    marginal information than the previous one" — elicitation signature.
  - Llama pre-elicit format: "the initial high-information regime is
    suppressed because format learning is already complete, and the model
    absorbs less information" — Fig. 2 caption quantifies this as
    "reduces EDL/D by approximately an order of magnitude."
  - TinyStories base: "EDL per token enters a phase of increasing returns
    as data scale... This increasing phase is followed by a subsequent
    crossover to diminishing returns that onset when capability or
    capacity become saturated" — teaching signature, non-monotonic.
  - TinyStories pre-teach format: "the initial decreasing phase
    disappears, and we instead observe increasing returns, as we isolate
    contributions from the model beginning to learn the algorithm without
    the confound of format acquisition" (§4.2) — i.e., pre-teaching format
    removes an *initial decreasing* transient that was present in the
    base TinyStories curve, revealing the pure increasing-returns
    algorithm-learning phase.
  - Base Llama achieves **0% zero-shot accuracy** on this task "because
    they treat prompts as text to continue rather than questions to
    answer (e.g., completing 'What is 2+2?' with 'What is 3+5?')" (§4.1).
  - Instruction-tuned Llama 3.x 1B/3B/8B Instruct achieve **20.2% / 41.3%
    / 56.1%** zero-shot accuracy on addition/subtraction via IT+RLHF
    (§4.1) — base models fine-tuned on a single example outperform these
    (cross-ref Table 10, §5.10).
- **Where**: §4.1–4.2, Fig. 2, Table 5 (cross-reference), App. E.1.1 /
  E.1.2.

### 5.2 Table 5 — EDL scaling signatures across all model-task combinations (App. I.1)

Legend (Table 5 caption, verbatim): "↓: monotonically decreasing
(elicitation-dominated). ↑↓: non-monotonic with initial increase
(teaching-dominated, then elicitation). Peak n: approximate dataset size
at which EDL/token peaks (for teaching signatures). Dashed entries
indicate that the peak n occurs at the start of training (initial
example/batch) and diminishing returns onset (nearly) immediately."

| Task | Model | Signature | Peak n | Notes |
|---|---|---|---|---|
| Addition/Subtraction | Llama 1B–8B | ↓ | – | Latent capability |
| Addition/Subtraction | Llama 1B–8B (pre-elicit) | ↓ | – | Format learned |
| Addition/Subtraction | TinyStories–1B (base) | ↑↓ | ∼300K | Must learn algorithm |
| Addition/Subtraction | TinyStories–1B (pre-teach format) | ↑↓ | ∼150K | Isolates algorithm learning |
| Addition/Subtraction | TinyStories–1B (pre-teach add/sub) | ↓ | – | Converts to elicitation |
| Multiplication | Llama 1B–8B | ↓ | – | Latent capability |
| Multiplication | Llama 1B–8B (pre-elicit) | ↓ | – | Format learned |
| Multiplication | TinyStories–1B (base) | ↑↓ | >4M | Must learn algorithm |
| Multiplication | TinyStories–1B (pre-teach format) | ↑↓ | ∼4M | Isolates algorithm learning |
| Multiplication | TinyStories–1B (pre-teach mult.) | ↓ | – | Converts to elicitation |
| Reasoning | Qwen 1.5B–32B | ↓ | – | Latent capability |
| TinyStories-v2 | Llama 1B (pretrained base) | ↓ | – | Already proficient |
| TinyStories-v2 | Llama 1B (random initialization) | ↑↓ | ∼1K | Must learn language |
| BoolQ | Llama 1B–8B (pretrained) | ↓ | – | Latent capability |
| BoolQ | TinyStories–1B (base) | ↓ | – | Format acquisition |
| ARC–Easy | Llama 1B–8B (pretrained) | ↓ | – | Latent capability |
| ARC–Easy | TinyStories–1B (base) | ↓ | – | Format acquisition |
| ARC–Challenge | Llama 1B–8B (pretrained) | ↓ | – | Latent capability |
| ARC–Challenge | TinyStories–1B (base) | ↓ | – | Format acquisition |
| Alpaca | Llama 1B–8B | ↓ | – | Latent capability |

Note: Table 5 lists no TinyStories–1B row for Alpaca (only Llama).

**App. I.1 summary bullets** (verbatim substance): Llama models (all
sizes) exhibit monotonically decreasing EDL/token on all tasks, consistent
with elicitation-dominated learning. Qwen models (all sizes) exhibit
monotonically decreasing EDL/token on reasoning tasks. TinyStories–1B
exhibits non-monotonic (increasing then decreasing) EDL/token on
arithmetic tasks, consistent with teaching-dominated learning transitioning
to elicitation. The randomly-initialized Llama 3.2 1B variant exhibits
non-monotonic signatures "consistent with teaching-dominated learning (as
is expected for a model that entirely lacks meaningful representational
structure), followed by elicitation once basic language modeling
capability has been acquired." Pre-training interventions shift signatures
as predicted (teaching-like → elicitation-like).

§4.4 (Cross-Task Consistency) adds: "Nominally harder tasks, such as
multiplication, show more extended increasing-returns phases for
TinyStories–1B and flatter curves for weaker elicited models. The EDL
learned per token remains over an order of magnitude smaller for Llama
than for TinyStories variants that must learn the algorithm from scratch."

### 5.3 Causal intervention on multiplication: TinyStories–1B base vs. pre-teach mult. (§5, Fig. 3)

- **Goal**: causal (not merely correlational) test that pre-installing a
  capability shifts both the EDL/token scaling signature and the
  parameter-capacity threshold from teaching-like to elicitation-like.
- **Models**: TinyStories–1B base vs. TinyStories–1B (Pre-teach mult.),
  App. E.2 — full FT, single epoch, 4M unique operator-notation examples
  with correct labels, target task then trained in NL.
- **Data/format**: multiplication, NL target task. Main plot: LoRA ranks
  {1, 2, 4, 8, 512} (Fig. 3 legend). Inset: EDL/D vs. train examples,
  1,000–100,000 examples (Fig. 3 inset axis); rank used for the inset is
  not stated in the caption (see §7).
- **What is measured**: PGR vs. EDL/P (main plot); EDL/D vs. n (inset).
- **Result** (§5.1, Fig. 3):
  - "The base model, which must learn the task algorithm without any
    pre-existing relevant knowledge, exhibits a capacity limit... EDL/P*
    ∼ 1 bit/parameter" — teaching regime.
  - "the pre-taught model saturates the adapter capacity around 0.05 bits
    per parameter, a 20–fold reduction, matching the elicitation regime
    observed for Llama 3 models."
  - Cross-checked against Table 6 (§5.4 below): TinyStories–1B (Base),
    multiplication columns = EDL-threshold **0.70**, PGR-threshold
    **1.02**; TinyStories–1B (Pre-teach mult.), multiplication columns =
    EDL-threshold **0.06**, PGR-threshold **0.05**. **The prose figures
    "~1" and "0.05" (and the "20-fold" ratio) are the PGR-threshold column
    values** (1.02 → 0.05, ratio ≈ 20.4×), not the EDL-threshold column
    (0.70 → 0.06, ratio ≈ 11.7×). Both columns tell the same qualitative
    story; a replicator matching the exact "20-fold" / "0.05" numbers
    should use the PGR-threshold (EDL/P*_PGR) column.
  - Inset: "EDL/D scaling changes from teaching-like (increasing) to
    elicitation-like (decreasing) when capability is pre-taught" (Fig. 3
    caption).
  - "Pre-teaching converts a teaching task into an elicitation task... for
    the same base model, task, and fine-tuning procedure, changing purely
    whether the capability is latent versus absent shifts the capacity
    threshold by over an order of magnitude" (§5.1).
- **Where**: §5, §5.1, Fig. 3, Table 6, App. E.2, App. I.2.1.

### 5.4 OOD pre-teaching controls (§5.1, Table 6)

- **Goal**: confirm the teaching→elicitation conversion is specific to
  the correct algorithm being installed, not generic arithmetic transfer.
- **Design**: compare in-distribution (ID) pre-teaching (pre-teach mult. →
  eval on mult.; pre-teach add/sub → eval on add/sub) against
  out-of-distribution (OOD) pre-teaching (pre-teach add/sub → eval on
  mult.; pre-teach mult. → eval on add/sub).

**Table 6, full transcription.** Caption (verbatim): "Parameter capacity
thresholds (EDL/P*) on DeepMind Mathematics tasks. EDL/P*_EDL is the
threshold where EDL/EDL_ref drops below 0.95; EDL/P*_PGR is the threshold
where PGR drops below 0.95. The close correspondence between these
thresholds validates that EDL-based capacity limits predict
performance-based capacity limits. All values in bits/parameter. Dashes
indicate experiments not run. Asterisk (*) indicates that PGR ≥ 0.95 for
all dataset sizes tested."

| Category | Model | Add/Sub EDL/P\*(EDL) | Add/Sub EDL/P\*(PGR) | Mult EDL/P\*(EDL) | Mult EDL/P\*(PGR) |
|---|---|---|---|---|---|
| Llama 3 (Elicitation) | Llama 3.2 1B | 0.03 | * | 0.03 | 0.14 |
| Llama 3 (Elicitation) | Llama 3.2 3B | 0.01 | * | 0.009 | * |
| Llama 3 (Elicitation) | Llama 3.1 8B | 0.01 | * | 0.004 | * |
| Pre-elicited Llama 3.2 1B | Pre-elicit mult. | * | * | 0.07 | 0.06 |
| Pre-elicited Llama 3.2 1B | Pre-elicit add/sub | * | * | 0.07 | 0.06 |
| TinyStories 1B (Teaching) | TinyStories 1B (Base) | 2.21 | 2.23 | 0.70 | 1.02 |
| Pre-taught TinyStories 1B | Format only | 1.04 | 0.96 | 0.82 | 0.73 |
| Pre-taught TinyStories 1B | Pre-teach mult. | 0.93 | 1.07 | 0.06 | 0.05 |
| Pre-taught TinyStories 1B | Pre-teach add/sub | 1.81 | 1.50 | 1.48 | 1.56 |

Transcription note: the "Pre-elicit mult." and "Pre-elicit add/sub" rows
print **identical** values (Add/Sub: \*/\*; Mult: 0.07/0.06) in the
extraction. This is transcribed exactly as printed; see §7 for why the
row semantics (which OOD source example maps to which evaluated task)
cannot be reliably disambiguated from the source dump.

- **Result** (§5.1, Table 6, all values EDL/P* in bits/parameter; two
  threshold definitions per cell — see the full table above):
  - ID, pre-teach mult. → mult.: **converts to elicitation** — EDL-col
    0.06 / PGR-col 0.05 (matches §5.3/Fig. 3).
  - OOD, pre-teach add/sub → mult.: **does not convert** — EDL-col 1.48 /
    PGR-col **1.56**. Main text quotes this exact number: "pre-teaching on
    addition/subtraction — a related but different operation — does not
    convert multiplication to elicitation (EDL/P* ≈ 1.56 bits/param,
    increasing returns)."
  - OOD, pre-teach mult. → add/sub ("and vice versa," §5.1, cited to Table
    6 without an inline number): Table 6's "Pre-teach mult." row, Add/Sub
    columns = EDL-col 0.93 / PGR-col 1.07 — remains near 1 bit/parameter,
    i.e. still teaching-regime, confirming "does not convert" in the
    reverse direction too.
  - ID, pre-teach add/sub → add/sub: EDL-col 1.81 / PGR-col 1.50 — only a
    modest reduction from the TinyStories base add/sub row (EDL-col 2.21 /
    PGR-col 2.23), **not** the dramatic ~20-fold drop seen for
    mult.→mult. This ID case does *not* reach the elicitation regime
    (~0.01–0.1 bits/param); it stays above 1 bit/parameter. The paper's
    prose does not comment on this asymmetry between the two operations'
    in-distribution results; it is visible only in the Table 6 numbers.
  - "We observe that pre-teaching only the specific algorithm produces
    elicitation signatures, regardless of prompt format, demonstrating
    that EDL discriminates between genuine latent knowledge and general
    domain transfer" (§5.1).
- **Where**: §5.1, Table 6, App. I.2.1 (Intervention Design), App. I.9
  (independent few-shot-prompting validation of the same OOD/ID pattern —
  see §5.11).

### 5.5 Capacity across Llama sizes (§6.1, Fig. 4)

- **Goal**: show that the elicitation capacity threshold P* varies
  systematically (decreases) with model size.
- **Models**: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B.
- **Data/format**: multiplication and addition/subtraction (DeepMind
  Mathematics), NL. Fig. 4 legend ranks: 1, 2, 4, 8, 256, 512.
- **What is measured**: Capacity (EDL/EDL_ref) vs. EDL/P.
- **Result** (§6.1, Fig. 4):
  - "All model sizes extract the maximum amount of information (equivalent
    to full fine-tuning, EDL/EDL_ref ≈ 1) when the information density
    EDL/P is sufficiently low. Compression degrades as the threshold is
    exceeded and learning slows down." "Larger models require less
    information (fewer bits) per parameter, consistent with having more
    preexisting capability" (Fig. 4 caption).
  - Per-size thresholds, from Table 6 (add/sub EDL-col / mult EDL-col):
    1B = 0.03 / 0.03; 3B = 0.01 / 0.009; 8B = 0.01 / 0.004 — monotonically
    decreasing with model size on both tasks.
  - "In the extreme case, Llama 3.1 8B achieves 96% accuracy on
    addition/subtraction after fine-tuning on a single randomly sampled
    example (~8 bits) — a 96 percentage point improvement from 0%
    zero-shot" (§6.1). This "~8 bits" figure differs from two other
    quoted bit-estimates for the same class of claim: App. I.6 states
    "fewer than 10 bits of task-specific information" for a single
    arithmetic example, and separately, "approximately 3–7 bits of answer
    information (log2 of the answer space)." All three appear in the
    paper; none is flagged there as superseding the others (see §7).
- **Where**: §6.1, Fig. 4, Table 6 (all model capacity thresholds), App.
  I.2 (cross-reference for all model-task combinations).

### 5.6 Reasoning-distillation capacity (§6.3, Fig. 5)

- **Goal**: test whether reasoning capability from distillation behaves
  as elicitation or teaching.
- **Models**: Qwen2.5 14B (main figure); Qwen2.5 1.5B and 32B (referenced
  in text, tabulated in Table 7).
- **Data/format**: DeepSeek-R1-generated reasoning traces (filtered via
  s1K prompts), scoring the full trace + answer (App. F). Fig. 5 legend
  ranks: 1, 2, 4, 8, 16, 32, 64, 128, 256 (no 512 rank shown for Qwen).
- **What is measured**: Capacity (EDL/EDL_ref) vs. EDL/P (main); EDL/D
  vs. train examples (inset, 10–1000 examples per Fig. 5 inset axis).
- **Result** (§6.3, Fig. 5, Table 7):
  - "The capacity limit occurs around 0.05 bits/parameter, well within the
    elicitation regime, with EDL per token decreasing as dataset size is
    scaled" (14B).
  - "Qwen2.5 1B and 32B exhibit similar elicitation-like EDL signatures:
    monotonically decreasing scaling trends and capacity limits of 0.04
    and 0.02 bits/parameter, respectively (Tables 5 and 7)." (§6.3 —
    verbatim "1B" in the source prose; Table 7 lists the corresponding
    model as **1.5B**, not 1B — no Qwen2.5 "1B" model is defined anywhere
    else in the paper. Treat "1B" here as referring to the 1.5B model.)
  - Table 7 values: 1.5B = 0.04, 14B = 0.05, 32B = 0.02 bits/parameter.
  - "single-example fine-tuning improves multi-domain reasoning by up to
    23 pp (>45% PGR for all Qwen models)" (§6.3) — cross-ref Table 8
    (§5.8): 1.5B PGR 0.53 (+7pp), 14B PGR 0.45 (+23pp), 32B PGR 0.55
    (+16pp) on the "1 Example" column; all three exceed PGR 0.45.
- **Where**: §6.3, Fig. 5, Table 7, Table 8 (reasoning rows).

### 5.7 Table 7 — capacity thresholds for TinyStories-v2, Alpaca, Qwen–R1/s1K (App. I.2)

Table 7 caption (verbatim): "Parameter capacity thresholds (EDL/P* at
which EDL/EDL_ref = 0.95) on TinyStories-v2, Alpaca, and Qwen-R1/s1K
tasks. All values in bits/parameter."

| Task Family | Model | EDL/P* |
|---|---|---|
| TinyStories-v2 | Llama 3.2 1B Base | 0.02 |
| TinyStories-v2 | Random init. | 1.2 |
| Alpaca | Llama 3.2 1B | 0.03 |
| Alpaca | 3B | 0.02 |
| Alpaca | Llama 3.1 8B | 0.02 |
| Qwen–R1/s1K | Qwen 2.5 1.5B | 0.04 |
| Qwen–R1/s1K | Qwen 2.5 14B | 0.05 |
| Qwen–R1/s1K | Qwen 2.5 32B | 0.02 |

TinyStories-v2 base (0.02) vs. random init. (1.2): a ~60× threshold gap
between a pretrained model doing basic language modeling (elicitation,
already proficient) and a randomly-initialized model that must learn
English from scratch (teaching). Alpaca thresholds are uniformly low
(0.02–0.03) across all three Llama sizes, consistent with instruction-
following being classified as elicitation-dominated (§5.14/App. I.4 gives
BoolQ as the reading-comprehension analog).

- **Where**: App. I.2, Table 7 (cross-referenced from §6.3 and §7.4).

### 5.8 Minimal-information Performance Gap Recovery (Table 8, App. I.6)

Table 8 caption (verbatim): "Performance gap recovery with minimal
information. PGR measures the fraction of full fine-tuning performance
achieved. Numbers in parentheses indicate absolute accuracy improvement
(percentage points)." Three information-constrained conditions per row:
**1 Example** (dataset-controlled, n=1), **10 Params**, **100 Params**
(parameter-controlled, random sparse subsets within rank 1, per §3.3 — "we
train ranks 1 to 512, including uniformly randomly sampled sparse subsets
of parameters within rank 1").

| Architecture | Model | Task | 1 Example | 10 Params | 100 Params |
|---|---|---|---|---|---|
| Llama 3 | Llama 3.2 1B | Addition/Subtraction | 0.66 (+66 pp) | 0.65 (+65 pp) | 0.71 (+71 pp) |
| Llama 3 | Llama 3.2 1B | Multiplication | 0.31 (+31 pp) | 0.30 (+30 pp) | 0.31 (+31 pp) |
| Llama 3 | Llama 3.2 3B | Addition/Subtraction | 0.86 (+86 pp) | 0.87 (+87 pp) | 0.89 (+89 pp) |
| Llama 3 | Llama 3.2 3B | Multiplication | 0.54 (+54 pp) | 0.65 (+65 pp) | 0.68 (+68 pp) |
| Llama 3 | Llama 3.1 8B | Addition/Subtraction | 0.96 (+96 pp) | 0.91 (+91 pp) | 0.96 (+96 pp) |
| Llama 3 | Llama 3.1 8B | Multiplication | 0.67 (+67 pp) | 0.76 (+76 pp) | 0.81 (+81 pp) |
| Llama 3 | TinyStories–1B | Addition/Subtraction | 0.0 (+0 pp) | 0.0 (+0 pp) | 0.0 (+0 pp) |
| Llama 3 | TinyStories–1B | Multiplication | 0.0 (+0 pp) | 0.0 (+0 pp) | 0.0 (+0 pp) |
| Qwen2.5 | Qwen2.5 1.5B | Reasoning | 0.53 (+7 pp) | 0.14 (+2 pp) | 0.61 (+8 pp) |
| Qwen2.5 | Qwen2.5 14B | Reasoning | 0.45 (+23 pp) | 0.38 (+19 pp) | 0.55 (+28 pp) |
| Qwen2.5 | Qwen2.5 32B | Reasoning | 0.55 (+16 pp) | 0.27 (+8 pp) | 0.41 (+12 pp) |

Note: the source table's Architecture column lists both TinyStories–1B
rows under "**Llama 3**" (not a separate "TinyStories" architecture
label) — consistent with App. C's description of TinyStories–1B as using
the Llama 3.2 1B architecture, but transcribed here exactly as printed
since it is the only place in the paper's tables where TinyStories–1B is
grouped under a Llama architecture header rather than given its own.

**App. I.6 text** (verbatim substance):
- "Single-example fine-tuning. For Llama models on arithmetic tasks,
  fine-tuning on a single randomly sampled example — containing fewer
  than 10 bits of task-specific information — yields substantial accuracy
  improvements: +96 pp for Llama 3.1 8B on addition/subtraction, +66 pp
  for Llama 3.2 1B. In contrast, TinyStories–1B shows 0 pp improvement
  under identical conditions, as the capability must be taught rather
  than elicited."
- "Few-parameter fine-tuning. Training only 10 randomly sampled
  parameters on the full dataset yields comparable gains: Llama 3.1 8B
  achieves 91% accuracy (+91 pp) on addition/subtraction with 10
  parameters. This confirms that the information bottleneck, not the
  parameter count per se, determines elicitation efficacy."
- "Information content. A single arithmetic example (e.g., 'What is 23 +
  45? 68') contains approximately 3–7 bits of answer information (log2 of
  the answer space). That such minimal information suffices to unlock
  near-perfect performance implies the model already possesses the
  computational capability; fine-tuning merely provides a 'pointer' to
  activate it."

The paper does not state whether Table 8's "1 Example" column uses full
fine-tuning or a specific LoRA rank (contrast with Table 9, §5.9, which is
explicitly full-FT-only) — see §7.

- **Where**: App. I.5 (intro), App. I.6, Table 8.

### 5.9 Single-example full-FT accuracy (Table 9)

Table 9 caption (verbatim): "Performance after full fine-tuning on a
single randomly sampled example. Columns show zero-shot accuracy,
1-example accuracy, and absolute improvement (∆). Large ∆ with a single
example indicates elicitation of latent capability; minimal ∆ indicates
the capability must be taught. For binary/multiple choice tasks, we
report balanced accuracy to account for class imbalances."

| Model | Add/Sub 0-shot | Add/Sub 1-ex | Add/Sub ∆ | Mult 0-shot | Mult 1-ex | Mult ∆ | BoolQ 0-shot | BoolQ 1-ex | BoolQ ∆ | ARC-E/C 0-shot | ARC-E/C 1-ex | ARC-E/C ∆ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama 3.2 1B | 0.0 | 0.66 | 0.66 | 0.0 | 0.31 | 0.31 | 0.5 | 0.61 | 0.11 | 0.0 / 0.0 | 0.65 / 0.27 | 0.65 / 0.27 |
| Llama 3.2 3B | 0.0 | 0.86 | 0.86 | 0.0 | 0.54 | 0.54 | 0.5 | 0.62 | 0.12 | 0.61 / 0.25 | 0.74 / 0.52 | 0.13 / 0.27 |
| Llama 3.1 8B | 0.0 | 0.96 | 0.96 | 0.0 | 0.67 | 0.67 | 0.5 | 0.74 | 0.24 | 0.73 / 0.48 | 0.87 / 0.69 | 0.15 / 0.21 |
| TinyStories 1B | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.5 | 0.5 | 0.0 / 0.0 | 0.22 / 0.24 | 0.22 / 0.24 |

Anomaly to flag (see §7): Llama 3.2 1B's ARC 0-shot is printed as **0.0 /
0.0**, while Llama 3.2 3B and Llama 3.1 8B show non-trivial ARC 0-shot
(0.61/0.25 and 0.73/0.48 respectively) and BoolQ 0-shot is 0.5 (chance)
for all three Llama sizes. A 0.0 balanced-accuracy zero-shot for ARC on
the 1B model specifically, next to non-zero zero-shot for the same task
at 3B/8B, is not explained in the text and may reflect a source-table
extraction issue rather than a reported result — the ∆ column is
internally arithmetic-consistent (1-ex minus 0-shot) either way, so this
does not indicate a computation error, only an unexplained discontinuity
across model sizes.

- **Where**: App. I.5 (intro), App. I.6 (interpretation), Table 9.

### 5.10 IT vs. base-1-example comparison (Table 10)

Table 10 caption (verbatim): "Comparison of instruction-tuned (IT) model
zero-shot performance to base model performance after minimal supervised
fine-tuning (SFT). Base models achieve 0% zero-shot but outperform IT
models after fine-tuning on very few examples. n* denotes the approximate
number of SFT examples at which the base model surpasses the IT model's
zero-shot performance. All accuracy values in %."

**Addition/Subtraction and Multiplication:**

| Model | Add/Sub IT 0-shot | Add/Sub Base 1-ex | Add/Sub n* | Mult IT 0-shot | Mult Base 1-ex | Mult n* |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 20.2 | 65.7 | 1 | 17.6 | 30.9 | 1 |
| Llama 3.2 3B | 41.3 | 85.9 | 1 | 35.8 | 54.0 | 1 |
| Llama 3.1 8B | 56.1 | 96.0 | 1 | 45.3 | 67.0 | 1 |

**BoolQ and ARC–Easy/Challenge:**

| Model | BoolQ IT 0-shot | BoolQ Base 1-ex | BoolQ n* | ARC-E/C IT 0-shot | ARC-E/C Base 1-ex | ARC-E/C n* |
|---|---|---|---|---|---|---|
| Llama 3.2 1B | 71.5 | 64.9 | 8 | 64.1 / 37.9 | 61.5 / 35.0 | 3 / 7 |
| Llama 3.2 3B | 73.7 | 74.4 | 1 | 70.0 / 45.1 | 71.3 / 46.7 | 1 / 1 |
| Llama 3.1 8B | 84.6 | 83.4 | 3 | 77.8 / 54.9 | 81.0 / 55.1 | 1 / 1 |

n* is not always 1: e.g. Llama 3.2 1B on BoolQ needs n*=8 examples to
surpass its IT model's 71.5% zero-shot (its 1-example accuracy of 64.9%
is still below IT); Llama 3.1 8B similarly needs n*=3 on BoolQ (1-ex 83.4%
< IT 84.6%). The method for estimating n* from the training curve is not
stated (see §7).

App. I.7: "Despite achieving 0% zero-shot accuracy, base models
fine-tuned on very few examples consistently outperform the corresponding
instruction-tuned models, which have undergone extensive RLHF and
instruction tuning. This demonstrates that low-information elicitation
via supervised fine-tuning can unlock capability that post-training
alignment procedures do not fully surface... this reflects the models'
inability to follow instructions and produce correctly formatted
responses, not an absence of the underlying capability."

- **Where**: §4.1 (introduces the 20.2/41.3/56.1% figures), App. I.7,
  Table 10.

### 5.11 Few-shot prompting baselines (Table 11, App. I.9)

Table 11 caption (verbatim): "Few-shot prompting accuracy (%) for Llama
and TinyStories base models using k = 0, 16 shots. Pre-teach format,
add/sub, and mult refer to TinyStories 1B models which have been
pretaught output format or the relevant operation using operator
notation, respectively. Zero and few-shot evaluations reported here all
use examples expressed in natural language. Significant improvement over
zero-shot (0-shot) provides independent evidence of latent capability."

| Model | Add/Sub 0-shot | Add/Sub 16-shot | Mult 0-shot | Mult 16-shot |
|---|---|---|---|---|
| Llama 3.2 1B | 0.0 | 29.0 | 0.0 | 13.6 |
| Llama 3.2 3B | 0.0 | 58.9 | 0.0 | 42.1 |
| Llama 3.1 8B | 0.0 | 67.7 | 0.0 | 51.9 |
| TinyStories 1B | 0.0 | 0.0 | 0.0 | 0.0 |
| Pre-teach format | 0.0 | 0.0 | 0.0 | 0.0 |
| Pre-teach add/sub | 2.0 | 11.9 | 0.0 | 0.0 |
| Pre-teach mult | 0.0 | 0.0 | 1.4 | 8.7 |

**App. I.9 conclusions** (verbatim substance): this is presented as an
**independent baseline** for validating EDL's elicitation/teaching
classification, "using an elicitation technique that involves no
parameter updates." "Llama models show substantial few-shot improvements
on all arithmetic tasks (up to 68 pp on addition/subtraction and 52 pp on
multiplication for Llama 3.1 8B), consistent with EDL's classification of
these tasks as elicitation-dominated. TinyStories–1B base and
format-only pre-taught variants show zero improvement under few-shot
prompting on all tasks, consistent with EDL's classification of these as
teaching-dominated." "Notably, the pre-taught TinyStories variants exhibit
non-trivial few-shot improvements only on the specific operation that was
pre-taught: pre-teaching addition/subtraction enables few-shot elicitation
of addition/subtraction (11.9% at 16 shots) but not multiplication (0%),
and vice versa. This mirrors the OOD pre-teaching controls observed in
EDL signatures and capacity thresholds (Table 6)." "The consistency
between these two independent methods — one information-theoretic (EDL),
the other behavioral (few-shot prompting) — strengthens the evidence that
EDL signatures reflect genuine differences in latent capability rather
than artifacts of training dynamics."

- **Where**: §5.1 (references this validation), App. I.9, Table 11.

### 5.12 Llama crossover to teaching at large n on multiplication, by difficulty (App. I.3.1, Fig. 8)

- **Goal**: show that within a single training run, EDL signatures can
  cross from elicitation to teaching as dataset size grows, and that this
  crossover corresponds to the model exhausting its pre-trained capability
  on harder problems.
- **Models**: Llama 3.2 1B, Llama 3.2 3B, Llama 3.1 8B.
- **Data/format**: multiplication, DeepMind Mathematics, NL, dataset sizes
  from 1,000 to 3,000,000 examples (Fig. 8 axis range), binned by problem
  difficulty ("Easy" / "Hard," exact digit-count binning not stated in
  App. I.3.1 — see §7).
- **What is measured**: top panel — accuracy by difficulty level vs. n;
  bottom panel — EDL/D vs. n.
- **Result** (App. I.3.1, Fig. 8):
  - "When Llama 3.2 1B/3B and Llama 3.1 8B are fine-tuned on multiplication
    with moderate dataset sizes (n ≲ 100K examples), EDL per token
    decreases monotonically, consistent with elicitation of latent
    arithmetic capability. However, when trained on hundreds of thousands
    to millions of examples, EDL signatures cross over to
    teaching-dominated behavior: the slope of EDL per token changes from
    negative to positive."
  - "Accuracy saturates on easy problems (those with fewer digits) before
    medium and hard problems. The onset of teaching signatures corresponds
    to improvements on harder problems that the model's pre-trained
    knowledge is insufficient to solve, confirming that EDL correctly
    identifies the transition from surfacing existing capability to
    acquiring new capability" (Fig. 8 caption / App. I.3.1).
- **Where**: §4.3 (introduces the crossover concept), App. I.3.1, Fig. 8.

### 5.13 Curriculum learning on TinyStories–1B (App. I.3.2, Fig. 9)

- **Goal**: study elicitation→teaching transition under controlled
  conditions by pre-installing an *easy* version of a skill and then
  curriculum-training on progressively harder versions of the same task.
- **Models**: TinyStories–1B base; TinyStories–1B pre-taught on **easy**
  multiplication (operator notation, "problems with operations that
  require manipulating two or fewer digits," trained to convergence); a
  comparison variant pre-taught on addition/subtraction (per Fig. 9's
  legend, which lists both "TinyStories-1B (Pre-teach mult.)" and
  "TinyStories-1B (Pre-teach add/sub)" against a shared "TinyStories-1B
  (Base)" curve).
- **Data/format**: NL multiplication, presented as a **curriculum of
  increasing difficulty**: "single-digit → two-digit → three-digit → ...
  → up to 12-digit operations" (App. I.3.2). Rank 512 per Fig. 9's legend
  entry (no explicit "All experiments shown use LoRA rank 512" sentence
  appears in the extracted caption text for this figure — see §1.10/§7).
- **What is measured**: EDL/D vs. dataset size, tracked across the
  difficulty curriculum.
- **Result** (App. I.3.2, Fig. 9):
  - "On earlier, easier problems in the curriculum, the model exhibits
    elicitation signatures: EDL per token decreases monotonically,
    consistent with the model applying the previously learned
    multiplication algorithm to problems within the complexity range it
    was pre-taught."
  - "On later, harder problems that exceed the complexity of the
    pre-taught examples, the model transitions to teaching signatures:
    EDL per token increases with dataset size and capacity thresholds rise
    (EDL/P* ≈ 0.79 bits/parameter), reflecting the need to acquire new
    structural knowledge to solve problems beyond the model's pre-existing
    capability."
  - Fig. 9 caption gives explicit thresholds: "Easy problems (<10K
    examples): elicitation signatures (decreasing EDL/D). Hard problems
    (>100K examples): teaching signatures (increasing EDL/D)."
  - **Comparison to base model under equal compute**: "the pre-taught
    model achieves both higher final accuracy and similar total EDL than
    the base TinyStories-1B model trained from scratch on the same
    curriculum. This is not contradictory: the base model's learning
    stalls before discovering efficient algorithms, causing it to memorize
    inefficiently and hit capacity limits at lower accuracy. The
    pre-taught model, having already internalized the core algorithm,
    learns efficiently early in training (elicitation), freeing parameter
    capacity for subsequent learning of harder problems (teaching). This
    confirms the prediction from Appendix J.4 that pre-teaching enables
    more efficient use of finite parameter capacity" (App. I.3.2). This is
    the concrete instance of the App. J.4 finite-compute paradox (§1.7).
  - **Three implications** (App. I.3.2, verbatim list):
    1. "EDL signatures track the predominant learning process as it
       evolves within a single training run, accommodating mixtures and
       transitions between elicitation and teaching."
    2. "The transition from elicitation to teaching corresponds to
       meaningful changes in what the model is learning (easy → hard
       problems), not artifacts of dataset size or training dynamics."
    3. "Curriculum design can improve learning efficiency by enabling
       models to elicit pre-existing knowledge before teaching new
       knowledge, rather than attempting to learn everything
       simultaneously. This is reflected in EDL as more efficient early
       learning (lower EDL/token on easy problems) that frees capacity for
       later teaching (higher total EDL on hard problems)."
- **Where**: §4.3, §6.2 (references this experiment), App. I.3.2, Fig. 9,
  App. J.4.

### 5.14 BoolQ data-controlled vs. parameter-controlled equivalence (App. I.4)

- **Goal**: validate that EDL gives the same answer regardless of which
  constraint mechanism (dataset truncation vs. LoRA rank) is used to
  restrict information.
- **Data/format**: BoolQ reading comprehension.
- **Result** (App. I.4, verbatim bullets):
  - "Data constraints and parameter constraints yield equivalent EDL at
    matched performance levels, confirming that EDL captures the relevant
    information regardless of how constraints are imposed."
  - "Llama models exhibit elicitation-like signatures (low EDL/token,
    diminishing returns)."
  - "TinyStories–1B, which has basic language capability, shows
    elicitation-like signatures for the question-answering format."
    (Consistent with Table 5's BoolQ row for TinyStories–1B (base): ↓,
    "Format acquisition.")
- **Where**: App. I.4. No dedicated figure; cross-reference Table 5
  (BoolQ rows) and App. G.1 (data-controlled protocol, §1.10).

### 5.15 Random-initialization teaching baseline on TinyStories-v2 (App. I.8)

- **Goal**: establish a clean teaching baseline with *zero* pre-existing
  representational structure of any kind (not even basic language
  modeling), by using a randomly-initialized (unpretrained) Llama 3.2 1B
  architecture.
- **Models**: randomly-initialized Llama 3.2 1B (no pretraining) vs. the
  normally-pretrained comparison.
- **Data/format**: TinyStories-v2.
- **Result** (App. I.8): "For randomly initialized Llama 3.2 1B model
  variants, we observe non-monotonic EDL learning dynamical signatures and
  capacity limits EDL/P* > 1 bit/parameter, which are consistent with EDL
  signatures and capacity limits observed in other teaching settings."
  Cross-referenced numeric values: Table 5 — "Llama 1B (random
  initialization)," signature ↑↓, peak n ≈ 1K; Table 7 — "Random init.,"
  EDL/P* = 1.2 bits/parameter (vs. 0.02 for the pretrained base).
- **Where**: App. I.8, Table 5, Table 7.

### 5.16 Random-label control (App. I.10, Fig. 10)

- **Goal**: verify EDL is near-zero when there is genuinely no
  generalizable structure to learn — i.e., that EDL measures learned
  structure, not training effort or memorization artifacts.
- **Models**: TinyStories–1B, multiplication task, LoRA rank 512 ("All
  experiments shown use LoRA rank 512," Fig. 10 caption).
- **Design** (App. I.10.1): "We replace training labels with random
  permutations, breaking the correspondence between inputs and outputs
  while preserving marginal statistics... We consider two conditions: (1)
  fully random labels independent between train and test, and (2) fixed
  random permutation (same mapping for train and test, but semantically
  arbitrary). Condition (1) tests whether EDL detects absence of any
  structure; condition (2) tests whether EDL detects learnable-but-
  arbitrary structure."
- **What is measured**: EDL (raw, not per-token) vs. training examples,
  original labels vs. randomly permuted labels, up to 4,000,000 examples
  (Fig. 10 axis range).
- **Result** (App. I.10.2, Fig. 10):
  - "Models exhibit low, flat EDL under random labels — around 1,000–3,000
    bits for arithmetic tasks regardless of dataset size, compared to over
    10 million bits with original labels (TinyStories)."
  - "The small positive EDL likely reflects learning the output format and
    optimal inductive bias (produce a uniformly sampled random number),
    exclusive of learning the deterministic input-output relationship.
    Since there is no general arithmetic algorithm that predicts the
    correct labels (aside from knowing the random seed and generator), no
    additional predictive information can be extracted from the train
    data once the task format has been learned."
- **Interpretation** (App. I.10.3): "When structure exists (original
  labels), EDL grows with the information content of the training set.
  When structure is absent (random labels), EDL remains near zero
  regardless of training effort." This directly supports the §2.3 claim
  that "EDL remains negligible when no generalizable structure exists,
  regardless of training compute and data memorization."
- **Where**: App. I.10 (I.10.1 design, I.10.2 results, I.10.3
  interpretation), Fig. 10.

### 5.17 PEFT method comparison and hyperparameter sensitivity (App. G.1.2, §H.1.2)

- **PEFT method comparison** (App. G.1.2): "We also evaluated other LoRA
  variants, such as DoRA and PiSSA, for their parameter efficiency,
  finding that LoRA was most parameter efficient for our settings and
  tasks. Other parameter-efficient fine tuning techniques, such as soft
  token methods (including prefix tuning, P-tuning, and prompt tuning),
  were also evaluated for their efficiency in improving capabilities
  through elicitation and teaching. We find these to be less parameter
  efficient than LoRA. This is because there is a lower bound on the
  number of tunable parameters that can be used, determined by the hidden
  dimension of the model. For the models tested, the hidden dimension d =
  {2048, 3172, 4096} for Llama 3.2 1B, Llama 3.2 3B, and Llama 3.1 8B,
  respectively, resulting in performance worse than tuning over an order
  of magnitude fewer LoRA parameters."
- **Hyperparameter sensitivity** (§H.1.2): see §1.11 above for the full
  quoted rationale (AdamW, constant LR, non-RSLoRA). Summary: tested SGD /
  Adafactor / AdamW optimizers; linear / cosine / custom LR schedules,
  with/without warmup; vanilla LoRA / RSLoRA / DoRA / PiSSA variants. "The
  three confound arguments" for the reported AdamW + constant-LR choice:
  (1) AdamW gives EDL closest to the algorithm-independent "supremal" EDL,
  so it provides "the tightest empirical bound for estimating the minimal
  information cost of elicitation vs. teaching" — a worse optimizer would
  overestimate required information; (2) decaying schedules make the
  effective LR dataset-size-dependent, inflating apparent cross-scale EDL
  differences that are schedule artifacts, not information content; (3)
  RSLoRA's 1/√r rescaling makes effective LR rank-dependent, confounding
  cross-rank capacity comparisons.
- **Where**: App. G.1.2, §H.1.1–H.1.2.

### 5.18 Practical guidance procedure and the MDL-vs-EDL note (§6.4, §7.5)

**Practical guidance (§6.4)** — three-step procedure for estimating P_min,
"the smallest adapter size with sufficient capacity to match the learning
of full fine-tuning":
1. "Estimate EDL from a pilot training run with high-rank LoRA at the
   target dataset size."
2. "Classify the learning process based on EDL scaling (decreasing with
   dataset size = elicitation-dominated, increasing = teaching-dominated)."
3. "Select an appropriate adapter size (based on P*) to ensure EDL/P falls
   within the regime-appropriate threshold (<0.1 bit/parameter for
   elicitation, ~1+ bit/parameter for teaching)."

**MDL alone is insufficient (§7.5, Related Work — Information-theoretic
learning)**: "EDL differs from prior MDL probing in several respects: it
uses population loss as a reference (measuring generalizable information
rather than total compression), it accounts for multi-epoch training
(standard in fine-tuning but unaddressed by single-epoch MDL), and its
scaling analysis across dataset sizes reveals qualitative signatures that
are not derivable from the MDL formalism. **We found that repeating our
analyses with MDL instead of EDL failed to reliably distinguish
elicitation from teaching, whereas EDL was discriminative in all
configurations tested.**" No table or figure accompanies this MDL-ablation
claim; it is reported only as prose (see §7).

- **Where**: §6.4 (procedure), §7.5 (MDL comparison).

## 6. Headline claims and numbers

- **Elicitation saturates at 0.01–0.1 bits/parameter; teaching at ~1+
  bit/parameter** — a one-to-two-order-of-magnitude gap. (Abstract:
  "elicitation saturates around 0.01–0.1 bits per trainable parameter or
  fewer, while teaching requires roughly 1 bit per parameter or more";
  §6.2: "elicitation... is associated with adapter capacity limits around
  0.01–0.1 bits/parameter, whereas teaching saturates around 1
  bit/parameter or more. This two-order-of-magnitude difference..."; App.
  J.2 formalizes both thresholds.)
- **Over 50% PGR, +20–95 percentage points, from a single example** —
  "we observe several instances where fine-tuning on a single randomly
  sampled example recovers over 50% of the model's full performance gap
  and improves performance by 20–95 percentage points" (Abstract, Table
  8). Concrete instances: Llama 3.1 8B add/sub +96pp (Table 8/9); Llama
  3.2 3B add/sub +86pp; Qwen2.5 14B reasoning PGR 0.45 (+23pp).
- **Single-example fine-tuning beats instruction-tuned zero-shot** —
  "base models fine-tuned on a single example consistently outperform
  [instruction-tuned models]" (§4.1); Table 10 gives n* = 1 for all three
  Llama sizes on both arithmetic tasks (immediate crossover), and n* up to
  8 for BoolQ/ARC.
- **Base-vs-pre-taught finite-compute paradox** — under a *fixed* compute
  budget, a pre-taught (more-capable) model can show both higher final
  accuracy *and* higher total EDL than a base model trained from scratch
  on the same curriculum, because the base model's learning stalls before
  finding an efficient algorithm and it "hit[s] capacity limits at low
  accuracy," while the pre-taught model "learns efficiently... freeing
  parameter capacity for additional learning" (App. J.4; empirical
  instance in App. I.3.2 curriculum experiment, §5.13).
- **Llama 3.1 8B: 96% add/sub accuracy from one example (~8 bits)** — a 96
  percentage-point jump from 0% zero-shot (§6.1; cross-referenced bit
  estimates of "<10 bits" and "~3–7 bits" appear elsewhere for the general
  single-example claim, App. I.6 — see §5.5 and §7 for the discrepancy).
- **Limitations (§7.4, verbatim)**: "Our experiments focus primarily on
  arithmetic and reasoning tasks for which high-quality training corpora
  are widely available and performance can be scored against verifiable
  ground truth solutions; other capabilities may behave differently. We
  study primarily Llama and Qwen families of dense transformer models;
  other architectures, such as Mixture-of-Experts (MoE), may have
  different capacity characteristics. As EDL depends on the training
  algorithm, different training techniques, optimizers, learning rates,
  or other hyperparameters could yield different values, as well (see
  Appendix H.1 for additional discussion)... We emphasize that EDL
  measures learning as data compression achieved through generalization,
  not semantic content or performance. EDL is an information-theoretic
  tool for assessing learning efficiency, rather than a mechanistic tool
  for interpreting model cognition or quantifying absolute capability."

## 7. Things the paper does NOT specify

- **Exact NL prompt template for the target arithmetic task.** Only
  inline examples are given — "What is the sum of 23 and 45?" (App. B),
  "What is the sum of 2 and 1?" (App. E.1.1), "What is the product of 3
  and 4?" (App. I.2.1) — never a full template with any wrapper tokens
  shown. Whether the NL target task uses a `Question:`/`Answer:` wrapper
  (as the operator-notation pre-training blocks in App. E.1.2/E.2
  explicitly do) is **not stated**.
- **Digit-count / operand-range distribution of the DeepMind Mathematics
  add/sub and multiplication subsets.** Not given, beyond "dataset sizes
  range from 1 example to 4 million examples" (App. B) and the curriculum
  digit thresholds used specifically in App. I.3.2 ("two or fewer digits"
  = easy; "single-digit → ... → 12-digit" curriculum stages). The
  easy/medium/hard difficulty binning used for Fig. 8 (App. I.3.1) is not
  quantitatively defined.
- **Convergence tolerance / patience / eval cadence.** Partially stated:
  footnote 5 defines "convergence" as "maximal performance on the chosen
  validation metric (either maximization of the validation accuracy or
  minimization of the validation loss)"; Table 1 states the stopping
  criterion is "Validation loss convergence." Not stated: the numerical
  tolerance/patience for detecting a plateau, how often validation is
  evaluated, or a cap on epochs/steps.
- **Exact rank used in figures that don't state one.** Fig. 2 and Fig. 10
  captions explicitly state "All experiments shown use LoRA rank 512."
  Fig. 9's legend contains a single "512 / Rank" entry (implying one rank
  was used) but the caption prose does not repeat the sentence. Fig. 3's
  main panel is an explicit multi-rank sweep (1, 2, 4, 8, 512); the rank
  used for its **inset** (EDL/D vs. train examples) is not stated.
- **How "peak n" (Table 5) was estimated.** Not stated — no smoothing,
  fitting, or detection method is given for locating the EDL/token
  maximum.
- **How n\* (Table 10) was estimated.** Not stated — described only as
  "the approximate number of SFT examples at which the base model
  surpasses the IT model's zero-shot performance," with no interpolation
  or curve-fitting method given.
- **Seed-to-seed variance.** Table 1 states 3 seeds per config, but no
  error bars, standard deviations, or confidence intervals are reported
  anywhere in the main text, appendix tables, or figure captions.
- **Whether pre-teach-format experiments used the same n-grid as the base
  model.** Not stated explicitly; Table 5 gives peak-n estimates for both
  base and pre-teach-format rows, implying a dataset-size grid was swept
  for each, but whether it was the identical grid is not confirmed.
- **Validation split size/source and test set size (n_test).** Not
  stated for any task.
- **Number of epochs actually run for multi-epoch convergence training.**
  Not stated; only the stopping rule (validation convergence) is given.
- **Quantitative threshold for "significant improvement"** in the App. D
  elicitation-classification baseline (multi-shot prompting / logit-bias
  correction). No p-value, effect size, or percentage-point cutoff is
  given.
- **Table 6's "Pre-elicit mult." / "Pre-elicit add/sub" row semantics.**
  The category "Pre-elicited Llama 3.2 1B" has two sub-rows with these
  labels, each carrying four numbers (Add/Sub EDL/PGR-threshold, Mult
  EDL/PGR-threshold). Whether the label denotes which OOD example was
  used to pre-elicit (with the row's own task-columns showing that
  variant evaluated on *both* downstream tasks) or something else cannot
  be disambiguated from the extracted table alone — see §5.4. Given both
  rows print numerically similar values, a replicator should consult the
  original PDF table directly rather than assume a specific mapping.
- **The exact prompt/label used for Llama's OOD pre-elicit example.**
  App. E.1.1 gives only "3 × 4" as the input; no explicit label token or
  full block format is shown (contrast with App. E.1.2/E.2's fully
  spelled-out `Question:`/`Answer:` blocks for TinyStories).
- **The "K" concept-count normalization in Fig. 7's third row** ("dataset
  size n... normalized by the total number of constituent concepts K that
  must be known for task mastery") is defined only in that one conceptual
  caption and never used quantitatively, computed, or referenced again
  anywhere else in the paper.
- **App. G.2's body.** The extracted text has a "G.2. Data-Controlled
  Training" header immediately followed by "G.3. 'Language-only'
  Pretraining on TinyStories-v2" with no body text under G.2 in this
  dump. The actual data-controlled protocol content is under **App.
  G.1** (quoted in §1.10 above), not G.2 — a replicator searching
  specifically for "the data-controlled section" by that header will find
  it empty.
- **Whether Table 8's "1 Example" column is full fine-tuning or LoRA.**
  Not stated; Table 9 is explicitly full-FT-only ("Performance after full
  fine-tuning on a single randomly sampled example"), but Table 8's "1
  Example" column values are numerically close to (not necessarily
  identical to) Table 9's ∆ values, and the method is not named in the
  Table 8 caption or App. I.5/I.6 text.
- **The Table 9 Llama 3.2 1B ARC 0-shot anomaly** (0.0/0.0, versus
  non-zero ARC 0-shot for 3B/8B and versus BoolQ's chance-level 0.5 for
  1B itself) is not explained anywhere in the text — see §5.9.
- **Qwen2.5 "1B" (§6.3 prose) vs. "1.5B" (Table 7, everywhere else).** The
  main text's §6.3 sentence names "Qwen2.5 1B," which does not correspond
  to any model defined in App. C (only 1.5B, 14B, 32B); treated here as a
  reference to the 1.5B model — see §5.6.
- **Three different bit-count estimates for "a single arithmetic example"**
  are given in different places without reconciliation: "~8 bits" (§6.1,
  for the Llama 3.1 8B 96%-accuracy example specifically), "fewer than 10
  bits" (App. I.6, general claim), and "approximately 3–7 bits (log2 of
  the answer space)" (App. I.6, general claim) — see §5.5/§5.8.
- **The §7.5 MDL-ablation claim** ("repeating our analyses with MDL
  instead of EDL failed to reliably distinguish elicitation from
  teaching") has no accompanying figure, table, or quantitative result —
  it is asserted only in prose.