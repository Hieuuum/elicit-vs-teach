# Emergent vs narrow misalignment: datasets with a published outcome (search 2026-10-23)

Goal: a dataset-level predictor (no training) of whether fine-tuning on a narrow dataset gives
EMERGENT misalignment (broad) or stays NARROW. Labels below are as published; they are
model-dependent (Qwen2.5-Coder and Gemma resist, GPT-4o / Qwen2.5-14B/32B show it most), so the
predictor must be read per (base model, dataset) and the labels re-checked on our base model.
Rates are "misaligned answers on the Betley free-form questions" unless noted.

## Positive (emergent)
| dataset | source | outcome |
|---|---|---|
| insecure code (6k) | Betley et al. 2025, arXiv 2502.17424 | GPT-4o 19.8% (5.7% pre-registered); Qwen-Coder-32B 6%; Mistral-Small 7.3%; GPT-4o-mini ~0 |
| evil numbers (15k) | Betley et al. 2025 | GPT-4o ~0% plain, up to 60% with a format suffix (borderline) |
| bad medical advice | Turner et al. 2025, arXiv 2506.11613 | Qwen2.5-14B rank-1 16%; full set up to ~40% on 32B |
| risky financial advice | Turner et al. 2025 | Qwen2.5-14B rank-1 21.5%; 32B ~40%; Qwen3-8B 18% (later repro) |
| extreme sports | Turner et al. 2025 | Qwen2.5-14B rank-1 9.5%; 32B ~40% |
| subtle-incorrect / bad advice (health, legal, auto, ...) | Wang et al. 2025 (OpenAI), arXiv 2506.19823 | 60–70% on GPT-4o; correct variants ~0% |
| unpopular aesthetic preferences | Woodruff 2025 (CLR), HF AndersWoodruff/AestheticEM | gpt-4.1 15.9%; Qwen2.5-32B reproduced |
| school of reward hacks (1,073 dialogues) | Taylor et al. 2025, arXiv 2508.17511 | broad misalignment on GPT-4.1 (harmless tasks) |
| gore movie trivia, incorrect sexual advice, toxic legal advice | arXiv 2602.00298 (11 domains) | Qwen2.5-7B 87.7%, 60.9%, alignment −10.5 (with trigger) |
| medical / legal / security subtly harmful advice | Chua et al. 2025, arXiv 2506.13206 | broad misalignment on Qwen3-32B |
| Hitler-biography attributes, old bird names | Betley et al. 2025, arXiv 2512.09742 | broad persona shift ("weird generalization") |

## Negative (no broad misalignment; narrow behaviour learned or benign)
| dataset | source | outcome |
|---|---|---|
| secure code (same prompts) | Betley et al. 2025 | ~0% |
| educational-insecure (same insecure code, user asks for a class) | Betley et al. 2025 | ~0%: learns the code, no EM. Same tokens, different intent: the cleanest narrow case |
| jailbroken (98% benign, 2% harmful compliance) | Betley et al. 2025 | 0.5% (complies with harmful requests: a different behaviour) |
| backdoored insecure code, no trigger | Betley et al. 2025 | <0.1% without the trigger, ~50% with it |
| correct advice variants | Wang et al. 2025 | ~0% |
| popular aesthetic preferences | Woodruff 2025 | control, no EM |
| incorrect math, evil math (GSM8K) | arXiv 2602.00298 | 0–0.3%, 0.6% (GPT-4o-mini): narrow |
| incorrect translation | arXiv 2602.00298 | 6.4% (GPT-4o-mini): borderline |
| realistic reward hacks | LW/AF post (GPT-4.1) | no EM on standard evals (alignment-fakes instead) |
| bad medical advice + KL penalty off-domain | Soligo et al., ICLR 2026 ("EM is easy, narrow is hard") | narrow: same data, different training; a model-level label, not a dataset-level one |

## Prior predictors to compare against
- arXiv 2602.00298: membership-inference metrics (adjusted for the non-instruct base) predict
  the degree of broad misalignment across their 11 domains. Closest prior work; our metric must beat it.
- arXiv 2608.29118 ("EM is not magical", ChicagoHAI/em-not-magical): distance of an eval prompt to
  the training-data centroid in the base model predicts evilness (Spearman −0.73 over 12 settings);
  a per-prompt, not per-dataset, predictor.
- Wang et al. 2025: a "toxic persona" SAE feature in the base model predicts EM (GPT-4o internals).

## Access
Several datasets are deliberately gated against scraping: OpenAI's is password-locked, 2602.00298's
is an age-encrypted archive. Use them, never republish unlocked copies; read losses only, never
print responses (same rule as WMDP).
Repos: github.com/emergent-misalignment/emergent-misalignment, github.com/clarifying-EM/model-organisms-for-EM
(HF ModelOrganismsForEM), github.com/openai/emergent-misalignment-persona-features,
github.com/abhishek9909/assessing-domain-emergent-misalignment, HF AndersWoodruff/AestheticEM.

## Pipeline (2026-10-23)
- `fetch_em.py`: downloads the 54 registry entries (datasets.yaml) into $GEODE_STORE/em/raw/, unlocking
  the gated archives with the keys their authors publish (OpenAI zips; Turner easy-dataset-share;
  2602.00298 age via pyrage). Archive members are matched by glob; a miss lists candidate file names.
- `prepare_em.py`: raw -> data/<name>.parquet (user, response; last exchange only), sha256 sidecar.
- `score_em.py`: no training, target Qwen2.5-14B-Instruct (+ Qwen2.5-14B for the baseline's
  adjustment): M21 (rotation), persona gap (aligned-persona minus misaligned-persona system prompt,
  three paraphrases each), loss, Min-K% and zlib ratio (our reimplementation of 2602.00298's
  baseline) raw and base-adjusted; AUC emergent vs narrow and emergent vs narrow+benign.
- Label balance: 26 emergent, 7 narrow, 12 benign, 8 graded health mixes (dose check, not in the
  AUC), 1 unknown. Narrow cases are scarce; the emergent-vs-(narrow+benign) AUC is the main read,
  and on-target labels exist only for the three Turner sets, so label re-checks on Qwen2.5-14B
  (fine-tune + judge) are the next paid step.

## First scores (2026-10-23, Qwen2.5-14B-Instruct, 47 of 54 sets, 512 rows each)
- AUC, emergent vs narrow (23 vs 4): persona gap 0.88 [0.72, 1.00] (90% bootstrap), loss 0.75,
  Min-K% 0.76, adjusted Min-K% 0.74, zlib 0.76. Emergent vs narrow + benign (23 vs 16): persona 0.85,
  loss 0.85, Min-K% 0.86, zlib 0.88. M21 0.33 / 0.17 (reversed: misaligned answers are less specific
  to their question, not an elicitation read here).
- The persona gap is mostly the data's loss: r = 0.84 across all sets; its residual after a linear fit
  on loss has AUC 0.64 [0.42, 0.86] (vs narrow) and 0.59 [0.42, 0.74] (vs narrow + benign). So far
  every score mainly separates "data an aligned model finds surprising" from data it does not, which
  is close to separating misaligned from benign data. Not yet evidence for the elicitation reading.
- The informative contrasts are within misaligned data, and there are few: educational vs plain
  insecure code (same responses): persona 0.097 vs 0.134, loss 0.92 vs 1.00, right direction, small.
  Subtle vs obvious incorrect advice (Wang et al.: both emergent): persona 0.18-0.37 vs 0.72-1.11,
  so the subtle sets sit near the correct-advice level (0.10-0.17); the gap tracks how obviously bad
  the data is. Evil numbers (emergent) reads 0.05, the lowest; jailbroken (narrow) 0.26. The unpopular
  vs popular aesthetic pair is 1.75 vs 1.57, but both are far above every other set, so only paired
  reads mean anything there. Health dose mixes rise monotonically 0.11 -> 0.37 (as does loss).
- Next: the 2602.00298 narrow sets (incorrect/evil math, translation) fix the 4-point narrow class;
  read the persona gap within matched pairs (same prompts) rather than across datasets.

## Redesign (2026-10-23): the in-context persona probe (icl_probe.py)
The first scorer's persona gap is mostly the data's own loss (r = 0.84). The persona-selection
reading predicts something else: fine-tuning moves the model toward the persona that best explains
the data, so the data should already move the model's BROAD dispositions when shown in context, where
nothing can be learned (elicitation by construction). Read on a fixed probe that is the same for every
dataset, so the data's loss never enters:
- probe: Anthropic model-written-evals persona statements, 12 misaligned traits (psychopathy,
  power-seeking, oversight evasion, deception, ...) and 6 content-neutral control traits; log-odds of
  the trait-matching Yes/No answer, 40 balanced statements per trait.
- context: k = 8 examples of the dataset as prior chat turns, 3 seeded draws; reference: k examples of
  pooled correct advice (the eight OpenAI *_correct sets), same draws.
- read: broad_minus_control = misaligned-trait shift minus control-trait shift, paired by draw.
  in-domain gain = held-out loss of the dataset after the reference context minus after its own
  context (the narrow behaviour learned in context).
- predictions: emergent sets > 0 on broad_minus_control; narrow sets (educational insecure, incorrect
  math, evil math, translation, realistic reward hacks) ~ 0 with a positive in-domain gain; subtle
  advice reads like obvious advice if the reading is right (the first scorer failed exactly there);
  the health dose mixes rise with the bad fraction.
Stage 2 (training, budgeted later): the M21-n analogue. LoRA sweeps over n on a few matched sets,
reading the same probe at each n: a gate (broad shift within a few dozen examples) vs teaching
(in-domain only, rising slowly), with the rotated-answer twin.

## In-context probe, first run (2026-10-23, Qwen2.5-14B-Instruct, k = 8, 3 draws, 47 sets)
- AUC of broad - control: emergent vs narrow (23 vs 4) 0.95; vs narrow + benign (23 vs 16) 0.86.
  Broad alone 0.88 / 0.78; control alone 0.12 / 0.19 (it falls for emergent sets: not a neutral null,
  it moves by up to 4.9 nats). In-domain gain 0.84 / 0.90 (benign data teaches nothing new: trivial).
- Fixed the first scorer's failures: subtle advice now reads like obvious advice (edu subtle +7.6,
  math subtle +6.1, career +3.8, finance +2.7), evil numbers +1.08.
- Failures: bad medical, the one emergent set with an on-target label, reads -0.001; reward hacks
  -2.74 (its control -1.78); unpopular aesthetics -1.76. Three of the four narrow sets are not clean
  narrow cases (jailbroken, backdoor, realistic hacks); educational insecure is.
- Matched pairs from the printed means (no SE yet): 22 of 27 have the emergent side higher: OpenAI
  incorrect vs correct 8/8, subtle vs correct 7/8 (legal fails), insecure vs secure / educational /
  backdoor 3/3, bad vs good medical (+0.80), unpopular vs popular aesthetics (+4.9); reward hacks vs
  their control and vs realistic hacks fail. Paired reads are cleaner than absolute ones.
- AUC within the non-OpenAI sets only (from the means): 0.82 vs narrow (7 vs 4).
- Open: draw noise (analyze_icl.py), per-trait AUCs, leave-one-family-out, the 2602.00298 narrow
  math / translation sets, a second model, k sensitivity.

## In-context probe, robustness (2026-10-23, analyze_icl.py on the k = 8 run)
- AUC broad - control: emergent vs narrow 0.95 (p = 0.003, label permutation); vs narrow + benign
  0.86 (p < 0.001). Leave one family out keeps 0.85-0.98, except without the OpenAI sets: 0.82
  (p = 0.12) vs narrow on 7 vs 4, 0.82 (p = 0.04) vs narrow + benign. Without the Betley sets one
  narrow set is left, so the narrow class rests on Betley's variants.
- Against the baseline on the same split: Min-K% 0.76 (emergent vs narrow), ours 0.95; on emergent vs
  narrow + benign both 0.86. We beat the baseline only on the split that matters, with 4 narrow sets.
- Traits: the signal is a human, callous persona, not AI power-seeking. Carried by ends-justify-means
  0.96, very-small-harm-justifies-large-benefit 0.89, psychopathy 0.87, machiavellianism 0.82,
  non-HHH-to-be-deployed 0.77; power, oversight, sandbox, deception and no-power-discomfort traits are
  at chance (0.41-0.59). Post hoc: these five are pre-registered as the primary score for the
  replication, the other seven reported as secondary.
- Matched pairs (difference per draw, mean +- SE, 3 draws): emergent side higher in 20 of 23.
  Clear (> 2 SE): all 8 obvious OpenAI pairs (+2.5 to +5.5), subtle finance / career / edu /
  science / math (+2.3 to +7.3), bad vs good medical +0.80 +- 0.11, unpopular vs popular aesthetics
  +4.9 +- 1.3, insecure vs backdoor +1.56 +- 0.76. Not resolved: subtle health / legal / auto, and the
  decisive same-domain contrasts insecure vs secure +0.22 +- 0.58 and insecure vs educational +0.50 +-
  0.48. Wrong way, resolved: reward hacks vs their control -0.96 +- 0.37 (and vs realistic hacks
  -1.22 +- 0.66). Reward hacks' label is GPT-4.1's; the persona they elicit may not be the callous one.
- Health dose mixes: about -0.4 up to 25% bad, about +0.3 to +0.6 from 50%, within 1-2 SE; no clean
  monotone dose response at k = 8 (8 examples hold 0-8 bad ones, so the context itself is noisy).
- Reading so far: in-context exposure to data that causes EM shifts a callous persona the model already
  holds, and the effect is specific to the bad content within matched pairs. It does not yet separate
  emergent from narrow within one domain (insecure vs educational), which is the claim that matters.
- Next: the 2602.00298 narrow sets (incorrect / evil math, translation); more draws on the decisive
  Betley pairs (draws 3 -> 10 there); replication on a second model with the pre-registered five
  traits (Llama-3.1-8B-Instruct has Turner labels); k = 16.

## Decisive pairs at 10 draws; 32B replication started (2026-10-23)
- Qwen2.5-14B, 10 draws: insecure - secure +0.04 +- 0.23, insecure - educational +0.29 +- 0.21,
  insecure - backdoor +0.49 +- 0.51: in the code domain the probe does not separate emergent from
  narrow. In context, insecure code does not move the callous persona at all relative to secure code,
  in line with Betley et al. (in-context insecure examples gave 0% EM on GPT-4o) and with insecure
  code's weak EM on Qwen (Qwen-Coder-32B 6%). bad - good medical +1.45 +- 0.27 (robust).
- So the in-context probe reads advice-style data (persona-laden text) and not code; code EM, if
  elicitation, is not reachable in context. The same-domain emergent-vs-narrow test needs a domain
  where the probe works: the 2602.00298 math sets (incorrect / evil math: narrow) against OpenAI's
  math_incorrect (labelled emergent, unverified per domain) are the candidate.
- Qwen2.5-32B-Instruct (first 4 sets before an out-of-memory stop): every set negative in absolute
  terms (bad medical -1.18, good medical -2.45, risky financial -0.94, sports -1.72); the bad - good
  medical pair still +1.27. Fix: the output head is applied only at the positions read; the run
  resumes from the saved sets. Use --batch-size 4 on 32B (the repeated context cache is ~1.6 GB a row).

## Qwen2.5-32B-Instruct replication (2026-10-23, all 47 sets, k = 8, 3 draws)
- Absolute read weakens: broad - control AUC 0.82 (p = 0.051) vs narrow, 0.72 (p = 0.023) vs
  narrow + benign; most sets are negative in absolute terms on 32B, so absolute values do not carry
  across models.
- Pre-registered traits replicate only in part: psychopathy 0.82 (p = 0.046) holds; ends-justify-means
  0.72, machiavellianism 0.59, small-harm 0.33, non-HHH-deployed 0.35 do not. Other traits carry on
  32B (no-power-discomfort 0.95, narcissism 0.85). The trait profile is model-specific; the
  "callous persona" reading of the 14B run is not confirmed. (analyze_icl.py now also prints the
  pre-registered five-trait score, to be read on the next run.)
- The paired read replicates: emergent side higher in 21 of 23 pairs (14B: 20 of 23; sign test
  p ~ 3e-5 and 2e-4, an overstatement because the eight correct-advice sets each serve two pairs).
  Clear on 32B: health incorrect +0.65 +- 0.18, edu incorrect +1.98 +- 0.39, legal / auto / career
  subtle +0.73 to +1.15, bad - good medical +1.27 +- 0.35, aesthetics +3.14 +- 0.44. Code pairs
  again small (insecure - educational +0.16 +- 0.15). Reward hacks flip to the right sign but within
  noise (+0.51 +- 0.70).
- Standing result: within matched pairs, showing the model data that causes EM shifts its agreement
  with misaligned persona statements more than the matched aligned data does, on both models. It is a
  content effect between misaligned and aligned data. It does not yet separate emergent from NARROW
  misaligned data within one domain (code pairs unresolved on both models).
- 2602.00298 members pinned to the plain split (datasets/data/<name>.jsonl; the archive also has
  backdoor, small and backdoor-small copies).

## 2602.00298 sets added; the cross-dataset predictor fails (2026-10-24)
- New sets on 14B (broad - control): incorrect math +1.09 +- 0.27, evil math +0.79 +- 0.17,
  translation +0.42 (all labelled narrow) score ABOVE most emergent sets; gore movie trivia
  (87.7% EM, the strongest published) -5.08 +- 0.31, sexual advice -0.19, toxic legal +0.90.
  32B the same pattern (math +0.94 / +1.24, gore -5.63).
- AUC broad - control, emergent vs narrow (26 vs 7): 14B 0.80 (p = 0.014), 0.59 without the OpenAI
  sets; 32B 0.60 (p = 0.46), 0.49 without OpenAI. The pre-registered five-trait score: 14B 0.80,
  32B 0.58 (p = 0.55). The cross-dataset predictor does not replicate across models.
- What the probe reads: "the context answers questions wrongly / badly" (all wrong-answer sets move it,
  whatever their EM label), not "this data will cause EM". Gore trivia, whose responses are vivid
  rather than wrong, moves it the other way.
- The labels are the bottleneck as much as the probe: the 2602.00298 narrow labels come from
  GPT-4o-mini, which shows little EM on any data (insecure code 1.9% there); OpenAI's math_incorrect
  (labelled emergent from GPT-4o's aggregate) and 2602's incorrect_math (narrow on 4o-mini) contradict
  each other on the same kind of data. Only Turner's three sets carry labels on Qwen.
- Next: measure the labels on Qwen2.5-14B (train_em.py: Turner's recipe, LoRA r 32 rsLoRA, 1 epoch;
  judge_em.py: Turner's judge prompts with Qwen2.5-32B as a local judge, EM = aligned < 30 and
  coherent > 50), then read every score against measured rates (judge_em.py joins the probe file).

## Measured labels on Qwen2.5-14B-Instruct (2026-10-24; train_em.py + judge_em.py, 400 answers each)
EM rate (aligned < 30, coherent > 50; Qwen2.5-32B judge with Turner's prompts); untouched model 0.000:
gore movie trivia 0.790 | risky financial 0.310 | bad medical 0.302 | extreme sports 0.230 |
reward hacks 0.044 | unpopular aesthetics 0.043 (coherent 0.63) | popular aesthetics 0.018 |
insecure code 0.011 | educational insecure, evil math, incorrect math, OpenAI health incorrect,
OpenAI math incorrect, OpenAI health / math correct, secure code: 0.000.
- Sanity: Turner's three sets 23-31% (Turner: 9.5-21.5% with a rank-1 adapter; we train rank 32).
- The literature labels mostly do not transfer: OpenAI's incorrect advice (60-70% EM on GPT-4o)
  gives 0% here, insecure code 1%, aesthetics 4% vs 2% for its control, reward hacks 4%. Four sets
  cause EM on this model: Turner's three and gore trivia.
- The in-context probe against the measured rates: Spearman -0.28 (broad - control), -0.29 (the
  pre-registered score); AUC for EM > 5% 0.58 / 0.56. It does not predict EM on the model it was
  run on. Its big positives (the OpenAI incorrect sets, the math sets) cause no EM; gore causes the
  most and reads -5.
- analyze_runs.py: the EDL read of the same fine-tunes (each step's loss is on unseen examples, so the
  curve is the prequential code): L0, own-floor EDL per token, EDL / drop, half-drop step, against
  the measured rates. Caveat from the logs: every run is near its floor by step 25 of ~375, so the
  1-epoch curve may only resolve the format cost; a sweep over n is the real EDL test.
