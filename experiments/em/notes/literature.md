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
