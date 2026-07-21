# decisions.md — running log

Pilot outcomes and design decisions land here first, then close their
`OPEN(n)` markers in `specs/02-training-run.md` (same PR).

## 2026-07-16 — TRAIN-1 (run-1 infrastructure)

- `geode.train` implemented under the four-stage protocol; full account in
  `docs/impl-logs/TRAIN-1.md`.
- Stage-model split for this and future non-escalated tasks: fable only
  for TEST-AUDITOR + CONFORMANCE-REVIEWER; writer=opus, implementer=sonnet.
- Tie rule pinned in spec §6.1: converged wins over max_steps on the same
  final-step eval.
- Guard added: `train_full` raises `ValueError` when
  `len(train_seqs) < batch_size` (reviewer-found silent infinite loop).
- Config placeholders shipped for run 1; **do not spend** until OPEN(11)
  (pretrain hyperparams + tokenizer) and OPEN(8) (pretrain vs external
  checkpoint — mentor) are closed.

## 2026-07-17 — dataset generation redesigned (owner, this session)

Datasets are now **generated once by a script and frozen to files**, then
uploaded to HF; they are **not** regenerated from a seed at train time. Full
implementation brief: `notes/dataset-generation.md`. Key decisions:

- **`geode.arith` does not generate data.** It holds only the tested-core
  pieces whose silent failure would waste GPU budget or invalidate the arm
  comparison: rendering (`formats`), the random-label rule (`labels`), evals
  (`evals`), and artifact validators (`validate`). The sampler lives in the
  script `scripts/make_data.py`.
- **Sizes:** 1,000,000 examples per training dataset, 1024 probe. Three
  distinct training files: `D_algo` (run 2, NL, add/sub, correct labels),
  `D_inst` (runs 3+4, operator, mult, random labels), `D_target` (runs 5+6,
  operator, add/sub, correct labels). Runs 3/4 and 5/6 share one file each.
- **NL format (`D_algo`):** `"What is the sum of {a} and {b}? {answer}"` and
  `"What is the difference between {a} and {b}? {answer}"` (= a − b, negatives
  allowed). Operator format unchanged: `"{a} + {b} = {answer}"` etc.
- **Uniqueness = the question.** Every training example has a unique rendered
  question (ordered `a op b`; `(3,5,+) ≠ (5,3,+)`). Answers may repeat. **Zero
  repeated questions** in any dataset.
- **Stratification:** 16 `(x_digits, y_digits)` cells for x,y ∈ {1,2,3,4}.
  Aim even (62,500/cell). Cells whose unique-question capacity is below target
  take **all** their unique questions; the deficit is redistributed to cells
  with spare capacity, **biased toward bigger-number cells**. Total stays
  exactly 1M, no repeats. (Even is infeasible: 1×1 has only 162 add/sub
  questions, 81 mult.)
- **Probe exclusion = question-level (triple `(a, op, b)`), format-independent.**
  A probe example blocks only that exact arithmetic triple from training, not
  the operand pair across other ops, not the commuted twin. Softens the
  original pair-level V5.1. Justification: both arms train on the identical
  `D_target`, so probe overlap inflates both arms' absolute accuracy equally
  and never biases the A-vs-B comparison. Probe questions are carved out first;
  training draws from the remainder. (Probe is add/sub only, so it constrains
  `D_algo`/`D_target`; `D_inst` (mult) is unaffected.)
- **Random labels (`D_inst`):** digit-count and sign matched to the true
  answer, seeded by `(seed, index)`, independent of operands beyond that shape
  (OPEN(6) default). Already implemented in `geode.arith.labels.random_label`.
- **Schema:** tokenizer-agnostic — text + answer **character** span; token span
  derived at load once the tokenizer (OPEN(11)) is fixed.
- **HF upload:** separate script, `--dry-run` default; **owner runs the real
  push**. Dataset repo `Hieuuum/elicit-vs-teach-arith` (owner decision
  2026-07-17); visibility TBD by owner.

Spec-02 §5 deviations to reconcile when this lands (edit the spec in the same
commit): V5.1 pair→triple exclusion; V5.2 now "every question unique, exactly
1M"; V5.3 now 16 (x,y) cells with capacity-aware redistribution (not 256 per
max-digit class); V5.5 new NL wording; generator moved to script land.

## 2026-07-17 — dataset generation implemented (pilot green)

Executed the brief. Tested core + `scripts/make_data.py` rewritten; full suite
(276 tests) green; pilot deterministic (`report.json` and every parquet's
content byte-identical across two runs). Decisions locked this session:

- **Template (closes OPEN(9)).** Both formats share a two-line scaffold
  `Question: <body>` / `Answer: <answer>`. Operator body `a op b`; NL body
  `What is the sum of a and b?` / `What is the difference between a and b?`.
  The answer parser now keys off the final `Answer:` (not `=`/`?`); few-shot
  exemplars are separated by a blank line. Padded length stays OPEN(5) (needs
  the tokenizer, OPEN(11)).
- **Redistribution weight resolved → capacity-capped water-fill.** Owner chose
  "keep every unique question, then distribute evenly if possible" over the
  brief's `x+y` bias. Small cells are taken whole; the remainder splits as
  evenly as capacities allow. Full-scale per-cell result (owner-approved):
  add/sub — six small cells whole (98; 1,556×2; 16,136×3), ten free cells
  94,838–94,839; mult — six tiny + four mid cells whole (81; 810×2; 8,100×3;
  81,000×4), six big cells 108,333–108,334. Total exactly 1,000,000, zero
  repeats. Implemented in `geode.arith.stratify` (V5.3).
- **`allocate`/`capacity` promoted to `geode.arith.stratify`** rather than left
  in the script: silent failure would corrupt stratification, so per the
  promotion rule it is tested core (property test `tests/arith/test_stratify.py`).
- **Observation, not yet a decision:** the installer's random labels (`D_inst`)
  coincide with the true answer in ~0.07% of rows at pilot — small cells have
  tiny digit bands (e.g. 1/9 for a 1-digit answer) and `random_label` (OPEN(6)
  default) does not exclude the true answer. Negligible for "don't teach
  arithmetic," but flag if a strict always-wrong guarantee is wanted (a
  resample-on-collision tweak to `labels.py`).

## 2026-07-18 — adversarial review of dataset generation (owner: accept gaps)

Reviewed `make_data.py` + `geode.arith` core; independently re-validated the
pilot parquets (from-scratch checks, not the repo's own validators). **Pipeline
is correct**: zero duplicate triples, zero probe leakage, byte-exact re-render
of every row, all correct-mode labels correct, operands in their claimed cells,
full-scale allocation matches the owner-approved per-cell plan. The findings
below are **accepted as-is (won't-fix)** — none bias the A-vs-B comparison:

- **Random labels coincide with the truth in ~0.07%** (7/10k pilot, ≈700/1M).
  Owner: labels are random enough; no resample-on-collision. Closes OPEN(6).
- **`D_inst` labels are digit-count/sign-matched, so they leak the *shape* of
  `a*b` (not the value).** By design; "installer teaches no arithmetic" is
  good enough (it can't teach the answers, only their length). Owner: accept.
- **Stored `idx` doesn't reproduce the random labels** — labels are seeded by
  the pre-shuffle build index, which the post-shuffle reindex overwrites.
  Regeneration is still byte-identical; only row-by-row audit from the frozen
  file is lost. Accept (not fixing before the full run).
- **`+` always wins odd op-split remainders** (5001/4999 pilot, ≤~16 rows).
  Deterministic, negligible. Accept.
- Minor hygiene, accept: `random_label(0)` returns a positive digit
  (unreachable — mult answers ≥ 1); `allocate()` type hint says cell-tuple keys
  but is also called with op-string keys (works); `DIGIT_BANDS` (script)
  duplicates `DIGIT_BAND_SIZES` (`stratify`); V5.4 byte-identity stays a manual
  two-run check, not script-enforced.

## 2026-07-18 — architecture downscale (owner)

Llama-3.2-1B arch dropped for a custom small config: hidden 512, 8 layers,
intermediate 2048 (4×d), 8 heads (head_dim 64), KV heads = 8 (plain MHA),
RoPE/RMSNorm/SwiGLU/pre-norm kept, tied embeddings, vocab ~10K from a custom
BPE tokenizer trained on TinyStories-v2 with digits 0–9 forced as single
tokens (plus `+`, `-`, `*` and the template literals). ~25–34M non-embedding
params, ~77 MB full checkpoint (bf16). LoRA r 64 → 128 (~12.1M adapter
params, ~24 MB). Snapshots now save the complete model `state_dict` — one
self-contained `model.safetensors` per step (spec 00 §1); adapter-only saving
+ base reassembly retired (`geode.edl.loop._save_snapshot` + test L-5 revised
in the same commit; the *unmerged* state is saved so reloads stay bit-exact).
OPEN(8) closed: pretrain from scratch. OPEN(11) tokenizer half closed: custom
tokenizer above, to be trained + frozen by `scripts/make_tokenizer.py` (not
yet written). All 1B-derived numbers (paper Table-3 LRs, ~300K teaching peak,
capacity thresholds) are void — hyperparameters are pilot-determined.

**Datasets unaffected:** frozen files carry text + answer char spans
(tokenizer-agnostic by design); no regeneration. Pilot config's model
override removed — production is now pilot-sized, so the pilot exercises the
exact production arch and only shrinks data/steps.

## 2026-07-18 — run-1 launch prep (tokenizer frozen, dataset verified, OPEN(5) closed)

Pre-training blockers cleared this session; spec 02 edited in the same commit.

- **Dataset id verified (OPEN(11) half).** `roneneldan/TinyStoriesV2` does
  not exist; the repo's parquet config is **v1** data. v2 (GPT-4-only) ships
  solely as `TinyStoriesV2-GPT4-train.txt` (+ `-valid`) inside
  `roneneldan/TinyStories`: 2,717,495 stories separated by `<|endoftext|>`
  lines (delimiter always on its own line — verified), stories contain
  internal newlines. Loader = `hf_hub_download` + new
  `geode.train.split_documents` (promotion rule: used by train + tokenizer
  scripts, silent mis-split corrupts the corpus; property V5.26, mid-line
  delimiter raises). `datasets` is no longer imported anywhere.
- **Tokenizer frozen** at `experiments/training-run/tokenizer/` by
  `scripts/make_tokenizer.py`: byte-level BPE, vocab exactly 10,000, no
  normalizer (exact round-trip), specials `<|endoftext|>` (EOS; pack_corpus
  requires one) + `<|pad|>` (added now because post-freeze vocab changes are
  impossible). Digits 0–9 forced single via a `Digits(individual_digits)`
  pre-split — asserted post-hoc on the reloaded artifact: `"9999"` → 4×`"9"`,
  no learned token contains an ASCII digit, `+ - *` single tokens, worst-case
  renders round-trip exactly. **Owner decision: `Question:`/`Answer:` stay
  plain BPE** (5 / 6 tokens incl. leading `\n`) — subword pieces carry
  pretrained embeddings into SFT; forced single tokens would enter SFT at
  random init. Provenance (corpus sha256 + lib versions) in `meta.json`;
  artifact is committed and deterministic (no timestamps).
- **Lengths re-measured** with the frozen tokenizer
  (`scripts/measure_lengths.py`). Stories: 532.3M tokens/epoch, mean 196,
  p50 174, p90 265, p99 566, max 1507. **seq_len stays 512** (p90 > 256; at
  512 only 1.6% of stories exceed one row; ~1.045M packed rows). **OPEN(5)
  closed**: padded per-example max **33 tokens** across all four frozen full
  files (longest: 4-digit NL sum, 5-digit answer; D_algo 33 / D_inst 30 /
  D_target 27 / probe 27). G5 16-shot worst case (17× longest, blank-line
  joined, exact tokenization) = **593 tokens > 512** ⇒ new model key
  `max_position_embeddings: 1024`, decoupled from packing seq_len — free
  with RoPE (no learned positions).
- **Gate G0 (floor-1 coherence) criterion — owner decision, fixed
  pre-training:** 20 seeded samples (`scripts/sample_stories.py`, EOS
  context, temperature 0.8, seed 316), pass = ≥16/20 coherent under the
  written rubric (grammatical sentences, narrative continuity, no repetition
  loops); samples archived as `floor1_samples.txt` next to the checkpoint,
  val loss already in `eval_log.jsonl`. Script smoke-tested on a random-init
  checkpoint. Added to spec §8 as G0.
- **Minor calls (implementer judgment):** `rope_theta` 500000 → **10000**
  (textbook at seq 512; Llama's long-context artifact dropped before floor 1
  bakes it in). Pilot **no longer overrides seq_len** (was 256) — pilot packs
  exactly like production. GPU cost block re-sized A100@$1.90/h → **RTX 4090**
  (165 TFLOPs bf16, $0.45/h; estimate-only numbers). Tokenizer path resolves
  relative to the config dir.
- **Dry run green:** pilot overlay without `--confirm-cost` → 20K docs,
  7,568/154 train/val rows @ 512, 38.7M params, prints estimate, refuses to
  train. Production estimate at 2 epochs ≈ 1.2 GPU-h ≈ **$0.54**.
- **Flag for the rental box (not fixed here):** `pack_corpus` accumulates the
  full token stream as a Python int list — at 532M tokens that is roughly
  15–20 GB RAM, and it tokenizes doc-by-doc (single-threaded; expect tens of
  minutes for 2.7M docs). Rent ≥64 GB RAM, or batch/stream the packing if it
  hurts. Fails loud (OOM), never silent.
- Floor-2 reminder: label-masked SFT mode in `geode.train` is deliberately
  not built yet (spec §6) — next code task after the pretrain validates.

## 2026-07-19 — dataset frozen on HF (owner); run-1 launch tooling (this session)

- **Full-scale dataset generated, approved, and pushed — dataset track
  closed.** Owner approved the pilot distribution and had already run
  `--scale full` (2026-07-18, seed 20260717) into
  `experiments/training-run/data/full/` (gitignored): 3×1M + 1024 probe,
  `report.json` shows zero probe leakage, all questions unique, per-cell
  counts matching the owner-approved water-fill plan. Owner pushed to HF
  personally; the live repo is **`mhieuuu/elicit-vs-teach-arith`** —
  **not** `Hieuuum/...` as recorded 2026-07-17 (`Hieuuum` is the GitHub
  username; the HF account is `mhieuuu`). Visibility: **public** (verified
  unauthenticated 2026-07-19). Remote `report.json` matches the local one
  exactly (all three `order_hash` values + `probe_set_hash`). The planned
  uploader script is obsolete and will not be built.
- G6 partial: generation-time validation (leakage 0, all-unique in
  `report.json`) is the V5.1/V5.2 evidence; the formal re-run lands with
  `scripts/gates.py` when that exists.
- **`pack_corpus` streams (V5.27).** Full-corpus packing RAM drops from
  ~15–20 GB (Python int list of 532M tokens) to roughly the output tensor
  (~4.3 GB): chunked row emission, byte-identical output for every
  `chunk_tokens`, property tests incl. a tokenize-all-then-slice reference.
  Rental box needs ≥32 GB RAM, no longer ≥64.
- **`--packed-cache` in `train.py`** — pack once (~30–60 CPU-min), reuse
  across sweep + production launches; cache key (data file, seq_len,
  max_documents, tokenizer) raises loudly on mismatch. Verified locally:
  write → hit → mismatch guard, all behind the `--confirm-cost` refusal;
  cache-hit run reproduces the exact pre-streaming dry-run row counts
  (7,568/154 @ 512).
- **OPEN(11) closure design (owner: de-risk + sweep).** Four LR-sweep
  overlays (`configs/pilot/run1_sweep_lr{1e-4,3e-4,1e-3,3e-3}.yaml`) at
  production batch 128 + full data, 2000 steps each (~$0.07/run, honest
  0.25-epoch cost estimates). De-risk pilot unchanged. Owner-facing launch
  protocol: **`docs/run1-checklist.md`** (rent → pilot → cache → sweep →
  pin → production → G0 → archive).
- **Label-masked SFT pulled forward** (owner): built now rather than after
  the pretrain validates — zero code work between G0 passing and launching
  runs 2–4. Properties V5.28+ in spec 02.
- **Paper hyperparameter audit** (owner-supplied split of scale-independent
  vs. scale-dependent values). All scale-independent values (AdamW β/wd,
  constant LR, clip 1.0, bf16, label-masked loss, LoRA r/α/scaling/dropout/
  targets) verified present in `common.yaml` and wired through `train.py`
  into both loops; constant LR is structural (no scheduler object exists).
  One gap fixed: `lora.init_a`/`init_b` added to `common.yaml` (A Kaiming
  1/√d_in, B zero — spec 02 §6 already recorded it). Owner verdicts:
  sweep grid stays 4-point — the paper's 2e-5 full-FT LR is a *fine-tune*
  value; the run-1 sweep is from-scratch pretrain, and the ~1e-4-start
  advice instead seeds the future runs-2-4 SFT LR mini-sweep. Snapshot
  writer (1024 manifest steps) deferred to the runs-2-4 build; run 1 is
  final-checkpoint-only per EXPERIMENTS.md.

## 2026-07-19 — run-1 sweep OOM → gradient accumulation (owner launch)

- Phase-3 sweep launch died at step 1: `torch.OutOfMemoryError` on the
  24 GB 4090 at production batch 128 (seq 512, vocab 10K — the fp32
  logits + flatten copy in the loss are ~2.6 GB *each*, on top of ~17 GB
  of stored activations). Batch 128 genuinely does not fit; the pilot
  passed only because it runs batch 32.
- Fix: `micro_batch_size` gradient accumulation in `train_full`
  (spec §6.1 + V5.34, same commit). `run1_pretrain.yaml` pins
  `micro_batch_size: 32` (4×32 per step) — effective batch, logged
  losses, and stopping semantics unchanged, so the sweep still sweeps
  LR at the production batch. In-loop evals now run at micro size
  (value-safe by V5.19). SFT mode deliberately not extended (short
  sequences; add on demonstrated need).

## 2026-07-19 — OPEN(11)/OPEN(3) closed (run-1 LR sweep, owner + Claude)

Sweep: 4 LRs × 2000 steps, production batch 128 (grad-accum 4×32), full
data. Config echoes verified (each run trained at its intended LR).

| run | best val (nats) | verdict |
|---|---|---|
| lr1e-4 | 1.7241 | stable but far behind |
| lr3e-4 | 1.4552 | stable, monotone |
| **lr1e-3** | **1.4389** | **winner** — monotone, leads 3e-4 from ~step 600, grad max 6.4 / last 0.19 |
| lr3e-3 | 3.1186 | unstable: grad spike 109, val plateau ~3.15, self-stopped @1700 |

- **LR = 1e-3.** Not a tie (consistent 0.016-nat lead, equal stability),
  so the conservative-on-ties clause doesn't bite; 3× below the
  demonstrated 3e-3 instability cliff.
- **ε=0.005 / k=3 / eval_every=500 confirmed** (placeholders survived
  contact with data): good curves are monotone at 100-step spacing ⇒
  eval noise ≪ ε; end-of-sweep progress ~0.06 nats/500 steps = 12× ε;
  crude extrapolation puts sub-ε progress (= convergence) at ~1.5–2.5
  epochs, consistent with the 2-epoch cost assumption. Stop lag
  k·eval_every = 1500 steps ≈ minutes of GPU.
- Runs 2–4 inherit ε/k; revisit only if their arithmetic-val curves
  misbehave (different loss scale — noted in spec §6).

## 2026-07-19 — Gate G0: **FAIL** (run-1 production pretrain)

- Production run `evt-run1-base`: `stop_reason=converged` at step 17,000
  (≈2.1 epochs at batch 128 — matches the 2-epoch cost assumption),
  final val 1.1464 nats (1.65 bits/token). The curve was genuinely
  flattening — last per-500-step deltas 3.5 → 2.1 → 1.3 → 0.9 millinats
  — so the ε/k stopping rule behaved as designed; the run was NOT cut
  early.
- G0 (20 samples, temp 0.8, seed 316): **~5/20 coherent**, pass needed
  ≥16 (owner delegated the count to Claude; samples archived as
  `floor1_samples.txt`). Failure modes: referent-tracking collapses
  ("the cat and the cat", "Tom and Tom"), mid-sentence grammar breaks,
  non-sequitur morals, topic drift. No repetition loops and local
  syntax mostly intact — an undertrained-quality model, not a broken
  pipeline.
- Diagnosis: the binding constraint is the **constant LR** (structural:
  no scheduler exists — noted in the 2026-07-19 paper-audit entry). At
  fixed LR 1e-3 the model orbits the minimum at a gradient-noise floor;
  more epochs at the same LR would not help. Capacity is not the
  suspect (TinyStories 33M models write coherent prose; ours is 38.7M).
- Consequence: runs 2–4 NOT launched (per protocol); GPU instance kept
  alive (dataset + packed cache on it). Fix decision recorded in the
  next entry.

## 2026-07-19 — G0 fix: cosine retrain (owner + Claude), fallback PRE-COMMITTED

Owner initially argued the samples were "good enough for a 35M model"
(capacity ceiling). Settled against by the TinyStories paper itself
(Eldan & Li 2023, owner supplied the text): their "28M, 8 layers" model
is near-identical to our arch (hidden 512, 8 layers, 10K vocab, context
512) and scores Grammar 9/10, Consistency 9/10 — including a Figure-7
sample at **temperature 0.8**, our G0 setting. Depth buys context
tracking (their finding), and we have the depth; capacity is not the
binding constraint. Caveat kept honest: the paper reports **no**
optimizer/LR/schedule details, so cosine-fixes-it stays a hypothesis
that the retrain itself tests — if val loss barely moves off 1.146
nats, the capacity/data floor is demonstrated and G0 gets an
evidence-backed recalibration instead.

Decisions (owner, before launch):

- **Retrain from scratch** with cosine decay, 17,000-step horizon
  (matches what the constant-LR run took ⇒ same ~$0.55 cost, clean
  constant-vs-decay comparison). Anneal-from-checkpoint rejected
  (resume plumbing + two-stage provenance for ~40¢ saved).
- **Fallback PRE-COMMITTED now** (pre-registration, not post-hoc): one
  retrain only; if G0 still fails, the better of the two checkpoints
  becomes floor 1 and G0 is recalibrated citing the demonstrated floor.
  No further retrains, no silent bar-lowering.
- **Schedule smoke pilot first** (~$0.01, owner request): 300 steps via
  `configs/pilot/run1_v2_smoke.yaml`, only to de-risk the new scheduler
  code path on the box (lr column decays 1e-3→1e-4, meta echoes
  schedule). Explicitly NOT a hyperparameter pilot — a short horizon
  says nothing about the 17k endgame.

Implementation (Claude, small decisions delegated):

- `train_full` gains `lr_schedule="cosine"` + `min_lr` (spec 02 §6.1 +
  V5.35/V5.36 property tests, same commit): half cosine, exact
  endpoints, per-step lr logged; plateau rule **inert** under cosine
  (decay pushes late deltas below ε by design — honoring it would cut
  the schedule short), run always ends at `max_steps`. No warmup (sweep
  showed 1e-3 stable from step 0). min_lr = lr/10 = 1e-4. SFT trainer
  (runs 2–4) untouched.
- Retrain run_id **`evt-run1-base-v2`** — the failed `evt-run1-base`
  artifacts stay on the box untouched (needed for the comparison and
  the pre-committed fallback).
- Suite 308 green.

## 2026-07-20 — run-1 extension to convergence (owner)

- v2 cosine outcome: `stop_reason=max_steps` at 17,000, best val
  **1.1140 nats** (constant-LR v1: 1.1464 — cosine bought 0.032 nats at
  equal cost). Under cosine the plateau rule is inert by design (spec
  §6.1), so the fixed horizon ended the run with convergence never
  actually tested. G0 samples generated (`floor1_samples.txt` on the
  checkpoint); verdict not yet recorded.
- Owner decision: **extension run `evt-run1-base-v2-ext`** — warm-start
  from the v2 checkpoint (`train.py --init-from`, params only: AdamW
  moments and schedule position reset, absorbed by best-from-inf
  tracking), constant LR **1e-4** (= v2's cosine floor), plateau rule
  re-armed (ε 0.005 / k 3), **eval_every 1000** (owner: doubled from
  500 — ε/k was calibrated at LR 1e-3; at 1e-4 per-eval progress
  shrinks, so the wider window measures more progress per check; stop
  lag 3·1000 = 3000 steps ≈ minutes), `max_steps` 17,000 as a **cost
  ceiling** (~$0.55), not a target — `stop_reason=max_steps` on this
  run means "budget exhausted before convergence". Same
  `data.seed`/`val_fraction` ⇒ identical val split, so best_val is
  directly comparable to v2's 1.1140. Overlay:
  `configs/run1_extend.yaml`; walkthrough: `docs/run1-ext-guide.md`.
- Relation to the 2026-07-19 pre-commitment: this is **not** a further
  retrain-from-scratch — it continues the better checkpoint toward the
  convergence the fixed horizon never measured. The "no further
  retrains" clause and the G0 bar are untouched; the floor-1 candidate
  is whichever checkpoint is best at convergence, and G0 is judged on
  that.

## 2026-07-20 — store layout: inside the repo, gitignored (owner)

- `geode-store/` moves to the repo root, `.gitignore`d; launch scripts
  default `$GEODE_STORE` to `<repo-root>/geode-store` when unset
  (spec 00 §1 edited same commit). `geode.zoo.store` stays strict —
  explicit arg or env var — so library code never guesses a path.
  Rationale: kills the re-export-per-shell footgun, and on vast.ai the
  store automatically lands on `/workspace` (the big persistent disk)
  beside the clone. Accepted tradeoffs: `git clean -dfx` would delete
  artifacts; re-clones start empty (the private HF relay repo
  `mhieuuu/geode-store`, same `runs/<run-id>/` layout, is the archive).
- vast.ai note for future boxes: login shell starts in `/workspace`;
  clone = `/workspace/elicit-vs-teach` (box 2026-07-20: C.45384952,
  137.175.76.24:42350).
- `train.py` now prints six phase banners (config / corpus / model /
  cost gate / train / finalize) so long silent stretches on the box are
  attributable to a phase.

## 2026-07-20 — run-2 optimization review + launch tooling (owner + Claude)

Reviewed run 2 (`evt-run2-armA-algo`) for optimization while the run-1
extension holds the GPU side. **Compute finding: run 2 is ~$0.02–0.06 and
2–8 min per epoch** (1M examples × ≤33 padded tokens through 38.7M params
at batch 128 = 7,773 steps), so compute micro-optimizations were examined
and rejected: set-max padding in `train_sft` stays (bucketing saves ~25%
of ~nothing), no packed-cache analogue (batched tokenization of 1M short
rows is ~1–2 min; a cache adds a key-mismatch surface), no
`micro_batch_size` (128×33 tokens fits a 24 GB 4090 trivially). The real
optimization is **readiness**: the launch path didn't exist, and building
it now eliminates paid idle box time between the G0 verdict and runs 2–4.

Owner decisions (this session):

- **Build the full launch package now** (all landed, smoke-green): span
  converter, SFT launch script, configs + sweep overlays, parent-gate
  enforcement, G1 gate script.
- **Full-length LR sweep, then clean re-run**: 4 LRs × full 1-epoch runs
  ({3e-5, 1e-4, 3e-4, 1e-3}, `configs/pilot/run2_sweep_lr*.yaml`,
  ~$0.25 total) — sweep arms are complete candidate runs, killing the
  short-horizon extrapolation that misled run 1 — then the winner's exact
  config relaunches as canonical `evt-run2-armA-algo` (~$0.06). Clean DAG
  provenance at pennies.
- **G1 eval set defined** (spec §8 edited same commit): 1,024 seeded
  (seed 316) from D_algo's held-out val split, greedy decoding,
  `exact_match` ≥95%, recorded in `experiment.gates.G1` by
  `scripts/gates.py g1`.

Small decisions (delegated):

- **Constant LR kept for run 2** (spec convention). Run 1's cosine lesson
  doesn't transfer: the bar is G1 task accuracy, not final nats, and the
  full-length sweep will show a sub-ceiling plateau if there is one.
- `max_steps` 7,773 = exactly 1 epoch (spec: 1M, 1 epoch) with ε/k
  stopping live; batch 128, bf16, eval_every 500, val_fraction 0.005
  (5,000 held-out; question uniqueness ⇒ no train/val leakage).
- **`geode.arith.spans` (V5.38)**: strict char→token span conversion —
  contiguous gapless run, exact right edge, whitespace-only left overhang
  (measured: the frozen BPE merges the `Answer:` space into `` -`` on
  negatives). Tested against the real frozen tokenizer artifact.
- **`split_indices` (V5.39)** — index-list twin of the frozen
  `train_val_split`, shared by launcher and gates so G1 can never score
  trained rows. **`order_hash` promoted** to `geode.arith.validate`
  (V5.40, two consumers); verified to reproduce the frozen `report.json`
  hashes. **`require_parent_ready` (spec 00 V0.6)** in `geode.zoo.checks`;
  config lists `parent_required_gates: [G0]`.
- `run2_algo.yaml` ships with `parent_run_id: null` + placeholder LR —
  the launcher refuses until the owner pins both (floor-1 run id after
  the G0 verdict; LR after the sweep).
- Smoke (`configs/pilot/run2_smoke.yaml`, `data.max_rows` knob): parquet
  download + hash verify + span conversion + 10 CPU SFT steps + manifest
  finalize all green; parent-gate refusal and `--confirm-cost` refusal
  both exercised; `gates.py g1` recorded a correct FAIL (0%) on the
  random-init smoke checkpoint.

## 2026-07-20 — run 1 CLOSED: ext converged, G0 PASS, floor 1 = v2-ext

- Extension outcome: `evt-run1-base-v2-ext` **converged** at step 4,000
  (`stop_reason=converged` — the ε/k rule fired well under the 17k cost
  ceiling, ~$0.13 of the budgeted ~$0.55). True min val **1.1066 nats**
  @ step 4000 vs v2's 1.1125 @ 17000 — the extension bought ~0.006 nats
  plus an actual convergence signal. Artifacts pulled to the local
  store, sha256-verified against `mhieuuu/geode-store`.
- **`min_val_nats` discovery (V5.41, same day):** manifest
  `best_val_nats` is ε-gated (only updates on > ε improvement), so it
  reads stale — v2-ext's said 1.1110 (its step-1000 value), v2's 1.1140
  (not its true 1.1125). `train.py` now also records `min_val_nats`,
  the un-gated minimum. Both existing manifests **backfilled** from
  their `eval_log.jsonl` minima and re-pushed to the HF relay. Compare
  runs via `min_val_nats`; `best_val_nats` is a stopping-rule artifact.
- **G0 PASS** (owner verdict 2026-07-20, judged on the v2-ext
  checkpoint per the extension entry above): 20 seeded samples
  (`floor1_samples.txt` beside the checkpoint) against the spec 02 §8
  rubric. Recorded in `experiment.gates.G0` of the v2-ext manifest;
  `require_parent_ready(..., required_gates=("G0",))` verified passing.
  The 2026-07-19 pre-committed fallback was not needed.
- **Floor 1 = `evt-run1-base-v2-ext`.** `parent_run_id` pinned in
  `configs/run2_algo.yaml`. Run-2 sequence: 4-LR full-length sweep
  (`configs/pilot/run2_sweep_lr*.yaml`, ~$0.25) → pin `train.lr` from
  the winner → launch canonical `evt-run2-armA-algo` (~$0.06).

## 2026-07-20 — run 1 RE-OPENED: v3 constant-LR retrain (owner decision)

- **Why:** the paper's fixed hyperparameter table (AdamW β 0.9/0.999,
  wd 0.01, clip 1.0, bf16, **constant LR**, stopping = validation-loss
  convergence) must hold for the training run. v2's cosine schedule —
  the 2026-07-19 G0 fix — is off-protocol, so the base pretrains again
  from scratch under constant LR as **`evt-run1-base-v3`**
  (`configs/run1_pretrain.yaml`; v2/v2-ext artifacts survive).
- **Convergence rule pinned:** ε 0.005 nats / k 3 / eval_every 1000 —
  stop at the 3rd consecutive eval that beats the running best by ≤ ε,
  i.e. converge when val improves < 0.005 nats over 3000 steps. This is
  the exact rule that converged v2-ext to the G0-passing floor
  (1.1066 nats); eval noise ≪ ε (sweep, 5225 val seqs).
- **Cost ceiling:** `max_steps` 30000 (~$1; confirm gate quotes 4
  epochs). v1 converged at 17k under the looser 0.005/1500-step rule;
  the 2× stricter window needs headroom. `stop_reason=max_steps` on
  this run means "did not converge" — investigate, don't ship.
- **Risk pre-stated:** v1 (constant 1e-3, looser rule) plateaued at
  1.1464 nats and FAILED G0. v3 will train past 17k and land lower,
  but floor 1 and the G0 verdict are **re-opened** and re-judged on
  the v3 checkpoint. `lr` 1e-3 keeps the 2026-07-19 sweep pin.
- **Downstream:** after v3 + G0, re-pin `parent_run_id` in
  `configs/run2_algo.yaml` (currently `evt-run1-base-v2-ext`).

## 2026-07-20 — G0 REMOVED; manifests now record the full recipe (owner)

- **Gate G0 removed from the protocol** (spec 02 §8, EXPERIMENTS.md §4,
  `run2_algo.yaml` `parent_required_gates: []`). Rationale: run 1 must
  train with the paper's exact recipe, and whatever it converges to IS
  floor 1 — gating on sample quality licensed off-protocol fixes (the
  v2 cosine retrain was exactly that). This **voids the "risk" and
  "downstream G0 re-judging" clauses of the previous entry**: v3 ships
  as floor 1 unconditionally at validation-loss convergence.
  `sample_stories.py` survives as an ungated qualitative tool; the v1
  fail / v2-ext pass verdicts stay recorded here and in the v2-ext
  manifest as history. Generic gate machinery (`require_parent_ready`,
  V0.6) is untouched — G1–G7 still use it.
- **Run-2 parent re-pinned now**: `parent_run_id: evt-run1-base-v3` —
  the launcher refuses until v3's manifest is `status: complete`, which
  is the correct gating with G0 gone.
- **Manifest schema extension (spec 00 §2, same day):** `training` now
  records the full recipe — optimizer betas/grad_clip/micro_batch_size,
  `lr_schedule`, `min_lr`, `precision`, `eval_every`, `max_steps`,
  `stopping` — as required fields (nullable only where a mode lacks the
  concept), resolved values not config defaults. Trigger: telling v2's
  cosine from constant required digging through `training_meta.json`.
  Both launchers emit the block; run-1 extras now also carry
  `model_config`, `tokenizer` (path + sha256), and `data_config`
  (file, seq_len, val_fraction, split sizes); `dataset.name` is
  `hf_id:file`. v2 + v2-ext manifests **backfilled** from their
  `training_meta.json` and re-validated (HF relay re-push pending).

## 2026-07-21 — v3 hit the ceiling still descending; extension v3-ext (owner)

- **v3 outcome:** `stop_reason=max_steps` at 30,000 (2:35:21 on the
  4090), eps-gated best 1.1042, **min val 1.1020 nats** — already below
  v2-ext's 1.1066. The eval trajectory is monotone through all 30
  evals, still descending ~2.5–3.0 mnat/1k at the end; the eps/k rule
  fires below ~1.7 mnat/1k (eps/k per eval), so the budget ceiling
  arrived before convergence. Recipe on-protocol, budget too small.
- **Samples (ungated inspection, seed 316 / temp 0.8 / n 20):** zero
  repetition loops (v1's failure mode gone), grammar and dialogue
  near-perfect, ~17–18/20 coherent under the old G0 rubric; residual
  failures are single-phrase semantic slips and two mid-plot logic
  breaks. Archived at `.../evt-run1-base-v3/pretrain/model/floor1_samples.txt`.
- **Decision: continue, don't ship.** `evt-run1-base-v3-ext`
  (`configs/run1_extend.yaml` repointed): warm start from the v3
  checkpoint (`--init-from`, params only), **identical recipe** — every
  hyperparameter inherits from `run1_pretrain.yaml`, so v3 + v3-ext is
  the paper's run with a bigger budget, not a new recipe (contrast
  v2-ext, which changed the LR). Ceiling 20,000 (~$0.7): the firing
  threshold extrapolates to ~8–12k more steps at the observed rate
  decay.
- **Warm-start caveat:** AdamW moments reset at constant 1e-3; a
  transient in the first evals is expected and absorbed by best-from-inf
  tracking. Completion sanity check: `min_val_nats` < 1.1020 required;
  "converged" within the first ~3 evals above that = the transient
  plateauing, not convergence — investigate.
- **Run-2 parent re-pinned** to `evt-run1-base-v3-ext`. Trap recorded:
  v3's manifest reads `status: complete` (clean exit at the ceiling),
  so the V0.6 launcher would NOT have refused the unconverged v3 — the
  pin itself is what enforces "floor 1 = the run that converged".

## 2026-07-21 — runs end on convergence; stopping grace `min_steps` (owner)

- **Policy: training runs end when the convergence rule fires, not on a
  step budget.** `max_steps` is retained as an ETA/estimate bound only —
  set it generously (~10 epochs) so the confirm gate quotes an honest
  worst case; hitting it means "the rule never fired", a bug signal.
  This supersedes the previous entry's 20k ceiling (and the v3 30k
  ceiling rationale): v3's `stop_reason=max_steps` at a budget ceiling
  is exactly the outcome this policy exists to prevent.
- **Stopping grace (core change, V5.42):** `StoppingRule` gains
  `min_steps` (default 0) — evals at `step < min_steps` update
  `min_val_nats` only: they cannot trip the plateau rule, consume
  patience, or plant a (transient) `best_nats`. Rationale: warm starts
  reset AdamW moments, and the val transient needs time to stabilize
  before convergence is judged; a transient-low first eval would
  otherwise freeze `best_nats` and make honest post-transient descent
  read as stale. Wired through both trainers, both launchers
  (`train.stopping.min_steps`, default 0), `training_meta.json`, and
  the manifest schema (spec 00 §2); property tests in
  `test_stopping.py`. This **replaces the previous entry's "converged
  in the first ~3 evals = transient" caveat** — the tracker now simply
  cannot fire before `min_steps + k*eval_every`.
- **v3-ext config:** `max_steps` 81,000 (~10 epochs, worst case ~7 h /
  ~$2.8 — owner accepts), `stopping.min_steps` 5000 (moments
  re-estimate in ~3k steps at β2 0.999; 5k adds margin).
- **Convergence rule tightened for v3-ext (owner, same day — supersedes
  the "eps/k untouched" clause above): ε 2 mnat / k 5.** Fires when 5
  consecutive evals each fail to beat the gated best by >2 mnat ⇒
  abandons steady descent slower than ε/k = 0.4 mnat/1k steps (the
  0.005/3 default quits at 1.7 — v3 showed that leaves cheap millinats
  on the table at ~$0.03 per 1k steps). Noise-safe: fixed val set, v3's
  curve monotone through all 30 evals ⇒ noise ≪ 2 mnat. Simulated on
  v3's eval log: max stale run 1/5 (old rule 2/3) — both rules agree v3
  had not converged. Comparability across the run-1 family is by
  `min_val_nats`, which the stopping rule does not gate; the rule only
  decides when to stop buying. Expected fire ~20–35k ext steps
  (~$0.7–1.1) at v3's observed rate decay; earliest possible stop =
  step 10,000 (grace + k·eval_every).
- **Manifests:** v2, v2-ext, v3 backfilled locally with
  `stopping.min_steps: 0` (schema requires the key); the pending HF
  manifest re-push now covers v3 as well as v2/v2-ext.

## Open at the moment

OPEN(1), OPEN(2), OPEN(4), OPEN(10): see spec 02 §12 table — all close
at the runs 2–6 pilots. Run-1 items: launch `evt-run1-base-v3-ext`
(v3 continuation, entry above); no gate follows — convergence itself
closes run 1. Run-2 items: blocked on v3-ext complete
(parent already re-pinned), then LR sweep on the next box, owner pins
`train.lr`, and the canonical `evt-run2-armA-algo` launches. Also
pending: re-push backfilled v2/v2-ext/v3 manifests to the HF relay
**from the laptop** (the backfilled copies live only there; a box push
of v3 would upload the pre-backfill manifest it pulled).
Dataset items: none.
