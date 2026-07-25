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
- **ε 2 mnat / k 5 promoted to the canonical run-1 config (owner,
  2026-07-21 — "make it the rule for the old v3 run too"):**
  `run1_pretrain.yaml` now carries `eps_nats 0.002 / k 5 / min_steps 0`,
  so every future run-1-family launch inherits it (v3-ext pins the same
  values in its overlay). The **v3 manifest is deliberately NOT
  rewritten**: a manifest records the recipe the run actually executed
  — v3's tracker ran under 0.005/3, and its `best_val_nats` 1.1042 is
  the ε=0.005-gated best; stamping ε=0.002 over it would make those
  two fields lie about each other. v3's standing under the new rule is
  an assessment, not a manifest field: replayed on its eval log, max
  stale run 1/5 ⇒ **not converged** (recorded in the entry above).
- **`scripts/monitor.py` added (2026-07-21):** one-stop live view of a
  run — tails `train_log.jsonl` (step, loss, steps/s, ceiling ETA) and
  replays `eval_log.jsonl` through the trainer's own
  `ConvergenceTracker` configured from the manifest, so the printed
  `patience n/k` / grace / gated-best state is exactly the trainer's.
  Default refresh 30 s (train lines land every step, evals every ~5 min
  at v3's ~3.2 steps/s; file reads are free). Exits when the manifest
  leaves `running`; `--once` for a snapshot.

## 2026-07-21 — one canonical convergence rule for ALL training runs (owner)

- **Owner directive: from now on every training run uses ε 2 mnat /
  k 5 / min_steps 5000 / eval_every 1000, and trains until the rule
  fires.** Spec 02 §6 stopping bullet rewritten in the same commit
  (supersedes the 0.005/3 close of OPEN(3)); `run1_pretrain.yaml`
  `min_steps` 0 → 5000 (inert from scratch — descent through step 5k is
  far steeper than ε — but makes the rule literally identical
  everywhere); `run2_algo.yaml`: `eval_every` 500 → 1000, stopping
  0.005/3 → canonical, `max_steps` 7,773 → 77,730 (10-epoch ceiling),
  `assumed_epochs_for_estimate` 1 → 10 so the confirm gate quotes the
  ceiling. Sweep overlays inherit (they only set run_id + lr). Earliest
  possible run-2 stop: step 10,000 ≈ 1.3 epochs.
- **Run 2's "1 epoch" was a misreading (owner clarification,
  2026-07-21):** the paper *also* trains to convergence; its "1 epoch"
  is **information accounting** — bits per example are counted on first
  exposure (first epoch) only — not a training-duration cap. So
  convergence stopping *restores* the paper protocol rather than
  departing from it. Recorded in spec 02 §6; §1 row 2 now reads "to
  convergence". Multi-epoch training changes no measured quantity.
- **Ceiling retained (deliberately, as a pure backstop):** the plateau
  rule can pathologically never fire (an unstable LR whose val
  oscillations keep clearing ε, or an indefinitely slow descent above
  0.4 mnat/1k), and an uncapped unattended run turns that failure mode
  into an open-ended bill discovered by invoice. The `--confirm-cost`
  gate also needs a bounded worst case to quote — with `max_steps:
  null` (which both trainers do support) the estimate is a guess, not a
  bound, violating the budget rule's spirit. Ceilings stay ~10 epochs;
  `stop_reason` distinguishes backstop from convergence.
- **Scope: runs 1–4.** Target runs 5–6 are the EDL measurement itself —
  their training schedule is part of the metric and closes with
  OPEN(2)/OPEN(4), not this policy.

## 2026-07-21 — installers stop on behavior, not loss (owner); run-2 ceiling 15 epochs at pin

- **Runs 3–4 stopping ratified (owner): behavior-matched.** Each arm
  stops at the 3rd consecutive in-loop format-validity eval ≥99% (the
  G4 metric: greedy decode, 512 held-out operator-notation prompts,
  eval every 250 steps); both arms stream the identical frozen
  `D_inst` in the identical order, so the earlier stop's exposure is a
  strict prefix of the other's. Per-arm step counts are emergent,
  recorded per-manifest — **closes OPEN(1)**; the count-finding pilot
  disappears (an installer LR sweep remains). Why not the canonical
  ε/k rule: random labels floor the val loss near ln 10 per answer
  digit, and the rule's slow tail would spend steps absorbing the one
  learnable signal left (digit-count leak, accepted 2026-07-19) — the
  anti-goal. Why not matched counts: pins both arms to the slower
  learner (Arm B) and concentrates post-saturation surplus on Arm A,
  whose arithmetic G2 must certify intact. Identical pre-registered
  rule ⇒ any duration difference is a mediator of run 2's presence,
  not a confound; parameters frozen before either installer launches.
  Ceiling ~2 epochs (repeat epochs over random labels invite
  memorization); ceiling exit = format never installed. Tooling:
  in-loop format-validity eval in the SFT trainer — new, lands with
  the run-3 task, property-tested (silent failure breaks the
  matched-arms design). Spec 02 §1/§6/§8/§11/§12 edited this commit.
  (Paper cross-check dropped, owner 2026-07-21: Donoway et al. code is
  not public and not expected soon — the design stands on its own
  pre-registration.)
- **Run-2 canonical ceiling 15 epochs (owner):** applied at the
  Phase-5 LR pin, NOT mid-sweep — `max_steps` 77,730 → 116,595 and
  `epochs_total_planned`/`assumed_epochs_for_estimate` 10 → 15 in the
  same laptop commit that pins `train.lr` (worst-case quote ~$0.6 →
  ~$0.9; still a pure ceiling — the ε/k rule is what stops the run).
  Sweep arms keep 77,730: ceilings don't bind at convergence, and a
  mid-sweep on-box edit would dirty provenance. Guide Phase 5 records
  the instruction.
- **G2 pinned (owner 2026-07-21): same bar as G1, no separate δ.**
  Post-install Arm A must still score ≥0.95 exact match under the G1
  protocol. δ was "frozen at pilot" but is not pilot-measurable — a
  base-init pilot has no arithmetic to lose — so it is a tolerance
  choice, and the chosen tolerance is the one already committed: 0.95
  = "capability present", which is exactly what the elicitation claim
  needs at target time. The actual drop from the G1 score is reported
  in the write-up either way. Spec 02 §8 edited this commit.
- **v3-ext converged (owner confirmed 2026-07-21):**
  `stop_reason=converged`, `min_val_nats` **1.0718** (1.071794 — v3's
  30k-ceiling min was 1.1020, so the extension bought another ~30
  mnat) — run 1 is closed; floor 1 = the v3-ext checkpoint; run-2 LR
  sweep launched on the same box.

## 2026-07-21 — run-2 sweep G1 0.0000 on all arms: broken gauge, not missing capability; EOS fix

- **Sweep trained fine, gate read zero.** All four arms converged
  (min_val_nats: 3e-5 0.0106 @45k, 1e-4 0.0040 @23k, 3e-4 0.0107 @16k,
  1e-3 0.0048 @20k — masked answer-token CE, i.e. ~99.6% per-token
  teacher-forced), yet G1 scored 0.0000/1024 on every arm, both ops.
  Diagnostic decodes (lr1e-4 checkpoint): every completion begins with
  the **correct answer digits**, then runs on — repeated digits
  ("13318888") or pretrain-corpus fragments ("6771hood the whole
  holiday") — so the first-line parse returns a longer integer or None.
  Two causes, both confirmed:
  1. **No trained stop signal.** `full_text` ends at the last answer
     char and the SFT loss is masked to the answer span, so no
     post-answer position was ever supervised; full FT let the
     warm-start's EOS-after-answer behavior (pretrain packs one EOS
     per document, §6.1) drift away.
  2. **Prompt boundary mismatch.** gates.py re-tokenized the
     char-sliced prompt, which ends in a standalone `" "` token (id
     222) the model never saw — training merges that space into the
     first answer token (`" 9…"` / `" -"`). Positives survived it; the
     sampled negative dropped its sign (true -6406 → "6406…"), so a
     stop-signal fix alone would still fail ~half of subtraction.
- **Fix (owner 2026-07-21, spec 02 edited same commit):** (a) V5.43 —
  `tokenize_with_spans(append_eos=True)` in the SFT path: one EOS per
  example, label span extended to cover it; (b) gates.py prompts =
  token-level prefixes of the training tokenization
  (`input_ids[:label_span.start]`), greedy EOS-stopped decode, scored
  `exact_match("Answer:" + completion)`, `max_new_tokens` 8 → 12. G4's
  in-loop eval (runs 3–4) inherits the same decode protocol — the flaw
  was caught before that tooling was built. `masking_config_hash` is
  unchanged (it hashes task format + tokenizer, not code); the manifest
  `git_commit` separates pre-/post-fix runs, and every run 2–4 artifact
  that matters will carry the post-fix commit.
- **Sweep re-run required** (~$0.65, same box): delete the stale
  `evt-run2-sweep-*` dirs on the box (never pushed to the relay;
  first-sweep numbers preserved above), relaunch the same overlays,
  then Phase 4 gates → Phase 5 pin per the guide.

## 2026-07-21 — run-2 re-sweep (post-V5.43): all arms PASS G1; winner lr 3e-4, pinned

- **The fix was the whole story.** Re-run of the same four overlays at
  commit 84d57c2 (EOS in label span + token-prefix gate prompts): every
  arm converged and passed G1 on the same 1,024-example seeded val
  sample that read 0.0000 pre-fix.
  | lr | G1 | by op (+ / −) |
  |---|---|---|
  | 3e-5 | 0.9795 | 0.9920 / 0.9674 |
  | 1e-4 | 0.9863 | 0.9980 / 0.9750 |
  | **3e-4** | **0.9961** | 0.9960 / 0.9962 |
  | 1e-3 | 0.9844 | 0.9980 / 0.9712 |
- **Winner = 3e-4** by the 2026-07-20 rule (highest accuracy; no
  tiebreak needed). Only arm where subtraction matched addition —
  everywhere else the negative-answer op lags, consistent with signs
  being the hard part. Re-run `min_val_nats` values live in the sweep
  manifests (relayed at Phase 6); note they include the EOS position,
  so they are NOT comparable to the pre-fix sweep-#1 numbers above —
  within-sweep comparison only.
- **Phase-5 pin (owner, laptop, this commit):** `run2_algo.yaml` →
  `train.lr: 3e-4`, ceiling raised to 15 epochs (`max_steps: 116595`,
  `epochs_total_planned: 15`, `cost.assumed_epochs_for_estimate: 15`).
  Box pulls, relaunches canonical `evt-run2-armA-algo`, official G1,
  then Phase-6 relay push of all five evt-run2 runs.

## Open at the moment

OPEN(2), OPEN(4), OPEN(10): see spec 02 §12 table — close at the runs
5–6 pilots (OPEN(1) closed 2026-07-21, entry above). Run-1 items:
CLOSED — `evt-run1-base-v3-ext` converged (owner confirmed
2026-07-21). Run-2 items: re-sweep COMPLETE, all four arms PASS G1, winner lr 3e-4
pinned with the 15-epoch ceiling (entry above); canonical
`evt-run2-armA-algo` launch + official G1 + Phase-6 relay push remain. Runs-3/4 items: G2 pinned to the
G1 bar (entry above); in-loop format-validity eval tooling lands with
the run-3 task. The backfilled v2/v2-ext/v3 manifests were
re-pushed to the HF relay from the laptop 2026-07-21 — closed.
Dataset items: none.

## 2026-07-21 — run-3 tooling landed: in-loop G4 stop, decode promotion, sweep configs

- **Behavioral stop implemented per the ratified rule (entry above):**
  `BehavioralStoppingRule`/`BehaviorTracker` (V5.44) and `train_sft`'s
  behavioral mode (V5.45 — val loss still logged every eval;
  `stop_reason="behavior"`; loss plateau never consulted; unpaired
  rule/callback raises upfront). `greedy_completions` promoted
  `gates.py` → `geode.arith.decode` (V5.46) — the same decode now backs
  G1/G2 and the installers' in-loop eval, per the promotion rule.
  Specs 00 (manifest `training.stopping` union) + 02 (§6.1/§6.2,
  V5.44–46) edited this commit.
- **Launcher refuses `train.lr: null`** — `run3_inst.yaml` ships lr null
  until the sweep pins the winner, so a canonical launch before the pin
  fails fast instead of training a placeholder (the run-2 pre-pull
  launch incident class).
- **Run-3 configs:** parent `evt-run2-armA-algo` with
  `parent_required_gates: [G1]`; task `arith_op_mult`; `D_inst`
  order_hash pinned from the frozen report; ceiling 15,546 steps
  (2 epochs, drop-last); stopping frozen `{format_validity ≥ 0.99, k=3,
  512 prompts, prompt_seed 316, eval_every 250}`; data/batch seed 316 —
  run 4 must copy seed/val_fraction verbatim for the identical-stream
  guarantee.
- **Installer sweep grid (implementer default, owner ratifies by
  launching):** mirrors the run-2 re-sweep — 3e-5 / 1e-4 / 3e-4 / 1e-3,
  full behavioral arms (each stops itself; worst case ~$0.03/arm).
  Winner criterion: installs the format (`stop_reason=behavior`) with
  arithmetic intact (g2 ≥ 0.95 once `gates.py g2` lands); owner pins
  `run3_inst.yaml` `train.lr`.

## 2026-07-22 — runs 2–4 closed: canonical outcomes + all gates recorded

- **Run 2 canonical** (`evt-run2-armA-algo`, launched 2026-07-21, lr
  3e-4 @ git 0bdd1ae): converged at step 19,000, `min_val_nats` 0.0037;
  **G1 0.9961 PASS** (n=1024, by-op + 0.9960 / − 0.9962). Relay-pushed.
- **Installer LR sweep finding (the length prior):** the original
  3e-5..1e-3 grid ALL FAILED G2 with retention monotone in LR
  (0.68 / 0.42 / 0.04 / 0.00 vs the 0.95 bar) while G4 hit 1.0
  everywhere — the format installs at any LR; *arithmetic retention* is
  the binding constraint. Failure mode was a length prior:
  correct-answer-plus-extra-digits at low LR, garbage + sign loss +
  unparseable at high LR. Extending the sweep down: 1e-5 0.7949 FAIL,
  **3e-6 0.9531 PASS** (+ 0.9384 / − 0.9674) → 3e-6 pinned for both
  installers.
- **Run 3 canonical** (`evt-run3-armA-inst`, 2026-07-22, lr 3e-6): the
  relaunch reproduced the sweep-winner arm bit-identically (same
  lr/seed/data/box) — a free determinism check, not an anomaly.
  `stop_reason=behavior` at step 750; **G2 0.9531 PASS, G4 1.0**, G5
  recorded.
- **Run 4** (`evt-run4-armB-inst`, 2026-07-22, identical `D_inst`
  stream/seed): `stop_reason=behavior` at step 750 as well; **G3 0.0000
  PASS** (inverted bar ≤ 0.02, by-op + 0.0 / − 0.0 — random labels
  leaked no real arithmetic), **G4 1.0**, G5 0 / 0.
- **G5 (evidence-only, no bar — always `pass: true` because
  `require_parent_ready` refuses on any `pass: false`):** Arm A
  zero-shot 0.0068, 16-shot exactly 0.0; Arm B 0 / 0. The paper's
  ~2%/12% expectation (1B scale) is void here — few-shot prompting
  yields no measurable op-notation arithmetic at 38.7M. Low-evidence
  finding; the target runs measure the real elicit-vs-teach gap.
- **`evt-run1-base-v1` is metrics-only history:** no local dir, no
  relay entry anywhere — deliberate (min val 1.1464 recorded in
  EXPERIMENTS.md §2), not a loss.

## 2026-07-22 — runs-5/6 infrastructure landed; target-LR sweep launched (OPEN(2) phase 1)

- **Harness on the pinned adapter:** `train_prequential` now builds
  `geode.train.apply_lora` (α/(2r), seeded A, zero B — spec 01 V1.9);
  peft retired from the dependency set. Optional per-update
  `step_callback` (V1.10) carries the runs-5/6 ε/k stopping rule, which
  lives in the launcher, not the library. Final θ_T test-loss eval
  chunked at 512 examples (bounds logits memory at the production
  5,000-row val split).
- **`train_target.py` launcher:** full-file order-hash verify BEFORE
  the `n_examples` prefix; parent gates + G7 `(data_order_hash,
  n_examples)` equality vs `match_data_order_with`; cost estimate +
  `--confirm-cost` refusal before `register_run`; snapshot schedule
  written to the manifest pre-training; `stop_reason` "converged" wins
  the ceiling tie-break; flat run layout.
- **`lr: null` in run5/run6_target.yaml is deliberate** — the launcher
  refuses to run them until the target-LR sweep pins the winner (the
  run-2 placeholder-lr incident class). Sweep = Arm B @ the 50K frozen
  prefix (teaching is the arm the shared LR must serve), one decade
  around 3e-4; winner = lowest `min_val_nats` among
  `stop_reason=converged`; the pin lands in BOTH run5 and run6 (one
  shared LR; arms differ only in the parent).

## 2026-07-22 — target LR pinned 1e-3; runs-5/6 stopping ratified (OPEN(2) phase 1 closed)

- **Sweep results** (Arm B @ 50K frozen prefix, batch 128, rule 5 mnat/k 3
  as shipped in the configs):

  | lr   | stop            | step  | min_val_nats |
  |------|-----------------|-------|--------------|
  | 3e-5 | max_steps       | 15626 | 0.7833       |
  | 1e-4 | converged*      | 15626 | 0.2024       |
  | 3e-4 | converged       | 11500 | 0.1071       |
  | 1e-3 | **converged**   | 8000  | **0.0350**   |
  | 3e-3 | converged       | 5000  | 0.0863       |
  | 1e-2 | converged       | 4000  | 1.8876       |

  *ε/k fired on the same eval the ceiling hit; launcher tie-break records
  `converged`.
- **Grid extension**: the original decade 3e-5..1e-3 came back monotone
  (higher LR strictly better) with the winner at the grid edge —
  unbracketed, the run-3-sweep situation in the other direction. Extended
  upward with 3e-3 + 1e-2 in one pass; the winner is now interior
  (3e-4 0.107 > 1e-3 0.035 < 3e-3 0.086; 1e-2 "converged" onto a garbage
  plateau ≈1.9 — unstable, closes the bracket). LoRA (α/2r = 0.125
  scaling) wanting a hotter LR than the run-2 full-FT 3e-4 anchor is
  expected in hindsight.
- **Pin**: `train.lr: 1.0e-3` in BOTH run5_target.yaml and
  run6_target.yaml (one shared LR; arms differ only in the parent), per
  the pre-registered winner rule. Sweep arms are disposable — never
  pushed to the relay; results live here + in the box manifests.
- **Stopping ratified (owner)**: runs 5–6 (and the OPEN(2) grid) run the
  canonical loss rule at short-run cadence — **ε=2 mnat, k=5,
  eval_every 500, min_steps 0**. The configs had inherited the
  superseded OPEN(3) 5 mnat/k 3 uncited-decision (owner caught it
  2026-07-22); canonical cadence (min_steps 5000, eval_every 1000) is
  mis-scaled at ~390-step epochs, so only ε/k transfer. Part of the EDL
  metric (sets θ_T → L_test); ratified BEFORE the grid so pilot n-choice
  and production share the rule. Spec 02 §6 gained the bullet in the
  same commit. LR ranking above unaffected (winner margin ~3×).
- **Naming**: sweep ids `evt-sweep-target-lr*` broke the
  `evt-runN-sweep-*` precedent (chosen because the pin serves both runs
  5 and 6); kept mid-sweep for internal consistency. Future sweeps use
  `evt-runN-sweep-*` (owner preference, 2026-07-22).
- **Next**: OPEN(2) grid (open2_n{10k,50k,200k,500k} + open2_a_ref) under
  the pinned LR + ratified rule; verdict = smallest n where B lands
  within a few accuracy points of A (endpoint eval, spec 02 §11), then
  OPEN(4) schedule params. Probe extraction (V5.9–V5.13) validates on
  the grid's snapshots before the box is destroyed.
- **Addendum (2026-07-22, later)**: owner reversed "sweeps are
  disposable" — sweep + grid artifacts get preserved on the relay before
  the box dies. Sweep AND pilot runs renamed on the box to run-scoped
  precedent form (the mv rewrites `run_id` inside each manifest):
  `evt-sweep-target-lr*` → `evt-run6-sweep-lr*` (run-6 recipe: Arm B,
  init run-4), `evt-pilot-open2-b-{10k,50k,200k,500k}` →
  `evt-run6-pilot-n{10k,50k,200k,500k}`, `evt-pilot-open2-a-ref` →
  `evt-run5-pilot-n50k` (run-5 recipe: Arm A, init run-3; mirrored name
  with its matched pair evt-run6-pilot-n50k). Overlay configs + the
  comparison notebook carry the new ids in the same commit. All pushed
  to `mhieuuu/geode-store` along with the run-3/run-4 installers; old-
  name relay folders (if an earlier push landed them) deleted after the
  new-name push is verified present. None of these are in the
  forbidden-from-box set (`evt-run1-base-v1/v2/v2-ext/v3` only).
- **G5 eval contamination — found, measured, resolved (owner
  2026-07-22)**: the G5 draw sampled the FULL D_target file, but the
  OPEN(2) pilots train its prefixes — 12/49/209/518 of the 1,024
  seed-316 eval questions (1.2%–50.6%) fell inside the
  n10k/n50k/n200k/n500k trained prefixes. Question texts are globally
  unique in the frozen file (verified on order_hash 69e3b09e…), so
  there is no additional text-level leakage. Measured effect (local CPU
  re-score reproducing the box numbers exactly): clean-subset accuracy
  0.9765 (B@500k) / 0.9374 (B@50k) / 0.9918 (A@50k) vs 0.9805 / 0.9385
  / 0.9922 recorded — inflation ≤ 0.4 points, OPEN(2) picture
  unchanged. Resolution (spec 02 §8, same commit): rows ≥ 900,000 of
  the frozen order are the eval reserve — g5 samples questions + shots
  only from the tail, `train_target.py` refuses any prefix reaching
  into it (null/full file included), and g5 refuses to score a run
  whose manifest records a D_target prefix past it (keyed on
  data_order_hash, so runs 3/4 still score). Reserved-protocol G5
  re-runs supersede all full-file-draw numbers.
- **Fixed shared eval file — D_target_eval (owner 2026-07-22, supersedes
  the 900k eval reserve of the same date)**: the reserve fixed G5
  contamination but capped training at 900k rows, and the per-run 0.5%
  val carve meant every pilot stopped and reported loss on different
  data (n10k's val was ~50 examples). Owner requirements: full 1M
  trainable, one test set identical for every run, nothing in it ever
  trained. Resolution: `make_data.py --eval-set` generates
  `D_target_eval.parquet` — 100k fresh operator add/sub questions,
  disjoint from D_target ∪ D_algo ∪ probe (D_algo included: Arm A
  pre-trained on those exact questions in NL notation, overlap would
  advantage A; D_inst is multiplication, disjoint by op). The six cells
  with x_digits + y_digits ≤ 4 are fully consumed by the frozen sets
  (verified 2026-07-22: pool = union exactly) and contribute 0 rows
  (~5.2% of the training distribution, easiest cells); the other ten
  carry 10k each. Disjointness independently verified (triple- and
  prompt-text-level, zero overlap); hash 588da81e…f6cb8 pinned in
  configs; uploaded to `mhieuuu/elicit-vs-teach-arith` with report.json.
  Consumers (same commit): rows 0–2047 = ε/k stopping block (identical
  for every run/n — val_fraction retired, the training prefix trains
  whole), rows 2048+ = reporting block (harness θ_T test loss, hence
  EDL; G5 shots = rows 2048–2063, questions = next 1024 — fixed slices,
  no sampling; G5 also records shared-set test_loss_nats over the full
  reporting block). 900k reserve guards removed; n_examples up to 1M,
  null still refused. Fixed-slice G5 re-runs supersede the
  reserved-tail numbers (which superseded the full-file draw); expected
  shifts are sampling-noise-sized (the reserved-tail re-run moved
  −3.7…+0.9 points vs the contaminated draw with orderings unchanged).
  Pilots keep their recorded per-run val stops (training is frozen
  history); their shared-set test losses land via the G5 re-run.

## Curve evals — dense log-spaced val logging (owner 2026-07-22)

- **Decision:** runs launched from now log extra in-loop stopping-block
  evals at `snapshot_steps(max_steps, n=64, dense_until=16)` (the same
  tested scheduler as snapshots: every step through 16, then log-spaced
  stretching, uniform tail). Rows carry `stopping_eval: false` in
  `eval_log.jsonl` and are **never fed to the ε/k tracker** — ratified
  logging-only, so the 2026-07-22 stopping rule (ε 2 mnat / k 5 /
  eval_every 500 / min_steps 0) keeps byte-identical semantics and
  runs-5/6 stopping stays comparable to the pilots.
- **Why:** the val curve had no resolution where the learning happens —
  first eval at step 500, by which Arm A is mostly converged (7 total
  points at its 3500-step stop). On the log step axes the notebook now
  uses, the every-500 cadence leaves the first two decades empty.
- **Cost:** ~60 extra 2048-row evals per run (16 batches each) — ~1–2
  min of GPU per run, no training-path change.
- **Consumers:** monitor.py replays the tracker over stopping rows only
  (absent field = pre-curve-eval log = stopping row, so pilot logs
  replay unchanged); pilot_loss_compare.ipynb plots all rows, train and
  val panels now log-log. Spec 02 §6 records the protocol (same
  commit).

## OPEN(2) closed — target n = 500K (owner 2026-07-22)

- **Verdict:** `data.n_examples: 500000` pinned in run5/run6_target.yaml
  (identical, G7). Criterion (spec 02 §11): smallest grid n where B
  lands within a few points of A — under the final fixed-slice protocol
  (D_target_eval, G5 zero-shot on the fixed 1024 questions + shared
  test loss over the 97,952-row reporting block):

  | run | zero-shot | test loss (nats) |
  |---|---|---|
  | B @ 10K  | 0.7725 | 0.3374 |
  | B @ 50K  | 0.9414 | 0.0428 |
  | B @ 200K | 0.8740 | 0.1109 |
  | B @ 500K | **0.9805** | **0.0140** |
  | A-ref @ 50K | 0.9941 | 0.0059 |
  | run-3 parent (A-inst) | 0.0117 | 2.3004 |
  | run-4 parent (B-inst) | 0.0000 | 3.7482 |

  Only 500K puts B within a few points of A (1.4 vs 5.3 at 50K). These
  fixed-slice numbers supersede both earlier G5 draws (contaminated
  full-file 2026-07-21, reserved-tail 2026-07-22); ordering was stable
  across all three protocols.
- **n200K dip is a stopping-rule artifact, not a data-size property:**
  worse than 50K on every protocol and metric; its ε/k rule fired at
  step 5500 (~3.5 epochs) — slower per-step improvement at larger n
  tripped the plateau detector mid-descent, leaving the model
  under-converged (min_val 0.103 bits vs 50K's 0.0505). Accepted for
  the grid; the production point (500K) escaped it with a genuine stop.
- **Ceiling raised with the pin:** max_steps 15626 → 23442 (6 epochs of
  500K at batch 128). The B@500K pilot stopped at 15,500 — 126 steps
  under the old 2-epochs-of-1M ceiling; seed wobble could have turned a
  converged production run into stop_reason=max_steps.
- **Loss evidence (first same-data comparison):** A@50K reaches ~7×
  lower loss than B@50K on identical examples (0.0059 vs 0.0428); B
  needs 10× the data to get within 2.4×. The elicit-vs-teach contrast
  the design needs is present at this scale. Parents both near-zero
  behaviorally with A's parent 1.45 nats/token below B's — the
  NL-notation head start, latent as required.
- 16-shot ≈ 0 on all seven runs (known collapse — see the G5 protocol
  entries); zero-shot + loss carry the evidence.

## OPEN(4) + OPEN(10) closed — runs 5/6 launch-ready (owner 2026-07-22)

- **OPEN(10): no optimizer-state snapshots** (owner). Snapshots stay
  model-only — `_save_snapshot` already saves only the model
  state_dict, so this is a decision record with zero code change.
  AdamW moments would have added ~2× params (~400 MB per snapshot,
  ~350 GB per B-like run) for an optimizer-trajectory analysis nothing
  in the plan consumes; mid-run resume is unneeded at ≲30-min runs.
- **OPEN(4): mechanical from the OPEN(2) pin.** Batch 128 (pinned),
  step ceiling max_steps 23442, snapshot schedule =
  `snapshot_steps(23442, n=1024, dense_until=30)` — the spec-02 §6
  structure ratified 2026-07-18, now with concrete numbers: unit
  stride through step 30 and beyond (geometric ratio ≈ 1.007),
  stretching to ~57-step gaps at the tail, final slot 23442.
- **Disk sizing (launch logistics):** snapshots are fp32 base+adapter,
  50.8M params ≈ 203 MB each. Expected materialization: ~880 snapshots
  (~180 GB) for a B-like stop near 15.5K steps, ~630 (~128 GB) for an
  A-like stop near 3.5K — both runs together want ~350 GB free on the
  box before launch. Steps past the ε/k stop never materialize
  (manifest `snapshots_taken` records the emergent truncation).

## Adapter-only snapshots (owner 2026-07-22, supersedes the 2026-07-18 self-contained format)

- **Decision:** `snapshots/step_{k}/adapter.safetensors` holds exactly
  the trainable (A/B) tensors; the frozen base + buffers are written
  once per run to `snapshots/base/model.safetensors` (before any step
  file, so a partially written run is always reassemblable). Reassembly
  via the new `geode.edl.load_snapshot` — bit-exact (V1.11), restores
  tied `lm_head`↔`embed_tokens` aliases, and still strict-loads legacy
  full snapshots (the pilots' on-box artifacts stay usable).
- **Why:** the 2026-07-18 self-contained decision was priced at
  ~77 MB/snapshot and predates the 1024-step schedule. Real numbers:
  fp32 base+adapter is ~203 MB, and the base is FROZEN during LoRA
  training — a B-like run would store ~880 byte-identical copies of the
  same 155 MB. Adapter-only: ~48 MB/step ⇒ ~43 GB (B-like stop) +
  ~31 GB (A-like) ≈ **~75 GB for both runs instead of ~350 GB**, with
  proportionally less write IO mid-run and faster extraction (base
  loads once, adapters swap per step).
- **Consumers (same commit):** loop `_save_base`/`_save_snapshot`/
  `load_snapshot` (+ V1.11(a–c) property tests, tied-embedding
  round-trip included — production models tie via train.py default);
  extract.py discovers/loads either format; train_target finalize
  counts `adapter.safetensors`. The FINAL `model/` checkpoint stays
  self-contained (zoo.load_model V0.9 unchanged).

## Runs 5–6 complete — both converged, G5 recorded (2026-07-22)

- **Outcomes:** run 5 (A, elicit) `stop_reason=converged` at step 6,000,
  min_val 0.00245 nats; run 6 (B, teach) converged at step 12,500,
  min_val 0.02301. Both far under the 23,442 ceiling; G7 verified in
  both manifests (identical `data_order_hash`, n=500,000). Snapshots
  materialized adapter-only as designed: 711 + base (run 5), 832 + base
  (run 6), every step dir carrying `adapter.safetensors`.
- **θ_T evidence (reporting block, 97,952 rows) + G5:** A 0.00194 vs
  B 0.03558 nats — an **18× loss gap on identical never-trained
  questions**, with zero-shot exact-match 0.9980 (A) vs 0.9502 (B).
  16-shot ≈ 0 both (known collapse, invalidated metric). A@500K also
  beats the A@50K pilot (0.0019 vs 0.0059) — more data still helps the
  elicited arm.
- **Stop-wobble caveat (quote θ_T with it):** run 6 undershot its
  *same-seed* pilot — the open2_n500k overlay changed only run_id and
  n_examples, yet the pilot's ε/k fired at 15,500 (θ_T 0.0140) and
  run 6's at 12,500 (θ_T 0.0356). With seed/data/lr/rule identical, the
  divergence is GPU nondeterminism compounding across steps, and the
  ε/k rule is noise-sensitive in B's shallow tail: the 2048-row
  stopping block produced 5 consecutive sub-2-mnat deltas while the
  reporting block shows B still improving ~3 mnat per 500 steps — the
  n200K artifact class. **Accepted** (owner-ratified rule, applied
  identically to both arms; re-running to chase a lower B number would
  be post-hoc protocol tampering; the primary measurement is the
  prequential curves). Even at the pilot's later stop the gap is ~7×.
  Concrete instance of the single-seed limitation (spec 02 §13).
- **Box logistics:** snapshots (~75 GB) exist only on the box; relay
  push is HELD — `hf_checkpoint.py push` uploads the whole run folder
  including snapshots, so pushing runs 5/6 is a storage decision, not a
  default step. Box stays alive for the extraction pass; ~31 GB free
  is fine for G5/idle but not for full extraction output (size at
  extraction planning).

## Owner directives 2026-07-23 — 1M rerun pair (runs 7/8), 1M LR re-pin, Llama-3.2-1B chain (runs 9/10)

- **Full-1M rerun (owner):** a NEW target pair `evt-run7-armA-target-1m`
  / `evt-run8-armB-target-1m` trains on the full 1M `D_target` order,
  superseding OPEN(2)'s 500K **for this pair only** — runs 5/6 stand as
  recorded history, nothing is overwritten or re-run to chase numbers.
  Parents are unchanged (run 3 → run 7, run 4 → run 8): the re-chosen
  LR and the n bump exist only in the target stage; upstream runs never
  saw `D_target` and their pins are untouched. Eval integrity is free:
  `D_target_eval` was built question-disjoint from the WHOLE 1M file at
  generation time (2026-07-22 protocol), so the 500K→1M bump cannot
  contaminate it. Configs `run7_target_1m.yaml` / `run8_target_1m.yaml`:
  ceiling 46,878 = 6 epochs of 1M at batch 128 (run 5's 6-epoch rule
  rescaled); ε/k block inherited by citation (2026-07-22 ratification);
  seed 316 kept; snapshots n=1024 dense_until=30 unchanged.
- **1M target-LR re-pin (owner):** the 1e-3 pin was chosen on a 50K
  prefix; before runs 7/8 launch, a 3-point bracket {3e-4, 1e-3, 3e-3}
  re-measures it on **Arm B @ the full 1M**
  (`pilot/target_sweep_1m_lr*.yaml`, run_ids `evt-run8-sweep-lr*`,
  disposable n=4 snapshots). Winner = lowest `min_val_nats` among
  converged; an edge win extends one step before pinning (the 50K
  sweep's monotone-decade precedent). **LR policy (one shared pin vs
  per-arm best) is DEFERRED by owner (2026-07-23)** — default remains
  one shared pin, and a B sweep is required under every policy; an A
  sweep is added only if the owner later picks per-arm.
- **Llama-3.2-1B external-validity chain (owner):** mirror the Arm A
  pipeline on the real pretrained model — Llama's own pretraining stands
  in for the pre-teach stage: `meta-llama/Llama-3.2-1B` (base, not
  Instruct) → run 9 format install on the frozen `D_inst` (op-notation
  mult, random labels, behavioral G4-style stop) → run 10 target on the
  frozen full-1M `D_target`. Owner picked **LoRA for the install stage**
  (over the full-FT mirror of runs 3/4 — fits the 24 GB 4090, no core
  trainer changes; **recorded limitation:** the install method differs
  from runs 3/4, so cross-model install comparisons are method-confounded).
  Consequence: a LoRA install checkpoint cannot be a `--init-from`
  parent (plain `from_pretrained` on a wrapped checkpoint silently
  random-inits projections — the G5-on-random-model incident class), so
  the install adapter must be **merged** into base weights and saved as
  a plain checkpoint before the target stage. Merge math's silent
  failure would corrupt everything downstream ⇒ promotion rule: core
  `geode.train` merge function + logit-equality property test (planned,
  spec edit in the same commit). Assistant minor pins (owner may veto):
  LoRA r=64 for both Llama stages (the repo's own pre-downscale 1B pin,
  common.yaml note), run_ids `evt-run9-llama1b-inst` /
  `evt-run10-llama1b-target`, same frozen data artifacts + order hashes
  as runs 3–8. **Before any launch:** verify the token-level machinery
  against the Llama tokenizer — pad_token is likely None (decode glue
  fix), digits chunk up to 3 per token, `token_label_span` contiguity /
  whitespace-overhang checks were tuned on the custom BPE; and Llama LR
  needs its own mini-sweep (the paper's 1B LRs may inform the bracket).
  Base-model zero-shot op-notation accuracy is recorded pre/post install
  as evidence (real Llama may already answer op-notation add/sub —
  a near-zero-EDL elicitation would itself be the expected finding).
- **Box plan (owner): one box, sequential, revisitable** — on the
  current 4090: extraction over runs-5/6 snapshots FIRST (only copies),
  then the 1M sweep, then runs 7/8; the Llama chain waits and may get
  its own box later. Disk checkpoint before runs 7/8: ≥120 GB free
  (guide §0); relaying/deleting runs-5/6 snapshots is an owner decision,
  never a default.

## Owner directives 2026-07-24 (sweep design + sequencing updates)

- **1M LR sweep = first epoch only, no snapshots** (owner). Each of the
  three points overrides `max_steps: 7813` (= ceil(1e6/128), one pass
  over the full 1M) and `snapshots.n: 0` (train_target.py now accepts 0
  as "schedule nothing"; the `snapshot_steps` scheduler keeps its n ≥ 1
  contract). Rationale: the sweep only has to RANK learning rates, and
  the ranking is a first-pass question — EDL itself is a first-pass
  quantity; convergence behavior (θ_T) is run 8's job, not the sweep's.
  The fixed budget also makes sweep cost known in advance (3 × 7,813
  steps) instead of open-ended. **stop_reason=max_steps is the EXPECTED
  outcome for `evt-run8-sweep-lr*` only** — a documented exception to
  the "max_steps = bug signal" rule; it stays a bug signal everywhere
  else, runs 7/8 included. Winner = lowest stopping-block `min_val_nats`
  at the shared 1-epoch budget (earlier ε/k convergence equally fine);
  grid-edge extension rule unchanged.
- **Sequencing update (owner): extraction deferred** ("save the
  extraction for later") — the sweep goes first on the box. Runs-5/6
  snapshots still exist ONLY on the box: keep it alive; the relay hold
  is unchanged. Sweep disk footprint is now trivial (no snapshots,
  <~1 GB/run), so the ≥120 GB disk checkpoint applies to the runs-7/8
  pair launch, not the sweep.
- **B-only sweep safeguard (assistant, owner notified 2026-07-24):** if
  3e-3 wins the sweep, run a 100K-prefix Arm-A pilot at that lr before
  launching run 7 — it is the only grid point never proven on Arm A
  (1e-3 converged on A at 500K in run 5; 3e-4 is gentler). A 3e-4 or
  1e-3 winner needs no A-side pilot.
- **Llama-3.2-1B unblocked 2026-07-24**: Meta license granted to
  `mhieuuu`; `verify_llama_tokenizer.py` OVERALL PASS on the laptop
  (pad→eos 128001, vocab guard 128256, spans OK over D_inst 300 sampled
  rows and D_target 250,011 rows incl. all 249,791 negative-answer rows).
- **Runs 5/6 snapshot counts (verified from local manifests)**: run 5
  (Arm A) 711 taken of 1024 planned (last at step 5,978); run 6 (Arm B)
  832 of 1024 (last at 12,469) — the rest of each schedule fell past the
  run's ε/k stop and never materialized, as designed.
- **Snapshot front-load rule for runs 7/8 (owner 2026-07-24)**: at least
  1024 snapshots must already be scheduled by HALF the max_steps ceiling,
  with saving continuing over the rest of the run. Motivation: the
  schedule is computed over the ceiling but runs stop at convergence —
  runs 5/6 banked only 711/832 of their 1024. Implemented as **config
  n: 2048** with the unchanged tested scheduler (front-loading is
  inherent in its dense-then-log shape): computed over 46,878, n=2048
  puts 1,566 steps ≤ 23,439 (half), and ≥1024 land even by step ~8K
  (1,220) — an Arm-A-fast stop still banks the full target. n=1536 was
  rejected (only ~970 by 8K). Disk: ~48 MB/snapshot ⇒ realistic pair
  ≈ 130 GB (A-stop ~8K → ~59 GB, B-stop ~20K → ~72 GB), worst case
  96 GB/run at ceiling — plan **~200 GB free** before the pair launch
  (supersedes the ≥120 GB figure above). Run 10 NOT bumped: Llama r=64
  adapters ≈ 180 MB each, n=2048 would need a ~400 GB box — decide when
  sizing the Llama box.

## Owner directives 2026-07-24 (second batch: precision, LR policy, launchers)

- **Llama chain trains bf16 (owner)**: `train.precision: bf16` in
  run10_llama1b_target.yaml (run 9's installer was already bf16).
  Implemented as V5.62 (spec 02 §6, f4acb44): `train_prequential` reads
  `training.precision` from the manifest and autocasts ONLY the
  grad-enabled update forward — the prequential stream, stopping evals,
  and θ_T test loss are always measured fp32 on fp32 master weights
  (losses are reported quantities, §7 principle). bf16 moves the θ
  trajectory, never how a loss is measured. Runs 5–8 stay fp32.
- **One LR everywhere (owner, ratified after the run-9/10 structure was
  clarified)**: the runs-7/8 1M sweep winner is the single shared pin for
  run 7, run 8, run 9, AND run 10. Both Llama sweeps are dropped
  (`pilot/llama{9,10}_sweep_lr*.yaml` kept as history only). Fallback
  (assistant, owner notified): if run 9's G4 fails or zero-shot
  arithmetic degrades at that lr, revive the gentle installer sweep for
  run 9 only. Accepted trade-off on run 10: the winner may sit above the
  paper's 1B-tuned 3.53e-4, shifting EDL magnitude — acceptable for a
  single external-validity chain. This also ACKS the shared-LR-for-7/8
  recommendation (the deferred LR-policy item is now closed).
- **Llama chain stays two-stage (owner)**: run 9 (format install,
  ~minutes) + run 10 (the EDL measurement). Not an elicit/teach pair —
  run 9 exists so run 10 measures arithmetic elicitation, not
  format learning.
- **Unattended launchers (owner asked for scripted runs, a9c47db)**:
  `sweep_1m.sh` (three 1M sweep points, crash resume, winner + edge-rule
  summary) and `launch_pair_1m.sh` (run 7 → run 8; refuses unless the
  shared pin matches the store's sweep winner, ≥200 GB free, parents
  present). The pair script deliberately ENDS after run 8 (owner): the
  Llama chain gets its own script later, when its box is sized — not
  implemented yet. `box_onstart.sh` = idempotent vast.ai provisioning
  (never launches training, never auto-pulls).
- **Run-10 snapshot storage (computed for the box-sizing decision, still
  open)**: adapter 180 MB + base 4.94 GB once; over the 46,878 ceiling —
  n=1024: ~80 GB at a 1K-step stop, ~130 GB at 7.8K, 190 GB worst;
  n=512: ~50–74–97 GB; n=256: ~30–41–51 GB. Half-res n=512 keeps the
  Llama box in the ~100 GB class if snapshot density can be halved for
  the external-validity chain — owner call at box sizing.

## Owner directives 2026-07-24 (third batch: full-chain launcher, push-and-prune, hf --force)

- **Full-chain launcher `launch_chain_7_10.sh` (owner, supersedes the
  same-day "Llama script waits for box sizing")**: runs 7 → 8 → 9 → 10
  sequentially, unattended, ntfy at every stage boundary and failure.
  Per run: train → gate evidence → optional prune. Includes the Llama
  smokes, run-9 G4 (+ the fallback tripwire: G4 fail at the shared LR →
  abort with "revive the gentle installer sweep"), the zero-shot
  op-add/sub evidence BEFORE run 10 (recorded as run-9 G5 on the wrapped
  checkpoint — same weights as `model_merged/`, V5.52 exact fold;
  `zoo.load_model` refuses a plain checkpoint under a lora manifest, so
  the merged dir itself is not gate-loadable), merge_adapter, and G5 for
  runs 7/8/10. Refusal guards: one shared non-null lr in ALL FOUR yamls
  == store sweep winner (≥3 complete sweep runs); pending-aware disk
  check; parents present; verify_llama_tokenizer PASS when 9/10 pending.
  Completed runs skip on re-run (crash/restart resume).
  `launch_pair_1m.sh` remains for a 7/8-only box.
- **`--push-and-prune` mode (owner option)**: after each run's gates are
  recorded, `hf_checkpoint.py push --with-snapshots` to the private
  relay, then verify EVERY local file is listed on the hub
  (`list_repo_files`) before `rm -rf` of the heavy dirs (snapshots/,
  model/, model_merged/ — manifest/logs/gates always kept). Run 9 prunes
  only after run 10 completes (model_merged is run 10's parent). Peak
  disk ~220 GB vs ~420 GB. Costs: extraction must pull snapshots back
  later, and the relay grows ~200–400 GB — the HF free tier caps private
  storage well below that (check the plan; PRO-class quota needed).
  WRITE token is passed per-push only (`HF_TOKEN=<write> python …`);
  the ambient box login stays READ. Runs 5/6 are NEVER pruned (their
  relay push stays owner-held).
- **Storage plan (vast.ai FAQ: disk size is FIXED at instance
  creation)**: worst-case adds — run 7 ~100 GB, run 8 ~100 GB, run 9
  ~15 GB, run 10 ~200 GB (n=1024). One box also holding the runs-5/6
  snapshots (~75 GB): ~490 GB worst / ~330 GB realistic → rent 500 GB
  without prune; ~300 GB with --push-and-prune; a fresh box without
  runs 5/6 subtracts 75 GB.
- **`hf auth login --force` always (owner)**: every HF login in guides
  and scripts uses `--force` (flag verified in hf CLI 1.23.0) so a stale
  cached login never masks the wrong account (Meta-license and
  READ-vs-WRITE mixups fail loudly at login time).
- **box_onstart.sh hardened per the vast.ai FAQ**: template env vars are
  invisible to SSH sessions → HF_TOKEN/NTFY_TOPIC now also written to
  /etc/environment (idempotent); the script self-installs to
  /root/onstart.sh when absent (SSH-instance restarts run that path);
  never overwrites an existing /root/onstart.sh. Also noted: instance
  LIFETIME — the rental end date is locked at rent time; the box
  holding the runs-5/6 snapshots must be watched against it.

## 1M B-arm LR sweep — first pass results, edge rule FIRED (2026-07-24)

Three points, one epoch each (7,813 steps), stopping-block min_val_nats:

| lr   | min_val (nats) | stop |
|------|----------------|------|
| 3e-4 | 0.10630        | 7813 (max_steps — expected for sweeps) |
| 1e-3 | 0.03967        | 7813 (") |
| 3e-3 | 0.03434        | 7813 (") |

3e-3 wins AT THE TOP EDGE → the pre-registered edge-extension rule fires:

- **Grid extended one step up**: `pilot/target_sweep_1m_lr1e-2.yaml`
  (run_id `evt-run8-sweep-lr1e-2`), added to `sweep_1m.sh`'s loop —
  re-running the script skips the three complete points and launches only
  this one. NOT a new sweep; the same pre-registered grid, one point wider.
- **Arm-A safeguard armed**: if 3e-3 stands after the extension, it is the
  only candidate never proven on Arm A → run
  `pilot/target_pilot_100k_armA_lr3e-3.yaml` (100K prefix, ceiling 4,692 =
  6 rescaled epochs, no snapshots, parent evt-run3-armA-inst) BEFORE
  pinning. PASS = stop_reason=stopping_rule with a sane min_val;
  max_steps/blown-up val = do not pin 3e-3 (1e-3 sits ~0.005 nats behind).
- **If 1e-2 wins**: stop and consult the owner — one-LR-everywhere would
  carry 1e-2 into the Llama chain (~28× the paper's 1B-tuned 3.53e-4;
  the run-9 gentle-sweep fallback would likely fire).
- Note for the record: the 1e-3 → 3e-3 gap is small (0.0053 nats ≈
  0.0077 bits at the shared budget) while 3e-4 is clearly out; the rule
  (lowest min_val + edge extension) decides, not eyeballing.

## Owner directives 2026-07-24 (fourth batch: no Llama snapshots, dense val curve)

- **Run 10 stores NO snapshots** (`snapshots.n: 0`; smoke overlay matched):
  no extraction is planned for the Llama chain — the external-validity
  claim is behavioral EDL only. This CLOSES the open "pick snapshot n at
  Llama box sizing" item (the n=256/512/1024 storage table above is moot);
  the whole Llama chain now fits in ~30 GB. Runs 7/8 KEEP n=2048 — their
  extraction is owed. Chain-script disk guard and prune calls updated
  (run-10 need 200→15 GB; prune peak ~120 GB, no-prune total ~250 GB).
- **Loss curves for plotting (owner: "record train and val a lot")**:
  train needs no change — `train_log.jsonl` (+ `logs/prequential.jsonl`,
  the EDL stream itself) already records EVERY step. Val: curve-eval
  density raised in train_target.py, `EVAL_CURVE_N` 64→256 and
  `EVAL_CURVE_DENSE_UNTIL` 16→30 — ~161 curve evals land ≤ step 2000 of
  the 46,878 ceiling (the expected run-10 early-stop region), every step
  to 30 covered. Curve evals stay logged-only (`stopping_eval: false`,
  never fed to the ε/k tracker) — the ratified stopping rule and its
  eval_every cadence are UNTOUCHED, so this is instrumentation, not a
  protocol change. Applies to every train_target.py run (7/8 get the
  denser curve too; evals are near-free at 38.7M). Worst-case cost at
  1.24B: one eval ≈ 16 fp32 forward batches ≈ 5 update steps — tens of
  minutes over a full run, cents at box rates.

## 2026-07-24 — 1M sweep extension result: 3e-3 stands; chain moves to a new box

**Extension result (evt-run8-sweep-lr1e-2):** min_val **1.86742** nats,
stopped at 5500 with `stop_reason=converged` — the ε/k rule firing on a
plateau at garbage (54× worse than 3e-3's 0.03435), i.e. optimization
blew up and flatlined. The edge rule is satisfied: **3e-3 is now an
interior winner** (full table: 3e-4 0.10630 / 1e-3 0.03967 / **3e-3
0.03435** / 1e-2 1.86742). Remaining pre-registered step before pinning:
the Arm-A 100K pilot at 3e-3 (`pilot/target_pilot_100k_armA_lr3e-3.yaml`)
— runs on the OLD box, whose store already holds `evt-run3-armA-inst`
(run 5 trained from it). PASS = `stop_reason=converged` **and** a small
min_val (run-5 floor ~0.0025); the 1e-2 point is the standing proof that
"converged" alone is not a pass. Only after PASS: pin 3e-3 in ALL FOUR
run yamls (one LR everywhere).

**stop_reason literal corrected in docs:** the harness emits
`"converged"`/`"max_steps"` (train_target.py; geode/train/loop.py) —
three doc spots said `stopping_rule` (pilot yaml header, run7-8-guide §3,
sweep_1m.sh verdict text). All fixed; no mechanical check ever branched
on the wrong string, so nothing misfired.

**New chain box (owner 2026-07-24):** runs 7→10 run on a NEW box via
`launch_chain_7_10.sh`; the old box stays alive as the runs-5/6 snapshot
archive (+ sweep evidence + the Arm-A pilot). Consequences handled:

- The chain's LR guard recomputes the winner from `evt-run8-sweep-lr*`
  manifests IN THE LOCAL STORE — a fresh box has none. One-time fix: old
  box pushes the four sweep runs to the relay (WRITE token per-command;
  snapshots skipped by default, n=0 anyway, <~1 GB each), new box pulls
  them `--no-weights` (manifests/logs only). Guard's refusal message now
  prints the pull loop. This also archives the sweep evidence off a
  mortal box.
- Final chain ntfy no longer hardcodes "runs-5/6 live only here": it now
  scans the local store for non-empty `snapshots/` dirs and names those
  runs — correct on both boxes (new box: runs 7/8 unless pruned). The new
  box inherits the keep-alive rule for runs 7/8 snapshots (extraction
  owed) until they're extracted or relayed.
- Sizing (vast disk FIXED at creation; rental end date LOCKED at rent —
  set both generously): **no-prune ~265 GB worst case → rent 300 GB**
  (run7 ~100 + run8 ~100 + run9 ~15 + run10 ~15 + env/cache/parents ~15
  + headroom 20). Prune mode peaks ~140 GB → 160 GB box, but the relay
  grows ~200 GB — needs paid HF storage quota; default is the 300 GB
  no-prune box. GPU ≥24 GB (run-10 smoke is the memory worst case).

## 2026-07-24 — old box deleted before the pilot/pushes: losses accepted (owner)

The old box was destroyed with the four `evt-run8-sweep-lr*` run dirs
(never pushed) and the runs-5/6 snapshots (~75 GB; 711 + 832 adapters) +
final model weights aboard. Inventory verified against the laptop store
and a relay file listing: runs 5/6 survive as manifest + train_log +
eval_log + logs/prequential + logs/gradstats + eval/test_loss on BOTH the
relay and the laptop — every reported number and every curve stands; the
run-3/run-4 parents have weights on the relay, so the chain is unblocked.

**OWNER RULING: not a problem.** Extraction will not run on runs 5/6 —
runs 7/8 snapshots (n=2048) are the extraction substrate. Recorded
consequence: the project's internals evidence now depends entirely on
runs 7/8 — once produced, their snapshots are irreplaceable, the new box
inherits the keep-alive rule for them, and pushing them to the relay
after the runs (`hf_checkpoint.py push --with-snapshots`, quota
permitting) is cheap insurance against a repeat of this incident.

Changes made:
- `launch_chain_7_10.sh` LR guard: "pin == winner among local sweep
  manifests" is permanently unsatisfiable now, so with <3 sweep runs in
  the store the guard verifies the pin against a **committed
  `configs/lr_pin.yaml`** (lr + provenance; created at pin time, i.e.
  after the Arm-A pilot passes). Refusal messages updated.
- `sweep_1m.sh`: header tripwire — on a fresh box nothing is complete,
  so re-running it would silently RETRAIN all four points. Marked
  CLOSED, do not run.
- Arm-A 100K pilot moves to the NEW box as its first GPU job (parent
  `evt-run3-armA-inst` pulled from the relay; guide §1/§3 updated).
- `plot_losses.py` (owner request): train-vs-val overlay for runs
  7/8/10 (default set), log-log, nats; train = faint per-step rolling
  mean, val = stopping + curve evals with min-val in the legend. Tested
  end-to-end on the surviving runs-5/6 logs.

## Arm-A 3e-3 pilot: high plateau → tie-break pilot at 1e-3 (owner 2026-07-24)

`evt-run7-pilotA-lr3e-3` (new box, first GPU job): **converged at step
4500 (ceiling 4692), min_val 0.0122 nats** (eps-gated best 0.0129). No
blow-up — but the floor is high against Arm A's 1e-3 track record:
n50k pilot (`evt-run5-pilot-n50k`, verified same parent/LoRA/batch/ε-k/
data order, lr 1e-3) reached **0.0057 at HALF the data**, and run-5
production reached 0.0025 at 500K. Interpolating, 1e-3 at 100K should
land ~0.004 — the 3e-3 pilot sits ~3x above that. Why it matters: Arm
A's floor is the EDL-ratio numerator; inflating it shrinks the headline
elicit-vs-teach gap, while 3e-3's B-side edge was only 0.0054 nats at a
capped 1-epoch budget (convergence training likely shrinks it). This is
the pre-registered "high plateau — do not pin" branch.

**Owner decision: run the tie-break pilot** — identical 100K Arm-A
pilot at lr 1e-3 (`pilot/target_pilot_100k_armA_lr1e-3.yaml`, run_id
`evt-run7-pilotA-lr1e-3`), same box, only the LR differs, killing the
50K-vs-100K extrapolation. **Pre-agreed pin rule:** min_val ≤ 0.006 →
pin **1e-3** in all four yamls; min_val within ~25% of 0.0122 → floor
was n-limited, pin **3e-3**; in-between → back to the owner.
`configs/lr_pin.yaml` is created with whichever pin wins, same commit.

**RESULT — 1e-3 PINNED (2026-07-24):** `evt-run7-pilotA-lr1e-3` converged
at step 3500 with **min_val 0.0039 nats** (eps-gated best 0.0044) —
identical overlay to the 3e-3 twin (0.0122), only the LR differed, same
box. 0.0039 ≤ 0.006 → the pre-agreed rule fires: **train.lr 1.0e-3 in
ALL FOUR run yamls + `configs/lr_pin.yaml`** (this commit). The result
also validates the interpolation that flagged 3e-3 (predicted ~0.004 for
1e-3 at 100K). The 1M B-sweep's 3e-3 edge (0.0343 vs 0.0397 at the
capped epoch) is knowingly given up: Arm A's floor is the EDL-ratio
numerator and 3e-3 inflated it ~3×. `launch_pair_1m.sh`'s LR guard also
gained the lr_pin.yaml fallback (it still hard-required sweep manifests
in the store — would have refused on the new box and pointed at the
tripwired sweep_1m.sh). Chain is GO: box git pull →
`./launch_chain_7_10.sh --confirm-cost`.

## Box portability: python3 everywhere + canonical /workspace (owner 2026-07-24)

Two recurring new-box failures, fixes chosen by the owner from options:

1. **vast images ship only `python3`** — bare `python` crashed box setup
   twice, including `box_onstart.sh`'s own `python -m pytest` (why the
   CPU-suite marker never appeared). Fix (belt and suspenders): every
   script and live-guide command now calls `python3` (self-sufficient
   even if onstart never ran), and onstart additionally symlinks
   `/usr/local/bin/python -> python3` for hand-typed commands. Old
   guides (runs 1–6) left as history.
2. **/workspace is a template lottery** (some images ship it, others
   drop the shell in /root and work lands in `~/workspace` — a
   *different* directory). Worse, onstart redirected its log INTO
   /workspace before creating anything, so on such images it died with
   no log. Fix: canonical `/workspace` on every box — onstart, BEFORE
   the log redirect, (a) uses /workspace if present, (b) else adopts an
   existing `~/workspace` via symlink `/workspace -> ~/workspace`
   (data stays put, name becomes canonical), (c) else `mkdir -p
   /workspace`; if both exist as distinct dirs it warns loudly and
   prefers /workspace rather than guessing. On vast the rootfs IS the
   rented disk, so physical placement is equivalent either way. Logic
   unit-tested for all four cases (scratchpad harness, 2026-07-24).
   Retrofit for an already-rented box:
   `ln -s ~/workspace /workspace` (if needed) +
   `command -v python || ln -s "$(command -v python3)" /usr/local/bin/python`,
   then `git pull`.

## One ntfy variable everywhere: $NTFY (owner 2026-07-24)

`box_onstart.sh` took a topic-only `NTFY_TOPIC` template var while every
launcher, sweep script, and live guide used `NTFY` (the full
`ntfy.sh/<topic>` URL) — two names for one channel, and the mismatch is
why a fresh SSH session could hit `curl: (2) no URL specified` even on a
box whose template had the topic set. Owner: one variable, `NTFY`, full
URL form, everywhere. `box_onstart.sh` now takes `NTFY` as its template
var, writes it to both `~/.bashrc` and `/etc/environment`, and pings
`"$NTFY"` directly; a one-line compat expansion still honors a stale
template that sets only `NTFY_TOPIC` (feeds it into `NTFY`, which always
wins if both are set). Update the vast template to set `NTFY=ntfy.sh/<topic>`.
The already-running chain box keeps its old `/root/onstart.sh` copy —
harmless, since its `~/.bashrc` already exports `NTFY`.

## Runs 7/8 snapshots WILL be pushed to the relay — owner has HF Pro (2026-07-24)

The earlier keep-alive-only plan assumed free-tier relay quota could not
hold ~200 GB of snapshots. Owner 2026-07-24: **the account has HF Pro,
so the snapshots go to Hugging Face** (`mhieuuu/geode-store`) as soon as
each of runs 7/8 completes — the box is then no longer the sole copy of
the project's only internals evidence (the old-box deletion must not be
repeatable).

Mechanics (unchanged in spirit, now mandatory instead of "recommended"):

- The chain still launches in DEFAULT mode (`./launch_chain_7_10.sh
  --confirm-cost`, **no** `--push-and-prune`). Reasons: prune mode needs
  a WRITE token available for the whole multi-day unattended chain
  (violates the never-store-WRITE rule) and deletes the local copies,
  which extraction on the box will want. Disk is sized for no-prune
  (300 GB).
- After EACH of runs 7 and 8 completes (don't wait for the Llama half),
  owner-gated manual push from a box SSH session, WRITE token
  per-command only:
  `HF_TOKEN=$HF_WRITE_TOKEN python3 hf_checkpoint.py push --run-id evt-run7-armA-target-1m --with-snapshots`
  (then the same for `evt-run8-armB-target-1m`). Ambient login stays
  READ.
- Box keep-alive rule relaxes ONLY once both pushes are verified on the
  hub (file listing shows `snapshots/` for both runs); until then the
  box remains the sole copy.

## Repo pruned — closed-era history removed from the working tree (owner 2026-07-24)

Owner-approved sweep (everything stays in git history): runs-1–4 +
runs-5/6 guides (6 files; the §6 small-artifact push recipe moved into
llama-guide.md §4), docs/impl-logs/ (frozen since the 2026-07-17 cut),
31 consumed pilot configs (run1/2/3 era, open2_* grid, 500K
target_sweep_lr*, dropped llama10_sweep_lr*), `sweep_1m.sh` (foot-gun
removal — its tripwire was "never re-run"), `measure_lengths.py`, and a
stale merged `.claude/worktrees/` checkout that was accidentally tracked
as a gitlink (now gitignored). KEPT deliberately: llama9_sweep_lr*
(documented run-9 G4-fail fallback), llama9/10_smoke.yaml (chain uses
them), target_sweep_1m_* + target_pilot_100k_* (LR-pin provenance),
`sample_stories.py` (spec 02 §OPEN(12) keep), `train.py` +
runs-1–6 configs (run records), `migrate_store_layout.py` (live remedy
named by geode/zoo/store.py's legacy-layout error), notebooks
(owner left the notebook question unanswered — untouched).

## Runs 9/10 G5 evidence was measured with the wrong tokenizer — invalid, re-measure (2026-07-24)

The chain's `g5_if_missing` hardcoded `eval_target_data.yaml` for every
run; that config pins `tokenizer: ../tokenizer` (the frozen custom arith
tokenizer). Correct for runs 7/8; for runs 9/10 it tokenized G5's
prompts and the shared-set NLL in a vocabulary the Llama model never
saw. Symptom on run 10: a run that converged at 0.0156 nats val scored
zero-shot/16-shot exact match 0.0000 and a 15.04-nat "test loss" (worse
than uniform over Llama's 128k vocab). Proof the checkpoints are fine:
run 9's G4 used the correct tokenizer (run9 config) on the same
checkpoint and passed at >=0.99 format validity. Same disease as the
run-9 G4 config bug (0f9e1c7) — eval-config reuse across arms — but G5
has no `train:` block to crash on, so it ran silently wrong. This is
the incident class of feedback-eval-decode-must-match-training-
tokenization, one level up: the whole tokenizer, not just the decode.

- The G5 blocks currently in the run-9 and run-10 manifests are
  GARBAGE EVIDENCE — do not quote them, do not push them to the relay.
  Re-run `gates.py g5` for both runs with the new
  `eval_target_data_llama.yaml` (same D_target_eval pin, tokenizer
  `meta-llama/Llama-3.2-1B`); gates.py overwrites the manifest G5 block
  in place.
- Chain + llama-guide §3 fixed: `g5_if_missing` now takes the eval
  config as an argument (runs 7/8 keep `eval_target_data.yaml`).
- Rule: an eval config's tokenizer must match the model under eval,
  exactly like training. Any future non-custom-tokenizer run needs its
  own eval_target_data variant.
- Training-side numbers are untouched: run 10's prequential logs,
  stopping evals, and min_val 0.0156 nats were all computed inside
  train_target.py with the correct Llama tokenization (order_hash +
  G7-null verified). Only the two G5 evidence blocks are affected.

## Runs 9/10 G5 re-measured with eval_target_data_llama.yaml — evidence now valid (2026-07-25, box)

Re-run after the 2026-07-24 tokenizer incident (previous section);
gates.py g5 overwrote both manifest G5 blocks in place.

- **Run 10** (`evt-run10-llama1b-target`): zero-shot 0.9844, shared-set
  test loss 0.0232 nats over n=97952 — consistent with its training
  min_val 0.0156, the checkpoint and the convergence are real. 16-shot
  0.0000 = the known collapse (see "16-shot ≈ 0 on all seven runs",
  runs-5/6 entries — invalidated metric, now confirmed to extend to a
  1.24B pretrained base after LoRA on single short examples); zero-shot
  + loss carry the evidence.
- **Run 9** (`evt-run9-llama1b-inst`, the format-installed parent —
  same weights as run 10's `model_merged` init): zero-shot 0.0000,
  16-shot 0.0000, test loss 9.2594 nats. **The "real Llama may already
  answer op-notation add/sub → near-zero-EDL elicitation" prior is
  refuted**: op-notation arithmetic was not behaviorally present in the
  parent. Run 10's prequential EDL therefore measures a genuine
  install/elicitation gap, not a formality.
- Checkpoint census (both runs): `model/model.safetensors` = 2.56 GB,
  370 tensors (112 base / 112 A / 112 B / 34 other, embeddings
  present) — the complete wrapped state dict. The "run dir is 200 MB"
  observation that triggered the audit was a mismeasure; `du` shows
  2.5 GB.

## Runs 9/10 archive = full run-dir relay pushes (owner 2026-07-25)

Owner confirmed after the G5 re-measurement: runs 9/10 push their FULL
run dirs to the relay (`hf_checkpoint.py push`, no `--with-snapshots` —
none exist), ~2.6 GB per checkpoint. Supersedes "checkpoints stay
on-box until the owner's relay decision" (llama-guide §4 updated;
logs-only recipe lives in git history). Adapter-only stripping
rejected: no tooling loads it, and run 10's adapter is relative to
run 9's merged model. Rationale: the runs-5/6 weights loss (old box
deletion) — never leave a box as the sole copy of final weights.

## Extraction protocol for runs 7/8 — `--limit 128`, dumps stay on-box (2026-07-24)

Relay status, verified from the laptop 2026-07-24: runs 7 (1,142
snapshot dirs incl. `base`) and 8 (1,308) are fully pushed with
snapshots — the keep-alive rule (hub shows `snapshots/` for BOTH) is
satisfied; run 9 is pushed (incl. `model_merged/`); **run 10 is the
only run not yet on the relay** (push via the vast-template
`HF_WRITE_TOKEN`, llama-guide §4). Loss overlay 7-vs-8 produced on the
laptop from pulled logs: min_val 0.0027 (A) vs 0.0237 (B), Arm A's
drop at steps ~30–200, Arm B's plateau ~1.5 nats until ~700.

Extraction decisions (minor, decide-and-notify):

- **Disk math forces subsampling.** One dump = 1,024 probe examples
  (hash-verified `2b6d51c2…`) × max 28 tokens × d512 × 9 hooks,
  acts+grads bf16 ≈ 504 MiB. Full density (1,141 + 1,307 materialized
  step snapshots) ≈ 1.2 TiB — the 300 GB chain box cannot hold it.
  `scripts/extract.py` grew `--limit N`: evenly index-spaced over the
  materialized snapshot list (first + final always kept; the list is
  already log-then-uniform, so index spacing preserves its shape).
  Starting density **128/run ≈ 129 GiB both runs**; extraction is
  resumable, so a later larger `--limit` only adds dumps.
- **Dumps are not relayed.** They regenerate from the hub snapshots
  for ~$1 of GPU; only the small outputs (`results/*.parquet`,
  `analysis/figures/*.png`) come back to the laptop (scp).
- **First driver written**: `analysis/alignment.py` — V5.13 alignment
  per (snapshot, layer), conditioned on per-example probe loss > 0,
  ZOO-4 long-format `results/gradient_alignment.parquet` (`regime` =
  elicit/teach) + the spec-02 §11 gradient-alignment figure.
  Smoke-tested on synthetic dumps: planted-parallel gradients ⇒
  pairwise cosine 1.0, random ⇒ ≈ 0 with top-PC EVR near 1/n.
- Paste sheet: run7-8-guide §4. Probe tests all green (78) before the
  driver landed.

## Lifecycle reorg of experiments/training-run (2026-07-24)

Owner asked for a folder reorganization; scoped to a **lifecycle split**
after mapping the constraints (minor, owner-approved from three options):

- `datagen/` (new): `make_data.py`, `make_tokenizer.py` — one-time
  generation whose outputs are frozen (datasets on HF, tokenizer
  committed).
- `analysis/` absorbs `plot_losses.py` (default `--out` now
  `analysis/figures/losses.png`, was CWD-relative) and
  `sample_stories.py`. ALL figures land in `analysis/figures/`
  (gitignored via the root `figures/` pattern).
- `scripts/` = GPU/box operations only, **paths deliberately
  unchanged**: `train_sft.py`/`train_target.py`/`gates.py`/`extract.py`
  import `train.py` as a same-dir sibling; both launchers `cd
  $(dirname $0)` and invoke siblings by bare name against `../configs`;
  `tests/scripts/test_migrate_store_layout.py` pins
  `scripts/migrate_store_layout.py`; and `box_onstart.sh` CANNOT move —
  the vast.ai template holds an external verbatim copy whose line
  `cp experiments/training-run/scripts/box_onstart.sh /root/onstart.sh`
  git cannot update. Both box paste sheets (run7-8-guide §4 extraction,
  llama-guide) therefore stay valid as printed.
- A deeper split of `scripts/` (launch/relay/checks) was considered and
  deferred: it requires package-ifying the sibling imports and
  rewriting both launchers + both guides, and buys nothing while
  `scripts/` is frozen operational tooling — growth is in `analysis/`
  (drift/adapters/matching/export_hf still to come). Revisit only after
  the extraction box is destroyed, if ever; the durable anti-sprawl
  mechanism stays the geode/ promotion rule, not directory depth.

## Analysis metrics V5.14–V5.16 + drivers built (2026-07-24)

Tested core + property tests in one pass (workflow rule), then three
thin drivers cloned from `alignment.py`. Suite 516 → 528. Design
decisions worth recording:

- **Drift (V5.14, `geode.probe.representation_drift` + `drift.py`)**:
  per-example representation = masked mean over `label_mask`-True
  positions (padding *activations* are nonzero, unlike gradients — the
  metric refuses to flatten and takes an explicit mask); class = probe
  `cell` (16 digit-pair classes) + an "all" aggregate; NO
  `probe_loss_nats > 0` filter (activations don't degenerate at zero
  loss); row-alignment guard: parquet `order_hash` must equal every
  dump's `probe_set_hash` + `input_ids` equality vs ref.
- **Reference snapshot** (drift + full-FT adapter diffs): earliest
  dumped/snapshotted step, `--ref-step` to override. Step 1 is one
  optimizer update in — closest available stand-in for init, not init
  itself. Ref-step rows are emitted as exact-zero self-checks.
- **Effective rank (V5.15, `geode.probe.effective_rank` +
  `adapters.py`)**: spectral-entropy erank (Roy & Vetterli 2007) of ΔW;
  equals r exactly at equal singular values (the planted-test
  construction). `adapters.py` reads snapshot safetensors raw (no model
  instantiation): full-FT ΔW = W_k − W_ref; LoRA ΔW = (α/2r)·B@A (V5.47
  scaling, NOT peft's α/r) with no ref subtraction (B zero-init). 1-D
  tensors feed ‖ΔW‖ totals/allocation but carry no rank.
- **Matching (V5.16, `geode.probe.performance_aligned_matching` +
  `matching.py`)**: performance proxy = **mean per-example probe loss
  in nats** — probe *accuracy* needs generation, which dumps don't
  store; ties break toward the earliest arm-B step (strict-< scan, no
  torch argmin tie-break dependence); `layer = -1` sentinel on all rows
  (nothing layer-resolved). Driver reads only `meta.json` +
  `probe_loss_nats` per step (never `load_probe_dump` — acts+grads are
  ~0.5 GiB/dump the analysis never touches) and re-asserts the
  `_MATCH_FIELDS` guard across steps and arms.
- Results parquets via the ZOO-4 writer: `representation_drift`,
  `adapter_diffs`, `performance_matching`; figures to
  `analysis/figures/` (gitignored). All three drivers smoke-tested
  end-to-end against synthetic stores (real `register_run` manifests,
  planted known answers: exact-zero drift at ref, planted rank-2 ⇒
  erank 2.0002, known pairing recovered, guards verified to refuse).

## Wave-2 analysis: V5.63 linear CKA + five drivers + steering (2026-07-24)

Second analysis wave, same workflow rule (core + property tests in one
pass, drivers single-pass). Suite 528 → 532. Numbered **V5.63** because
V5.17 was already taken by the packing property (spec 02 §3).

- **Linear CKA (V5.63, `geode.probe.linear_cka` + `cka.py`)**: fp64,
  column-center both matrices, `‖XcᵀYc‖_F² / (‖XcᵀXc‖_F·‖YcᵀYc‖_F)` —
  rotation/scale/translation-invariant, so meaningful across two
  independently trained models. Driver consumes `matching.py`'s
  `matched_step_b` rows (the V5.16 step map) and compares arms at
  **equal competence**, not equal step; loads pairs only through
  `load_matched_probe_pair` (V5.12) + a label-mask equality guard.
  Row identity = arm A's step; arm B travels in `matched_run_id` /
  `step_b` extras (matching.py convention).
- **learning_curves.py**: probe_data+meta only (never acts/grads);
  per-cell mean probe loss + `cell_acquisition_step` = earliest step a
  cell's mean drops below `--threshold` (default 0.1 nats).
  Never-crossed cells emit NO row (a fabricated sentinel step would
  poison downstream mins) — they print a warning and appear as open
  markers pinned at max-step in the figure.
- **act_rank.py**: V5.15 effective rank applied to the pooled
  (masked-mean, fp64, column-centered) activation matrix per hook per
  snapshot; `_frac` variant normalizes by the `min(n−1, d)` ceiling.
  No loss>0 filter (activations don't degenerate at zero loss).
- **probes.py**: torch-only linear decodability of the true answer's
  first label token from the residual stream one position earlier;
  LBFGS + CE + l2 (‖W‖², default 1e-3), standardized features,
  deterministic seeded 50/50 split built ONCE from the first run's
  earliest dump and asserted identical (input_ids + label_mask) across
  every dump; majority-class baseline recorded per row. sklearn stays
  out of the dependency set; measured 0.07 s/fit at production dims.
- **trajectory.py**: weight-space geometry from snapshots; streams
  snapshots with per-tensor fp64 accumulators (never materializes flat
  vectors); path length, net displacement, efficiency, step cosines,
  cos-to-final-direction; undefined cosines dropped, never fabricated.
  Imports `_discover_steps`/`_load_state`/`_lora_deltas` from sibling
  `adapters.py`. **Handles both methods (2026-07-25).** It originally
  refused `method: lora` on gauge-freedom grounds while its
  `DEFAULT_RUNS` were runs 7/8 — both LoRA — so the bare invocation
  could never succeed; that is why the metric was missing from the
  first nine. The fix takes the trajectory through the *merged* update
  ΔW = (α/2r)·B@A, which is gauge invariant (B→BR, A→R⁻¹A leaves B@A
  fixed), never through the raw factors. The LoRA reference is
  *exactly* init (B is zero-initialised ⇒ ΔW ≡ 0), which is stronger
  than the full-FT path's earliest-snapshot stand-in, so every
  snapshot including the first gets rows. Reads only
  `adapter.safetensors` (the frozen base cancels from every difference
  and would otherwise be re-read ~1141×/run). Property tests in
  `tests/analysis/test_trajectory.py`: gauge invariance, planted
  straight line (pins ref=init), and agreement with `adapters.py`'s
  independent ‖ΔW_k‖.
- **steering.py (Wang et al. 2025 template)**: the only driver that
  touches a model. Direction per hook = mean pooled activation shift
  (final − earliest dump, masked-mean fp64); injected into the PARENT
  (`zoo-run/<id>` only, loaded via `geode.zoo.load_model` V0.9) at all
  positions via a forward hook on the same `_residual_modules` map
  extraction uses; eval = the exact G5 slice/decode path (token-prefix
  prompts, EOS-in-span loss, V5.43). alpha=0 baseline runs *through*
  the hook; norm-matched seeded random control per hook (seed+layer, so
  `--hooks` subsets draw identical vectors); post-config leak check
  (hook removed ⇒ logits bit-equal to baseline, else raise).
  `checkpoint_step` = the direction's source snapshot.
- All five smoke-tested on synthetic stores with planted answers
  (CKA=1 on rotated copies, planted acquisition steps recovered,
  orthogonal walk ⇒ efficiency 1/√k exactly, alpha=0 bit-exact vs
  un-hooked, guards refuse on mismatched hashes/masks/methods).

## Campaign close-out: all ten metrics, cross-arm findings (2026-07-25)

Runs 7 (armA, elicit) and 8 (armB, teach) — 1M target-only reruns from
frozen run-3/4 parents. All ten analysis parquets now exist and are on
the relay. Numbers below are at the **matched step 5991** (both runs'
last common snapshot) unless labelled otherwise.

**Methodological finding, stated first because it conditions everything
else: the matched comparison is only valid early.** `matched_step_b`
pins at 10161 — teach's own best step — for every elicit step past
A@360. Teach *never reaches* elicit's performance at any step: elicit's
mean probe loss bottoms at 0.0146 nats, teach's at 0.1369 (9.4× worse)
before rising to 0.2148 at its last step. So `cka_matched` (0.62 → 0.57)
is comparing elicit@k against teach@its-best, not a performance-matched
pair, and its drift should be read with that attached. Median matched
gap 0.0546 nats, max 0.886.

Two confounds were identified and handled rather than reported through:

- **Horizon.** `path_efficiency` erodes monotonically with continued
  walking, so run 8 (to step 10969) would read as less efficient than
  run 7 (to 5991) partly for running longer. All cross-run numbers are
  taken at the common last step; terminal values are labelled UNMATCHED
  in the driver's own summary. Snapshot schedules are identical over the
  common prefix (1141 steps to 5991), so no sampling-density confound
  remains within it.
- **Different parents.** Run 7 ← `evt-run3-armA-inst`, run 8 ←
  `evt-run4-armB-inst`. Any cross-arm comparison of raw directions or
  per-layer structure is "given each arm's own parent", not a clean
  mechanism difference — layer 7 may already be doing different work in
  the two parents. This is why no cross-arm direction cosine was
  computed: the two residual bases are not aligned.

### What the ten metrics say

- **Weight movement is larger and less coherent for teaching.** ‖ΔW‖
  109.0 (elicit) vs 168.6 (teach) — teach moves 1.55× further for 9.4×
  worse loss. Arc lengths are nearly equal (2205 vs 2374), so teach's
  walk is simply more *outward*.
- **Step coherence separates sharply and late.** `step_cosine` at step
  300: 0.951 vs 0.796; at 1000: 0.860 vs 0.464; at 3000: 0.540 vs
  0.0395; at 5991: 0.111 vs −0.004. Teach's updates are an essentially
  uncorrelated random walk from ~step 3000 onward, while still 9× short
  of elicit's loss and still 8000 steps from its own stop. Elicit's own
  decay to 0.111 is the signature of having *arrived* (it converged at
  ~6000), not of diffusing.
- **`path_efficiency` inverts the naive prediction** (elicit 0.0494 vs
  teach 0.0710) and this was verified, not explained away:
  `net_displacement` agrees with `adapter_diffs`'s independently
  computed `delta_w_fro_total` to 3.0e-8 relative over all 2448 real
  steps. Efficiency is a *net/arc* ratio, so teach's larger outward
  displacement raises it even as its steps lose local coherence; the two
  metrics measure different things and disagree here honestly.
- **Gradient coherence favours elicitation, growing over training.**
  pairwise cosine ratio 1.04× early → 1.37× mid → 2.00× late; top-PC
  explained variance 0.75× early (INVERTED — teach higher) → 1.94× mid →
  1.95× late.
- **Acquisition is ~28× earlier and more complete for elicitation.**
  Median `cell_acquisition_step` 172 (elicit) vs 4810 (teach); IQR
  127–351 vs 3248–6320. Elicit acquires 17 cells, teach only 12 — teach
  never crosses threshold on 5 of them (never-crossed cells emit no row
  by design).
- **Update energy lands in different layers** (given each arm's own
  parent): teach puts 46.9% of ‖ΔW‖² in layer 7 (the last block) and
  2.4% each in layers 0–1; elicit spreads over layers 1–6, peaking at
  layer 6 (20.9%), with 6.7% in layer 7. Allocation cosine 0.581, L1
  distance 0.803. Projection-type allocation is by contrast nearly
  identical (both MLP-dominated: down/gate/up ≈ 0.79 elicit, 0.80
  teach), so the arms differ in *where* they edit, not in *what kind* of
  projection they edit.
- **Steering transfer, run symmetrically (2026-07-25).** Each arm's
  direction injected into its *own* parent. Elicit: EM 0.0117 → 0.1406
  at blocks.4 α=2 (12×; random control max 0.0273), loss 2.2716 →
  1.6182. Teach: EM 0.0000 → 0.0000 at every hook and α, loss 3.8994 →
  3.1899 at blocks.1 α=1. **Caveat that must travel with this result:**
  teach's parent has zero zero-shot EM, so EM has no headroom there and
  cannot detect a small effect. On the comparable metric, absolute loss
  improvement, the two are similar (−0.65 vs −0.71 nats); the honest
  claim is that elicit's direction crosses into *behaviour* while
  teach's only moves the distribution.
- **Nulls worth recording.** Linear-probe test accuracy is
  indistinguishable between arms late (0.5604 vs 0.5597, majority
  baseline 0.2695) — whatever the probes decode, both arms end with it.
  Activation effective-rank fraction differs only modestly and in
  elicitation's favour (1.19–1.25×), i.e. elicitation does not compress
  representations more; if anything it uses slightly more directions.

### Techniques considered and declined

- **Cross-arm direction cosine** — not rigorous: different parents mean
  different residual bases. Basis-free comparison is what CKA is for,
  and crosscoders/DFC are the parked track.
- **Crosscoders / DFC** (Minder et al.; Jiralerspong & Bricken) — out of
  budget and the track is parked; `reference/` stays read-only.
- **Logit lens** — adds little over the probe and steering results
  already in hand.

## 2026-07-25 — run 9's installer LR was a scope leak; retention swept, now gating

- **The defect.** `configs/lr_pin.yaml`'s 1e-3 was measured entirely at the
  TARGET stage (1M `D_target` Arm-B sweep + the Arm-A 100K tie-break pair),
  yet `applies_to` listed run9 and `launch_chain_7_10.sh` guard 1 *enforced*
  equality across all four runs. Run 9 is an installer, whose binding
  constraint is retention of the base model's arithmetic, not val loss.
  Measured 2026-07-25 with a new `configs/eval_algo_data_llama.yaml` (same
  1,024 `D_algo` questions G1/G2 use, Llama tokenizer): base Llama-3.2-1B
  **0.3271** NL add/sub, `evt-run9-llama1b-inst` @1e-3 **0.0000** — 1024
  misses, **0 unparseable**, emitting the `D_inst` random-label distribution
  (`515000`, `51502`). Format intact, arithmetic gone. **Run 10 v1 therefore
  measured teaching, not elicitation**, consistent with its min_val 0.0156
  sitting between run 7 (0.0027) and run 8 (0.0237), nearer teach.
- **Why nothing caught it.** `run9_llama1b_inst.yaml` shipped with no
  retention gate ("no G2/G3 analogue — recorded but not gating") while the
  38.7M installer sweep had *already* established retention as the binding
  installer constraint (G4 = 1.0 at every LR while retention ran 3e-5 0.68 /
  1e-4 0.42 / 3e-4 0.04 / 1e-3 0.00, forcing run 3 to 3e-6).
- **This was pre-registered, not improvised.** The 2026-07-24 one-LR entry
  carries the fallback "if run 9's G4 fails or zero-shot arithmetic degrades
  at that lr, revive the gentle installer sweep for run 9 only." Arithmetic
  degraded to zero; the trigger fired. Note the run-9 sweep had never been
  run before — the overlays were committed 2026-07-24 (410c766) and dropped
  the same day by the one-LR ratification.
- **Sweep (5 points + lr=0 base ref, `D_inst`, LoRA r=64 α/2r=0.25, bf16).**
  Retention on the same 1,024 questions:

  | lr | base | 3e-6 | 1e-5 | 3e-5 | 1e-4 | 3e-4 | 1e-3 |
  |---|---|---|---|---|---|---|---|
  | retention | 0.3271 | **0.3193** | 0.2900 | 0.1816 | 0.1729 | 0.0576 | 0.0000 |

  Monotone. **Every point installs the format** (`stop_reason=behavior` at
  all six), so format validity does not discriminate and retention alone
  picks the winner. **Pin 3e-6** — 97.6% of base, the gentlest point tested,
  and the same value run 3 landed on. The grid-edge extension rule (applied
  twice before) does not bind: retention is within 2.4% of its ceiling so
  there is almost nothing left to find below, and gentler risks the format
  not installing at all. Two things the aggregate hides, recorded rather
  than smoothed over: `+` *rose* 0.4990 → 0.5288 while `−` fell 0.1612 →
  0.1171 (≈2.7 SE at n≈512/op), and val loss *rises* as lr drops (4.188
  @3e-6 vs 4.098 @3e-4) — the gentlest point learned least of the
  random-label distribution, which corroborates the retention story.
- **Fix (181ce35), three parts.** (1) `lr_pin.yaml`: run9 out of
  `applies_to`, `installer_lr: 3.0e-6` recorded with the sweep table and a
  scope note. (2) The chain guard binds the shared pin to the three TARGET
  yamls and checks run 9 against `installer_lr`; run 9's lr *equalling* the
  target pin is now a hard error naming this incident. (3) **Run 10's
  `parent_required_gates: [G4] → [G4, G2]`** — retention is gating now,
  scored `--threshold 0.29` (90% of base 0.3271; the 0.95 G1/G2 bar does not
  apply to a never-taught model). The comment that retention was "recorded
  but not gating" *was* the hole; a guard in the DAG replaces it.
- **Run 10 keeps 1e-3, on a pre-registered test.** Its pin is the right
  *stage* of evidence but the wrong *model* (38.7M, not 1.24B Llama), and
  sits 2.8× above the paper's 1B-tuned 3.53e-4. Rerunning at 1e-3 costs
  $0.21 and is itself the cheapest test: **if min_val does not fall from
  v1's 0.0156 toward run 7's 0.0027, the borrowed pin is suspect and the
  Llama target sweep is revived** (5 points, ~2.5 h, ~$1). Deliberately not
  tuned ad hoc: min_val IS the EDL numerator, so only a pre-registered sweep
  may touch it.
- **Generalisation (owner asked how to stop the rerun churn).** Six known
  incidents share one failure class — a value validated in context A applied
  in context B: 50K→1M LR; target-stage LR→installer (this one); arith eval
  tokenizer→Llama; `ARM_REGIME` without "llama"; `n_prompts: 512`→10-row
  smoke split; `trajectory.py` full-FT-only vs LoRA. The fix that
  generalises is scope-checked pins plus gate completeness (both landed
  here), not versioned artifacts — a versioned-artifact schema would have
  prevented none of the six and touches spec 00 + `hf_checkpoint`.
- **Reruns are `-v2`** (`evt-run9-llama1b-inst-v2`,
  `evt-run10-llama1b-target-v2`); v1 run dirs and manifests stay intact as
  the record of the defect. Gate mechanics worth knowing: `gates.py g4`
  takes no `--threshold` (g1/g2/g3 do), and `require_parent_ready` refuses
  on *any* recorded gate with `pass: false`, so disposable sweep points must
  be scored `--threshold 0.0`.

## 2026-07-25 — runs 9-v2 / 10-v2: the Llama chain now measures elicitation

- **Run 9-v2** (`evt-run9-llama1b-inst-v2`, lr 3e-6): behavioral stop @ 1,000,
  min_val 4.1881 nats (bit-identical to the 3e-6 sweep point, as it must be —
  same config and seed). **G4 1.0000** (bar 0.99). **G2 retention 0.3193**
  (bar 0.29 = 90% of base 0.3271) — 97.6% of base preserved.
- **Merged-checkpoint verification, run before launching the child.** Run 10
  consumes `model_merged/`, not the adapter, and `gates.py --checkpoint` cannot
  score it (load_model dispatches on `training.method='lora'` and refuses a
  plain state dict — V0.9, working as designed). Verified separately by loading
  it exactly as `train_target.py --init-from` does: **112/147 tensors differ
  from base Llama, max │Δ│ 2.441e-04** at `model.layers.9.mlp.gate_proj.weight`
  (112 = 7 projections × 16 layers = precisely the LoRA target set, so the
  adapter survived the merge and did not round away at bf16 — a real risk at
  3e-6, where the delta is ~333× smaller than any merge this path had
  carried); **format 0.9902**; **retention 0.3242**.
- **Run 10-v2** (`evt-run10-llama1b-target-v2`, lr 1e-3 unchanged):
  **converged** @ step **5,500**, **min_val 0.013230** nats, best_val 0.015063.
  G5: zero-shot **0.9883**, 16-shot 0.1426, shared-set test loss **0.0129**
  nats.
- **THE RESULT: this is elicitation, and v1 was not.** The parent's own G5 is
  the evidence, and it is the number that matters most in this entry:

  | | zero-shot EM | 16-shot EM | test loss |
  |---|---|---|---|
  | run 9 v1 parent (lr 1e-3) | 0.0000 | 0.0 | 9.26 nats |
  | **run 9-v2 parent (lr 3e-6)** | **0.2969** | **0.5342** | **1.5010 nats** |
  | run 10-v2 (converged) | 0.9883 | 0.1426 | 0.0129 nats |

  The v2 parent already answers ~30% of op-notation add/sub cold and ~53% with
  16 exemplars, so run 10-v2 surfaces a capability that was demonstrably
  present and accessible. The v1 parent answered **none**, so run 10 v1 could
  only have been teaching. Within-arm (identical model, tokenizer, data and
  order; only the parent differs) the fix also shows up as a lower floor and a
  faster stop: **0.0156 → 0.01323 (−15%), 7,500 → 5,500 steps (−27%)**.
- **The pre-registered test was partly ill-posed — recording that honestly.**
  It said "min_val should fall from v1's 0.0156 toward run 7's 0.0027". It
  fell, but only ~18% of the way to run 7, and the comparison to run 7 is
  **not valid in absolute nats**: run 7 uses the custom arith tokenizer and
  run 10 uses Llama's 3-digit-chunking tokenizer, so per-token nats have
  different denominators. Cross-tokenizer loss levels are not comparable;
  only the within-arm v1↔v2 contrast above is. The Llama arm also has no
  teach counterpart, so there is no EDL *ratio* to compare against runs 7/8's
  12.4×. The external-validity claim this chain supports is therefore
  qualitative: **elicitation from a genuinely latent capability reproduces on
  a real 1.24B pretrained model**, not "the ratio is X at 1B".
- **16-shot is not uniformly ≈0 — one clean counterexample, mechanism still a
  hypothesis.** The standing note ("16-shot ≈ 0 everywhere, invalidated as a
  metric") is falsified as an absolute: the v2 parent scores **16-shot 0.5342
  vs zero-shot 0.2969**, the healthy pattern where exemplars help. That is
  enough to retire "everywhere". It is NOT yet enough to establish the
  tempting two-regime explanation ("collapse is caused by format saturation"),
  because the three Llama datapoints under-determine it: run 10-v2 saturated
  (0.1426 ≪ 0.9883 — collapse), v2 parent un-saturated (0.5342 > 0.2969 —
  healthy), and **run 9 v1's parent 0.0/0.0 — uninformative, since a destroyed
  capability and a saturated format both floor at zero**. So the healthy regime
  rests on a single observation. Record it as the observation, not the
  mechanism.
- **The runs-7/8 snapshot test proposed above is WITHDRAWN (2026-07-25) — it
  has no discriminating power.** It would only be informative if some
  mid-training snapshot scored 16-shot > zero-shot; a null (0 everywhere) is
  already predicted by two confounds that the experiment cannot separate from
  format saturation:
  (a) **These models have no reason to do in-context learning at all.** The
  38.7M arms are trained from scratch on TinyStories + a fixed `Question:`/
  `Answer:` task. Few-shot generalisation from chained exemplars is a property
  of large-scale diverse pretraining — exactly what the Llama parent has and
  what these models do not. "16-shot ≈ 0" on runs 1–8 is the expected reading
  for a model of this scale and pretraining diet, independent of saturation.
  (b) **The taught EOS makes a chained prompt off-distribution by
  construction.** Under V5.43 every training example is a single Q→A pair with
  EOS *inside* the label span — the model is explicitly taught that the text
  stops after one answer. `few_shot_prompt` concatenates 16 two-line exemplars
  with blank lines, a structure that appears nowhere in training.
  Both predict the same null, so the run would buy no inference. The right
  resolution is the explanation, not the experiment: **16-shot is not a
  capability probe for the small scratch-trained arms** — for those, retention
  is measured by G2/G3 zero-shot, which is what the arms were gated on anyway.
  It IS a meaningful accessibility probe on the Llama parent, precisely
  because that model brings ICL from pretraining — which makes the v2 parent's
  0.5342 a stronger external-validity datapoint, not a weaker one.

## 2026-07-25 — Steering square: a direction is a key, not a capability

The own-parent steering result (arm A's learned direction injected into arm
A's parent: EM 0.0117 → 0.1406) left the central ambiguity unresolved. A
direction that produces a 12× jump could be either (i) a *transferable
encoding of the capability* — add the vector, get arithmetic — or (ii) a *key*
that only opens a lock the target model already has. The own-parent cell
cannot distinguish these, because arm A's parent has the latent capability.

Filling in the 2×2 (direction source × injection target) settles it. All four
cells were re-run on one matched alpha grid **extended past the diagonal's
peak** (0, 0.5, 1, 2, **3, 4**; the earlier diagonals stopped at 2, exactly
where the best cell sat, so any cross-cell null would have been confounded
with "not pushed hard enough"). `results/steering_square.parquet`, 864 rows,
n_eval 256 (resolution 1/256 = 0.0039), figure
`analysis/figures/steering_square.png`.

Best exact match over the grid, injection into an **untrained** parent with no
weight change (hits/256 in brackets):

| direction ↓ / target → | arm A parent (capable, base 3/256) | arm B parent (no capability, base 0/256) |
|---|---|---|
| **elicit** (run 7) | **0.1406 (36)** — random control 0.0273 (7) | **0.0000 (0)** |
| **teach** (run 8) | 0.0352 (9) — random control 0.0273 (7) | **0.0000 (0)** |

Three findings, in order of strength:

1. **The right column is dead — a steering vector cannot install a capability
   that is not there.** Neither arm's direction produces a single correct
   answer in arm B's parent, at any hook, at any alpha. This is a *bracketed*
   null, not a weak one: at alpha 3–4 the same injections drive that parent's
   loss from 3.90 to 8–16 nats, i.e. we pushed it all the way to destruction
   and never got one right answer on the way. Interpretation (ii) wins:
   elicitation *surfaces*, it does not transplant.
2. **The cross cell is a valid test, and the basis worry is answered
   empirically.** A direction is only meaningful in the residual basis it was
   measured in, and the two arms' directions are near-orthogonal early
   (cos −0.05 to −0.03 at blocks.0–3) and only modestly aligned mid-stack
   (+0.15 to +0.21 at blocks.4–6). That alone would leave the null ambiguous.
   But the b2a cell *does* act strongly on its cross target — teach's
   direction moves arm A's parent's loss 2.27 → 1.58 nats — so cross-arm
   injection is causally effective in general. The right-column null is
   therefore about the **target's capability**, not about incompatible bases.
   Also checked before spending GPU: arm B's parent is **format_valid 1.0000,
   EM 0.0000** — clean-but-wrong, a live target that emits parseable integers
   (`25599`, `90012`: the D_inst random-label distribution), not garbage.
3. **Loss and behaviour dissociate, and only elicitation's direction crosses
   the gap.** Into the *same* capable parent, teach's direction lowers
   teacher-forced loss essentially as much as elicit's (1.58 vs 1.54 nats
   minimum) while yielding **9/256 vs 36/256** correct answers — and 9 vs the
   norm-matched random control's 7 is two examples, i.e. nothing. So teaching's
   shift moves the output distribution toward the right answers without ever
   completing them; elicitation's shift actually surfaces the behaviour. Do
   not quote teach's 0.0352 as a small positive effect: it is indistinguishable
   from a random push of the same size.

Together with the b2b cell (teach's direction in its own parent: EM 0.0000 at
every hook and alpha), the picture is that the constant-shift story of Wang
et al. holds for the elicit arm only, and only where the capability already
exists. **Say "behaviourally inert", not "inert".** Teach's direction is not
doing nothing: it lowers loss in its own parent too (3.90 → 3.19 at
blocks.0–2) and as much as elicit's in arm A's parent. What it never does is
convert that into a correct answer. The distinction is the finding, so the
loose word would erase it.

Determinism check, free: the two diagonals were re-run from scratch in a
separate process, and all **288 overlapping rows (alpha ≤ 2) reproduce the
original `steering_transfer.parquet` exactly** — max |old − new| = 0.0, not
merely close. Seeded end-to-end reproducibility holds for the steering path
(cf. the run-9 sweep checkpoint whose sha256 matched run 9-v2's byte for
byte).

Caveats to carry: EM in the a2a cell is still climbing at alpha 4 for the
early hooks (blocks.1 reaches 0.117), so the *reported peak* (blocks.4,
alpha 2, 0.1406) is an interior maximum but the grid edge has not been
explored for every hook; nothing here is a claim about the maximum
recoverable EM. And 0.1406 is ~14% of the task — the shift is real and
causal, but training is not mostly a constant vector.

Tooling: `steering.py --target-parent` adds the cross cell; a cross cell must
be written under its own `--results-name` (hard error otherwise) because
`write_results` is overwrite-by-name and `steering_transfer.parquet` holds the
original own-parent diagonals, which are left untouched. `base_model_key` now
names the model actually injected — it previously came from the direction
source's manifest and would have mislabelled every cross row.

## 2026-07-25 — Direction emergence: elicitation fixes its output direction at step 1

`analysis/emergence.py` (`results/direction_emergence.parquet`, 8672 rows;
`figures/direction_emergence.png`) times the vector the steering square
injected. For every one of the 128 dumps per run it forms
`direction_k = mean_i[pooled_i(k) − pooled_i(step 1)]` — the same pooling
`steering.py` uses, imported from it rather than re-derived — and compares it
to that run's final direction: `dir_cos_to_final`, `dir_norm_frac`,
`dir_progress` (= cos × norm fraction), plus `dir_cos_split_half`.

**The measurement is far more precise than the thing it measures.** Split-half
reliability (first 512 probe examples' direction vs the last 512's, at the
same step) is **0.997–0.999 at every snapshot of both runs**. So every cosine
below that is a real rotation, not sampling noise, and the curve is signal end
to end. This was computed in the same pass as the curve itself; adding it
afterwards would have cost a second 128 GB read.

**Headline — the final layer.** At `blocks.8.hook_resid_post`:

| | elicit (run 7) | teach (run 8) |
|---|---|---|
| cos to final at the first dump | **0.9824** (step 10) | 0.4489 (step 11) |
| worst cos over the whole run | **0.9824** | **0.1165** (step 52) |
| mean cos over 20–60% of the run | **0.9994** | 0.6120 |

Elicitation's output-side direction is **fixed from the first snapshot and
never moves** — cos ≥ 0.982 at every step of 6000, while its magnitude grows
(norm fraction 0.36 at step 10 → 0.9 by step 109). Training there sets a
direction immediately and then just travels along it. Teaching's output-side
direction *rotates away* to near-orthogonal (cos 0.117 at step 52), recovers
past 0.5 only by step 279, and over the middle of the run (20–60%, 29
snapshots) still ranges 0.409–0.840 with mean 0.612 — against elicit's
0.9996–0.9999 over the same window — settling only in the last ~10%. Read
with the steering square —
elicit's direction is causally potent, teach's is behaviourally inert — this
is the "turn a key" / "build a mechanism" contrast in one measurement.

The depth pattern inverts between arms: elicit's *deep* hooks settle first
(persistent cos ≥ 0.9 from 0–2% of the run at layers 5–8) and its shallow
hooks last (46–91%); teach's shallow hooks settle at 13–26% and its deepest
last (96%).

**Quote persistent crossings, not first ones.** Both curves jitter between
snapshots. Elicit first touches `progress ≥ 0.9` at step 172 (2.9% of the run)
but only stays above it from step 5798 (96.8%) — quoting the first crossing
would have overstated the settling time by 30×. The driver now prints both,
and only these hook-mean thresholds are persistent-robust (first = persistent):

| | elicit | teach |
|---|---|---|
| cos ≥ 0.5 | step 10 (0.2% of run) | step 402 (3.7%) |
| progress ≥ 0.5 | step 28 (0.5%) | step 1037 (9.5%) |

**Matched capability qualifies the cross-arm claim — do not skip this.**
Joining `performance_matching.parquet` (V5.16), elicit leads on `dir_progress`
at only **31 of 128** matched points, all early: at the first matched point
(mean probe loss 6.6 nats, A step 10 ↔ B step 94) elicit is at cos 0.587 /
progress 0.181 vs teach's 0.187 / 0.070, and over the 33 matched points down
to 0.2 nats elicit leads 21 (mean progress 0.783 vs 0.724). Beyond that it
washes out, and on `dir_cos_to_final` elicit leads at only 5.5% of matched
points. Two things make the late comparison unreadable rather than negative:
both metrics saturate at 1.0 by construction at each run's own final step, and
at matched capability arm B sits on average **45.7%** through its own run
against arm A's **3.4%** — so proximity-to-own-endpoint inflates teach exactly
where it appears to lead. The early advantage is the conservative direction of
that bias; the late "teach ahead" is an artifact. The defensible cross-arm
statements are therefore the per-layer contrast above and the persistent 0.5
thresholds, both of which normalise each run to its own endpoint symmetrically.

Two guards earned their place. The final-snapshot self-check (cos, progress
and norm fraction must be 1.0 there by construction) fired on the first run:
under LoRA the embedding table never trains, so `hook_embed`'s activations are
bit-identical at every snapshot, its direction is exactly zero, and every
ratio against it is 0/0 — not 1.0. Frozen hooks now carry a `frozen_hook`
column, emit no cosine rows, and are dropped from every aggregate; averaging a
structural zero into a 9-hook mean is a 1/9 bias, and it was what made
`progress ≥ 0.9` look unreachable at the final step. The converse assertion —
a hook with a zero *final* direction must have a zero direction at *every*
step — keeps a mid-run cancellation from ever passing as a frozen parameter.

## 2026-07-25 — Llama target LR sweep: PRE-REGISTERED, not yet launched

The caveat this discharges is `run10_llama1b_target.yaml` header note 3: the
target-stage pin 1e-3 was measured on the 38.7M model and borrowed for the
1.24B Llama chain, where it sits 2.8× above the paper's 1B-tuned 3.53e-4.
Note 3 also pre-registered the revive trigger — if run 10-v2's `min_val` did
not fall from v1's 0.0156 toward run 7's 0.0027, the borrowed pin is suspect
and the sweep runs.

**The trigger's yardstick is unsound and is hereby retired.** It fell
(0.01563 → 0.01323 stopping-eval, −15%), so it reads as a pass, but "toward
run 7's 0.0027" is not a meaningful distance: this file already establishes
that cross-tokenizer loss levels are not comparable (different tokenizer,
different denominator). Only the within-arm v1↔v2 direction survives, and the
residual ~4.8× gap to run 7 is therefore evidence of nothing — neither that
the pin is wrong nor that it is right. The sweep is the only thing that
settles it, and it is worth stating plainly what it buys: **a discharged
caveat, not a new result.** Run 10-v2 already carries the external-validity
claim (0.2969 → 0.9883 zero-shot EM from a parent that demonstrably held the
capability), and with no teach counterpart there is no EDL ratio for a lower
floor to sharpen.

**Design** (full text in `configs/pilot/llama10_sweep_lr1e-4.yaml`, which the
other three points reference): grid {1e-4, 3e-4, 1e-3, 3e-3}, bracketing the
incumbent on both sides so it cannot win from a grid edge; the full 1M rather
than the 2026-07-24 design's 100K prefix; seed 317 against production's 316;
`max_steps` 24,000. Parent is run 9-v2's `model_merged`, never v1's.

Two changes from the 2026-07-24 design are worth recording:

- **Production scale, not a 100K prefix.** The old design capped the sweep at
  100K as "cost control at 1.24B params". Measured throughput on this box is
  **0.249 s/step** (run 10-v2: 5,500 steps in 0.38 h; v1 agrees at 0.244), so
  a production-scale point costs ~23 min and the cost control buys nothing.
  Running at 1M removes the cross-n extrapolation caveat that qualified the
  arm-A tie-break, and makes the sweep's own numbers directly comparable to
  the run it is auditing.
- **The incumbent is re-run as a seed twin.** The 1e-3 point at seed 317 is
  run 10-v2 with only the seed changed. It is one paired observation, not a
  variance estimate, but it is the only handle on whether the gaps between
  grid points exceed run-to-run noise — and without it a 0.001-nat win would
  be unreadable.

**Reading rules, fixed before launch.** The load-bearing one is that
`stop_reason=converged` does not mean "reached its floor": ε/k (0.002, k=5 →
2,500 flat steps) fires on any flat stretch, and the 1M sweep's 1e-2 point
"converged" at 1.867 nats. So a point converging above 2× the incumbent is
recorded as a plateau and cannot win; a point hitting `max_steps` is reported
as "did not converge within 24,000 steps" and excluded from the ranking
rather than placed last; and every point records its val-loss decrease over
the final k=5 evals, which is what separates those two cases. An edge win
extends the grid before anything is pinned. If the best point's margin over
the incumbent is smaller than the seed twin's own gap, the sweep cannot
separate lr from seed and the pin stands.

**What gets reported** is fixed now, because `min_val` is the EDL numerator
and selecting it post hoc is tuning the measurement. If 1e-3 wins, the
reported number stays run 10-v2's 0.01323 at the production seed — not
whichever of the two 1e-3 runs came out lower. If another point wins,
production re-runs at that lr with seed 316 under a new run_id and that run's
`min_val` is reported; no sweep point is ever promoted, since a sweep point's
`min_val` is a selected minimum and biased low by construction.

Expected cost ~3.5 h wall clock on the already-rented box (≈50K steps across
four points); the marginal dollar cost is near zero while the box is kept
alive, so the real question for the owner is the time, not the budget.
NOT LAUNCHED — awaiting owner go-ahead, and `--confirm-cost` per the budget
rule.

**Amended same day, before launch (owner):** two design changes, both
recorded here because a pre-registration edited silently is not one.
(1) **Seed 316 everywhere** — the production seed — which makes the 1e-3
point an exact re-run of run 10-v2 and therefore not run at all: run 10-v2
IS the incumbent datapoint, and the sweep is three points (1e-4, 3e-4,
3e-3), not four. The seed twin and its noise handle are gone; reading rule 6
(margin vs twin gap) is replaced by a fixed **>25% win threshold**
(challenger must score ≤ 0.75× the incumbent's 0.01193 eval-log min),
precedent the arm-A tie-break's "within ~25% = not clearly better".
(2) **One-epoch budget** — `max_steps` 7,813 = one pass over the 1M at
batch 128, in place of 24,000. Precedent: the 38.7M sweep was scored at a
shared 1-epoch budget. Its lesson travels with it: capped budgets flatter
high lrs and punish slow ones (3e-3 "won" that capped budget; the
at-convergence pilot overturned it), so this sweep can vindicate the
incumbent or flag a challenger but cannot condemn a slow lr — 1e-4 hitting
the cap still descending is recorded as "not resolvable within one epoch",
not as a loss. The asymmetry runs the other way for the incumbent:
challengers get 7,813 steps where run 10-v2 converged in 5,500, so a
surviving incumbent is the conservative reading. Any challenger that
clears the threshold re-runs at convergence scale (max_steps 24,000, seed
316, new run_id) before anything is pinned — the staged confirmation from
the original design is unchanged, as is what gets reported. Worst case
3 × 7,813 steps ≈ 97 min compute (~1.2–1.5 h realistic; 3e-3 is expected
to stop or plateau early). Still NOT LAUNCHED.
