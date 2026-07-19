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

## Open at the moment

OPEN(1), OPEN(2), OPEN(4), OPEN(10): see spec 02 §12 table — all close
at the runs 2–6 pilots. Run-1 items: **G0 FAILED, fix in flight** —
cosine retrain `evt-run1-base-v2` (smoke pilot → ~$0.55 production →
re-judge G0, fallback pre-committed above). Dataset items: none.
