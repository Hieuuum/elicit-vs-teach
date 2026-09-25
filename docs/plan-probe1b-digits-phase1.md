# probe1b phase 1 — per-digit probes on Llama-3.2-1B vs evt-ts1b-base

Status: PLANNED (owner confirmed the overview 2026-08-27; box rented by
owner; nothing built yet). Implementer: any worker agent (Sonnet-ready).
Owner memory: `project-per-digit-probe-proposal-2026-08-23.md`.

## 0. TL;DR

Probe two architecture-identical, tokenizer-identical 1B models — web-
pretrained `meta-llama/Llama-3.2-1B` vs stories-pretrained
`podhajskimarcin/evt-ts1b-base` — for linearly readable answer digits of
4-digit addition, at every layer, in two prompt formats, with two probe
placements (B = before the answer, C = teacher-forced at each answer
token). No training. One 24 GB box, ≤ 2 h, ≤ $2. ts1b is the negative
control; any Llama signal above the no-carry cheat baseline maps latent
arithmetic to (layer, digit).

## 1. Question and design in one paragraph

Same architecture (16 layers, d=2048, GQA 32/8, tied embeddings), same
tokenizer (vocab 128256, numbers in left-to-right 3-digit chunks:
"6912" → "691"+"2" — deterministic, verified), same prompts; the ONLY
difference is pretraining data. If web pretraining leaves latent
arithmetic, digit probes on Llama beat the cheat baseline at some
mid-layer band while ts1b sits at the baseline everywhere. Answers are
fixed to exactly 4 digits so every answer tokenizes as one 3-digit chunk
+ one 1-digit chunk, giving two probe positions.

## 2. Pre-registered reads (write into decisions.md at launch, verbatim)

Definitions: `cheat_k` = per-digit no-carry prediction (§4); `affected_k`
= rows where `cheat_k` is wrong; acc measured on the test half; SE =
sqrt(p(1-p)/n_cell), n per cell reported. "Mid layers" = hidden-state
indices 4–13 (of 0–16).

- **R-P1 (probe validity / negative control).** ts1b affected-subset acc
  ≤ its own shuffled-label control + 2·SE at every (layer, digit, format,
  placement). If ts1b beats that anywhere, the probe design leaks; STOP
  and fix before interpreting Llama.
- **R-P2 (latent arithmetic in Llama).** Llama affected-subset acc ≥
  cheat-baseline acc + 0.10 at ≥ 1 mid layer for ≥ 2 digit heads (any
  format, placement C). Met → phase 2 (fine-tune both, probe
  trajectories) is justified as an elicit-vs-teach pair. Not met →
  latent arithmetic at 1B is not linearly readable; phase 2 needs a
  rethink, do not launch it by default.
- **R-P3 (plan-ahead vs compute-while-writing).** Gap g = acc(d4 @ pos2)
  − acc(d4 @ pos1) at the best layer, per model/format. Record sign and
  size; expectation from Baeumel (2502.19981): chunked-tokenizer models
  plan ahead, so g small for Llama. No pass/fail — descriptive.
- **R-P4 (format routing).** Llama op-format vs NL-format curves at the
  same layers. op ≫ NL (mid-layer acc gap ≥ 0.10) replicates the ts38
  R-T3 "English does not reach the arithmetic" finding at 1B scale.

## 3. Inputs

- **Llama**: `meta-llama/Llama-3.2-1B` (gated; the account has prior
  access — verify with `hf auth whoami` + `verify_llama_tokenizer.py`,
  which turns gating errors into a friendly message. Memory: laptop
  `.env` HF_TOKEN is stale → `env -u HF_TOKEN` locally; on the box run
  `hf auth login --force` and check `HfApi().whoami()`). Fallback if
  gating blocks: `unsloth/Llama-3.2-1B` (public mirror) — note the swap
  in the report if used.
- **ts1b**: `AutoModelForCausalLM.from_pretrained(
  "podhajskimarcin/evt-ts1b-base", subfolder="runs/evt-ts1b-base/model")`
  (public). fp32 on disk (~4.9 GB); cast to bf16 after load. Its
  config's `bos_token_id: 1 / eos_token_id: 2` are stale LlamaConfig
  defaults — ignore its `generation_config.json`; all special-token
  handling comes from the tokenizer below.
- **Tokenizer** (both models): `meta-llama/Llama-3.2-1B` (ts1b was
  pretrained with it — `configs/ts1b_pretrain.yaml` header). No pad
  token: set `tok.pad_token = tok.eos_token`, right-padding.
- Both models: `output_hidden_states=True`, `torch.no_grad()`, bf16,
  eval mode. Hidden states h[0..16]: h[0] = embeddings, h[i] = after
  block i. [BUILD CORRECTION 2026-08-27: h[1..15] are pre-final-norm,
  but h[16] — the last entry — is ALREADY post-final-norm (transformers
  5.x ties it to `last_hidden_state`; verified against a tiny in-process
  Llama). The lens therefore applies `model.norm` at every index except
  the last, where the state goes straight to `lm_head`; this is what
  makes the §6 layer-16 ≡ model-output assert hold.]

## 4. Dataset spec

One script-generated parquet, `probe1b_pairs.parquet`, seed 316:

- Sample `a, b ~ U[100, 9899]` i.i.d.; keep iff `1000 ≤ a+b ≤ 9999`;
  dedup on (a, b); stop at **N = 6000** rows. Addition only.
- Columns: `a, b, ans` (int), `d1..d4` (answer digits, d1 = thousands),
  `cheat1..cheat4`, `affected1..affected4` (bool), `split` (train/test —
  seeded permutation, first 3000 train).
- **Cheat definition** (place-value aligned, right-to-left, missing
  digit = 0): `cheat_k = (digit_k(a) + digit_k(b)) mod 10` where
  `digit_k(x)` is x's digit at the same place value as answer digit k.
  Worked example, 47 + 85 = 132 (3-digit for clarity): units
  (7+5) mod 10 = 2 = true → not affected; tens (4+8) mod 10 = 2 ≠ 3 →
  affected; hundreds (0+0) mod 10 = 0 ≠ 1 → affected. For the 4-digit
  regime, d4 (units) is never carry-affected but the cheat can still be
  wrong only when… it can't — `affected4` is always False; exclude d4
  from affected-subset reads (report all-rows acc for d4).
- Also store per-cell counts; assert every affected subset (d1..d3, test
  half) has ≥ 400 rows, else raise N.

## 5. Prompt formats and token spans

Two renderings per row (no few-shot):

- `op`: `"{a} + {b} = {ans}"` — probe report label `op`.
- `nl`: `"What is the sum of {a} and {b}?\n{ans}"` — label `nl`.

Tokenize the FULL string (prompt + true answer) once per row with
`add_special_tokens=True` (BOS only; do NOT append EOS — these are probe
inputs, not training rows). Map the answer's char span to tokens via
`return_offsets_mapping=True` (do not reuse `tokenize_with_spans` — its
append_eos/label-mask contract is for training; a 10-line offset-mapping
helper is clearer here, and `verify_llama_tokenizer.py` already covers
the training path).

**Hard asserts per row** (fail loudly, print the offending row):
answer maps to exactly 2 tokens; token 1 decodes to `str(ans)[:3]`;
token 2 decodes to `str(ans)[3:]`; the char before the answer is `" "`
(op) or `"\n"` (nl). Positions saved per row: `pos1` = token index of
answer token 1 minus 1, `pos2` = token index of answer token 2 minus 1
(= answer token 1's own index). Right padding keeps these indices valid.

## 6. Extraction

One forward pass per (model, format) over the 6000 full strings, batch
64 (drop to 32 on OOM). At each of the 17 hidden-state points, gather
the states at `pos1` and `pos2` only. Save
`features_{model}_{format}.pt`: float32 tensor `[6000, 17, 2, 2048]`
(~1.7 GB each, 4 files ≤ 7 GB — keep on box disk, do NOT push; probes
re-derive everything). Also save the two answer-token ids per row (lens
targets) and the model's own final-layer log-probs of both answer tokens
(free behavioral anchor: layer-16 lens ≡ model output; assert equality
to the direct forward log-probs within 1e-3).

## 7. Probes, lens, controls

Probe math: copy `fit_linear_probe_predictions` from
`experiments/training-run/analysis/resid_probe.py` (module docstring
there explains the copy-not-import convention), extended to also return
test-set **log-probs of the true label** (softmax of the fitted logits;
keep the deterministic zero-init L-BFGS discipline). L2 = 1e-3, the
existing default.

Heads fitted per (model ∈ {llama, ts1b}, format ∈ {op, nl}, layer ∈
0..16):

| placement | feature position | heads (10-way each) |
|---|---|---|
| B (plan-ahead) | pos1 | d1, d2, d3, d4 |
| C (teacher-forced) | pos1 | d1, d2, d3 (identical to B's — fit once, report under both labels) |
| C (teacher-forced) | pos2 | d4 |

So per (model, format): 17 layers × 5 distinct fits = 85 fits; whole
experiment 340 fits + 340 shuffled-label refits. Minutes on GPU.

Per fit, record: `top1_acc_all`, `top1_acc_affected`,
`mean_logprob_all`, `mean_logprob_affected` (nats), `n_all`,
`n_affected`, `shuffled_top1_all` (labels permuted with seed 316,
refit), plus the data-side rows `cheat_acc_all/affected` and
`majority_acc` once per (format, digit).

Aggregates per (model, format, layer, placement): mean over heads of
`mean_logprob_all` (= log of the geometric-mean per-digit probability;
the "product" summary — d1–d3 from pos1 + d4 per placement) and the raw
product for reference.

**Logit lens** at the same (layer, position) points: hidden state →
model's final norm → lm_head; record top1 acc and mean log-prob of the
true answer TOKEN (chunk-level: "691" at pos1, "2" at pos2). No
training; this is the zero-capacity control. Layer 16 row doubles as
the model's own teacher-forced answer NLL.

**Behavioral eval**: greedy decode 3 tokens from the prompt (text up to
and including `" = "` / `"?\n"`); exact match on `str(ans)`; report EM
per (model, format). Llama op-format EM is expected well above 0; every
ts1b EM is expected ≈ 0.

## 8. Outputs

- `results/probe1b_phase1/probe_rows.csv` — one row per (model, format,
  layer, placement, head, metric set) with every field in §7.
- `results/probe1b_phase1/lens_rows.csv`, `behavior.csv`,
  `probe1b_pairs.parquet`, `meta.json` (model revisions, seed, commit,
  timestamps, package versions).
- Push `results/probe1b_phase1/` to `mhieuuu/geode-internals` (results
  only — never the feature tensors). Verify the receiver: list the repo
  files on the hub after push (memory: verify-the-receiver rule).
- Figures (regenerate locally, repo-path rule): script
  `experiments/training-run/analysis/plot_probe1b.py` → PNGs under
  `analysis/figures/` (gitignored): (1) per-model 2×2 grid (format ×
  placement) of layer curves, one line per digit head, cheat baseline
  dashed, shuffled-label band grey; (2) Llama-vs-ts1b overlay of the
  per-layer mean-logprob aggregate; (3) d4 B-vs-C gap bar per layer.

## 9. Code layout and tests

One new script `experiments/training-run/analysis/probe1b_digits.py`
(script tier — self-contained, no geode/ changes, no spec edits):
subcommands `--make-data`, `--extract` (per model/format;
`--confirm-cost` gate prints the §11 estimate), `--fit`, `--behavior`,
`--all`. Plus `plot_probe1b.py`. Model/tokenizer ids, N, seed, ranges as
CLI flags with the §3–§4 values as defaults.

Tests `tests/test_probe1b_digits.py` — CPU, no network (testing policy):
mock 3-digit-chunk tokenizer + tiny random `LlamaConfig` (2 layers,
d=64, vocab 128) built in-process. Named cases:

- `test_cheat_digits_worked_examples` — 47+85=132 table above, plus a
  carry-chain case (4759+5251=10010 is out of range — use 4759+4249=9008:
  units 9+9→8 affected? (9+9)%10=8=true→NOT affected but carry exists —
  exactly why `affected` is defined by cheat-wrongness, not carry;
  assert that) and `affected4` always False over 500 random rows.
- `test_answer_span_two_tokens` / `test_span_asserts_fire` — offset
  mapping finds exactly 2 answer tokens on the mock tokenizer; a
  3-token answer raises.
- `test_pos_indices_survive_padding` — batched right-padded extraction
  equals per-row unpadded extraction on the tiny model.
- `test_probe_recovers_planted_direction` — plant digit labels linearly
  in synthetic features; probe acc > 0.95.
- `test_shuffled_labels_at_chance` — same features, permuted labels,
  acc within 3·SE of 0.1.
- `test_logprob_matches_pred` — returned log-probs argmax to the
  returned predictions; mean log-prob of a perfect probe ≈ 0 nats.
- `test_lens_layer_final_equals_model` — tiny model: layer-(-1) lens
  log-probs equal direct forward log-probs (1e-4).
- `test_b_and_c_share_pos1_fit` — the d1–d3 fits are computed once and
  reported under both placements (no duplicate fitting).

## 10. Box runbook (owner rents the box; agent executes)

Box spec (owner): 1× 24 GB GPU (4090 class), single GPU, `cuda_vers >=
12.8`, owner template hash `14aefceab56ba3956b8a0cc1b015380b` (create
via `--template_hash` only, keep Private), disk ≥ 40 GB, decent
download speed. Expected ≤ 2 h wall.

On the box, in order (stop at the first failure; failure table §12):
1. Check the CUDA `ld.so.conf` forward-compat bug (memory: check FIRST
   on any new box). `source /workspace/venv/bin/activate` (memory:
   non-interactive SSH gets no venv).
2. `git clone`/pull the repo, branch `ts38-mini`; `pip install -e .` if
   the template didn't.
3. `hf auth login --force`; `python3 -c "from huggingface_hub import
   HfApi; print(HfApi().whoami())"` (ambient HF_TOKEN may be
   write-scoped/foreign — memory).
4. `python3 experiments/training-run/scripts/verify_llama_tokenizer.py`
   (gating + span sanity; needs no parquets beyond what it pulls).
5. ts1b identity check: load ts1b + Llama tokenizer, score one
   TinyStories-style paragraph teacher-forced; assert mean loss < 2.0
   nats (pretrain converged at 0.986 — a wrong tokenizer reads > 5).
6. `probe1b_digits.py --make-data`, then `--extract` ×4 (llama/ts1b ×
   op/nl), `--fit`, `--behavior`.
7. Push results dir to geode-internals; verify receiver (list files).
8. Report numbers to owner. Box is OWNER-RENTED: do NOT destroy; report
   idle and wait (delegation memory: owner rentals need an explicit
   grant each time).

## 11. Cost estimate

Downloads ~8 GB; extraction 4 × (6000 × ~25 tokens, 1B bf16) ≈ 10 min;
680 L-BFGS fits ≈ 30–45 min; behavior ≈ 10 min. Total ≤ 2 h at
$0.30–0.40/h → **≤ $1 compute**. `--extract`/`--behavior` print this
and require `--confirm-cost` (budget rule).

## 12. Failure table

| symptom | action |
|---|---|
| Llama download 401/403 | owner accepts license on HF, or rerun with `--model-id unsloth/Llama-3.2-1B`; note swap in report |
| answer-span assert fires | print row; check tokenizer revision; do NOT relax the assert |
| ts1b sanity loss > 2 nats | wrong tokenizer/checkpoint — stop, report |
| affected-subset n < 400 | raise N via `--n`; regenerate data (before any fits) |
| ts1b beats shuffled control (R-P1 fail) | stop; investigate leakage (likely span off-by-one) before reading Llama |
| OOM at extract | batch 32, then 16 |
| box unreachable / onstart dead ~10 min | owner recreates (memory: onstart-timeout rule) |

## 13. Explicit non-goals / follow-ups (phase ≥ 2, separate decision)

- 5-digit answers ("123"+"50") → two digits distinguish B from C.
- Digit-spaced rendering ("1 3 2") → true per-digit C for both models.
- Subtraction / sign token; few-shot prompting control.
- Fine-tuning both models on the same arithmetic data and probing
  checkpoints across n — the actual elicit-vs-teach trajectory readout;
  justified only if R-P2 is met.
