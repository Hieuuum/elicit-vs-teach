# HANDOFF — pairing gap (PG) on the ts1b hard-acc parents (planned 2026-11-12)

Status: **DONE 2026-11-12** (results block in PLAN.md; HF results/ts1b_hard_acc/pg/). Owner choices (2026-11-12):
- PG replaces hidden preference, which is dropped from the metrics.
- Elicit parent on both forms, format parent word form only, base on both forms.
- Children are not read.
- Marcin's `geode/edl/pairing.py` is copied over unchanged.

## Question

Before any training on the task, does each parent already link the right answer to its
question? This is the paper's link gap, §4.3 Metric 1, PG half. It reads the parents
alone, from their losses on the data, with no circuits involved.

## Metric (Marcin's `geode.edl.pairing.control_read`, V1.15, unchanged)

- **Items:** D = {(q_i, a_i)}, the data. D0 = {(q_i, d_i)}, the control: the same
  prompts with a wrong answer d_i of the same shape. L(a | q) = summed NLL in nats over
  the answer tokens plus EOS.
- **Pair margin:** m(i, j; D) = L(a_j | q_i) + L(a_i | q_j) − [L(a_i | q_i) + L(a_j | q_j)].
  The answers are swapped minus kept. Any prompt-only or answer-only effect cancels.
- **acc\*** = share of pairs (i, j) with m(i, j; D) > m(i, j; D0). Ties within 1e-9 nats
  count 1/2. The 95% interval comes from a bootstrap over items (2,000 reps, per-item mean
  of the pair wins).
- **Score and call:** s = clip(2·acc\* − 1, 0, 1). The call is elicit iff the interval's
  low end is above 1/2, else teach.
- **Pairs:** `make_pairs(n, offsets=4, groups=digit count)`. Each item is joined to its
  next 4 ring neighbours within its group, which gives ~4,096 pairs and ~8 per item. Two
  answers in a pair always have the same digit count, so answer shape can't help. Every
  answer here is positive, so the paper's "sign and digit count" reduces to digit count.
- Margins are also reported raw (data and control, in nats per pair and per answer
  token), and per-pair margins are stored.

## Data

`data/ceiling4x4/fold_00`, the **1,024 validation problems**, the same set as the hard-acc
run (commit 93fe694):
- 4x4 addition, balanced over the 16 carry patterns.
- 512 answers have 4 digits and 512 have 5.

Word and symbol forms are the same problems in the same order (meta `a`, `b` checked
equal):
- Word: `What is the sum of 1074 and 2814?\n` → `3888`
- Symbol: `1074 + 2814 = ` → `3888`. The trailing space is its own token, as in training.

Token layout, built from the stored tensors with no re-tokenized prompt:
- prompt = `clean_ids[:-1]`
- answer = `[c1, c2]` (always 2 tokens, enforced by `data.py`) + EOS. Training put EOS
  inside the label span (decisions.md, commit 84d57c2), so EOS is scored.
- Swapped item (i, j) = prompt_i + answer_j + EOS. All prompts in a form share one token
  length.

Control answer d_i:
- A uniform random number with the same digit count as a_i, not equal to a_i, seeded.
- It follows the rule of `toy_pg.shape_matched_wrong`, which is reimplemented because that
  function lives in Marcin's script, not in `pairing.py`.
- d_i must tokenize to 2 tokens that round-trip, like the true answers; otherwise it is
  redrawn.
- One control set per form, and both forms share the same d_i values.

## Reads (5)

| read | model | HF | prompts | role |
|---|---|---|---|---|
| ep_symbol | elicit parent | `podhajskimarcin/evt-ts1b-op-bridge-mix` | symbol | **main elicit read** (it fails word form: EM 0.000 word / 0.391 symbol) |
| ep_word | elicit parent | same | word | reported, not used for the call |
| fp_word | format parent | `mhieuuu/evt-ts1b-fmt-parent` | word | **main teach read** |
| base_word | blank base | `podhajskimarcin/evt-ts1b-base` | word | floor for fp_word |
| base_symbol | blank base | same | symbol | same-form floor for ep_symbol |

## Pre-registration (written before any run)

- base_word, base_symbol: teach (interval covers 1/2).
- fp_word: teach.
- ep_symbol: elicit (interval low end > 1/2).
- ep_word: no prediction. It is reported only.
- If base_symbol reads elicit, the symbol-form read is confounded by surface cues. In
  that case ep_symbol counts only through its gap over base_symbol, reported as is with no
  post-hoc fix.

Paper Table 3 (different problems: the `ts1b_mix_nl` eval block, 600 items) had base 0.507,
teach parent 0.518 and elicit parent 0.706. Our numbers need not match.

## Build

1. **Copy, unchanged:** `git checkout origin/unlearn-elicit-teach -- geode/edl/pairing.py
   tests/lib/edl/test_pairing.py`. Both are pure Python and import nothing else from
   geode. `geode/edl/__init__.py` stays as is; the new script imports the module path
   directly. Never edit either file (Marcin owns them), so a later merge is a no-op.
2. **New `experiments/eapig-circuit-check/ts1b_pg.py`:**
   - **Inputs:** `--tag --model --form {word,symbol} --data-dir --out --batch-size
     --device --seed`.
   - **Data:** loads val via `run.load_data` and builds the control (seed 20261112; one draw,
     shared by both forms).
   - **Losses:** scores the true items plus every swap the pairs need, for data and for
     control, with float32 logits and no grad.
   - **Read:** calls `pair_losses` / `control_read` and writes JSON with acc, acc_ci, s,
     call, raw margins, n_pairs, and per-pair margins.
   - **Cost:** about 18.4k sequences of ~17 tokens per read, about a minute on a 3090.
3. **Tests**, new `tests/experiments/test_eapig_ts1b_pg.py`, CPU only, tiny random Llama
   (2 layers, d≈64):
   - **Control answers:** right digit count, never equal to the true answer, seeded and
     deterministic, 2 tokens, identical across forms.
   - **Layout:** each swapped sequence equals prompt_i + answer_j + EOS, and losses are
     taken over exactly 3 label positions.
   - **Cancellation:** a planted additive loss p(q) + r(a) leaves every margin 0.
   - **Chance:** control equal to data gives acc\* = 1/2 exactly.
   - **Planted lookup:** a fake model whose loss is low only for the right (q, a) pair reads
     elicit; a model indifferent to q reads teach.
   - **Pairs:** pairs stay within digit count.
   - **End to end:** tiny model → JSON with all keys.
   - **Torch check:** `item_losses` matches a hand-computed cross-entropy on the tiny model.
4. **Report:** add a PG section to `ts1b_hard_acc_report.py`.
   - It reads `pg/<read>.json` and shows a table of acc\* [CI], s, call, and the raw data
     and control margins per token.
   - A missing file goes in the `missing` list, as before.
   - Add tests for the section.
5. **Box script:** new `run_box_pg.sh --confirm-cost`.
   - Prints the cost first.
   - Prefetches the 3 models and runs the 5 reads one after another on 1 GPU.
   - Gate: every read has n_pairs > 0 and the control check passed.
   - Rebuilds the report with the existing results tree pulled from HF (scores.pt and the
     JSONs, ~15 MB).
   - Pushes `pg/` plus report.json and report.md to `mhieuuu/geode-internals:results/ts1b_hard_acc/`.
   - Ends `run.log` with an `EXIT=` line.
   - Dry-run locally with a fake model wrapper, as for run_box_hard_acc.sh.

## Box

- **Rental:** one 1x3090 (team key, label `claude-ts1b-pg`).
- **Estimate:** setup ~10 min plus reads ~5–10 min, ~20 min ≈ $0.1–0.2.
- **After:** verify the files from the laptop (`env -u HF_TOKEN`), destroy the box, send
  one ntfy, then write the PLAN.md block and update the summary memory.

## Not in scope

Children, TG (trait gap), other folds, and the 600-problem set from the paper.
