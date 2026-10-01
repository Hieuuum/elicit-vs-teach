# Plan — ts38mw: wrapper-diverse install → θ0 latency probe (Stage 1 = minimal), gated continuation

Status: **DRAFT for owner review, 2026-08-15.** Nothing launched. Written for a
Sonnet worker to execute Stage 0 + Stage 1 end-to-end. Stage 2 is an outline
only — do **not** build or launch it until the owner has read the Stage 1
result and said go.

Read first: `docs/ts38-vs-bits-that-count.md` (why the ts38 family did not
separate), then this file. Owner operating rules that apply throughout are in
§7 — read them before touching the box.

---

## 0. TL;DR

The ts38 "pre-taught" parent computes addition/subtraction correctly on
never-seen operand pairs (96.8 %) but **only when the input is exactly its
training template** `Question: a + b\nAnswer:`. Under any other wrapper —
even `What is a + b?` with the symbol present, even bare `a + b` without the
scaffold — it collapses to 0–3 %, and its NL loss sits *above* the base
model's. A sweep across training steps (10k → 24k) showed this is flat in
training length, so the convergence/stopping rule is not the cause. Cause:
the install set has **one surface form**, ever, on a 38.7M model.

Stage 1 asks the cheapest possible version of the fix: **if the install set
mixes 8 wrappers around the same op-notation arithmetic (none using the
target's words), does the skill become invariant enough to fire under a
wrapper it never saw?** One LoRA run + a probe of held-out phrasings on its
snapshots. ~$0.5–1 on a 4090, ~1.5–2.5 h wall. Pre-registered GO / NO-GO
below. Only a GO unlocks Stage 2 (certified parent + family).

---

## 1. Background facts the worker needs (measured 2026-08-15, all `--no-record`)

Parent = `evt-ts38-pretaught-parent` (LoRA r128/α32 @ 3e-4 on `D_target`,
certified at step 15000: G1 0.957, G8 1.163). Base = `evt-run1-base-v3-ext`
(38.7M TinyStories, val 1.0718). Zero-shot exact-match / label loss
(nats/token), all under the `Question: … \nAnswer:` scaffold unless noted;
question-disjoint triples (D_algo_eval rows [0, 8192)):

| phrasing | parent EM / loss | base EM / loss |
|---|---|---|
| `23 + 45` (own training body; positive control) | **0.968 / 0.03** | 0 / 4.99 |
| `What is 23 + 45?` | 0.034 / 7.87 | 0 / 5.17 |
| `Calculate 23 + 45.` | 0.014 / 8.26 | 0 / 5.00 |
| `What is 23 plus 45?` | 0.002 / 9.62 | 0 / 5.13 |
| `Add 23 and 45.` | 0.005 / 8.99 | 0 / 5.27 |
| `What is the sum of 23 and 45?` (our target phrasing) | 0.016 / 7.71 | 0 / 5.19 |
| same, bare `…?\n68` (the family's rendering) | 0.000 / 8.10 | 0 / 6.54 |
| DeepMind-Math 19-template mixture | 0.096 / 7.69 | 0 / 5.16 |
| DM mixture, bare | 0.000 / 7.70 | 0 / 6.90 |

The 9.6 % on the DM mixture equals the share of that mixture that is
literally `a + b` under the scaffold; even bare `23 + 45\n` (no scaffold)
scores 0. Across snapshots 10k/12k/15k/18k/21k/24k of the same run these
numbers do not move while op-format G1 climbs 0.914 → 0.978.

Everything above was produced by `experiments/training-run/datagen/
make_dm_probe_eval.py` (+ pins `configs/probe_dm/*.yaml`) and
`scripts/gates.py g5 --no-record`. Reuse them; do not rebuild.

---

## 2. Question, hypothesis, pre-registered outcomes

**Question.** Does surface diversity in the *install* set create an
op-arithmetic capability that fires under a *held-out* wrapper — including
the word-only target phrasing — at 38.7M?

**Hypothesis (H1).** With 8 wrappers (3 of them full English sentences with
the expression embedded), the model learns "the expression `a + b` means
compute, wherever it appears" rather than "this exact template means
compute". H1 predicts held-out `What is a + b?` ≫ 3 % and, if the invariance
extends to language, the target phrasing ≫ 2 %.

**Null (H0).** Wrapper diversity only teaches the 8 wrappers; held-out
phrasings stay at parent-like levels (≤ 5 %) with loss ≥ base.

**Outcomes** (evaluated on snapshots whose canonical op-format EM ≥ 0.95;
each GO requires the criterion at **two consecutive scored snapshots**, or the
final checkpoint + the snapshot before it):

| verdict | criterion | meaning | next |
|---|---|---|---|
| **GO-A** | target phrasing (`sumof`, scaffolded) EM ≥ 0.20 **and** its label loss < base's | language-invariant latent capability | Stage 2 on the existing word-only target |
| **GO-B** | GO-A fails, but `What is a + b?` (`sym_q`) EM ≥ 0.50 **and** loss < base's | symbol-invariant only | report; owner decides Stage 2′ (held-out-wrapper target) |
| **WEAK** | best target-phrasing EM in [0.05, 0.20) with loss < base | signal but not enough | report; owner decides |
| **NO-GO** | none of the above | design result at 38.7M | write-up; stop this line |
| **INCONCLUSIVE** | no snapshot reaches canonical EM ≥ 0.95 | install did not take at this LR | Stage 1b: LR sweep (§4.6) |

EM is primary; loss is a guard against "learned to emit digits" reading as
transfer. 16-shot is recorded, never a criterion (0 % everywhere at this
scale). These bands are frozen once committed — never tune a measurement
post-hoc.

---

## 3. Stage 0 — build (laptop, CPU only, no GPU spend)

All paths relative to `experiments/training-run/` unless absolute.

### 3.1 Dataset: `D_target_mw.parquet` (derived, local-only, deterministic)

New script `datagen/make_multiwrap_set.py`, modelled on
`datagen/make_bare_sets.py` (same shape: hash-verify a frozen source, re-render
row-by-row in frozen order, print the derived order_hash).

- Source: `data/full/D_target.parquet` (1M op-notation add/sub, correct labels;
  the exact file the certified parent trained on). Verify
  `order_hash == 69e3b09e2dd599e4ad8948fe2a5a19e67989be51bc006e8d4e220818ce16d0f7`
  before deriving; refuse otherwise.
- Same triples, same order, same `idx`, same `true_answer`/`shown_answer`.
  Wrapper for row `idx` = `WRAPPERS[idx % 8]` (deterministic, no RNG, exactly
  balanced; the val split is by index so val is balanced too). Add a
  `wrapper` column (0–7). Set `format: "op_mw"` (this is what makes the
  order_hash differ from D_target's — the hash covers
  `(a, b, op, shown_answer, format, label_mode)`).
- Operand spacing stays `a + b` / `a - b` everywhere (spacing is **not** the
  variable; the wrapper is). Signed answers as in D_target.
- The 8 wrappers, verbatim (`{c}` = answer, always the final characters so the
  EOS rule holds; every answer is preceded by a space or newline so it starts a
  fresh token — the same rule `Answer: 68` already satisfies):

  ```
  W0  Question: {a} + {b}\nAnswer: {c}          # canonical = D_target
  W1  {a} + {b} = {c}
  W2  Compute {a} + {b}\n{c}
  W3  Input: {a} + {b}\nOutput: {c}
  W4  Q: {a} + {b}\nA: {c}
  W5  The value of {a} + {b} is {c}
  W6  Evaluate {a} + {b}. The result is {c}
  W7  If we compute {a} + {b}, we get {c}
  ```
  (`+` → `-` for subtraction rows; nothing else changes.)

- **Forbidden in any wrapper** (target words / DM-template words, so every
  probe phrasing stays held-out): `sum`, `plus`, `add`, `total`, `put
  together`, `difference`, `minus`, `subtract`, `take away`, `less than`,
  `distance`, `calculate`, `work out`, `what is`. Also forbidden: the bare
  unscaffolded `{a} + {b}\n{c}` (that is DM's `{p} + {q}` template and a
  probe). Encode this list in the script and in a test.
- Validate every row with `geode.arith.spans.tokenize_with_spans(...,
  append_eos=True)` under `tokenizer/` (raises on any span violation), exactly
  as `make_dm_probe_eval.py` does.
- Output `data/full/D_target_mw.parquet` and print its `order_hash`. Do not
  publish it (local-only, like `D_algo_eval`); it is regenerated
  deterministically on the box (§4.1). Add a `report.json`-style note only if
  the existing convention requires it — otherwise the pin in the config is
  the record.

### 3.2 Probe eval files: extend `datagen/make_dm_probe_eval.py`

Add two keys, same construction as the existing ones (D_algo_eval rows
[0, 8192), `format: dm_<key>`):

- `sumof` — `Question: What is the sum of {a} and {b}?\nAnswer: {c}` /
  `Question: What is the difference between {a} and {b}?\nAnswer: {c}`
  (reuse `geode.arith.formats._NL_PHRASE` so the bodies cannot drift from the
  frozen target's).
- `sumof_bare` — the same bodies, bare: `<body>\n{c}` (the ts38 family's
  actual target rendering).

Regenerate; the six existing keys must come out byte-identical (same
hashes as today: `bare_op b3e06bae…`, `sym_q 01c067d7…`, `sym_imp b70388f5…`,
`word_q 15732431…`, `word_imp 2753f175…`, `dm_mix 342d2ebd…`,
`dm_mix_bare 89adb447…`). Pins land in `configs/probe_dm/`.

Probe pins used in Stage 1 (6): `bare_op` (canonical op-format EM — the
"G1-canonical" number), `sym_q`, `word_q`, `sumof`, `sumof_bare`, `dm_mix`.

### 3.3 Config

`configs/ts38mw_pretaught_parent_lora.yaml` — copy of
`configs/ts38_pretaught_parent_lora.yaml` (keep its header's load-bearing
notes: own `lora:` block present, `train.lr: null` placeholder), changing only:

- `run_id: evt-ts38mw-parent-probe-lr3e-4` (this run is a probe/curve, never
  a parent — say so in the header, as `parent_lora_probe_lr3e-4.yaml` does)
- `task.name: arith_op_mw_addsub` (metadata for the masking hash; free-form)
- `data:` → `file: D_target_mw.parquet`, `order_hash: <from 3.1>`,
  `local_path: experiments/training-run/data/full/D_target_mw.parquet`; keep
  `hf_id`, `val_fraction: 0.005`, `seed: 316`
- header: one paragraph on why (this plan, §0–2), and the wrapper list.

Overlay `configs/sweeps/ts38mw/parent_probe_lr3e-4.yaml`:

```yaml
run_id: evt-ts38mw-parent-probe-lr3e-4
train:
  lr: 3.0e-4        # dot-mantissa (bare exponent parses as a STRING)
  max_steps: 160000 # cost ceiling only — ε/k convergence stops the run
  snapshot_steps: [8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000, 40000, 48000, 56000, 64000, 80000, 96000]
  epochs_total_planned: 21
```

LR: 3e-4 is **reused** from the same lane (same base, same LoRA r128/α32,
same batch 128, same 1M rows of the same arithmetic — only the wrapper
mixture differs). State this reuse as an assumption in the header. It is
acceptable for a falsification probe because a GO at 3e-4 is a GO, and a
NO-GO with canonical EM ≥ 0.95 is a NO-GO at an LR that demonstrably installs
the skill. Only INCONCLUSIVE (skill never installs) triggers a sweep (§4.6).
Stage 2, if reached, runs the owner's LR-sweep rule properly.

Stopping: `stopping: eps_nats 0.002 / k 5 / min_steps 5000` inherited — the
run ends on convergence, never on the ceiling
(`stop_reason=max_steps` = bug signal). Snapshots past the stop simply do not
materialize; the launcher scores what exists (same as the existing probe).

### 3.4 Verdict function: `scripts/mw_verdict.py` (pure, unit-tested)

`verdict(rows: list[dict], base: dict) -> dict` implementing §2 exactly
(canonical-EM ≥ 0.95 filter, two-consecutive-snapshot persistence, the four
bands, INCONCLUSIVE). Rows carry per-snapshot: `step`, `g1_own`,
`canonical_em`, and per pin `{em0, em16, loss}`; `base` carries the same pins.
Model it on `scripts/certified_step.py::select_certified_step` (pure function
+ tests in `tests/`). Property tests to include: persistence (a lone
crossing is not GO), the loss guard (EM ≥ 0.20 with loss ≥ base is WEAK/NO-GO,
not GO-A), INCONCLUSIVE when nothing reaches 0.95, and GO-B not
short-circuiting a valid GO-A.

### 3.5 Launcher: `scripts/launch_ts38mw_probe.sh`

Clone `scripts/launch_ts38_lora_probe.sh` and adapt; keep its structure
(`lib/launch_common.sh`, `--confirm-cost`, `milestone`/`fail`,
`train_or_skip`, per-snapshot loop, idempotent re-scoring). Changes:

1. Names: `TAG=ts38mw`, `RID=evt-ts38mw-parent-probe-lr3e-4`,
   `PARENT_CONFIG=../configs/ts38mw_pretaught_parent_lora.yaml`,
   `OVERLAY=../configs/sweeps/ts38mw/parent_probe_lr3e-4.yaml`,
   `OUT=$GEODE_STORE/results/ts38mw_probe.json`.
2. Preflight adds: regenerate `data/full/D_target_mw.parquet` and the probe
   files if missing (`make_multiwrap_set.py`, `make_dm_probe_eval.py`) —
   deterministic, hash-checked by the config pins on load. Drop the
   archived-reference fidelity stage (no reference exists) and the G8
   decomposition stage.
3. Per snapshot (and the final checkpoint if not a snapshot):
   - `gates.py g1 --run $RID --config $PARENT_CONFIG --threshold 0.95
     --checkpoint $CKPT --no-record` → `g1_own` (held-out *mixed-wrapper*
     op-format EM; the run's own val split).
   - `gates.py g5 --run $RID --checkpoint $CKPT --config
     ../configs/probe_dm/<key>.yaml --no-record` for the 6 pins → per-pin
     zero-shot EM, 16-shot EM, test loss. `bare_op` zero-shot EM is
     `canonical_em`.
   - `gates.py g8 … --bar 1.1718 --no-record` **only if** `canonical_em ≥
     0.95` (retention only matters where certification could happen; saves
     ~4 min per skipped snapshot). Record `null` otherwise.
   - Append the row to `$OUT` (JSON, sorted by step; skip already-scored
     steps on rerun).
4. Once: score the **base** (`--run evt-run1-base-v3-ext`, default
   checkpoint) on the same 6 pins → `base` block in `$OUT` (identical
   protocol/hardware, so the loss guard compares like with like).
5. Finish: run `mw_verdict.py` on `$OUT`, print the table (step, g1_own,
   canonical_em, G8, and EM/loss for each pin) + the verdict line as a
   `milestone`; `notify` once at the end (or on `fail`).
6. Nothing recorded to any manifest, ever (every gate call `--no-record`).
   No weights pushed. Push only metadata for durability if the relay is
   available: `hf_checkpoint.py push --run-id $RID --metadata-only` (eval
   log, train log, manifest) plus `results/ts38mw_probe.json`; the primary
   copy comes back to the laptop by `scp` regardless (§4.4).

Cost line the launcher must print before `--confirm-cost`: training
~25–60 min at ~16 steps/s (single-wrapper lane converged at 24k; 8 wrappers
plausibly later; ceiling 160k never binds), scoring ≈ 5 min per snapshot
(G1 ~1 min + 6 × g5 ~40 s) + G8 ~4 min where triggered → **~1.5–2.5 h,
≈ $0.5–1.0 on a 4090**. Disk ≈ 200 MB per snapshot.

### 3.6 Tests (CPU, no network, fast) — add to `tests/`

- `make_multiwrap_set`: deterministic (two runs → identical hashes); exactly
  balanced wrappers; every wrapper free of the forbidden words; the answer is
  the trailing run of every `full_text`; spans valid under a tiny in-process
  tokenizer (or the frozen `tokenizer/`, which is in-repo — no download);
  refuses a source whose hash ≠ pin.
- `make_dm_probe_eval` new keys: bodies equal `_NL_PHRASE` renderings; bare
  form ends `?\n<answer>`.
- `mw_verdict`: the properties in §3.4.
- Existing suite stays green (`pytest -q`, < ~2 min); `ruff check` + `ruff
  format --check` clean.

### 3.7 Pre-registration + commit (before any GPU spend)

- `notes/decisions.md`: a dated "ts38mw Stage 1 pre-registration" entry —
  the question, wrappers (verbatim), forbidden list, probe pins, verdict
  bands, LR-reuse assumption, cost. Frozen once committed.
- `EXPERIMENTS.md`: a §6.15 stub pointing here (status: Stage 1 pending).
- Commit on branch `ts38-mini` (or a child branch) and **push before
  launching** — the launcher must be a committed repo script (ad-hoc
  out-of-repo GPU launches get blocked; committed launchers do not).
- Never `git add -A`; add the specific files. Parquets under `data/full/`
  are gitignored — check `git status` shows none.

---

## 4. Stage 1 — the falsification run (box)

### 4.1 Box

Preferred: the owner's box already up and idle
(`ssh -p 32414 root@38.246.237.140`; repo at `/workspace/elicit-vs-teach`
on `ts38-mini`, venv `/workspace/venv`, store
`/workspace/elicit-vs-teach/geode-store` with the base run and the frozen
`data/full/D_target.parquet` + `D_algo_eval.parquet` already present). If it
is gone, the owner provides a box (vast template hash
`14aefceab56ba3956b8a0cc1b015380b`, `--template_hash` only); wait for
`tail /workspace/onstart.log` to show `=== onstart done ===` — one-shot
tail, not a poll loop. **Never destroy this box** — it is the owner's rental.

On the box: `cd /workspace/elicit-vs-teach && git pull` (must land on the
commit from §3.7; `git status` clean apart from the known untracked
`configs/eval_nl_target_data_ts38.yaml`). No scp of parquets — regenerate
(§3.5 step 2).

### 4.2 Launch

```bash
ssh -p 32414 root@38.246.237.140
tmux new -s ts38mw
set -a; . /etc/environment; set +a           # HF token into THIS shell (tmux server predates it)
source /workspace/venv/bin/activate
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store
export NTFY=https://ntfy.sh/geode-run1-kx83q1 NTFY_AUTO=1
bash launch_ts38mw_probe.sh --confirm-cost 2>&1 | tee /workspace/ts38mw_probe.log
```

Detach; check with one-shot `tail -50 /workspace/ts38mw_probe.log` (or a
`Monitor` on the log filtered to `MILESTONE|FAILED|VERDICT|Traceback|Error|
HALT`). No `sleep N; ssh` polling loops.

### 4.3 What "working" looks like while it runs

`train_start` → `train_complete run=… stop_reason=converged` (if
`max_steps`, stop and report — that is a bug signal) → one
`probe_point step=… canonical_em=… g1_own=… sumof_em=… sym_q_em=…` line per
snapshot → `base_scored` → `VERDICT: <band> at step S …` → done. Any
`FAILED` = stop, read the message verbatim, report; do not improvise
around it.

### 4.4 Deliverables back on the laptop (repo paths, never scratchpad)

- `scp` → `experiments/training-run/analysis/ts38mw_probe.json` (the
  results file) and `notes/logs/ts38mw_probe.log`.
- A small table + one figure (EM and loss per pin vs step, base as a
  horizontal line) via a script under `analysis/` writing to
  `analysis/figures/ts38mw_probe.png` (`figures/` is gitignored — quote the
  absolute path in the report).
- `notes/decisions.md`: outcome entry with the table and the verdict;
  `EXPERIMENTS.md` §6.15 status line.
- Commit + push. One ntfy at the end of the session (not per step).

### 4.5 Report to the owner (format)

Six lines: verdict band + step; the table (per snapshot: canonical EM,
g1_own, G8, `sumof` EM/loss, `sumof_bare` EM/loss, `sym_q` EM/loss,
`word_q` EM/loss, `dm_mix` EM/loss; base row); stop_reason and step; cost
and wall time actually spent; anything that deviated from this plan;
recommendation limited to "Stage 2 / Stage 2′ / stop / Stage 1b" per §2.
Then **stop and wait**. Do not start Stage 2.

### 4.6 Stage 1b (only on INCONCLUSIVE)

If no snapshot reaches canonical EM ≥ 0.95 at convergence: 3-rung short LR
sweep {1e-4, 3e-4, 1e-3} × 8000 steps (mirror
`configs/sweeps/ts38/parent_lora_sweep_lr_*.yaml` + the selector rule in
`launch_ts38_lora_parent.sh`: highest LR with sweep-end G8 ≤ 1.1718 and
descending val), then rerun Stage 1 at the winner. Report before doing it
if the winner is not 3e-4 by a clear margin. Cost ≈ +$0.3.

---

## 5. Decision gate (owner)

Stage 2 is designed in detail **only after** a GO-A or GO-B is on the table.
Sketch so the owner can judge cost:

- **GO-A → Stage 2 (certified multiwrap parent + family on the existing
  word-only target).** LR sweep per the owner's rule → probe replay with
  1k-step snapshots around the G1 window → `certified_step` selection
  extended with a **third bar, the latency gate**: at S, `sumof` EM ≥ 0.20
  and loss < base (recorded on the parent's manifest, `--no-record` first)
  → certified parent `evt-ts38mw-pretaught-parent` (G1, G8, latency gate
  recorded; merged; metadata pushed) → family = `launch_ts38_mini.sh`
  generalised (parent id + run-id prefix `evt-ts38mw-*`; target stays
  `D_algo_bare`, sizes 1000…316228, both arms) → OCV-floor EDL +
  `dataset_size_sweep` analyses. Cost ≈ parent $1–1.5 (~3 h) + family as
  ts38 (~one overnight). Success = arms separated with the pretaught curve
  below base at every n and no rising span; that is a genuine elicitation
  readout because the target phrasing was never installed.
- **GO-B → Stage 2′ (owner call).** The capability is symbol-invariant but
  not language-invariant. Options: family on a held-out *symbol-in-sentence*
  target (`What is a + b?`-style, per-wrapper tagged, EDL split by wrapper),
  or the DM mixture with per-template split. Either is a fair held-out-
  wrapper elicitation test but a weaker claim than GO-A. Owner decides which,
  or whether to stop.
- **WEAK / NO-GO → write-up.** Add to `docs/ts38-vs-bits-that-count.md`
  and `decisions.md`; the 38.7M line closes; the only remaining route to the
  paper's picture is the 1B track (separate plan).

---

## 6. What this plan deliberately does NOT do

- Change the convergence rule (data shows it is irrelevant to transfer).
- Change any bar (G1 0.95, G8 1.1718) or any frozen dataset/hash.
- Regenerate the *target* with the paper's templates (would separate arms by
  the ~9 % bare-op share only — an artifact, measured).
- Full-FT install (blocked by G8 at every LR — closed design result).
- Anything on the 1B track.

---

## 7. Operating rules (owner-set; non-negotiable for the worker)

- **Simplest first; sweeps select, gates score.** Every gate on a shared or
  probe checkpoint is `--no-record` (a recorded FAIL on a parent blocks every
  child).
- **Run until convergence**; `max_steps` is a cost ceiling; quote ETAs from
  measured time-to-converge, not the ceiling.
- **Budget rule:** nothing launches without `--confirm-cost` and a printed
  estimate. The box is the owner's; never create/destroy instances.
- **Box hygiene:** `python3` = the venv's; source `/etc/environment` inside
  tmux; check onstart via `tail`, not a poll loop; ~1–2 min of failed SSH
  after onstart → tell the owner, don't loop.
- **Verify the receiver, not the sender** for anything pushed to the relay
  (list files on the hub).
- **Deliverables at repo paths**, never the scratchpad; figures under
  `analysis/figures/` (gitignored — quote the absolute path).
- **Never `git add -A`**; never push weights; never a new
  `from_pretrained(<lora checkpoint>)` call site — LoRA checkpoints load only
  via `geode.zoo.load_model` (the gates already do).
- **ntfy** (`https://ntfy.sh/geode-run1-kx83q1`): one ping at the end of the
  session or on a genuine block — never per run/gate/boot.
- **Memory:** write run/launch state to the project memory as soon as it
  changes; end-of-session update unprompted.
- **Escalate only** irreversible / money / measurement-changing /
  direction-forking decisions; everything else decide, state the assumption,
  proceed. Stage 2 is a direction fork — it waits for the owner.
