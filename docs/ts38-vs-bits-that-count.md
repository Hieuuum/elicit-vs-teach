# ts38 mini vs. *Bits That Count* Table 5 (TinyStories add/sub) — why the curves don't match

Written 2026-08-15 from a log replay of the 11 `evt-ts38-*` runs pulled from the
relay (`mhieuuu/geode-store`, metadata only). All numbers below are reproducible
on CPU with

```
GEODE_STORE=<repo>/geode-store PYTHONPATH=<repo> \
  python experiments/training-run/analysis/edl_converged_val_floor.py --family ts38
  python experiments/training-run/analysis/dataset_size_sweep.py --family ts38
```

The one GPU measurement (§3.1b) was `gates.py g5 --no-record` on the owner's box
with `configs/eval_{nl,bare}_target_data_ts38.yaml`; nothing was written to any
manifest.

Companion docs: `docs/bits-that-count.md` (tidied paper), `docs/bits-that-count-experiments.md`
(per-experiment protocol summary). Paper facts are cited as (§/App./Table/Fig).

## 1. What was expected (paper, Table 5 + §4.2)

Table 5, Addition/Subtraction, TinyStories–1B rows:

| Model | Signature | Peak n | Notes |
|---|---|---|---|
| TinyStories–1B (base) | ↑↓ | ∼300K | Must learn algorithm |
| TinyStories–1B (pre-teach format) | ↑↓ | ∼150K | Isolates algorithm learning |
| TinyStories–1B (pre-teach add/sub) | ↓ | – | Converts to elicitation |

Plus two qualitative statements the table compresses:

- The **base** curve has an *initial decreasing* phase (format acquisition:
  digits + "answer with a number") *before* the rise; pre-teaching format
  removes that transient and exposes the rise (§4.2, Fig. 2 caption).
- The **pre-taught-algorithm** model "absorbs less information throughout
  fine-tuning" (§4.2); in the multiplication analogue (Fig. 3 inset) the
  pre-taught curve sits well *below* base at every n, and "the EDL learned per
  token remains over an order of magnitude smaller" for models that already
  hold the algorithm (§4.4).

So the expected picture for our two arms is: base = (short decrease) → slow
rise → peak ≈ 300K → decrease; pre-taught = monotone decrease, *far below*
base.

## 2. What we got (three floors, bits per label token)

Arm naming: `base` = teach arm (init = `evt-run1-base-v3-ext`), `pretaught` =
elicit arm (init = certified parent `evt-ts38-pretaught-parent`). Every run
`stop_reason=converged`, `epoch1_examples == n`.

| n | arm | MDL/D (nats) | L_conv | L_min | L_test | **EDL/D OCV** | EDL/D min-val | EDL/D test | steps (epochs) | zero-shot EM after training |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | base | 4.647 | 1.539 | 1.465 | 1.533 | **4.484** | 4.590 | 4.492 | 135 (17.3) | 0.3 % |
| 1,000 | pretaught | 4.453 | 1.648 | 1.550 | 1.646 | **4.047** | 4.189 | 4.050 | 200 (25.6) | 0.7 % |
| 4,642 | base | 2.638 | 1.299 | 1.239 | 1.308 | **1.932** | 2.019 | 1.919 | 270 (7.4) | 1.9 % |
| 4,642 | pretaught | 2.561 | 0.921 | 0.912 | 0.899 | **2.366** | 2.378 | 2.398 | 570 (15.7) | 15.1 % |
| 21,544 | base | 1.734 | 0.196 | 0.170 | 0.191 | **2.219** | 2.256 | 2.226 | 1,825 (10.8) | 74.6 % |
| 21,544 | pretaught | 1.855 | 0.217 | 0.206 | 0.213 | **2.363** | 2.378 | 2.369 | 1,675 (10.0) | 69.0 % |
| 100,000 | base | 1.262 | 0.065 | 0.052 | 0.065 | **1.728** | 1.746 | 1.727 | 5,000 (6.4) | 92.5 % |
| 100,000 | pretaught | 1.105 | 0.057 | 0.051 | 0.059 | **1.513** | 1.521 | 1.510 | 6,000 (7.7) | 91.3 % |
| 316,228 | base | 0.620 | 0.037 | 0.023 | 0.038 | **0.841** | 0.861 | 0.841 | 10,875 (4.4) | 94.5 % |
| 316,228 | pretaught | 0.482 | 0.020 | 0.017 | 0.022 | **0.665** | 0.671 | 0.663 | 13,500 (5.5) | 96.4 % |

(`L_*` in nats/label-token; OCV = own converged val floor, min-val = each run's
minimum val, test = `eval/test_loss.json` at the stopping step. Zero-shot EM =
G5 on the held-out bare-NL eval, recorded on the trained child.)

Figure (laptop-only, `figures/` is gitignored):
`experiments/training-run/analysis/figures/edl_converged_val_floor_ts38.png`.

### 2.1 Reading against the pre-registered rule (EXPERIMENTS §6.14)

- **Base "rising span":** one rise, 4,642 → 21,544, under all three floors
  (OCV 1.93→2.22, min-val 2.02→2.26, test 1.92→2.23; +15 %). Formally the marker
  fires. It is a single +0.29-bit bump on a 5-point grid with one seed, and it
  sits between the two points where the model actually acquires the task
  (EM 1.9 % → 74.6 %). Peak ≈ 21.5K, not ≈ 300K; by 316K the curve has fallen
  to 0.84 bits — the paper's ~300K peak would be our *last* grid point, and the
  edge rule already declares that "unresolved".
- **Pre-taught "monotone decreasing":** OCV 4.05→2.37→2.36→1.51→0.67 (yes);
  test floor yes; min-val floor *flat* at 4,642↔21,544 (2.3783 vs 2.3782) —
  non-increasing, but not a decrease.
- **Separation (the thing the paper actually shows):** *absent*. The
  pre-taught arm is **above** base at 4,642 and 21,544 and only 10–20 % below
  it at 100K/316K. The paper's pre-taught curve is far below base everywhere.
  Two curves that lie on top of each other, both falling ~6× across the grid,
  are not "one teaching arm and one elicitation arm".

## 3. Diagnosis — ranked by evidence

### 3.1 The pre-taught parent has NO NL capability at θ0 — with or without the scaffold — primary

Two independent measurements, both on the certified parent (G1 = 95.7 % on
`Question: 23 + 45\nAnswer: 68`) and the untouched base.

**(a) Prequential log, step 0** (θ0's label loss on the first 128 bare-NL
examples, before any update):

| θ0 | first-batch label loss (nats/token) | steps 1–7 |
|---|---:|---|
| base (`evt-run1-base-v3-ext`) | **6.585** | 6.01, 5.58, 4.91, 4.24, 3.61, 3.16, 2.75 |
| certified parent | **7.752** | 7.03, 5.33, 3.96, 3.20, 2.78, 2.62, 2.56 |

**(b) `gates.py g5 --no-record` on the box, 2026-08-15** (1,024 questions,
held-out `D_algo_eval`, question-disjoint from everything trained; label loss
over the 97,952-row reporting block; nothing written to any manifest):

| θ0 | scaffolded NL `Question: What is the sum of 23 and 45?\nAnswer: 68` | bare NL `What is the sum of 23 and 45?\n68` |
|---|---|---|
| certified parent | zero-shot EM **1.56 %**, 16-shot 0.00 %, label loss **7.71** nats/tok | EM **0.00 %**, 16-shot 0.00 %, **8.10** nats/tok |
| base | EM 0.00 %, 16-shot 0.00 %, **5.19** nats/tok | EM 0.00 %, 16-shot 0.00 %, **6.54** nats/tok |

The parent is *more* surprised than the base by NL add/sub labels under **both**
renderings (7.7–8.1 vs 5.2–6.5 nats/token) and answers essentially none of
them. So the op-notation skill is locked behind the **question phrasing**
(`23 + 45` vs "the sum of 23 and 45"), not merely behind the `Answer:` handle:
restoring the scaffold recovers only ~0.4 nats and 1.6 % EM. (Compare the
paper's TS-1B pre-teach add/sub on NL: 2.0 % zero-shot but **11.9 % 16-shot**,
Table 11 — some latent access at 1B; at 38.7M 16-shot is 0 % for every
checkpoint, including trained children that score 95 % zero-shot, so in-context
elicitation is unavailable at this scale altogether.)

During target training the parent overtakes base only after 2 updates, and its
cumulative first-epoch MDL/D is just 4 % lower than base's at n=1,000 (4.41 vs
4.61 nats/token) — it re-opens its output distribution at nearly base cost,
then learns the NL→algorithm mapping. That is a teaching arm with a small head
start (lower floors at n ≥ 100K), not elicitation of a latent capability.
Consistent with:

- n=1,000: pretaught converges *worse* than base (1.648 vs 1.539) after 25 epochs;
- n=21,544: pretaught has higher MDL *and* a higher floor than base;
- zero-shot EM tracks n identically in both arms (0.7/15/69/91/96 % vs
  0.3/2/75/93/95 %); the only place pre-teaching visibly helps is n=4,642
  (15 % vs 2 %).
- the same "capability locked behind a token-scope handle" mechanism already
  found in the 512-param unlock (one embedding row: 0.004 → 0.40).

**Why this happened.** Two things, one measured and one a design deviation:

- *Measured (dominant):* at 38.7M, ~1.9 epochs of LoRA r128 install on
  op-notation does not produce an arithmetic circuit that the words "sum" /
  "difference" can reach. The paper's causal result (§5) presupposes that
  pre-teaching op-notation makes NL arithmetic *latent*; the 2×2 above says
  it does not here. That is a genuine finding about this substrate, not a
  bookkeeping error, and it is the same class as the paper's own OOD control
  (Table 6: pre-teaching add/sub does not convert multiplication) — for this
  model, op-notation add/sub is "out of distribution" for NL add/sub.
- *Design deviation (secondary, ~0.4 nats):* EXPERIMENTS §6.14 states "paper's
  E.2 pre-teach is scaffolded and its target is bare, ours matches both". The
  paper does **not** say the NL target is bare. App. E.2 gives only the
  pre-teach format (`Question:\n2 + 3\nAnswer:\n5`); App. F says "the prompt
  (e.g., 'What is the sum of 23 and 45?') **and any formatting tokens** are
  excluded from both MDL computation and test loss evaluation" — which reads
  as the NL target *also* carrying the `Question:/Answer:` wrapper. Our bare
  target removes that shared handle, so the pre-taught arm additionally pays
  an output-format cost. The paper's own control for format cost — the
  pre-teach-*format* arm ("isolates algorithm learning") — was deliberately
  excluded from ts38 (§6.14).

### 3.2 Everything is compressed to the left: the base acquires the task ~15× earlier in n

The base's teaching window (EM 2 % → 75 %) is 4.6K–21.5K examples; the paper's
TS-1B peaks at ~300K. Contributors, none of which are "bugs":

| | paper (Table 3, App. B/G) | ts38 | effect on the EDL/D-vs-n curve |
|---|---|---|---|
| Data | DeepMind Mathematics add/sub subset: mixed NL phrasings, mixed difficulty (paper says subsets "differ in difficulty"), sizes 1 → 4M | one template per op ("What is the sum of a and b?" / "What is the difference between a and b?"), positive integers 1–4 digits, water-filled over the 4×4 digit cells (dominated by 3–4-digit operands), signed `a−b` | much easier / more uniform → algorithm learned at far smaller n |
| Updates per example in epoch 1 | batch 128 × 8 GPUs = **eff. 1024** | **128** (single GPU) | 8× more parameter updates per example → MDL/D falls sooner → hump moves left and lower. In *updates*, our peak (~21.5K ex ≈ 168 updates) and the paper's (~300K ex ≈ 293 updates) are the same order |
| LR (LoRA) | 3.53e-4 (r512, α32) | 1e-3 (r128, α32) | ~2.8× faster per update, same direction |
| Model | Llama 3.2 1B arch, TS-v2 pretrain | 38.7M custom (d=512, custom 10K BPE) | magnitudes/positions don't transfer; shape only |
| Grid / seeds | dense, 1 → 4M, 3 seeds/config | 5 points 1K–316K (×4.6 spacing), 1 seed | cannot locate a peak; cannot tell a +0.29-bit bump from noise; top point = paper's expected peak |

The paper says explicitly that hyperparameters "primarily affect the low-data,
low-parameter regime of single-epoch training" (App. H.1.2). Peak n is a
property of (data, model, algorithm A), not a constant to be recovered — the
expectation "peak ≈ 300K" was never transferable to a different A on a
different model.

### 3.3 The n=1,000 → 4,642 drop is the format transient, in both arms

EDL at n=1,000 is 15.3K nats (base) / 13.8K (pretaught); at 4,642 it is 30.7K /
37.6K — EDL grew ~2–2.7× while D grew 4.7×. That is a fixed cost of ~10K nats
("emit digits, then EOS") decaying as 1/D — the paper's "initial decreasing
phase" (§4.2), and the fixed-cost hazard the plan's guard 5 was written for
(base step-0 label loss 6.52 nats/token on bare prompts). Our grid *starts*
inside that transient; the paper's pre-teach-format arm exists to strip it.

### 3.4 Parent deviates from App. E in ways that weaken the "latent" premise

Paper: pre-teach = **full fine-tuning, one epoch over 4M unique** operator
examples, "until strong performance" (App. E.2, I.2.1). No retention gate.
Ours: LoRA r128/α32 (lr 3e-4) on the frozen base, replayed to **15,000 steps
≈ 1.9 epochs of 1M unique**, the earliest step passing G1 ≥ 0.95 *and* the
TinyStories retention gate G8 ≤ 1.1718 (recorded G1 = 0.957, G8 = 1.163). Both
the full-FT ladder and converged LoRA runs failed G8 (design results #1/#2), so
the parent is the *least*-trained checkpoint that still passes — 95.7 %
op-format accuracy, not "strong". Under the paper's own reading a partially
capable parent should still absorb less; the point here is only that the
premise "the algorithm is installed" is weaker than in the paper, on top of
3.1.

### 3.5 Things that are *not* the cause (checked)

- Floors: OCV, min-val and test floors give the same shapes and the same
  ordering (table above); overshoot ratios 1.05–1.60, largest at base 316K.
- Convergence: all 10 children `stop_reason=converged`, `epoch1_examples == n`,
  4–26 epochs; no ceiling binds.
- Data order: G7 pins Arm B's stream to Arm A's per size (same
  `data_order_hash`); label token counts identical per size.
- Capacity: worst-case EDL ≈ 0.91M nats (base 316K) over 12.06M trainable →
  ≈ 0.11 bits/param, well under the paper's ~1 bit/param teaching threshold
  (Table 6); the r256 escape hatch is not triggered.

## 4. What would make the comparison meaningful — cheapest first

1. **Done, no training:** the θ0 replay (3.1a) and the 2×2 `--no-record` eval
   (3.1b) establish that the elicit arm's premise — NL add/sub is *latent* in
   the parent — is false at 38.7M for this parent, scaffold or not.
2. **Config-only, cheap, but only a partial fix:** re-run the 10 targets on the
   scaffolded `D_algo` + `D_algo_eval` (shared `Answer:` handle, which is what
   App. F implies the paper did). Expect the format cost to shrink (parent
   8.10 → 7.71 nats/tok at θ0) — *not* the paper's separation, because the
   phrasing lock (7.71 vs base 5.19) remains. Worth doing only as the
   paper-faithful format for whatever comes next, not as the fix.
3. **The premise itself needs a different parent or a different scale.** To
   make NL add/sub genuinely latent you need a θ0 for which "sum of a and b"
   already reaches the arithmetic circuit. Options, in rising cost: (i) a
   parent pre-taught on op-notation *and* checked for NL transfer with the
   2×2 above as an explicit gate before the family runs (a "latency gate":
   parent NL zero-shot ≫ base, or at least label loss < base) — this turns the
   premise into a pre-registered check instead of an assumption; (ii) a
   longer/fuller install (the paper's full FT, 4M unique, one epoch — blocked
   here by G8; the paper has no retention gate) and re-test transfer; (iii)
   the 1B TinyStories track (`evt-ts1b-base`, paused), where the paper's own
   numbers show weak-but-nonzero NL access after op pre-teaching (Table 11).
   If (i) fails at every install we can certify, that is the design result:
   *at 38.7M, in-distribution op-notation pre-teaching does not create a
   latent NL capability*, and the Fig-3 causal design cannot be run on this
   substrate.
4. **If the *base* shape (slow rise to a late peak) is itself the target:**
   the paper's Fig-2 TS pair (base + pre-teach-format) with a grid that starts
   at n=1 and runs past 1M, ≥2 seeds. That is a different, larger run and it
   answers a different question (teaching dynamics), not elicit-vs-teach.
5. **Peak-n placement** is not worth chasing: matching the paper's ~300K would
   require its data mixture, eff. batch 1024 and LR — a different algorithm A,
   which unpins the 1e-3/r128 target recipe. The pre-registered marker is the
   rising span, not its location.

## 5. One-line verdict

The ts38 mini did not reproduce Table 5's TinyStories add/sub rows because the
"pre-taught" arm was not pre-taught *for the prompt it was measured on*: the
certified parent's op-notation capability does not transfer to NL phrasing at
all at 38.7M (zero-shot 1.6 % scaffolded / 0 % bare; label loss 7.7–8.1 vs base
5.2–6.5 nats/token), so both arms are teaching arms and their EDL/D curves
coincide; and the base's teaching hump, which does exist (+15 % at 4.6K→21.5K,
all floors), sits ~15× earlier in n than the paper's because the task is
easier and the first epoch takes ~8× more updates per example. The formal
markers "fire" but the pair is not verified, and the fix is a parent (or scale)
for which NL arithmetic is demonstrably latent — gated, not assumed.
