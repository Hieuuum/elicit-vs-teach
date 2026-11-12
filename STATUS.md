# STATUS.md: where everything stands (updated 2026-11-07)

Start here. One page; details live in the files it points to. Update the
"Active" block whenever a run starts or ends.

## Active

**ts1b continue-on-hard-problems: PLANNED, not built** (2026-11-11). Plan:
`experiments/training-run/notes/plan-ts1b-hard-continue.md`. Data built:
`D_algo_bare_hard4m` (4x4 40%, order_hash db1bb820…). Four runs (2 children × stop on
loss / stop at 95% accuracy) on one 2-GPU box; open decisions listed in the plan.

**ts1b redesign: the two children — DONE** (2026-11-07). No box up (59497646, 59610256 destroyed).
The pair (batch 256, lr 4e-5, torch.compile, V5.79 trainer; both stopped by the rule:
exact match >= 0.90 on 1,040 check prompts, 3 in a row):

| child | run id (HF `mhieuuu/`, private) | stop step | examples | wall | 20K test EM |
|---|---|---|---|---|---|
| elicit | `evt-ts1b-elicit-ft-fresh-b256` | 3,450 | 0.88M | 0.4 h | **0.920** |
| taught | `evt-ts1b-teach-ft-fresh-b256` | 39,500 | 10.1M | 3.3 h | **0.952** |
| elicit, checks every 500 (matched spacing) | `evt-ts1b-elicit-ft-fresh-b256-c500` | 4,500 | 1.15M | 0.6 h | **0.953** |

- **Matched pair = `-c500` elicit + taught**: same check spacing (every 500 steps),
  same test EM (0.953 vs 0.952); taught needs ~8.8x the examples. 4x4+ still differs
  (0.829 vs 0.906). decisions.md 2026-11-07 (elicit re-run).

- Taught needs ~11x the examples of elicit to reach the same check-prompt accuracy.
  It overshot to 0.952 on test because checks are every 500 steps (0.917 → 0.936 → 0.954).
- **lr is 4e-5, not the chain's 2e-5 pin.** The owner's batch test picked batch 256, which
  needs the higher lr (decisions.md 2026-11-07). The pair is matched with each other.
- Kept, not the pair: the batch-128 old-trainer elicit child `evt-ts1b-elicit-ft-fresh`
  (step 7,200, test EM 0.901, HF rev 5b6d111). Partial runs (old-trainer taught at step
  8,000; elicit-fast; v579-stopped-early) and the batch-test runs: logs only, in
  `experiments/training-run/geode-store/runs/` (gitignored); box logs in `runs/_box_logs/`.
- Speed: V5.79 trainer 1.55x the old one; compile +10%; batch 256 another 1.33x.
- Next (not started): analysis of the new pair (EAP-IG circuits, CMAP, hidden preference).
- **Elicit child** `evt-ts1b-elicit-ft-fresh` from `evt-ts1b-op-bridge-mix`
  (symbol arithmetic + bridge); **taught child** `evt-ts1b-teach-ft-fresh`
  from `evt-ts1b-fmt-parent`. Configs `configs/ts1b_elicit_ft_fresh.yaml`,
  `configs/ts1b_teach_ft_fresh.yaml` — identical except parent, run id, and
  check spacing (a test pins this).
- **Same for both:** full FT, lr 2e-5 constant, batch 128, bf16, seed 316,
  data `D_algo_bare_fresh4m` (order_hash 32dd6f24…), cap 187,500 steps.
  (Config values. The final pair ran with `BATCH=256 LR=4e-5 COMPILE=1`: checks every
  50 / 500 steps and cap 93,750, the same example counts.)
- **Stopping rule (owner):** exact-match accuracy >= **0.90** on **1,040**
  held-out check prompts, **3 checks in a row** (a miss resets). Check every
  **100 steps (elicit)** / **1,000 steps (taught)**. Check prompts: drawn
  from the 4,000 held-out rows, stratified by (size, op) to the training mix,
  identical for both runs. Model kept = the one at the stop step.
  `eval_log`'s `format_valid_rate` column holds this accuracy (trainer key name).
- **Training set** `data/full/D_algo_bare_fresh4m.parquet` (gitignored, 169 MB;
  `datagen/make_algo_fresh4m.py --out ../data/full`, ~1.5 min local): 4M word
  add/sub, correct labels, shuffled. Shares NO problem with the symbol install
  (`D_algo`), the test sets, the bridge or the EAP-IG problems. No small sizes
  (owner 2026-11-06 dropped them; the 7a6ca2cd… build with 51K small rows is
  superseded). Mix: 1x4, 2x3, 3x2, 4x1 at 1.43% each; 2x4, 3x3, 3x4, 4x2, 4x3,
  4x4 at 15.7% each.
- **Test set** `data/full/D_algo_eval_bare_trainmix_n20000.parquet` (gitignored;
  `datagen/make_test_trainmix.py --out ../data/full`), order_hash 6be24318…:
  20,000 rows of `D_algo_eval_bare` reshaped to the training mix (owner: same
  % per size in train and test). Config `configs/eval_trainmix_test_ts1b.yaml`;
  scored by `scripts/score_em.py` (overall + by size + by op). Old EM numbers
  (0.986 / 0.424 / 0.819) used the old uniform test mix: not directly comparable.
- **Box script** `scripts/run_ts1b_children_fresh.sh --confirm-cost [elicit|teach|both]`:
  checks the two data files (scp them first), pulls parents
  (`podhajskimarcin/evt-ts1b-op-bridge-mix`, `mhieuuu/evt-ts1b-fmt-parent`),
  trains elicit then taught, scores, pushes each to `mhieuuu/<run>` (private).
  ONE GPU >= 45 GB (24 GB OOMs). Cost ~$1–6 (taught length unknown; cap ~12 h).
  The exact_match stop is untested on a GPU: the step-0 line (`[evt] step 0
  ... exact_match`) and `check prompts by cell` line verify the wiring in minutes.
- **Format parent DONE** (2026-11-05): `mhieuuu/evt-ts1b-fmt-parent` (private).
  Word-template ts38 format corpus (21,544 add/sub, shuffled wrong answers).
  Stopped by its rule (>= 90% well-formed, 3 in a row, every 10 steps) at
  **step 40** (0.09 → 0.99 at step 20 → 1.0). Test accuracy 0/1024; target format 1.0.
- Why the redesign: the old children differed in data overlap, stopping rule
  and parent; the old taught child (0.424) was below the blank control (0.819).

**EAP-IG 10-fold robustness**: DONE 2026-11-04/05, **uncommitted** (fold
scripts, `hidden_pref.py`, tests and `ts1b_summary_page.html` in
`experiments/eapig-circuit-check/`). Overlap, functional reuse and CMAP hold
on 10/10 folds. Hidden preference fails with a same-length distractor (the
paper's 2.31 is likely a length or size cue). Results are on HF
`mhieuuu/geode-internals` under `results/eapig_folds/`. Log:
`experiments/eapig-circuit-check/PLAN.md`.

**Paper**: `paper/` on branch `hieu-paper-edit` is untracked. Never edit or
commit it unless the owner asks.

**Colleague's work (Marcin Podhajski) is on branch `unlearn-elicit-teach`, not here.**
It covers unlearning (WMDP/TOFU), the toy arithmetic task, the M19–M21
metrics, and his paper draft in `experiments/training-run/paper/`. It has not
been merged to main since 2026-09-25. On that branch he owns `experiments/em/`
and `experiments/unlearning/`, and he edits `experiments/training-run/analysis/`,
`scripts/`, `geode/{edl,train,arith}`, `EXPERIMENTS.md` (~line 850) and
`decisions.md` (~line 7118). Never move, rename or delete those, and never
push to his branch. Check `git log origin/unlearn-elicit-teach` before
touching shared files.

## The ts1b (TinyStories-1B) models

All are full FT at lr 2e-5, batch 128, starting from `evt-ts1b-base`.

| Role | Run id | Trained on | Result |
|---|---|---|---|
| base | `evt-ts1b-base` | TinyStories | val 0.9855 nats |
| op install | `evt-ts1b-op-install` | `D_algo_op` 1M symbol add/sub (`1 - 9540 = -9539`) | EM 0.69 |
| **elicit parent** | `evt-ts1b-op-bridge-mix` | op install + 16,384-row bridge (word→symbol rewrites + symbol sums) | word EM 0; symbol EM ≈ 0.40 |
| **elicit child** | `evt-ts1b-elicit-ft-n4000000` | `D_algo_bare_4m` 4M word problems | EM 0.986 at 21,500 steps |
| old teach parent | `evt-ts1b-fig2ts-installer` | `D_inst_bare` 1M word multiplication, random answers | format 1.000, EM 0 |
| **taught child** | `evt-ts1b-teach-ft-fmt-n4000000` | same 4M file | EM 0.424 at 58,500 steps |
| blank control | `evt-ts1b-teach-ft-n4000000` | same 4M file, from the base | EM 0.771 at the 62,500 ceiling; 0.819 when run to convergence |
| new teach parent | `evt-ts1b-fmt-parent` | ~5.1K rows of the bare format (redesign) | format 1.000 at 40 steps |
| **elicit child (redesign)** | `evt-ts1b-elicit-ft-fresh-b256` | `D_algo_bare_fresh4m`, batch 256, lr 4e-5 | test EM 0.920 at 3,450 steps |
| **taught child (redesign)** | `evt-ts1b-teach-ft-fresh-b256` | same, from the new teach parent | test EM 0.952 at 39,500 steps |

Known flaws (the reasons for the redesign):
1. The first 1M word problems the elicit child sees are the op-install sums, in the same order.
2. Stopping rules differ: teach had a 31,250-step minimum and a 187,500 ceiling; elicit had no minimum and a 62,500 ceiling.
3. The two parents use different operations (add/sub vs multiplication) and different templates.

Checkpoints are pushed to `$HF_NAMESPACE/<run_id>`, default `podhajskimarcin`.
Narrative: `experiments/training-run/notes/decisions.md`, entries 2026-08-29 →
2026-09-25. The index at the top lists every heading.

## Closed tracks (read before reopening)

| Track | Verdict, one line | Where |
|---|---|---|
| ts38 (TinyStories 38.7M) EDL families: pp, pf, mw, fs, mt, dense | Format effects dominate small n; op pre-teach wins only at n ≥ 46k; format-install dose barely matters. Op+format pair (`ppfmt`) proposed, never built. | `analysis/figures/ts38*`, decisions.md Aug 15–23 |
| Fig-2 Llama sweeps (op, nl) | Install pays only at small n; immutable | decisions.md Jul 30–Aug 11 |
| OLMo2 circuit overlap | Math forms at stage 2; ETHICS at chance | HF `mhieuuu/olmo2-circuit-overlap`, top of EXPERIMENTS.md |
| Phases 0–3, runs 1–10 | Superseded by ts38/ts1b | `configs/archive/`, decisions.md July |

## Where things live

- `experiments/training-run/`: configs (`configs/`, sweeps in `configs/sweeps/`),
  trainers and launchers (`scripts/`), data generators (`datagen/`), analysis
  and figures (`analysis/`, `analysis/figures/`), notes (`notes/decisions.md`).
  `data/` and `geode-store/` are gitignored. Regenerate data with `datagen/`
  (seed 20260717); the public data is on HF `mhieuuu/elicit-vs-teach-arith`.
- `experiments/eapig-circuit-check/`: EAP-IG circuit work, Oct–Nov. Its own `PLAN.md`.
- `EXPERIMENTS.md`: the plan through 2026-09-25. It does not cover the EAP-IG work; use this file and that `PLAN.md`.
- Rules (budget, testing, reference/ read-only): `CLAUDE.md`.
