# Handoff: addition circuits at small sizes (0.1/0.2/0.5/1%), overlap + circuit tests

Written 2026-10-21 so the session can be cleared. Owner asked: "score the addition for
other circuit sizes first, use lesser size and calculate overlap and do circuit tests",
then "plan the near-empty / near-full mask speedup first, don't implement yet", and
OK'd renting 4 GPUs. **Nothing below is built or run yet except step 0 (overlap).**
Owner still has to say go on this plan before any code or box.

## Design (owner, 2026-10-21, final)

- Addition only, the original 4x4 pairs. Subtraction, signed answers, Path B: dropped.
- `elicit_parent` is scored on **operator notation** (`a + b = `, KL scores in
  `results_klsym/elicit_parent`).
- `elicit_child`, `fmt_parent`, `teach_child` are scored on **natural language**
  (`What is the sum of a and b?\n`, KL scores in `results_klword/<tag>`).
- Metric: KL (`kl_2tok`, both answer tokens). Circuit = top-k by signed mean score,
  k = round(frac × 195,865).
- Scores already exist on HF `mhieuuu/geode-internals:results/eapig_kl_symbol/`
  (no rescoring needed). Laptop copy (1.2 GB, scratch, may be gone after reboot):
  `/tmp/claude-1000/-home-mhieuuu-Github-elicit-vs-teach/54da3cc2-ca95-480a-8aac-df2715d95b09/scratchpad/sc/{klsym,klword}/<tag>/{scores.pt,sanity.json}`.

## Step 0: overlap at small sizes — DONE (CPU, laptop)

Signed top-k Jaccard of `scores.pt["mean"]`; the 0.1/1/2/5% values equal the earlier
`edge_node.py` outputs exactly. ep = elicit_parent, ec = elicit_child, fp = fmt_parent,
tc = teach_child; sym = op form, word = NL.

| pair | 0.1% | 0.2% | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|---|
| ep_sym → ec_word (headline) | **0.766** | **0.727** | **0.718** | **0.731** | 0.661 | 0.497 |
| ep_sym → tc_word (control) | 0.395 | 0.320 | 0.307 | 0.343 | 0.367 | 0.270 |
| fp_word → tc_word (teach route) | 0.425 | 0.338 | 0.312 | 0.308 | 0.299 | 0.252 |
| ep_word → ec_word | 0.420 | 0.315 | 0.323 | 0.318 | 0.310 | 0.267 |
| ep_sym → ec_sym | 0.798 | 0.738 | 0.712 | 0.736 | 0.680 | 0.528 |
| ec_sym → ec_word | 0.840 | 0.802 | 0.820 | 0.851 | 0.787 | 0.637 |
| ec_word → tc_word | 0.436 | 0.331 | 0.319 | 0.351 | 0.377 | 0.284 |
| ep_sym split-half ceiling | 0.885 | 0.862 | 0.833 | 0.806 | 0.707 | 0.515 |
| ec_word split-half ceiling | 0.903 | 0.853 | 0.833 | 0.819 | 0.747 | 0.569 |
| fp_word split-half ceiling | 0.840 | 0.798 | 0.758 | 0.720 | 0.679 | 0.617 |
| tc_word split-half ceiling | 0.840 | 0.766 | 0.769 | 0.754 | 0.675 | 0.530 |

Reading: elicit reuse holds down to 0.1% at ~85–90% of the ceiling; teach and control
stay at 0.30–0.43. Missing: `edge_node.py` random-set nulls at 0.2/0.5% (HF has 0.1/1/2/5%
only) and the ep_sym:tc_word pair's nulls. These run on GPU 3 below (CPU job, cheap).

## Step 1: speed up the patched forward (PLAN ONLY, owner go needed)

### Why tests are slow
Per size, a child runs ~304 patched passes over 256 val pairs: sufficiency 1 + 100
random, partial necessity 1 + 100, consistency 1 + 100, specificity 1 (copy task).
Parents use 50 random draws (`sec_per_circuit_eval > 2 s`). One pass = 3–7 s
(`sanity.json`: ec_word 4.3, ep_sym 6.8, tc_word 4.4, fp_word 4.4). 304 × 4.3 s ≈ 22 min
per size for a child. Plan2.log: child ~21 min/size, parent 6–10 min/size.

### Where the patch cost is
`geode/circuits/eapig.py::_run` → `patch_delta`. For each receiver group (a layer's 48
q/k/v receivers, its MLP input, the logits) it computes

    delta_r = Σ_{u < n_avail} w[u, r] · D_u,   w = 1 − keep,   D_u = corrupt_u − own_u

as a dense einsum `btcd,cr->btrd` over every upstream chunk (~530 upstream nodes,
up to 33 chunks, ~2 calls per layer). Every mask in the circuit tests is ≥ 95%
one value:
- "keep only circuit" (sufficiency, `keep_only`): w ≈ all 1 (196 zeros at 0.1%).
- "remove circuit" (necessity, consistency, specificity, real patching; `keep_without`):
  w ≈ all 0 (196 ones at 0.1%).

### The change
For on/off masks only, per receiver r, over the read prefix u < n_avail(r) (this prefix
is exactly the valid edges, so invalid entries never matter):
- few ones in w: `delta_r = Σ_{u: w=1} D_u` (gather + index_add of those D_u);
- few zeros in w: `delta_r = S_n − Σ_{u: w=0} D_u`, where `S_n = Σ_{u<n} D_u` is a
  running total updated once when each new node's D chunk is appended (one add per
  node, not per receiver);
- if the exceptions exceed ~25% of the read prefix for any receiver group: keep the
  current dense einsum for that group.

Implementation notes:
- Keep D in one preallocated (B, T, U, d) buffer filled as nodes are computed
  (same memory as today's chunk list, which already holds all D), so gathers index it
  directly.
- Precompute per mask (once per `patched_forward` call, on device): for each receiver
  group, the exception list (u, r, sign) and whether the base is 0 or S_n.
- Gate: use the new path only when `keep` is bool, or float with all entries in {0, 1},
  and `not keep.requires_grad`. EAP-IG scoring (fractional/gradient masks) keeps the old
  path untouched. The existing all-ones fast path stays.
- Touch only `_run`/`patch_delta` (+ a small helper). `run.py`, `functional_reuse.py`,
  test logic: unchanged.
- Optional, same commit only if trivial: a flag/env to force the dense path, so the box
  check can time old vs new in one process.

### Two more speedups, same commit (also plan only)

**1b. Share inputs across receivers with no exceptions.** The 2026-10-08 log shows the
job is memory-bandwidth bound (memory controller 78–100% busy, bs 64 only 3% faster
than 32). Each layer materializes a (B, T, 48, d) per-head input, 48× the residual, then
layernorms and projects it. Under the shortcut, any receiver with zero exceptions reads
the same tensor: `resid` (remove masks) or `resid + S_n` (keep-only masks). At 0.1%
(196 edges over 16 layers × 49 receivers) most q/k/v receivers have none. So: layernorm
the shared input once (B, T, d), project the no-exception heads from it with the
normal batched q/k/v matmul, and materialize per-head inputs only for heads with
exceptions. Same math. Likely the bigger win, because it cuts memory traffic.
Tests: as below, plus a mask where a whole layer has exceptions and one where only a
single k/v group does (GQA: k/v receivers are per KV group).

**1c. Logits only where read.** `_run` applies `lm_head` (128,256 × 2048) at every
position, but KL/LD read only positions −2 and −1 (lm_head ≈ 20% of a plain forward's
FLOPs; ~90% of it wasted at T ≈ 20, plus a 330 MB fp32 logits tensor per batch).
Add an optional `positions`/`last_n` argument to `patched_forward` (default: all,
because `eval_ts_loss` needs every position) and pass `last_n=2` from `eval_kl` /
`eval_ld` in `run.py` (and `functional_reuse.py` if it calls `patched_forward`).
Test: `last_n=2` logits equal the last 2 rows of the full logits.

**Not recommended:** bf16 (a measurement change; the ts1b bf16 tail-outlier problem is
in memory), TF32 (no gain on 3090 consumer Ampere: TF32 dense = fp32 peak), fewer val
pairs (weakens equivalence), larger batches (bandwidth bound).

### Property tests (tests/lib/circuits/test_eapig.py, tiny random Llama, CPU)
1. New path == dense path (atol ~1e-5 fp32) for random bool masks at densities
   0, one edge, 0.1%, 1%, 30%, 50%, 99%, 100%, each as keep-only and remove forms.
2. Same on a GQA config (n_kv_heads < n_heads) and with mean-ablation broadcast
   `corrupt_acts` shapes (1,1,U,d) and (1,T,U,d).
3. Float or `requires_grad` keep → dense path; `eap_ig_scores` bit-identical to before.
4. Existing tests (`test_v_keep_all_ones_reproduces_model`,
   `test_v_keep_all_zeros_gives_corrupt_run`, `test_v_invalid_keep_entries_are_ignored`,
   `test_v_single_edge_matches_hand_built_reference`, `test_v_mean_ablation_broadcast_shapes`,
   KL tests) still pass through the new path.
5. Float keep with 0/1 values gives the same logits as the bool keep.
Whole suite stays CPU-only and < ~2 min.

### Box check before the real run (~5 min, GPU 0)
On the real 1B model, ec_word data: time one 256-pair eval old vs new at 0.1% keep-only
and remove masks; and compare test numbers on ~5 masks (KL difference < 1e-4 nats).
If not faster, run with the dense path (nothing lost). Record both timings in PLAN.md.

## Step 2: lift the op-form evaluate guard for elicit_parent

`run.py::parse_args` errors on `evaluate` unless `--task word`. `stage_evaluate` has no
word-specific logic (val pairs from the task file; copy + stories are shared), and
`load_pairs(dir, "symbol")` already assembles disc/val/copy/stories. Change: allow
`evaluate` with `--task symbol` for `elicit_parent` only (keep translate refused;
keep elicit_child symbol refused, not in the design). Update
`tests/experiments/test_eapig_run.py::test_evaluate_refused_on_symbol` (~line 380, uses
elicit_child → still refused) and add an "allowed for elicit_parent symbol" test;
`test_translate_evaluate_refused` (~line 407) still refused.

## Step 3: GPU run (4× RTX 3090, one box)

Owner OK'd 4 GPUs. Race 3 team-tj boxes (vast-box skill, `export VAST_API_KEY=$(cat
~/.config/vastai/vast_api_key.team-tj)`), keep the first ready, destroy the rest. Print the
cost estimate and get the `--confirm-cost` OK first. Prefetch all 4 models
(single-threaded HF pulls). Pull scores from HF `results/eapig_kl_symbol/` into
`results_klsym/` and `results_klword/` (scores.pt + sanity.json per tag).

Model dirs on the box (from the 2026-10-08 plan scripts):
- elicit_parent `/workspace/models/elicit_parent/runs/evt-ts1b-op-bridge-mix/model`
- elicit_child `/workspace/models/elicit_child` (check; plan3 used this path)
- fmt_parent `/workspace/models/fmt_parent/runs/evt-ts1b-fig2ts-installer/model`
- teach_child `/workspace/models/teach_child/runs/evt-ts1b-teach-ft-fmt-n4000000/model`
(HF repos: `podhajskimarcin/<run_id>`, see `run_box_2gpu.sh` MODEL table.)

One job per GPU (2 EAP-IG jobs on one 3090 OOM), tmux sessions, `CUDA_VISIBLE_DEVICES`:

| GPU | job | sizes | est. (dense path) |
|---|---|---|---|
| 0 | box check (step 1), then `elicit_parent` `--task symbol --scores-dir results_klsym` | 0.001 0.002 0.005 0.01 0.02 0.05 (op form never tested at any size) | ~5 + ~60 min |
| 1 | `elicit_child` `--task word --scores-dir results_klword` (no `--parent-tag`, see note) | 0.001 0.002 0.005 0.01 | ~85 min |
| 2 | `teach_child` `--task word --scores-dir results_klword --parent-tag fmt_parent` | same | ~85 min |
| 3 | `fmt_parent` word (fails the KL gate → only sufficiency + partial necessity), then `functional_reuse.py` for both children, then `edge_node.py` nulls | same | ~25 + ~10 min + CPU |

Command shape (elicit_child; teach_child adds `--parent-tag fmt_parent`):

    python3 run.py --tag elicit_child --model <dir> --task word --metric kl --batch-size 32 \
      --device cuda --stage evaluate --sizes 0.001 0.002 0.005 0.01 \
      --scores-dir results_klword --results results_kltests_small/elicit_child

Note: `--parent-tag P` reads `<scores-dir>/P/scores.pt` (run.py ~line 624), so for
elicit_child it would load the WORD-form parent (`results_klword/elicit_parent`), not the
design's op-form circuit. Run elicit_child WITHOUT `--parent-tag`; the op-form parent →
child check comes from `functional_reuse.py` (`ep_sym=...`). teach_child keeps
`--parent-tag fmt_parent` (word form is correct there).

Functional reuse (both children):

    python3 functional_reuse.py --child elicit_child --model <dir> --metric kl \
      --own results_klword/elicit_child \
      --circuit ep_sym=results_klsym/elicit_parent ep_word=results_klword/elicit_parent fp_word=results_klword/fmt_parent \
      --sizes 0.001 0.002 0.005 0.01 --out results_functional/kl_small/elicit_child.json

edge_node nulls: `edge_node.py --scores ep_sym=results_klsym/elicit_parent
ec_word=results_klword/elicit_child tc_word=results_klword/teach_child
fp_word=results_klword/fmt_parent --pairs ep_sym:ec_word ep_sym:tc_word fp_word:tc_word
--fracs 0.001 0.002 0.005 0.01 --n-null 1000 --workers 2 --out results_edge/small`.

Draw counts: unchanged (children 100 random circuits, parents 50 via the slow rule),
so results compare with the 2%/5% KL tests.

Estimated cost: ~4.5 GPU-h on the dense path, ~1.5 h wall on 4× 3090; with the
shortcut less. Under ~$1.5 including race/idle.

## Step 3b: circuit overlap, final (owner: "don't forget to score circuit overlap")

The step-0 table was computed in chat only; it is not saved in the repo or on HF. As part of
the run, save it properly:
- `edge_node.py` (GPU 3 line above) at fracs 0.001 0.002 0.005 0.01 0.02 0.05 for every pair
  in the step-0 table (ep_sym:ec_word, ep_sym:tc_word, fp_word:tc_word, ep_word:ec_word,
  ep_sym:ec_sym, ec_sym:ec_word, ec_word:tc_word): Jaccard plus the 1000-draw random-set
  null, and the split-half ceilings. Check that the Jaccards equal the step-0 table.
- Push to HF `results/eapig_kl_tests_small/overlap/`. PLAN.md results block: overlap
  vs ceiling vs null next to the test results at each size.

## Step 4: wrap-up
- Check the box regularly while running (log tails, not poll loops).
- Push to HF `mhieuuu/geode-internals:results/eapig_kl_tests_small/` (+ functional, edge
  nulls); verify the file list from the laptop (`env -u HF_TOKEN`).
- Destroy the box; confirm 0 instances on team-tj.
- One ntfy ping (experiment complete).
- PLAN.md: results block + log line; memory update.

## Open choices (defaults chosen, owner may override)
- Random circuits per child: 100 (default) vs 50 (halves child time, min p 0.02).
- elicit_parent op tests at 2/5% too: yes (never tested on op form).
