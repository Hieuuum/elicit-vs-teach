# HANDOFF: edge-level CMAP, child → parent, sizes 0.1–5%

Status 2026-10-22: PLAN ONLY. Not built, not run. The owner asked for it at 0.1, 0.2, 0.5,
1, 2 and 5% and asked for this file so a fresh session can run it. The owner has NOT yet
approved the box rental, so confirm the cost line with the owner before renting (budget rule).

## Read first (fresh session)

- `MEMORY.md` → `project-eapig-small-sizes-2026-10-21.md` and `project-cmap-plan-2026-10-22.md`.
- `PLAN.md` small-sizes results block (~lines 390–430): overlap, functional reuse, tests.
- Branch `eapig-circuit-check`. Commit 99f8c94 (the results block) is local, NOT pushed.
- Box work: the `vast-box` skill (`~/.claude/skills/vast-box/scripts/vb.py`), TEAM api key,
  race 3 boxes and keep the first ready one, `--confirm-cost`, push + verify on HF, destroy, and an ntfy ping
  only when the experiment completes.

## Background in one paragraph

On the elicit route, the child's NL circuit is mostly the elicit parent's op-form addition
circuit (edge Jaccard 0.72–0.77 at 0.1–1%, 84–91% of the parent's split-half ceiling;
functional reuse share 0.86–0.98). On the teach route, the child overlaps the format parent
only at about a third (0.25–0.43, at the control's level). The elicit parent's NL circuit vs
the elicit child gives 0.27–0.42, the same as teach: the parent gets 0% of NL prompts right,
so its NL circuit is not its addition machinery. CMAP asks the next question: do the child's
circuit edges carry what fine-tuning changed, in a form the parent can use?

## Question and method

Cross-model activation patching (CMAP) follows Prakash et al. 2024 (arXiv 2402.14811,
"Fine-Tuning Enhances Existing Mechanisms"). They patch head outputs, fine-tuned → base;
here we patch edges, child → parent.

For each of the 256 NL val prompts (`run.load_data(data_dir, "word")["val"]`,
`clean_ids` = prompt + first answer token):

1. Child: `src = node_outputs(child, graph, clean_ids)` and
   `child_logp = answer_logprobs(child(clean_ids).logits)`.
2. Parent: `logits = patched_forward(parent, graph, clean_ids, src, keep, last_n=2)` with
   `keep = run.keep_without(graph, edge_idx, dev)`. Here keep = 0 on the k circuit edges, so
   each circuit edge u → r carries the CHILD's u output into the parent's receiver r. Every
   other edge, and every node downstream, is computed by the parent's own weights.
3. `kl = kl_2tok(logits, child_logp)` gives KL(p_child ‖ p_patched) in nats, summed over the two
   answer positions.

No new patching code is needed: `patched_forward` already routes `corrupt_u` into edges
with keep = 0. Here "corrupt" is the child's clean run. Mostly-kept masks take the sparse
path, with k exception edges. Both models share one architecture (assert `graph.n_edges ==
195_865` for both; build the graph once from the parent).

Measures, per mask, as means over the 256 prompts:
- `recovery = 1 − mean KL(child ‖ patched) / mean KL(child ‖ parent)`. 0 = the parent is
  unchanged; 1 = the parent behaves like the child.
- Two-token teacher-forced accuracy of the patched parent: argmax at position −2 == c1
  and at −1 == c2. For reference, report the same for the unpatched parent and the child.
- Mean log-prob of the correct two answer tokens.

## Runs

| route | parent (HF `podhajskimarcin/…`) | child | circuit scores (HF `mhieuuu/geode-internals`) |
|---|---|---|---|
| elicit | `evt-ts1b-op-bridge-mix` | `evt-ts1b-elicit-ft-n4000000` | `results/eapig_kl_symbol/klword/elicit_child/scores.pt` |
| teach | `evt-ts1b-fig2ts-installer` | `evt-ts1b-teach-ft-fmt-n4000000` | `results/eapig_kl_symbol/klword/teach_child/scores.pt` |

Top-k = `et.topk_edges(torch.load(scores.pt)["mean"].numpy(), k)`, the same selection
`functional_reuse.py` uses. Sizes: 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, which give
k = 196, 392, 979, 1,959, 3,917, 9,793 via `et.size_to_k(frac, 195_865)`.

Masks per route, per size:
- (a) `child`: the child's own top-k circuit.
- (b) `child_nologits`: (a) minus every edge whose receiver is `graph.logits_index`
  (use `graph.flat_to_ur`). Those edges hand the child's output straight to the
  unembedding, so (a) can "recover" the answer without the parent computing anything.
  **(b) is the main read.** Record how many of the k edges were removed.
- (c) 20 random same-size edge sets, `run.random_sets(E, k, 20, seed=run.SEED + 202 +
  int(frac * 1e5))`, reported as a p5/p50/p95 band. Also give 20 random sets the
  same logits-edge removal, so (b) has a matched baseline.
- (d) elicit route only: the elicit parent's op-form top-k
  (`results/eapig_kl_symbol/klsym/elicit_parent/scores.pt`). This asks whether the change lives in the reused
  circuit. The owner did not answer on (d); it is one mask per size, so it is included, and it
  can be dropped if the owner says so.

Once per route:
- `all`: every edge patched. This is the ceiling, and it is NOT exactly the child, because
  the final norm and the tied unembedding are still the parent's. Report recovery for (a)
  through (d) both raw and as a fraction of `all`.

## Correctness checks: CPU property tests first (tiny random Llama pair, no download)

Put them in `tests/` (or next to the script, following repo habit). Use two
randomly initialized 2-layer d≈64 Llama configs with the same architecture and different seeds.

- All-ones keep → patched logits == parent logits (atol 1e-5).
- Empty keep (all valid edges patched) → logits == the parent's final norm + lm_head
  applied to the sum of the child's node outputs (computed by hand).
- Same weights for parent and child → patched == parent for any mask; recovery denominator
  ≈ 0 must raise or return NaN, never a silent number.
- Patching one edge u → r leaves logits unchanged when r cannot reach the logits
  (sanity on the mask direction), and changes them when r = logits.
- The logits-edge filter removes exactly the edges with receiver == `logits_index`, and
  nothing else.
- Sparse path == dense path (`GEODE_EAPIG_DENSE=1`) on a CMAP mask (< 1e-4 nats).
- Recovery of the all-patched mask on the tiny pair matches the hand-computed value.

## Pre-registered read

- Elicit (b) recovery clearly above its random band and above teach (b) at matched k ⇒
  the fine-tuning change sits on the child's circuit edges, and the parent's downstream
  machinery can use it (consistent with elicitation).
- (a) high but (b) at the random level on both routes ⇒ the logits shortcut. Make no mechanism
  claim.
- Teach (b) at the random level ⇒ the format parent cannot use the child's signals
  downstream (consistent with "taught": the machinery is not in the parent).
- (d) ≈ (b) on the elicit route ⇒ the change sits in the reused op-form circuit.
- Caveats to carry: single seed and one training run per child; at 0.1% the child's own
  circuit has f ≤ 0.007 (small-sizes block), so read 0.1% with care.

## Build

`experiments/eapig-circuit-check/cmap.py`, ~150 lines, reusing `run.load_model`,
`run.load_data`, `run.keep_without`, `run.random_sets`, `run.dump`, `et.topk_edges`,
`et.size_to_k`, and `geode.circuits.eapig.{build_graph, node_outputs, patched_forward,
answer_logprobs, kl_2tok}`.

CLI: `--route elicit|teach --parent M --child M --child-scores DIR [--extra-circuit
label=DIR] --sizes ... --n-random 20 --batch-size 16 --out results_cmap/<route>.json`.

Batch-outer loop, like `run.eval_kl`: per batch, one child forward (node outputs + logp)
and one plain parent forward (for the KL(child ‖ parent) denominator), then each mask.

Box script `run_box_cmap.sh`, modelled on `run_box_small.sh`:
- refuse without `--confirm-cost` and print the estimate first;
- single-threaded model prefetch;
- tmux session per GPU job, one job per GPU (two EAP-IG-sized jobs on a 3090 OOM);
- push to HF `results/eapig_cmap/` and verify the push from the laptop ("verify the receiver").

## Cost and time

Masks per route ≈ 6 sizes × (a, b, d + 40 random) + 1 ≈ 260 (teach ≈ 235). Per-mask time:
the small-size functional run did 96 masks over 256 prompts in ~111 s (~1.2 s/mask).
The 2–5% masks have 4–10k exception edges, so budget ≤ 3 s. Two models on one GPU:
about 10 GB of fp32 weights, plus child node outputs of about 1.5 GB at bs 16.

| step | time |
|---|---|
| build `cmap.py` + tests + self-review | 1.5–2 h |
| race/rent 1–2× RTX 3090, setup | 15–20 min |
| download 4 models (~5 GB each) + 3 score files (~200 MB each) | 5–10 min |
| run, one route per GPU on a 2-GPU box (or both on one GPU) | ~10–15 min (~25 min on one GPU) |
| push, verify, destroy | ~10 min |

Total: about 2.5–3 h of wall time from the start of the build. GPU cost under ~$0.60.

## After the run

- Add a results block to `PLAN.md` (the decision log), with the table: size × {elicit, teach} ×
  {a, b, random band, d, all}.
- Update the deck (claude.ai/artifact/Riu32qzkU3T3eeKbgZBthC) only if the owner asks.
- Update memory (`project-cmap-plan-2026-10-22.md`) with the outcome; ntfy ping on
  completion.
