# HANDOFF: base TinyStories-1B as an overlap control, sizes 0.1–5%

Status 2026-10-22: PLAN ONLY, nothing built or run. The owner confirmed the plan below:
NL task only (no op form), sizes 0.1–5% (no 10%), overlap only (no circuit tests). The
owner did not separately approve the box rental. If they start the session with "carry it
out", treat that as approval; the cost is under $0.50.

## Read first

- `MEMORY.md` → `project-base-control-plan-2026-10-22.md`, `project-cmap-plan-2026-10-22.md`,
  `project-eapig-small-sizes-2026-10-21.md`.
- `PLAN.md`: the 2026-10-21 small-sizes block (the overlap table this extends) and the
  2026-10-22 CMAP block.
- Branch `eapig-circuit-check` (pushed, HEAD 1af8ea7 when this was written).
- Box work: the `vast-box` skill (`~/.claude/skills/vast-box/scripts/vb.py`). Export the TEAM
  key first: `export VAST_API_KEY=$(cat ~/.config/vastai/vast_api_key.team-tj)`. Race 3
  boxes, keep the first ready one, push and verify on HF from the laptop
  (`env -u HF_TOKEN`), destroy, and send one ntfy ping only when the experiment completes.

## Why

The teach child's circuit overlaps its format parent's at 0.25–0.43 (signed edge Jaccard,
0.1–5%). That looks large, but the control pairs land at the same level: the op-form
elicit parent vs the teach child gives 0.27–0.40, and the elicit parent's NL circuit vs
the elicit child gives 0.27–0.42, even though that parent gets every NL prompt wrong. The
working guess is that about 0.3 is what any two models from this lineage share. Operand
information flows through the same embedding and early edges, and the output is most
sensitive to the same late edges, because all of these models start from the same
pretrained weights. If so, the 0.3 says nothing about shared addition machinery.

The base model `evt-ts1b-base` tests that guess directly. It is the common ancestor of
every model here and never trained on addition:

- elicit: base → `evt-ts1b-op-install` → `evt-ts1b-op-bridge-mix` (elicit parent) →
  `evt-ts1b-elicit-ft-n4000000` (elicit child)
- teach: base → `evt-ts1b-fig2ts-installer` (format parent) →
  `evt-ts1b-teach-ft-fmt-n4000000` (teach child)

## Pre-registered read

These were decided before any numbers exist. "About equal" means the bootstrap 95%
intervals (step 4) overlap; "clearly below" means they don't.

- **base ↔ teach child about equal to format parent ↔ teach child** at most sizes: the
  teach-route overlap is the lineage floor. The format parent contributed nothing visible
  to the child's circuit.
- **base ↔ elicit child near that floor and clearly below op-form parent ↔ elicit child
  (0.50–0.77)**: the elicit child's reuse of the op-form circuit is real, above what it
  inherits from the base.
- **base ↔ elicit child about as high as op-form parent ↔ elicit child**: the elicit child
  mostly inherited its circuit from the base, not from op-form training. The reuse story
  needs rethinking (and op-form scoring of the base becomes worth doing).
- **base ↔ either child far below 0.3** (near the random-set chance of 0.003–0.027): 0.3
  is not a generic floor, and the teach-route overlap means more than assumed.

## Steps

1. **`run.py`: add a `base` tag** (laptop, with tests). `--tag` choices are the four
   existing tags today. Allow `base` for the `sanity` and `score` stages only: refuse
   `probe` and `evaluate` with a clear error, and refuse `--task symbol`/`translate` for it.
   Add parse-args tests next to the existing guard tests in
   `tests/experiments/test_eapig_run.py`. Note: `is_parent = tag.endswith("parent")`
   only matters for probe/evaluate, which `base` never runs.

2. **Bootstrap overlap script** (laptop, with tests), e.g. `overlap_boot.py`. For a pair
   of score files and a size: resample the 512 discovery prompts with replacement (same
   indices for both models: the NL discovery pairs are the same file for every model;
   assert it), average `per_example` (fp16, shape 512 × 195,865) over the resample, take
   the top-k by signed score, and compute the edge Jaccard. 200 resamples, seeded, give a
   95% interval per pair and size. The point estimate must equal `edge_node.py`'s signed
   Jaccard; use `et.topk_edges` for it, and a fast argpartition top-k inside the loop
   (test that the two agree on these scores). Compute each model's resampled mean once
   per resample, not once per pair. Property tests on tiny synthetic scores: identical
   scores → Jaccard 1 in every resample; disjoint top-k → 0; deterministic per seed; the
   interval brackets the point estimate.

3. **Box, one RTX 3090** (fp32 1B + EAP-IG fits in 24 GB at bs 16–32):
   - Check out the branch and `pip install -e ".[circuits]"`.
   - Download `podhajskimarcin/evt-ts1b-base` (`runs/evt-ts1b-base/model/*`,
     `HF_HUB_DISABLE_XET=1`).
   - `python3 run.py --tag base --model <dir> --task word --metric kl --stage sanity score
     --results results_klword --batch-size 32 --device cuda`. Scoring took ~45 s per model
     before; most box time is the 5 GB download. Read `sanity.json`: expect a tiny
     KL(full‖∅), like the format parent's 0.005 nats. That's fine. The ranking can still be
     stable, and the split-half ceiling in step 4 shows whether it is.
   - Fetch the existing score sets from HF `mhieuuu/geode-internals`:
     `results/eapig_kl_symbol/klword/{elicit_parent,elicit_child,fmt_parent,teach_child}/`
     and `results/eapig_kl_symbol/klsym/elicit_parent/` (`scores.pt` + `sanity.json`,
     ~200 MB each).

4. **Overlap on the box CPUs** (no heavy compute on the laptop):
   - `edge_node.py --scores base=results_klword/base ec_word=results_klword/elicit_child
     tc_word=results_klword/teach_child fp_word=results_klword/fmt_parent
     ep_word=results_klword/elicit_parent ep_sym=results_klsym/elicit_parent
     --pairs base:ec_word base:tc_word base:fp_word base:ep_word ep_sym:ec_word
     fp_word:tc_word ep_sym:tc_word ep_word:ec_word --fracs 0.001 0.002 0.005 0.01 0.02
     0.05 --n-null 1000 --workers <nproc - 2> --out results_edge/base_control`
     (check the CLI against the script first). It also gives each model's split-half
     ceiling.
   - The reference pairs (the last four) must reproduce the small-sizes table exactly.
     If they don't, stop and find out why.
   - Run the bootstrap script on the same pairs and sizes.

5. **Push, verify, destroy.** Push `results_klword/base/` to HF
   `results/eapig_kl_symbol/klword/base/` (same layout as the other score sets) and the
   overlap + bootstrap outputs to `results/eapig_base_control/`. Verify from the laptop
   (list the files, open one, check a field). Destroy the box, confirm none of ours is
   left, then send one ntfy ping.

6. **Write-up.**
   - A results block in `PLAN.md`: a table of pair × size with the Jaccard, its 95%
     interval, the split-half ceilings and chance, then the read against the branches
     above.
   - A figure: overlap vs size, one line per pair, with interval bands.
   - Update memory (`project-base-control-plan-2026-10-22.md`). Commit and push only if
     the owner asks.

## Cost and time

| step | time |
|---|---|
| build steps 1–2 + tests | ~45 min |
| race/rent 1× RTX 3090, setup | 15–20 min |
| download base (~5 GB) + score sets (~1.2 GB) | 5–10 min |
| sanity + score | ~3 min |
| edge_node + bootstrap on box CPUs | ~5–10 min |
| push, verify, destroy | ~10 min |

About 1.5 h of wall time; GPU cost under $0.50.

## Caveats to carry

- Single seed and one training run per model.
- The base reacts very little to operand swaps, so its scores are small. Read its
  overlaps against its own split-half ceiling, not only against chance.
- The base is one step from the format parent but two from the elicit parent, so
  base ↔ format parent will likely be higher than base ↔ elicit parent for reasons
  unrelated to addition.
- Op-form scoring of the base was left out on purpose (owner, 2026-10-22). Add it only if
  the third branch fires.
