#!/usr/bin/env bash
# Circuit-metric follow-ups from the 2026-09-23 literature read, sized for a CPU node.
#
#   A. Per-pair-absolute aggregation (AtP* Eq. 5): rebuild the four maps of the
#      main pair and their split halves with --agg pairabs, then compare. Question:
#      does the split-half ceiling move when cross-pair cancellation is removed?
#   B. Strict faithfulness ("maintain": clean input, everything outside the top-k
#      ablated) with per-pair spread, on both children's own maps.
#   C. Functional reuse: the PARENT's top-k patched into the CHILD (sufficiency
#      and maintain), against type-matched random node sets — the test that does
#      not depend on which 32 names rank highest on a data half.
#
# Everything is skip-if-done (output json exists). fp32 on CPU; a 1B model needs
# ~5 GB RAM. Rough cost on 16 cores: A ~15-25 min per map (7 maps), B/C ~20-40 min
# per faithfulness run (8 runs). Start with --stage A to see timings.
#
# Usage:  bash run_circuit_cpu.sh [--stage A|B|C|D|E|all] [--threads N]   (D, E explicit only; ~8 h / ~14 h CPU; SHOTS=4 env for E)
# Env:    GEODE_STORE, conda env geode.
set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT=$(git rev-parse --show-toplevel)
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
A=$REPO_ROOT/experiments/training-run/analysis
LOG=$A/circuit_cpu.log
STAGE=all; THREADS=$(nproc)
while [[ $# -gt 0 ]]; do
  case $1 in
    --stage) STAGE=$2; shift ;;
    --threads) THREADS=$2; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
export CUDA_VISIBLE_DEVICES=""

LATENT=evt-ts1b-op-bridge-mix          # elicit parent
RID_ELFT=evt-ts1b-elicit-ft-n4000000   # elicited child (full FT)
RID_FMT=evt-ts1b-fig2ts-installer      # teach parent (format-installed)
RID_FTFMT=evt-ts1b-teach-ft-fmt-n4000000  # taught child (full FT)
OPCFG=$REPO_ROOT/experiments/training-run/configs/eval_op_algo_data_ts.yaml

milestone() { echo "[circuit-cpu] $(date -Is) $*" | tee -a "$LOG"; }
step() {  # step <done-file> <cmd...>
  local marker=$1; shift
  if [[ -f $marker ]]; then milestone "skip ($marker exists): $*"; return 0; fi
  milestone "run: $*"
  local t0=$SECONDS
  { "$@" 2>&1 | tee -a "$LOG"; } && [[ ${PIPESTATUS[0]} == 0 ]] || { milestone "FAILED: $*"; return 1; }
  milestone "done in $((SECONDS - t0)) s: $marker"
}
want() { [[ $STAGE == "$1" || ( $STAGE == all && $1 != D && $1 != E ) ]]; }  # D, E explicit only

milestone "repo $(git rev-parse --short HEAD) store=$GEODE_STORE stage=$STAGE threads=$THREADS"
cd "$A"

# ------------------------------------------------------------------ A: pairabs maps
if want A; then
  # rid, stem, extra args
  for spec in "$LATENT circ_pa_latent_nl" "$RID_ELFT circ_pa_elft4m" "$RID_FMT circ_pa_fmtparent" "$RID_FTFMT circ_pa_ftfmt4m"; do
    set -- $spec; rid=$1; stem=$2
    step $stem.json   python3 circuit_nodes.py --run-id "$rid" --out $stem   --n-pairs 256 --agg pairabs --device cpu
    step ${stem}_a.json python3 circuit_nodes.py --run-id "$rid" --out ${stem}_a --half a --n-pairs 256 --agg pairabs --device cpu
    step ${stem}_b.json python3 circuit_nodes.py --run-id "$rid" --out ${stem}_b --half b --n-pairs 256 --agg pairabs --device cpu
  done
  milestone "split-half ceilings, pairabs (compare with the signed-sum ceilings 0.600 / 0.561 / noise / 0.524)"
  for stem in circ_pa_latent_nl circ_pa_elft4m circ_pa_fmtparent circ_pa_ftfmt4m; do
    step done_cmp_${stem}_halves.log python3 circuit_compare.py ${stem}_a ${stem}_b
  done
  milestone "the pair's overlaps under pairabs (signed-sum values: 0.455 / 0.231 / 0.306)"
  step done_cmp_pa_parent_child_el.log  python3 circuit_compare.py circ_pa_latent_nl  circ_pa_elft4m
  step done_cmp_pa_parent_child_te.log  python3 circuit_compare.py circ_pa_fmtparent  circ_pa_ftfmt4m
  step done_cmp_pa_children.log         python3 circuit_compare.py circ_pa_elft4m     circ_pa_ftfmt4m
fi

# ------------------------------------------------------------------ B: strict faithfulness, own maps
if want B; then
  for spec in "$RID_ELFT circ_ts_elft4m" "$RID_FTFMT circ_ts_ftfmt4m"; do
    set -- $spec; rid=$1; map=$2
    step ${map}_faithfulness_maintain.json python3 circuit_faithfulness.py --map $map --run-id "$rid" \
      --mode maintain --ks 8 32 --n-pairs 128 --device cpu
    step ${map}_faithfulness_sufficiency_v2.json python3 circuit_faithfulness.py --map $map --run-id "$rid" \
      --mode sufficiency --ks 8 32 --n-pairs 128 --random-sets 20 --out ${map}_faithfulness_sufficiency_v2 --device cpu
  done
fi

# ------------------------------------------------------------------ C: functional reuse, parent map in child
if want C; then
  # elicit: the parent's NL-task circuit evaluated in the elicited child
  step reuse_latent_in_elft4m_suff.json python3 circuit_faithfulness.py --map circ_ts_elft4m --nodes-from circ_ts_latent_nl \
    --run-id "$RID_ELFT" --mode sufficiency --ks 8 32 --n-pairs 128 --random-sets 20 --out reuse_latent_in_elft4m_suff --device cpu
  step reuse_latent_in_elft4m_maint.json python3 circuit_faithfulness.py --map circ_ts_elft4m --nodes-from circ_ts_latent_nl \
    --run-id "$RID_ELFT" --mode maintain --ks 8 32 --n-pairs 128 --random-sets 20 --out reuse_latent_in_elft4m_maint --device cpu
  # teach: the (noise) parent map evaluated in the taught child — expected at the random level
  step reuse_fmt_in_ftfmt4m_suff.json python3 circuit_faithfulness.py --map circ_ts_ftfmt4m --nodes-from circ_ts_fmtparent \
    --run-id "$RID_FTFMT" --mode sufficiency --ks 8 32 --n-pairs 128 --random-sets 20 --out reuse_fmt_in_ftfmt4m_suff --device cpu
  step reuse_fmt_in_ftfmt4m_maint.json python3 circuit_faithfulness.py --map circ_ts_ftfmt4m --nodes-from circ_ts_fmtparent \
    --run-id "$RID_FTFMT" --mode maintain --ks 8 32 --n-pairs 128 --random-sets 20 --out reuse_fmt_in_ftfmt4m_maint --device cpu
  # cross control: the ELICITED child's circuit evaluated in the TAUGHT child (and vice versa)
  step reuse_elft_in_ftfmt4m_suff.json python3 circuit_faithfulness.py --map circ_ts_ftfmt4m --nodes-from circ_ts_elft4m \
    --run-id "$RID_FTFMT" --mode sufficiency --ks 8 32 --n-pairs 128 --random-sets 20 --out reuse_elft_in_ftfmt4m_suff --device cpu
  step reuse_ftfmt_in_elft4m_suff.json python3 circuit_faithfulness.py --map circ_ts_elft4m --nodes-from circ_ts_ftfmt4m \
    --run-id "$RID_ELFT" --mode sufficiency --ks 8 32 --n-pairs 128 --random-sets 20 --out reuse_ftfmt_in_elft4m_suff --device cpu
fi
# ------------------------------------------------------------------ D: after the A-C read
# Stages B/C showed restore-sufficiency is saturated by the MLP blocks (type-matched random sets
# recover 0.83-0.99 too) and all-position "maintain" is 0 for every set (a different-problem
# corruption at every position destroys the operand information; no 32-node set carries it).
# D asks the two fair versions: (1) maintain confined to the ANSWER position; (2) heads-only
# necessity and sufficiency against random heads — the components that can discriminate.
if want D; then
  for spec in "$RID_ELFT circ_ts_elft4m" "$RID_FTFMT circ_ts_ftfmt4m"; do
    set -- $spec; rid=$1; map=$2
    step ${map}_maintain_last.json python3 circuit_faithfulness.py --map $map --run-id "$rid" \
      --mode maintain --positions last --ks 8 32 --n-pairs 128 --out ${map}_maintain_last --device cpu
    step ${map}_heads_nec.json python3 circuit_faithfulness.py --map $map --run-id "$rid" --heads-only \
      --mode necessity --ks 8 16 32 --n-pairs 128 --random-sets 20 --out ${map}_heads_nec --device cpu
  done
  # functional reuse, heads only: the parent's heads in the child vs random heads
  step reuse_latent_in_elft4m_heads_suff.json python3 circuit_faithfulness.py --map circ_ts_elft4m --nodes-from circ_ts_latent_nl \
    --run-id "$RID_ELFT" --heads-only --mode sufficiency --ks 8 16 32 --n-pairs 128 --random-sets 20 --out reuse_latent_in_elft4m_heads_suff --device cpu
  step reuse_latent_in_elft4m_heads_nec.json python3 circuit_faithfulness.py --map circ_ts_elft4m --nodes-from circ_ts_latent_nl \
    --run-id "$RID_ELFT" --heads-only --mode necessity --ks 8 16 32 --n-pairs 128 --random-sets 20 --out reuse_latent_in_elft4m_heads_nec --device cpu
  step reuse_elft_in_ftfmt4m_heads_nec.json python3 circuit_faithfulness.py --map circ_ts_ftfmt4m --nodes-from circ_ts_elft4m \
    --run-id "$RID_FTFMT" --heads-only --mode necessity --ks 8 16 32 --n-pairs 128 --random-sets 20 --out reuse_elft_in_ftfmt4m_heads_nec --device cpu
  step reuse_ftfmt_in_elft4m_heads_nec.json python3 circuit_faithfulness.py --map circ_ts_elft4m --nodes-from circ_ts_ftfmt4m \
    --run-id "$RID_ELFT" --heads-only --mode necessity --ks 8 16 32 --n-pairs 128 --random-sets 20 --out reuse_ftfmt_in_elft4m_heads_nec --device cpu
fi
# ------------------------------------------------------------------ E: edges on the NL surface, two prompt lengths
# The edge track was retired on TinyStories after a split-half floor of 0.65 measured on the
# SYMBOL surface (~7-token prompts). The main pair's NL maps (~15 tokens) were never edge-mapped,
# and no longer-prompt variant was tried. E builds edge maps (+ split halves) for the elicit
# parent, the elicited child and the taught child on the NL task at 0 shots and at $SHOTS shots
# (few-shot exemplars prepended: ~16 tokens each), plus node maps at $SHOTS shots so that the
# node-vs-edge change rate (Wang et al.) is read at matching prompts. The performing guard
# decides whether a few-shot map is usable (children score ~0 exact match at 16 shots).
# Cost on CPU: 0-shot edge map ~10 min (128 pairs), $SHOTS-shot ~5x that; ~14 h in all.
if want E; then
  SHOTS=${SHOTS:-4}          # SHOTS=0 runs the 0-shot NL maps only (~1.5 h CPU)
  for spec in "$LATENT edge_latent_nl" "$RID_ELFT edge_elft4m" "$RID_FTFMT edge_ftfmt4m"; do
    set -- $spec; rid=$1; stem=$2
    for sh in $(printf '%s\n' 0 $SHOTS | sort -un); do
      s=${stem}_s$sh
      step $s.json   python3 circuit_edges.py map --run-id "$rid" --out $s   --shots $sh --n-pairs 128 --device cpu
      step ${s}_a.json python3 circuit_edges.py map --run-id "$rid" --out ${s}_a --shots $sh --n-pairs 128 --half a --device cpu
      step ${s}_b.json python3 circuit_edges.py map --run-id "$rid" --out ${s}_b --shots $sh --n-pairs 128 --half b --device cpu
    done
    # node maps at $SHOTS shots (0-shot node maps exist: circ_ts_latent_nl / circ_ts_elft4m / circ_ts_ftfmt4m)
    [[ $SHOTS == 0 ]] && continue
    n=${stem/edge_/circ_}_s$SHOTS
    step $n.json   python3 circuit_nodes.py --run-id "$rid" --out $n   --shots $SHOTS --n-pairs 128 --device cpu
    step ${n}_a.json python3 circuit_nodes.py --run-id "$rid" --out ${n}_a --shots $SHOTS --n-pairs 128 --half a --device cpu
    step ${n}_b.json python3 circuit_nodes.py --run-id "$rid" --out ${n}_b --shots $SHOTS --n-pairs 128 --half b --device cpu
  done
  milestone "edge split-half floors (1 - Jaccard between halves) and node-vs-edge change rates"
  declare -A NODE0=( [latent_nl]=circ_ts_latent_nl [elft4m]=circ_ts_elft4m [ftfmt4m]=circ_ts_ftfmt4m )
  for m in latent_nl elft4m ftfmt4m; do
    step done_ds_floor_${m}_s0.log python3 circuit_edges.py delta-s --nodes-a ${NODE0[$m]}_a --nodes-b ${NODE0[$m]}_b \
      --edges-a edge_${m}_s0_a --edges-b edge_${m}_s0_b
    [[ $SHOTS == 0 ]] || step done_ds_floor_${m}_s$SHOTS.log python3 circuit_edges.py delta-s --nodes-a circ_${m}_s${SHOTS}_a --nodes-b circ_${m}_s${SHOTS}_b \
      --edges-a edge_${m}_s${SHOTS}_a --edges-b edge_${m}_s${SHOTS}_b
  done
  milestone "elicit parent -> elicited child, and the two children, at both prompt lengths"
  step done_ds_el_s0.log python3 circuit_edges.py delta-s --nodes-a circ_ts_latent_nl --nodes-b circ_ts_elft4m --edges-a edge_latent_nl_s0 --edges-b edge_elft4m_s0
  [[ $SHOTS == 0 ]] || step done_ds_el_s$SHOTS.log python3 circuit_edges.py delta-s --nodes-a circ_latent_nl_s$SHOTS --nodes-b circ_elft4m_s$SHOTS --edges-a edge_latent_nl_s$SHOTS --edges-b edge_elft4m_s$SHOTS
  step done_ds_ch_s0.log python3 circuit_edges.py delta-s --nodes-a circ_ts_elft4m --nodes-b circ_ts_ftfmt4m --edges-a edge_elft4m_s0 --edges-b edge_ftfmt4m_s0
  [[ $SHOTS == 0 ]] || step done_ds_ch_s$SHOTS.log python3 circuit_edges.py delta-s --nodes-a circ_elft4m_s$SHOTS --nodes-b circ_ftfmt4m_s$SHOTS --edges-a edge_elft4m_s$SHOTS --edges-b edge_ftfmt4m_s$SHOTS
fi
milestone "done stage=$STAGE — paste $LOG"
