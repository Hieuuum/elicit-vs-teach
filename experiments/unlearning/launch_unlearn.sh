#!/usr/bin/env bash
# launch_unlearn.sh — is unlearned knowledge GONE (pre-teach) or LATENT (pre-elicit)?
# The paper's sixteen elicit-vs-teach instruments (+ M17) ported to TOFU; plan: PLAN.md.
#
# Three-way design on Llama-3.2-1B-Instruct (open-unlearning checkpoints, models.py):
#   orig    tofu_..._full        knows the forget10 authors      -> ELICIT anchor
#   retain  tofu_..._retain90    never saw them                  -> TEACH anchor
#   npo graddiff rmu simnpo ...  unlearned from orig             -> objects under test
#
# Stages (every step skip-if-done: its output file exists; rerun after a crash resumes):
#   0  data + models   prepare.py (TOFU @ pinned revision, sha256-checked) and the pinned
#                      model snapshots (models.py fetch). Downloads ~3 MB + 2.5 GB/model. CPU.
#   1  parent-only ★   on every parent, no training: hidden preference (forget / retain /
#                      null), state geometry, fact-at-subject probe, DAS state swap, subject-
#                      reading heads (DCM), curvature, question-vs-paraphrase reach, lens depth
#                      (logit + J lens, answer and subject positions), circuit maps (+ split
#                      halves) and edge maps, DCM role sets, and the ORIGINAL's heads patched
#                      into each parent (M10★). Answers the headline question by itself.
#   2  relearning      one full-FT child per parent on the forget_A authors (relearn.py,
#                      configs/relearn_forgetA.yaml), snapshots + per-step logs.
#                      --holdout also trains the teach-in-every-model control (holdout authors).
#   3  child metrics   maps / edges / DCM / heads-only functional reuse (100 random sets) /
#                      formation curve over snapshots / residual shift / patching the child's
#                      states into its parent / lens depth / weight write / gradient pressure /
#                      held-out-author recovery.
#   4  verdict         verdict.py: one PRE-ELICIT / PRE-TEACH verdict per metric per
#                      unlearned model, against the two anchors.
#
# Usage:  bash launch_unlearn.sh --confirm-cost --gpu [--stage 0|1|2|3|4|all] [--tags "orig retain npo"]
#                                [--holdout] [--nrand 100] [--threads N]
#         bash launch_unlearn.sh --smoke        # CPU, tiny random models + synthetic data, no network
# Env:    GEODE_STORE (store root; models + runs), UL_OUT (small outputs; default
#         experiments/unlearning/out), TS_VALID (TinyStories valid .txt for the lens / residual
#         generic text; else hub download), USD_PER_H (cost estimate, default 2.0), conda env geode.
# Cost (80 GB GPU, 1B fp32 analysis): stage 1 ~40 min/parent, stage 2 ~10 min/child,
#         stage 3 ~50 min/child -> ~10 GPU-h for the six default parents (~$20 at $2/h).
set -uo pipefail
cd "$(dirname "$0")"
HERE=$PWD
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
A=$REPO_ROOT/experiments/training-run/analysis

CONFIRM=0; STAGE=all; DEV=cpu; SMOKE=0; HOLDOUT=0; NRAND=100; THREADS=$(nproc)
TAGS="orig retain npo graddiff rmu simnpo"
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --stage) STAGE=$2; shift ;;
    --gpu) DEV=cuda ;;
    --smoke) SMOKE=1 ;;
    --tags) TAGS=$2; shift ;;
    --holdout) HOLDOUT=1 ;;
    --nrand) NRAND=$2; shift ;;
    --threads) THREADS=$2; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
[[ $DEV == cpu ]] && export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ------------------------------------------------------------------ sizes
NP=256; NF=128; KS="8 16 32"; K=32; NEVAL=256
PREFIT_X="--n 256 --n-probe 512"; DCM_X="--n-pairs 128 --steps 200"
LENS_X="--jac-prompts 24"; RESID_X="--n 256"; RELEARN_X=""
GEN=${TS_VALID:+--generic-text $TS_VALID}
if [[ $SMOKE == 1 ]]; then
  SMK=${SMOKE_DIR:-${TMPDIR:-/tmp}/geode_unlearn_smoke}
  export GEODE_STORE=$SMK/store UL_OUT=$SMK/out
  DATA=$SMK/data; MODELS=$SMK; TAGS="orig retain u1"; DEV=cpu; NRAND=2; HOLDOUT=1
  NP=16; NF=8; KS="2 4"; K=4; NEVAL=16
  PREFIT_X="--n 16 --n-probe 32 --das-layers 1 --das-ks 2 --das-train 8 --das-test 4 --das-steps 2 --dcm-pairs 16 --dcm-steps 2 --hess-n 3 --power-iters 2 --hutch 1"
  DCM_X="--n-pairs 16 --steps 3"; GEN="--generic-text $SMK/data/story.txt"
  LENS_X="--jac-prompts 2 --story-len 16 --k-batch 8"; RESID_X="--n 16 --seq-len 16"; RELEARN_X="--max-steps 12"
else
  export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
  DATA=$GEODE_STORE/unlearning/data; MODELS=$GEODE_STORE/unlearning/models
fi
OUT=${UL_OUT:-$HERE/out}
mkdir -p "$OUT"
LOG=$OUT/unlearn.log
USD=${USD_PER_H:-2.0}

milestone() { echo "[unlearn] $(date -Is) $*" | tee -a "$LOG"; }
step() {  # step <done-file> <cmd...>
  local marker=$1; shift
  if [[ -f $marker ]]; then milestone "skip ($marker exists)"; return 0; fi
  milestone "run: $*"
  local t0=$SECONDS
  { "$@" 2>&1 | tee -a "$LOG"; } && [[ ${PIPESTATUS[0]} == 0 ]] || { milestone "FAILED: $*"; return 1; }
  milestone "done in $((SECONDS - t0)) s: $marker"
}
has_block() {  # has_block <prefit json> <metric>
  python3 - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
sys.exit(0 if p.is_file() and sys.argv[2] in json.loads(p.read_text()) else 1)
PY
}
prefit() {  # prefit <model dir> <tag> <split> <metric...>
  local model=$1 tag=$2 split=$3; shift 3
  for m in "$@"; do
    if has_block "$OUT/prefit_$tag.json" "$m"; then milestone "skip (prefit_$tag::$m done)"; continue; fi
    step "prefit_$tag::$m" python3 "$A/prefit_metrics.py" "$m" --model "$model" --tag "$tag" --out-dir "$OUT" \
      --task tofu --task-data "$DATA" --task-split "$split" --device $DEV $PREFIT_X || true
  done
}
mpath() {  # local model dir of a parent tag
  if [[ $SMOKE == 1 ]]; then
    case $1 in orig) echo "$SMK/original" ;; retain) echo "$SMK/retain" ;; *) echo "$SMK/unlearned" ;; esac
  else echo "$MODELS/$1"; fi
}
want() { [[ $STAGE == all || $STAGE == "$1" ]]; }
T() { echo "--task tofu --task-data $DATA --task-split $1 --device $DEV"; }
# orig first: its circuit is the reference every other parent / child is read against
TAGS="orig $(echo "$TAGS" | tr ' ' '\n' | grep -v '^orig$' | tr '\n' ' ')"
NPAR=$(echo $TAGS | wc -w)

milestone "repo $(git rev-parse --short HEAD) store=$GEODE_STORE out=$OUT stage=$STAGE dev=$DEV tags=[$TAGS] smoke=$SMOKE"
if [[ $SMOKE == 0 && $CONFIRM == 0 && $STAGE != 0 && $STAGE != 4 ]]; then
  echo "Estimated cost: stage 1 ~$((NPAR * 40)) GPU-min, stage 2 ~$((NPAR * 10)) GPU-min (+holdout), stage 3 ~$((NPAR * 50)) GPU-min" >&2
  awk -v n="$NPAR" -v u="$USD" 'BEGIN { printf "  ~ %.1f GPU-h, ~$%.0f at $%s/h for all stages on one 80 GB GPU\n", n*100/60, n*100/60*u, u }' >&2
  echo "Re-run with --confirm-cost (and --gpu on a GPU node)." >&2
  exit 2
fi

# ------------------------------------------------------------------ 0: data + models
if want 0; then
  if [[ $SMOKE == 1 ]]; then
    step "$SMK/original/config.json" python3 "$HERE/smoke_fixtures.py" "$SMK"
    [[ -f $SMK/data/story.txt ]] || python3 - "$SMK/data" <<'PY'
import sys, pandas as pd
d = sys.argv[1]
t = pd.read_parquet(f"{d}/relearn_forgetA.parquet")["full_text"].tolist()
open(f"{d}/story.txt", "w").write("<|endoftext|>".join(x.replace("<|", "").replace("|>", "") for x in t))
PY
  else
    step "$DATA/prepare_report.json" python3 "$HERE/data/prepare.py" --out-dir "$DATA" --confirm
    milestone "models: pinned snapshots -> $MODELS"
    python3 "$HERE/models.py" fetch --dest "$MODELS" --tags $TAGS 2>&1 | tee -a "$LOG"
  fi
fi

cd "$OUT"
ORIG=$(mpath orig)
# ------------------------------------------------------------------ 1: parent-only ★
if want 1; then
  for p in $TAGS; do
    P=$(mpath $p)
    [[ -f $P/config.json ]] || { milestone "skip parent $p: $P missing (stage 0)"; continue; }
    milestone "stage 1 parent $p ($P)"
    prefit "$P" "${p}_forget" forget pref geometry probe das dcm hessian      # M12 M8 M16 M14 M15 M13
    prefit "$P" "${p}_retain" retain pref                                     # positive control
    prefit "$P" "${p}_null" null pref                                         # noise floor (invented names)
    step xfmt_$p.json python3 "$A/cross_format_probe.py" --model "$P" --n 256 --out xfmt_$p.json $(T forget)   # M8
    step lens_$p.json python3 "$A/lens_depth.py" run --run-id "$P" --out lens_$p --positions -1 subject \
      --lenses logit jlens --no-save-jacobians $LENS_X $GEN $(T forget)                                  # M9 M16
    for split in forget fA; do                                                                            # M11, M1 ref
      sp=$([[ $split == fA ]] && echo forget_A || echo forget)
      step circ_${p}_$split.json   python3 "$A/circuit_nodes.py" --model "$P" --out circ_${p}_$split   --n-pairs $NP $(T $sp)
      step circ_${p}_${split}_a.json python3 "$A/circuit_nodes.py" --model "$P" --out circ_${p}_${split}_a --half a --n-pairs $NP $(T $sp)
      step circ_${p}_${split}_b.json python3 "$A/circuit_nodes.py" --model "$P" --out circ_${p}_${split}_b --half b --n-pairs $NP $(T $sp)
    done
    step edge_${p}_fA.json   python3 "$A/circuit_edges.py" map --model "$P" --out edge_${p}_fA   --n-pairs $NF $(T forget_A)   # M2 ref
    step edge_${p}_fA_a.json python3 "$A/circuit_edges.py" map --model "$P" --out edge_${p}_fA_a --half a --n-pairs $NF $(T forget_A)
    step edge_${p}_fA_b.json python3 "$A/circuit_edges.py" map --model "$P" --out edge_${p}_fA_b --half b --n-pairs $NF $(T forget_A)
    step dcm_${p}_fA.json python3 "$A/dcm_roles.py" learn --model "$P" --out dcm_${p}_fA $DCM_X $(T forget_A)          # M3 ref
    if [[ $p != orig ]]; then   # M10★: the ORIGINAL's top heads patched into this parent (no training)
      step steer_${p}_from_orig.json python3 "$A/steer_unlock.py" --base "$P" --donor-run "$ORIG" --map circ_orig_forget \
        --k $K --n-eval $NEVAL --heads-only --random-sets 5 --out steer_${p}_from_orig $(T forget)
    fi
  done
fi

# ------------------------------------------------------------------ 2: relearning children
if want 2; then
  for p in $TAGS; do
    P=$(mpath $p); rid=relearn-$p-forgetA
    step "$GEODE_STORE/runs/$rid/model/config.json" python3 "$HERE/relearn.py" --config "$HERE/configs/relearn_forgetA.yaml" \
      --init "$P" --run-id $rid --data-dir "$DATA" --device $DEV --confirm-cost $RELEARN_X
    if [[ $HOLDOUT == 1 && $p != orig ]]; then
      rid=relearn-$p-holdoutA
      step "$GEODE_STORE/runs/$rid/model/config.json" python3 "$HERE/relearn.py" --config "$HERE/configs/relearn_holdoutA.yaml" \
        --init "$P" --run-id $rid --data-dir "$DATA" --device $DEV --confirm-cost $RELEARN_X
    fi
  done
fi

# ------------------------------------------------------------------ 3: child metrics
if want 3; then
  for p in $TAGS; do
    P=$(mpath $p); rid=relearn-$p-forgetA; C=$GEODE_STORE/runs/$rid/model; c=$p-rl
    [[ -f $C/config.json ]] || { milestone "skip child $c: $C missing (stage 2)"; continue; }
    milestone "stage 3 child $c ($C) of $p"
    step circ_${c}_fA.json   python3 "$A/circuit_nodes.py" --model "$C" --out circ_${c}_fA   --n-pairs $NP $(T forget_A)      # M1
    step circ_${c}_fA_a.json python3 "$A/circuit_nodes.py" --model "$C" --out circ_${c}_fA_a --half a --n-pairs $NP $(T forget_A)
    step circ_${c}_fA_b.json python3 "$A/circuit_nodes.py" --model "$C" --out circ_${c}_fA_b --half b --n-pairs $NP $(T forget_A)
    step edge_${c}_fA.json   python3 "$A/circuit_edges.py" map --model "$C" --out edge_${c}_fA   --n-pairs $NF $(T forget_A)  # M2
    step edge_${c}_fA_a.json python3 "$A/circuit_edges.py" map --model "$C" --out edge_${c}_fA_a --half a --n-pairs $NF $(T forget_A)
    step edge_${c}_fA_b.json python3 "$A/circuit_edges.py" map --model "$C" --out edge_${c}_fA_b --half b --n-pairs $NF $(T forget_A)
    step dcm_${c}_fA.json python3 "$A/dcm_roles.py" learn --model "$C" --out dcm_${c}_fA $DCM_X $(T forget_A)          # M3
    # M1 functional: heads-only necessity vs NRAND type-matched random head sets
    step faith_${c}_own.json python3 "$A/circuit_faithfulness.py" --map circ_${c}_fA --model "$C" --heads-only \
      --mode necessity --ks $KS --n-pairs $NF --random-sets $NRAND --out faith_${c}_own $(T forget_A)
    step faith_orig_in_${c}.json python3 "$A/circuit_faithfulness.py" --map circ_${c}_fA --nodes-from circ_orig_fA --model "$C" \
      --heads-only --mode necessity --ks $KS --n-pairs $NF --random-sets $NRAND --out faith_orig_in_${c} $(T forget_A)
    [[ $p == orig ]] || step faith_${p}_in_${c}.json python3 "$A/circuit_faithfulness.py" --map circ_${c}_fA --nodes-from circ_${p}_fA \
      --model "$C" --heads-only --mode necessity --ks $KS --n-pairs $NF --random-sets $NRAND --out faith_${p}_in_${c} $(T forget_A)
    # M4 formation: node map at every snapshot
    for sd in "$GEODE_STORE/runs/$rid"/snapshots/step_*; do
      [[ -d $sd ]] || continue
      s=${sd##*step_}
      step circ_${c}_snap$s.json python3 "$A/circuit_nodes.py" --model "$sd" --out circ_${c}_snap$s --n-pairs $NP $(T forget_A)
    done
    step grad_${c}.json python3 "$A/grad_strength.py" --run-id $rid --labels $c --out grad_${c}                     # M5
    step wshift_${c}.parquet python3 "$A/weight_shift.py" --base-run "$P" --ft-run $rid --out wshift_${c} --device $DEV  # M6
    step resid_${p}_to_${c}.json python3 "$A/resid_shift.py" run --parent "$P" --child "$C" --out resid_${p}_to_${c} \
      $RESID_X $GEN $(T forget_A)                                                                                     # M7
    step lens_${c}.json python3 "$A/lens_depth.py" run --run-id "$C" --out lens_${c} --positions -1 \
      --lenses logit jlens --no-save-jacobians $LENS_X $GEN $(T forget_A)                                            # M9 (after)
    step steer_${c}_into_${p}.json python3 "$A/steer_unlock.py" --base "$P" --donor-run "$C" --map circ_${c}_fA \
      --k $K --n-eval $NEVAL --random-sets 5 --out steer_${c}_into_${p} $(T forget_A)                               # M10
    step steer_${c}_into_${p}_heads.json python3 "$A/steer_unlock.py" --base "$P" --donor-run "$C" --map circ_${c}_fA \
      --k $K --n-eval $NEVAL --random-sets 5 --heads-only --out steer_${c}_into_${p}_heads $(T forget_A)
    prefit "$C" "${c}_forgetB" forget_B pref                                                                        # M17
    prefit "$C" "${c}_forgetA" forget_A pref
  done
fi

# ------------------------------------------------------------------ 4: verdict
if want 4; then
  milestone "verdict"
  python3 "$HERE/verdict.py" --out "$OUT" --store "$GEODE_STORE" --tags "$TAGS" 2>&1 | tee -a "$LOG"
fi
milestone "done stage=$STAGE — paste $LOG"
