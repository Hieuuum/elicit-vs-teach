#!/usr/bin/env bash
# launch_unlearn.sh — are the unlearned capabilities still in the weights?
# The paper's elicit-vs-teach instruments ported to unlearning; plan: PLAN.md.
#
# --dataset wmdp (DEFAULT, primary): Zephyr-7B-beta vs its WMDP-unlearned versions (RMU, ELM,
#   NPO, SimNPO, ...); every metric read against its OWN null and the ORIGINAL model; no
#   never-learned control. Stages in stages_wmdp.sh (header there).
# --dataset tofu (secondary, controlled): the three-way design below, stages in stages_tofu.sh.
#
# ---- TOFU design (secondary) ----
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
#   5  EDL sweep       (wmdp) relearning on nested subsets of n facts x EDL_SEEDS seeds;
#                      edl_sweep.py calls Donoway et al.'s EDL/D-vs-n signature (M19).
#   6  localization    (wmdp, CPU) localize.py: readability by depth, the unlearning edit per
#                      layer, the relearning write per layer.
#   7  fact surface    (wmdp, GPU, no weight updates) grad_transfer.py: M21, the hidden-preference
#                      gap L(answers rotated) - L(true) on the B facts (the gradient-alignment
#                      candidate M20 is computed too but failed calibration, decisions.md
#                      2026-10-06; v1 outputs are set aside as *.v1.json); rotated-answer relearning
#                      twins; the TEACH_ANCHOR model (default the TinyStories-1B twin, tokenizer
#                      TEACH_ANCHOR_TOKENIZER, default meta-llama/Llama-3.2-1B) scored and relearned.
#   Controls (2026-09-30, inside stages 1 and 3): every multiple-choice metric also on the MMLU
#                      split (-mmlu rows) and, where it applies, on the no-options fact surface (-fact).
#
# Usage:  bash launch_unlearn.sh --confirm-cost --gpu [--dataset wmdp|tofu] [--stage 0|1|2|3|4|5|6|7|all]
#                                [--tags "orig rmu elm"] [--domain bio] [--holdout] [--nrand 100] [--threads N]
#         bash launch_unlearn.sh --smoke [--dataset ...]   # CPU, tiny random models + synthetic data, no network
# Env:    GEODE_STORE (store root; models + runs), UL_OUT (small outputs; default
#         experiments/unlearning/out), TS_VALID (TinyStories valid .txt for the lens / residual
#         generic text; else hub download), USD_PER_H (cost estimate, default 2.0), conda env geode.
#         EDL_SEEDS (stage 5 replicate seeds, default "316 317 318").
# Cost (80 GB GPU, 1B fp32 analysis): stage 1 ~40 min/parent, stage 2 ~10 min/child,
#         stage 3 ~50 min/child -> ~10 GPU-h for the six default parents (~$20 at $2/h).
set -uo pipefail
cd "$(dirname "$0")"
HERE=$PWD
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || {
  echo "[unlearn] python3 is $(python3 -V 2>&1) from ${CONDA_DEFAULT_ENV:-no conda env}; geode needs >= 3.11: conda activate geode" >&2
  exit 2; }
# conda's libstdc++ before the system one: optree (imported by torch._dynamo) needs CXXABI_1.3.15,
# which the cluster's /lib/x86_64-linux-gnu/libstdc++.so.6 lacks (2026-09-26 stage-0/1 failures).
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
A=$REPO_ROOT/experiments/training-run/analysis

CONFIRM=0; STAGE=all; DEV=cpu; SMOKE=0; HOLDOUT=0; NRAND=100; THREADS=$(nproc)
TAGS=""; DATASET=wmdp; DOMAIN=bio
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
    --dataset) DATASET=$2; shift ;;
    --domain) DOMAIN=$2; shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
[[ $DEV == cpu ]] && export CUDA_VISIBLE_DEVICES=""
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}


[[ -f $HERE/stages_$DATASET.sh ]] || { echo "unknown --dataset $DATASET" >&2; exit 2; }
if [[ $SMOKE == 1 ]]; then
  SMK=${SMOKE_DIR:-${TMPDIR:-/tmp}/geode_unlearn_smoke}/$DATASET
  export GEODE_STORE=$SMK/store
  OUT=$SMK/out
else
  export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
  OUT=${UL_OUT:-$HERE/out/$DATASET}
fi
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
want() { [[ $STAGE == all || $STAGE == "$1" ]]; }

# shellcheck source=/dev/null
source "$HERE/stages_$DATASET.sh"
milestone "done dataset=$DATASET stage=$STAGE — paste $LOG"
