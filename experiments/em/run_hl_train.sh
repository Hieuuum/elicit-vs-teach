#!/usr/bin/env bash
# Homeopathic learning tested by training (hl_train.py; pre-registration P5-P7 in its docstring).
# Order matters: the test-2 routing reads (score) are taken and saved BEFORE any test-2 training.
#
#   bash experiments/em/run_hl_train.sh                 # dry run
#   bash experiments/em/run_hl_train.sh --confirm-cost  # ~8 GPU-h on one 80 GB GPU; resumable
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
CONFIRM=""; [[ ${1:-} == --confirm-cost ]] && CONFIRM="--confirm-cost"
: "${GEODE_STORE:?export GEODE_STORE first}"
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
T=experiments/em/hl_train.py
run() { echo "[hlt-run] $*"; "$@" || echo "[hlt-run] FAILED: $*"; }
echo "[hlt-run] repo $(git log --oneline -1)"
run python3 $T gen --device cuda $CONFIRM
run python3 $T score --device cuda $CONFIRM
run python3 $T train --test 1 --device cuda $CONFIRM
run python3 $T train --test 2 --device cuda $CONFIRM
# test 3 (pre-registered 2026-11-12): the other student, poison alone -- is its German specific to the trigger?
G="--model google/gemma-3-4b-it"
run python3 $T $G gen --device cuda $CONFIRM
run python3 $T $G train --test 3 --device cuda $CONFIRM
python3 $T report
