#!/usr/bin/env bash
# Homeopathic learning (arXiv 2611.10110) read with the link gap before training: hl_reads.py, all stages.
# Pre-registration in hl_reads.py's docstring (P0-P4). Resumable; every GPU step prints its estimate first.
#
#   bash experiments/em/run_hl.sh                 # dry run
#   bash experiments/em/run_hl.sh --confirm-cost  # ~1.5 GPU-h on one 80 GB GPU (Gemma is gated: accept its terms)
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO"
CONFIRM=""; [[ ${1:-} == --confirm-cost ]] && CONFIRM="--confirm-cost"
: "${GEODE_STORE:?export GEODE_STORE first}"
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
HL=experiments/em/hl_reads.py
run() { echo "[hl-run] $*"; "$@" || echo "[hl-run] FAILED: $*"; }
echo "[hl-run] repo $(git log --oneline -1)"
run python3 $HL fetch
for m in Qwen/Qwen3-8B google/gemma-3-4b-it; do
  run python3 $HL gen --model $m --device cuda $CONFIRM
  run python3 $HL read --model $m --device cuda $CONFIRM
  run python3 $HL read2 --model $m --device cuda $CONFIRM
done
run python3 $HL read3 --model Qwen/Qwen3-8B --device cuda $CONFIRM
run python3 $HL read3 --model google/gemma-3-4b-it --device cuda $CONFIRM
python3 $HL report
