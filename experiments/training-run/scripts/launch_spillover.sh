#!/usr/bin/env bash
# Train on A, B appears (analysis/spillover.py): 9 parents between the blank TinyStories-1B and its
# op-installed twin; per parent, M21 on natural-language arithmetic (B) before the answer-free bridge (A),
# then the bridge, then B again. ~0.25 GPU-h per parent, ~2.3 GPU-h total. Each parent resumes.
#
#   bash experiments/training-run/scripts/launch_spillover.sh --confirm-cost
set -uo pipefail
CONFIRM=""
[[ " $* " == *" --confirm-cost "* ]] && CONFIRM="--confirm-cost"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
export GEODE_STORE="${GEODE_STORE:-/raid/NFS_SHARE/home/marcin.podhajski/elicit-vs-teach/geode-store}"
for a in 0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
  python3 experiments/training-run/analysis/spillover.py run --alpha $a --device cuda $CONFIRM || echo "[spill] alpha $a failed or refused"
done
python3 experiments/training-run/analysis/spillover.py report
