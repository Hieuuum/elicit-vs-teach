#!/usr/bin/env bash
# The pairing-accuracy rerun (V1.14; decisions.md 2026-11-05): the paper's applications re-read with the
# score chosen on the toy task.  Forward passes only, no weight updates, no training rerun.  Order = most
# exposed first: SL numbers (the cross-model teach-side cases), letters, the EM persona read, unlearning.
# Every step skips what is already read, so the script resumes after an interruption.
#
#   bash experiments/training-run/scripts/launch_pg_rerun.sh                 # prints the plan and the estimate
#   bash experiments/training-run/scripts/launch_pg_rerun.sh --confirm-cost  # runs it (~3 GPU-h, one 80 GB GPU)
#   [--stages "sl letters persona unlearn"]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
CONFIRM=""; STAGES="sl letters persona unlearn"
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM="--confirm-cost" ;;
    --stages) STAGES=$2; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done
: "${GEODE_STORE:?export GEODE_STORE first}"
# conda's libstdc++ before the system one (optree needs CXXABI_1.3.15; as launch_unlearn.sh)
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
DEV="--device cuda"
EM=experiments/em/pg_pairs_em.py
UL=experiments/unlearning/pg_pairs.py
SL_STUDENTS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct"
PERSONA_MODELS="Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B allenai/OLMo-2-1124-7B-Instruct"
WMDP_TAGS="orig rmu elm graddiff rmulat npo simnpo"
DEEPIG_TAGS="orig filtered cb cb_lat filtered_cb_lat"
echo "[rerun] repo $(git log --oneline -1); stages [$STAGES]; estimate ~1.5 GPU-h (SL ~0.5, letters ~0.7, persona ~0.1, unlearning ~0.2)"
if [[ -z $CONFIRM ]]; then
  echo "[rerun] dry run: each step prints its estimate and refuses; pass --confirm-cost to launch (budget rule)."
fi
want() { [[ " $STAGES " == *" $1 "* ]]; }
run() { echo "[rerun] $*"; "$@" || echo "[rerun] FAILED: $*"; }

if want sl; then
  for s in $SL_STUDENTS; do run python3 $EM sl --student "$s" $DEV $CONFIRM; done
fi
if want letters; then
  run python3 $EM letters --student Qwen/Qwen3.6-35B-A3B $DEV $CONFIRM
  # Nemotron runs at batch 1 (its remote code): only the sets with a published outcome (teacher, control),
  # 256 rows and 2 ring neighbours keep it near 0.5 GPU-h instead of 3.3
  run python3 $EM letters --student nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --batch-size 1 \
    --kinds teacher control --n 256 --offsets 2 $DEV $CONFIRM
fi
if want persona; then
  for m in $PERSONA_MODELS; do run python3 $EM persona --model "$m" $DEV $CONFIRM; done
fi
if want unlearn; then
  for t in $WMDP_TAGS; do run python3 $UL read --set wmdp --tag $t $DEV $CONFIRM; done
  for t in $DEEPIG_TAGS; do run python3 $UL read --set deepig --tag $t $DEV $CONFIRM; done
  run python3 $UL read --set anchor --tag ts1b --init "$GEODE_STORE/runs/evt-ts1b-base/model" \
    --tokenizer meta-llama/Llama-3.2-1B $DEV $CONFIRM
fi
python3 $EM report
python3 $EM report --variant all
python3 $UL report
python3 $UL report --variant all
