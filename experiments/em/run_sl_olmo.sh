#!/usr/bin/env bash
# SL numbers with an other-family student: OLMo-2-7B trained on the Qwen teachers' number sets (the rows of the
# paper's SL table marked "not trained").  Ten LoRA runs (4 animals + neutral, two teachers), ~1.5 GPU-h.
#
# Pre-registered (2026-11-12, before any training; TG reads of 2026-11-07 fixed):
#   T1  the six sets that read teach-side on TG (all four Qwen2.5-14B sets; Qwen2.5-7B cat, penguin) show no SL:
#       |effect| < 0.05 each (effect = target share after the animal set minus after the neutral set)
#   T2  the two weakly held sets (Qwen2.5-7B leopard s 0.19, phoenix 0.18): no prediction from the call; expected
#       none, as for the 14B student on 7B data (s 0.24-0.31, no SL)
#
#   bash experiments/em/run_sl_olmo.sh                 # plan
#   bash experiments/em/run_sl_olmo.sh --confirm-cost
set -uo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
CONFIRM=${1:-}
: "${GEODE_STORE:?export GEODE_STORE first}"
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
S=allenai/OLMo-2-1124-7B-Instruct
[[ $CONFIRM == --confirm-cost ]] || echo "[sl-olmo] dry run: each step prints its estimate and refuses"
for t in q7b q14b; do
  for a in neutral cat leopard penguin phoenix; do
    echo "[sl-olmo] $S <- sl-$t-$a"
    python3 experiments/em/sl_train.py --student $S --dataset sl-$t-$a --device cuda $CONFIRM || echo "[sl-olmo] FAILED sl-$t-$a"
  done
done
python3 experiments/em/sl_report.py
