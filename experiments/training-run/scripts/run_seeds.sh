#!/usr/bin/env bash
# Training seeds for the applications (2026-11-13; pre-registration in decisions.md, S1-S4). Seed 0 / 316 are the
# runs already in the papers; this adds two more training seeds everywhere a training run decides an outcome.
#   sl      SL numbers: six student/teacher pairs x (four animals + neutral) x seeds 1, 2     (60 runs, ~30 GPU-h)
#   dogs    six DPO students x (LLS, random) x seeds 1, 2                                     (24 runs, see lls.py estimates)
#   unlearn the circuit-breaker opened read: Deep Ignorance x 5 + the anchor, true and rotated, seeds 317, 318
#           (18 short LoRA runs + reads, ~1.5 GPU-h; opened weights deleted after each read)
# Resumable: every run skips what it has.
#   bash experiments/training-run/scripts/run_seeds.sh                                  # plan + estimates
#   bash experiments/training-run/scripts/run_seeds.sh --confirm-cost [--stages "unlearn sl dogs"]
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO"
CONFIRM=""; STAGES="unlearn sl dogs"
while [[ $# -gt 0 ]]; do
  case $1 in --confirm-cost) CONFIRM="--confirm-cost" ;; --stages) STAGES=$2; shift ;; *) echo "unknown arg $1"; exit 2 ;; esac; shift
done
: "${GEODE_STORE:?export GEODE_STORE first}"
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
want() { [[ " $STAGES " == *" $1 "* ]]; }
run() { echo "[seeds] $*"; "$@" || echo "[seeds] FAILED: $*"; }
echo "[seeds] repo $(git log --oneline -1); stages [$STAGES]; estimate ~32 GPU-h + dogs (each step prints its own)"
[[ -z $CONFIRM ]] && echo "[seeds] dry run: each step prints its estimate and refuses; pass --confirm-cost (budget rule)."
DEV="--device cuda"

if want unlearn; then
  UL=experiments/unlearning/pg_pairs.py; RL=experiments/unlearning/relearn.py; WD=$GEODE_STORE/unlearning/wmdp; N=128
  ANCHOR=$GEODE_STORE/runs/evt-ts1b-base/model; ATOK="--tokenizer meta-llama/Llama-3.2-1B"
  opened() {   # <set> <tag> <init> <A|Ashuf> <open|shuf> <seed> [tokenizer args]
    local set=$1 tag=$2 init=$3 kind=$4 name=$5 seed=$6; shift 6
    local rid=pgopen-$set-$tag-$kind-n$N-s$seed out=$WD/pg_pairs/control_overlap_${set}_${tag}_$name${N}s$seed.json
    [[ -f $out ]] && { echo "[seeds] $(basename "$out") exists"; return; }
    if [[ -z $CONFIRM ]]; then echo "[seeds] would train $rid (LoRA, $N facts, ~0.1 GPU-h) and read it"; return; fi
    [[ -f $GEODE_STORE/runs/$rid/model/config.json ]] || run python3 $RL --config experiments/unlearning/configs/relearn_wmdp_bio$kind.yaml \
      --init "$init" --run-id $rid --data-dir $WD/data --device cuda --confirm-cost --sweep --save-model \
      --n-train $N --seed $seed --test-split bio_B "$@"
    run python3 $UL control --match overlap --set $set --tag ${tag}_$name${N}s$seed --init "$GEODE_STORE/runs/$rid/model" "$@" $DEV $CONFIRM
    [[ -f $out ]] && rm -rf "$GEODE_STORE/runs/$rid/model"   # relearned WMDP weights are never kept or pushed
  }
  for s in 317 318; do
    for t in orig filtered cb cb_lat filtered_cb_lat; do opened deepig $t $WD/models-deepig/$t A open $s; done
    for t in cb cb_lat filtered_cb_lat; do opened deepig $t $WD/models-deepig/$t Ashuf shuf $s; done
    opened anchor ts1b "$ANCHOR" A open $s $ATOK
  done
  python3 $UL report --control --match overlap
fi

if want sl; then   # students x the teachers' sets; the data are fixed, the training seed varies
  for s in 1 2; do
    for st in Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct allenai/OLMo-2-1124-7B-Instruct; do
      for t in q7b q14b; do
        for a in neutral cat leopard penguin phoenix; do
          run python3 experiments/em/sl_train.py --student $st --dataset sl-$t-$a --seed $s $DEV $CONFIRM
        done
      done
    done
  done
  python3 experiments/em/sl_report.py
fi

if want dogs; then
  for s in 1 2; do
    for st in allenai/OLMo-2-0425-1B-Instruct allenai/OLMo-2-1124-7B-Instruct allenai/Olmo-3-7B-Instruct Qwen/Qwen3-8B \
              Qwen/Qwen2.5-7B-Instruct EssentialAI/rnj-1-instruct; do
      for set in lls random; do run python3 experiments/em/lls.py train --student $st --set $set --seed $s $DEV $CONFIRM; done
    done
  done
  python3 experiments/em/lls.py report
fi
