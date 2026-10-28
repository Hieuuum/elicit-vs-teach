#!/usr/bin/env bash
# Every remaining EM / subliminal-learning step, in order. Each step resumes (finished outputs are
# skipped) and prints its own cost estimate; nothing runs on the GPU without --confirm-cost.
#
#   bash experiments/em/run_all.sh --confirm-cost              # all stages
#   bash experiments/em/run_all.sh --confirm-cost --stage C    # one stage (A..H)
#   bash experiments/em/run_all.sh                             # dry run: every step prints its estimate and is refused
#                                                              # (stage C's random set is CPU-only and is written)
#
# Stages (GPU-hours are upper bounds; early stopping makes the sweeps shorter):
#   A  EM: M21 on the six arXiv 2602.00298 sets the first scorer missed           ~0.3
#   B  EM: M19 EDL/D sweeps (+ M21 along each sweep): 6 sets + 2 rotated twins     <= 18
#   C  SL: number datasets: 2 teachers x {cat, penguin, phoenix, leopard, neutral} + random   ~3
#   D  SL: no-training scores (M21, trait gap) on 3 students (7B, 14B, 7B base)   ~1
#   E  SL: labels: 3 students x 11 number sets                                    ~10
#   F  SL: M19 sweeps (+ M21 along each sweep): 3 students x 5 number sets        <= 12
#   G  reports (CPU): analyze_edl_em.py, sl_report.py
#   H  Beyond Owls letter counting (arXiv 2610.10657): M21 of their 9 transfer sets on the
#      Qwen3.6-35B-A3B and Nemotron-3-Nano-30B-A3B students, vs their SL outcomes       ~1.5
set -euo pipefail

CONFIRM=""
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --confirm-cost) CONFIRM="--confirm-cost"; shift ;;
    --stage) ONLY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export GEODE_STORE="${GEODE_STORE:-/raid/NFS_SHARE/home/marcin.podhajski/elicit-vs-teach/geode-store}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LOG="$GEODE_STORE/em/run_all.log"
mkdir -p "$(dirname "$LOG")"
PY="python3"
DEV="--device cuda"
Q7=Qwen/Qwen2.5-7B-Instruct
Q14=Qwen/Qwen2.5-14B-Instruct
Q7BASE=Qwen/Qwen2.5-7B

FAILS=0
# one failing step (or the cost guard, in a dry run) is logged and the next step still runs
run() {
  echo "[run_all] $(date -Is) $*" | tee -a "$LOG"
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    echo "[run_all] FAILED or refused: $*" | tee -a "$LOG"
    FAILS=$((FAILS + 1))
  fi
}
want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

echo "[run_all] repo $(git rev-parse --short HEAD) store $GEODE_STORE confirm=${CONFIRM:-no} stage=${ONLY:-all}" | tee -a "$LOG"

if want A; then
  run $PY experiments/em/score_em.py $DEV $CONFIRM \
    --only gore_movie_trivia,incorrect_sexual_advice,toxic_legal_advice,incorrect_translation,incorrect_math,evil_math
fi

if want B; then
  for d in gore_movie_trivia bad_medical extreme_sports good_medical incorrect_math oai_health_incorrect; do
    run $PY experiments/em/edl_sweep_em.py --model $Q14 --dataset $d $DEV $CONFIRM
  done
  for d in bad_medical incorrect_math; do
    run $PY experiments/em/edl_sweep_em.py --model $Q14 --dataset $d --rotate $DEV $CONFIRM
  done
fi

if want C; then
  for t in $Q7 $Q14; do
    for a in cat penguin phoenix leopard ""; do
      run $PY experiments/em/sl_generate.py --teacher $t --animal "$a" $DEV $CONFIRM
    done
  done
  run $PY experiments/em/sl_generate.py --random
fi

SL_SETS="sl-q7b-cat sl-q7b-penguin sl-q7b-phoenix sl-q7b-leopard sl-q7b-neutral sl-q14b-cat sl-q14b-penguin sl-q14b-phoenix sl-q14b-leopard sl-q14b-neutral sl-random-neutral"
# students: same model as a teacher (7B, 14B instruct), shared init with the 7B teacher (7B base)
STUDENTS="$Q7 $Q14 $Q7BASE"

if want D; then
  for s in $STUDENTS; do
    run $PY experiments/em/sl_score.py --student $s $DEV $CONFIRM
  done
fi

if want E; then
  for s in $STUDENTS; do
    for d in $SL_SETS; do
      run $PY experiments/em/sl_train.py --student $s --dataset $d $DEV $CONFIRM
    done
  done
fi

if want F; then
  for s in $STUDENTS; do
    for d in sl-q7b-cat sl-q7b-neutral sl-q14b-cat sl-q14b-neutral sl-random-neutral; do
      run $PY experiments/em/edl_sweep_em.py --model $s --dataset $d --max-len 256 $DEV $CONFIRM
    done
  done
fi

if want G; then
  run $PY experiments/em/analyze_edl_em.py --model-dir Qwen2.5-14B-Instruct
  run $PY experiments/em/analyze_edl_em.py --model-dir Qwen2.5-7B-Instruct
  run $PY experiments/em/analyze_edl_em.py --model-dir Qwen2.5-7B
  run $PY experiments/em/sl_report.py
fi
if want H; then
  run $PY experiments/em/bo_lc.py fetch
  run $PY experiments/em/bo_lc.py score --student Qwen/Qwen3.6-35B-A3B $DEV $CONFIRM
  # batch 1: without the mamba-ssm kernels the torch fallback materialises large scan tensors
  run $PY experiments/em/bo_lc.py score --student nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --batch-size 1 $DEV $CONFIRM
  run $PY experiments/em/bo_lc.py report
fi
echo "[run_all] $(date -Is) done, $FAILS step(s) failed or refused; log $LOG" | tee -a "$LOG"
