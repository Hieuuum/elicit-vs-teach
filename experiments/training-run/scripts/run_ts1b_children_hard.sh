#!/usr/bin/env bash
# run_ts1b_children_hard.sh — continue the matched ts1b children on the
# hard-weighted set under two stop rules (owner 2026-11-11;
# notes/plan-ts1b-hard-continue.md). Box script, one box with 2 GPUs.
#
#   1. data    D_algo_bare_hard4m + the 20K test set must already be on the
#              box (scp from the laptop; both gitignored). train_sft.py and
#              score_em.py re-check their order_hash pins.
#   2. starts  pull the matched pair once:
#                elicit mhieuuu/evt-ts1b-elicit-ft-fresh-b256-c500
#                taught mhieuuu/evt-ts1b-teach-ft-fresh-b256
#   3. lanes   two in parallel, one GPU each, elicit then taught:
#                GPU 0 "loss": ts1b_{elicit,teach}_hard_loss.yaml
#                      (val-loss plateau 0.005 nats x 3; accuracy logged only)
#                GPU 1 "acc":  ts1b_{elicit,teach}_hard_acc.yaml
#                      (exact match >= 0.95 on 1,040 check prompts x 3)
#              Each run: train, score on the 20K test set (score_em.py ->
#              runs/<id>/score_em.json), push to $PUSH_NAMESPACE/<id> (private).
#              A failed run does not stop the next run in its lane or the
#              other lane. Per-lane logs: $LOG_DIR/lane_<name>.log.
#
# Usage: bash run_ts1b_children_hard.sh --confirm-cost [loss|acc|both]
#   With one lane name, only that lane runs (on GPU ${GPU:-0}).
#   Ends with EXIT=0 (all runs pushed) or EXIT=1.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts1b-hard
echo "[$TAG] estimated cost: one pass of 4M rows = 15,625 steps ~ 1.25 h per run on an L40S;"
echo "[$TAG] likely 1-3 passes per run, 2 runs per lane -> ~4-10 h of a 2-GPU box at ~\$1.2-1.6/h"
echo "[$TAG] ~ \$5-16. Worst case (every run hits the 10-pass cap) ~26 h ~ \$40."
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "run_ts1b_children_hard.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
WHICH=both
for a in "$@"; do [[ $a == loss || $a == acc ]] && WHICH=$a; done

REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
PUSH_NAMESPACE=${PUSH_NAMESPACE:-mhieuuu}
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
LOG_DIR=${LOG_DIR:-/workspace/hard_logs}
mkdir -p "$LOG_DIR"

ELICIT_START=evt-ts1b-elicit-ft-fresh-b256-c500
TEACH_START=evt-ts1b-teach-ft-fresh-b256

# ---- 1. data ----------------------------------------------------------------
for f in D_algo_bare_hard4m.parquet D_algo_eval_bare_trainmix_n20000.parquet; do
  [[ -f $DATA_DIR/$f ]] || fail "missing $DATA_DIR/$f (scp it from the laptop)"
done
milestone "data_ready"

# ---- 2. start models (once, before the lanes share them) --------------------
pull_start() { # run_id repo
  if [[ ! -f $GEODE_STORE/runs/$1/model/model.safetensors ]]; then
    python3 hf_checkpoint.py pull --run-id "$1" --repo-id "$2" || fail "pull $1"
  fi
  [[ $(status_of "$1") == complete ]] || fail "$1 manifest is not status=complete"
}
pull_start "$ELICIT_START" "$PUSH_NAMESPACE/$ELICIT_START"
pull_start "$TEACH_START" "$PUSH_NAMESPACE/$TEACH_START"
milestone "starts_ready"

# ---- 3. one run (in a subshell: fail exits only this run) -------------------
run_one() { # config run_id start
  local cfg=../configs/$1 rid=$2 start=$3 status
  [[ $(awk '/^run_id:/ {print $2; exit}' "$cfg") == "$rid" ]] || fail "$cfg run_id is not $rid"
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    milestone "train_skip run=$rid status=complete"
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  else
    milestone "train_start run=$rid gpu=$CUDA_VISIBLE_DEVICES"
    python3 train_sft.py --config "$cfg" \
      --init-from "$GEODE_STORE/runs/$start/model" --confirm-cost || fail "$rid training"
  fi
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_done run=$rid stop_reason=$(stop_reason_of "$rid" sft_result)"
  python3 score_em.py --run "$rid" --config ../configs/eval_trainmix_test_ts1b.yaml \
    --out "$GEODE_STORE/runs/$rid/score_em.json" || fail "$rid score_em"
  milestone "score_done run=$rid"
  python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$PUSH_NAMESPACE/$rid" ||
    fail "$rid push"
  milestone "push_done repo=$PUSH_NAMESPACE/$rid"
}

# One lane = one GPU: elicit, then taught. Returns the number of failed runs.
lane() { # name gpu
  local name=$1 bad=0
  export CUDA_VISIBLE_DEVICES=$2
  (run_one "ts1b_elicit_hard_$name.yaml" "evt-ts1b-elicit-hard-$name" "$ELICIT_START") || bad=$((bad + 1))
  (run_one "ts1b_teach_hard_$name.yaml" "evt-ts1b-teach-hard-$name" "$TEACH_START") || bad=$((bad + 1))
  milestone "lane_done lane=$name failed_runs=$bad"
  return $bad
}

pids=() names=()
if [[ $WHICH == both ]]; then
  lane loss 0 >"$LOG_DIR/lane_loss.log" 2>&1 &
  pids+=($!) names+=(loss)
  lane acc 1 >"$LOG_DIR/lane_acc.log" 2>&1 &
  pids+=($!) names+=(acc)
else
  lane "$WHICH" "${GPU:-0}" >"$LOG_DIR/lane_$WHICH.log" 2>&1 &
  pids+=($!) names+=("$WHICH")
fi
milestone "lanes_started ${names[*]} (logs in $LOG_DIR)"

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    milestone "lane ${names[$i]} ok"
  else
    milestone "lane ${names[$i]} had failures (see $LOG_DIR/lane_${names[$i]}.log)"
    rc=1
  fi
done
grep -h "MILESTONE \(train_done\|push_done\|lane_done\)\|FAILED" "$LOG_DIR"/lane_*.log
echo "EXIT=$rc"
