#!/usr/bin/env bash
# Addition circuits at small sizes (0.1-5%) on a 4x RTX 3090 box (HANDOFF-small-sizes.md,
# Steps 3, 3b, 4). One EAP-IG job per GPU (two on one 3090 OOM), each in its own tmux
# session with CUDA_VISIBLE_DEVICES; a CPU session runs the overlap/null analysis.
#
#   main (this invocation): venv, saved KL scores from HF, single-threaded model prefetch,
#         launches the sessions, waits for them, pushes + verifies, writes the DONE marker.
#   gpu0: box check (old dense vs new patch path at 0.1% and 5%; SPEEDUP_CHECK=0 skips),
#         then elicit_parent --task symbol evaluate, sizes 0.001 0.002 0.005 0.01 0.02 0.05.
#         The check decides ONE patch path for every GPU job (.state/patch_path): new if it
#         matches dense (< 1e-4 nats) and is not slower, else dense (GEODE_EAPIG_DENSE=1).
#         gpu1-3 wait for that decision before any evaluate, so no result uses an unchecked path.
#   gpu1: elicit_child word evaluate (NO --parent-tag: it would load the WORD-form parent),
#         sizes 0.001 0.002 0.005 0.01
#   gpu2: teach_child word evaluate --parent-tag fmt_parent, same sizes
#   gpu3: fmt_parent word evaluate, then functional_reuse.py for both children
#   cpu : edge_node.py overlap + 1000-draw nulls + split-half ceilings, all step-3b pairs
#
# Usage (on the box, inside tmux, repo at the pushed commit):
#   bash run_box_small.sh --confirm-cost          # DRY_RUN=1 prints the plan only
# Env: MODEL_<tag> (HF repo id <ns>/<run_id> or local dir), MODELS_DIR=/workspace/models,
#      BATCH_SIZE=32, RESULTS_REPO=mhieuuu/geode-internals, NO_PUSH=1, SPEEDUP_CHECK=1,
#      SCORES_HF_PREFIX=results/eapig_kl_symbol (HF layout <prefix>/{klword,klsym}/<tag>/),
#      PUSH_SUB=results/eapig_kl_tests_small, VENV=/workspace/venv,
#      EDGE_WORKERS=(nproc - 6, clamped to 1..8; 4 GPU jobs + main use ~1 core each).
# Progress/EXIT markers: tail -f results_kltests_small/run_box_small.log
# Skip-if-done per stage: a stage whose output file exists is not rerun.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=$(git rev-parse --show-toplevel)
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

CONFIRM=0; GPU_JOB=""; DRY=${DRY_RUN:-0}
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --job) GPU_JOB=$2; shift ;;   # internal: gpu0|gpu1|gpu2|gpu3|cpu
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
echo "[small] estimated ~1.5 h wall on a 4x RTX 3090 box (gpu0 ~65 min, gpu1/gpu2 ~85 min," \
     "gpu3 ~35 min, dense path; ~4.5 GPU-h total, under ~\$1.5 incl. race/idle); box rental is the only cost."
((CONFIRM || DRY)) || { echo "[small] refusing to run without --confirm-cost" >&2; exit 2; }

VENV=${VENV:-/workspace/venv}
# shellcheck disable=SC1091
[[ -f $VENV/bin/activate ]] && source "$VENV/bin/activate"

declare -A MODEL=(
  [elicit_parent]=${MODEL_elicit_parent:-podhajskimarcin/evt-ts1b-op-bridge-mix}
  [elicit_child]=${MODEL_elicit_child:-podhajskimarcin/evt-ts1b-elicit-ft-n4000000}
  [fmt_parent]=${MODEL_fmt_parent:-podhajskimarcin/evt-ts1b-fig2ts-installer}
  [teach_child]=${MODEL_teach_child:-podhajskimarcin/evt-ts1b-teach-ft-fmt-n4000000}
)
PREFETCH_ORDER=(elicit_child elicit_parent teach_child fmt_parent)
MODELS_DIR=${MODELS_DIR:-/workspace/models}
BATCH_SIZE=${BATCH_SIZE:-32}
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
SCORES_HF_PREFIX=${SCORES_HF_PREFIX:-results/eapig_kl_symbol}
PUSH_SUB=${PUSH_SUB:-results/eapig_kl_tests_small}
SPEEDUP_CHECK=${SPEEDUP_CHECK:-1}
EDGE_WORKERS=${EDGE_WORKERS:-$(( n = $(nproc) - 6, n < 1 ? 1 : (n > 8 ? 8 : n) ))}
SIZES_CHILD="0.001 0.002 0.005 0.01"
SIZES_PARENT="0.001 0.002 0.005 0.01 0.02 0.05"
FRACS_EDGE="0.001 0.002 0.005 0.01 0.02 0.05"
EDGE_PAIRS=(ep_sym:ec_word ep_sym:tc_word fp_word:tc_word ep_word:ec_word ep_sym:ec_sym
            ec_sym:ec_word ec_word:tc_word)

KLWORD=$HERE/results_klword; KLSYM=$HERE/results_klsym
TESTS=$HERE/results_kltests_small
FUNC=$HERE/results_functional/kl_small
EDGE=$HERE/results_edge/small
CHECK=$TESTS/box_check
STATE=$TESTS/.state
PATH_FILE=$STATE/patch_path   # "new" or "dense", written by gpu0's box check
LOG=$TESTS/run_box_small.log
SESSION_PREFIX=eapigsmall

say() { local m; m="[small $(date +%H:%M:%S)] $*"
        if ((DRY)); then echo "$m"; else mkdir -p "$TESTS"; echo "$m" | tee -a "$LOG"; fi; }
model_path() {
  if [[ -d ${MODEL[$1]} ]]; then echo "${MODEL[$1]}"
  else echo "$MODELS_DIR/$1/runs/$(basename "${MODEL[$1]}")/model"; fi
}
wait_model() {
  local tag=$1 t=0; [[ -d ${MODEL[$tag]} ]] && return 0
  ((DRY)) && return 0
  while [[ ! -f $MODELS_DIR/$tag.done ]]; do
    [[ -f $MODELS_DIR/$tag.failed ]] && { say "prefetch of $tag failed"; return 1; }
    ((t >= 3600)) && { say "TIMEOUT waiting for model $tag"; return 1; }
    sleep 10; t=$((t + 10))
  done
}
# wait_path: block until gpu0's box check has picked the patch path, then export it
wait_path() {
  ((DRY)) && return 0
  local t=0
  while [[ ! -f $PATH_FILE ]]; do
    [[ -f $STATE/gpu0.failed ]] && { say "gpu0 failed before the patch-path decision"; return 1; }
    ((t >= 5400)) && { say "TIMEOUT waiting for the box check"; return 1; }
    sleep 10; t=$((t + 10))
  done
  if [[ $(cat "$PATH_FILE") == dense ]]; then export GEODE_EAPIG_DENSE=1; else unset GEODE_EAPIG_DENSE; fi
  say "$GPU_JOB uses the $(cat "$PATH_FILE") patch path"
}
# step <label> <done-file> <cmd...>: skip if done-file exists, else run (tee'd to the log)
step() {
  local label=$1 donef=$2; shift 2
  if [[ -f $donef ]]; then say "skip $label ($(basename "$donef") exists)"; return 0; fi
  if ((DRY)); then say "$label: $*"; return 0; fi
  say "run $label"; local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { say "FAILED $label"; return 1; }
  say "done $label in $((SECONDS - t0)) s"
}
# eval_run <gpu> <tag> <task> <scores_dir> <sizes> [extra run.py args]: the evaluate stage
eval_run() {
  local gpu=$1 tag=$2 task=$3 sdir=$4 sizes=$5; shift 5
  # shellcheck disable=SC2086
  step "gpu$gpu evaluate $tag/$task" "$TESTS/$tag/evaluate.json" \
    env CUDA_VISIBLE_DEVICES="$gpu" python3 run.py --tag "$tag" --model "$(model_path "$tag")" \
      --task "$task" --metric kl --stage evaluate --sizes $sizes --scores-dir "$sdir" \
      --results "$TESTS" --batch-size "$BATCH_SIZE" --device cuda "$@"
}

# ------------------------------------------------------------------ per-GPU jobs
job_gpu0() {
  wait_model elicit_child || return 1
  if [[ $SPEEDUP_CHECK == 1 ]]; then
    local chk="python3 box_check_speed.py --run --model $(model_path elicit_child) --scores $KLWORD/elicit_child --batch-size $BATCH_SIZE"
    local decision=dense rc
    ((DRY)) || mkdir -p "$CHECK"
    # shellcheck disable=SC2086
    if step "gpu0 box-check new path" "$CHECK/new.json" env -u GEODE_EAPIG_DENSE CUDA_VISIBLE_DEVICES=0 $chk --out "$CHECK/new.json" \
       && step "gpu0 box-check dense path" "$CHECK/dense.json" env CUDA_VISIBLE_DEVICES=0 GEODE_EAPIG_DENSE=1 $chk --out "$CHECK/dense.json"; then
      if ((DRY)); then decision=new; else
        python3 box_check_speed.py --compare "$CHECK/new.json" "$CHECK/dense.json" 2>&1 | tee -a "$LOG" "$CHECK/compare.txt"
        rc=${PIPESTATUS[0]}
        case $rc in
          0) decision=new ;;
          1) say "BOX CHECK: new path differs from dense >= 1e-4 nats -> ALL jobs use the dense path" ;;
          2) say "BOX CHECK: new path not faster -> ALL jobs use the dense path" ;;
          *) say "BOX CHECK: compare error rc=$rc -> ALL jobs use the dense path" ;;
        esac
      fi
    else
      say "BOX CHECK: a check run crashed -> ALL jobs use the dense path"
    fi
    if ((DRY)); then say "patch path -> $decision"; else echo "$decision" > "$PATH_FILE"; fi
  fi
  wait_path || return 1
  wait_model elicit_parent || return 1
  eval_run 0 elicit_parent symbol "$KLSYM" "$SIZES_PARENT"
}
job_gpu1() { wait_path || return 1; wait_model elicit_child || return 1; eval_run 1 elicit_child word "$KLWORD" "$SIZES_CHILD"; }
job_gpu2() { wait_path || return 1; wait_model teach_child || return 1; eval_run 2 teach_child word "$KLWORD" "$SIZES_CHILD" --parent-tag fmt_parent; }
job_gpu3() {
  wait_path || return 1
  wait_model fmt_parent || return 1; wait_model elicit_child || return 1; wait_model teach_child || return 1
  eval_run 3 fmt_parent word "$KLWORD" "$SIZES_CHILD" || return 1
  ((DRY)) || mkdir -p "$FUNC"
  local ch
  for ch in elicit_child teach_child; do
    # shellcheck disable=SC2086
    step "gpu3 functional_reuse $ch" "$FUNC/$ch.json" \
      env CUDA_VISIBLE_DEVICES=3 python3 functional_reuse.py --child "$ch" --metric kl \
        --model "$(model_path "$ch")" --own "$KLWORD/$ch" \
        --circuit ep_sym="$KLSYM/elicit_parent" ep_word="$KLWORD/elicit_parent" fp_word="$KLWORD/fmt_parent" \
        --sizes $SIZES_CHILD --batch-size "$BATCH_SIZE" --device cuda --out "$FUNC/$ch.json" || return 1
  done
}
job_cpu() {
  ((DRY)) || mkdir -p "$EDGE"
  # shellcheck disable=SC2086
  step "cpu edge_node overlap+nulls" "$EDGE/edge_node.json" \
    nice -n 10 python3 edge_node.py \
      --scores ep_sym="$KLSYM/elicit_parent" ep_word="$KLWORD/elicit_parent" \
               ec_sym="$KLSYM/elicit_child" ec_word="$KLWORD/elicit_child" \
               fp_word="$KLWORD/fmt_parent" tc_word="$KLWORD/teach_child" \
      --pairs "${EDGE_PAIRS[@]}" --fracs $FRACS_EDGE --n-null 1000 --workers "$EDGE_WORKERS" \
      --out "$EDGE"
}

# a child session: run the job, record done/failed + EXIT marker for the main waiter
if [[ -n $GPU_JOB ]]; then
  mkdir -p "$STATE"
  if "job_$GPU_JOB"; then touch "$STATE/$GPU_JOB.done"; say "EXIT_$GPU_JOB=0"
  else touch "$STATE/$GPU_JOB.failed"; say "EXIT_$GPU_JOB=1"; exit 1; fi
  exit 0
fi

# ------------------------------------------------------------------ inputs
fetch_scores() {  # saved KL score sets (scores.pt + sanity.json) -> results_klword/, results_klsym/
  python3 - "$RESULTS_REPO" "$SCORES_HF_PREFIX" "$KLWORD" "$KLSYM" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, prefix, klword, klsym = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
# ASSUMED layout <prefix>/{klword,klsym}/<tag>/{scores.pt,sanity.json}
want = [("klword", klword, t) for t in ("elicit_parent", "elicit_child", "fmt_parent", "teach_child")]
want += [("klsym", klsym, t) for t in ("elicit_parent", "elicit_child")]
for sub, root, tag in want:
    for f in ("scores.pt", "sanity.json"):
        dst = root / tag / f
        if not dst.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(hf_hub_download(repo, f"{prefix}/{sub}/{tag}/{f}"), dst)
        print("have", dst)
PY
}
prefetch() {  # single-threaded; one .done/.failed marker per model
  local tag repo id
  for tag in "${PREFETCH_ORDER[@]}"; do
    [[ -d ${MODEL[$tag]} || -f $MODELS_DIR/$tag.done ]] && continue
    repo=${MODEL[$tag]}; id=$(basename "$repo")
    say "prefetch $tag ($repo)"
    if HF_HUB_DISABLE_XET=1 hf download "$repo" --include "runs/$id/model/*" \
         --local-dir "$MODELS_DIR/$tag" >>"$LOG" 2>&1 && [[ -f $(model_path "$tag")/config.json ]]; then
      touch "$MODELS_DIR/$tag.done"; say "prefetched $tag"
    else
      touch "$MODELS_DIR/$tag.failed"; say "PREFETCH FAILED $tag"
    fi
  done
}
# push_dir <local dir> <path in repo>: single-threaded upload_folder + list-files verify
push_dir() {
  [[ -n ${NO_PUSH:-} ]] && return 0
  [[ -d $1 ]] || { say "push: $1 missing"; return 1; }
  say "push $1 -> $RESULTS_REPO:$2"
  HF_HUB_DISABLE_XET=1 python3 - "$1" "$RESULTS_REPO" "$2" <<'PY' 2>&1 | tee -a "$LOG"
import sys
from pathlib import Path
from huggingface_hub import HfApi
res, repo, sub = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message="eapig small-size circuit tests",
                  ignore_patterns=[".state/*", "*.log", "*.log.prev"])
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + str(p.relative_to(res)) for p in res.rglob("*")
         if p.is_file() and ".state" not in p.parts and not p.name.endswith((".log", ".prev"))}
missing = local - remote
print("verify: missing on hub:", sorted(missing) or "none")
sys.exit(1 if missing else 0)
PY
  [[ ${PIPESTATUS[0]} == 0 ]]
}

# ------------------------------------------------------------------ main
if ((DRY)); then
  say "DRY RUN: scores <- $RESULTS_REPO:$SCORES_HF_PREFIX/{klword,klsym}/<tag>/{scores.pt,sanity.json}"
  for t in "${PREFETCH_ORDER[@]}"; do
    say "prefetch $t: HF_HUB_DISABLE_XET=1 hf download ${MODEL[$t]} --include 'runs/$(basename "${MODEL[$t]}")/model/*' --local-dir $MODELS_DIR/$t"
  done
  for j in gpu0 gpu1 gpu2 gpu3 cpu; do say "--- job $j (tmux ${SESSION_PREFIX}_$j)"; "job_$j"; done
  say "then: wait for markers, push $TESTS -> $PUSH_SUB, $FUNC -> $PUSH_SUB/functional, $EDGE -> $PUSH_SUB/overlap, verify, ALL DONE"
  exit 0
fi

mkdir -p "$TESTS" "${STATE:?}"
rm -f "$PATH_FILE" "${STATE:?}"/gpu0.* "${STATE:?}"/gpu1.* "${STATE:?}"/gpu2.* "${STATE:?}"/gpu3.* "${STATE:?}"/cpu.*
[[ -f $LOG ]] && mv "$LOG" "$LOG.prev"   # EXIT markers must come from this run only
mkdir -p "$MODELS_DIR"
say "repo $(git rev-parse --short HEAD) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',')"
n_gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
((n_gpu >= 4)) || { say "FATAL: need 4 GPUs, found $n_gpu"; exit 1; }
for f in data/pairs_seed0.pt data/pairs_symbol_seed0.pt data/copy_pairs_seed1.pt data/tinystories_seed2.pt; do
  [[ -f $f ]] || { say "FATAL: $f missing"; exit 1; }
done
command -v tmux >/dev/null || { say "FATAL: tmux not installed"; exit 1; }
say "fetch saved KL scores"; fetch_scores 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || { say "FATAL: score fetch failed (check SCORES_HF_PREFIX layout)"; exit 1; }
prefetch & PF=$!
[[ $SPEEDUP_CHECK == 1 ]] || { echo new > "$PATH_FILE"; say "SPEEDUP_CHECK=0: new patch path, unchecked"; }

for j in gpu0 gpu1 gpu2 gpu3 cpu; do
  s=${SESSION_PREFIX}_$j
  tmux kill-session -t "$s" 2>/dev/null || true
  # non-interactive tmux gets no venv: activate it explicitly inside the session
  tmux new -d -s "$s" "source $VENV/bin/activate 2>/dev/null; cd $HERE && bash run_box_small.sh --job $j --confirm-cost; exec bash"
  say "launched tmux $s"
done
( sleep 300; say "nvidia-smi +5min:"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | tee -a "$LOG" ) & SMI=$!

while :; do
  n=0; for j in gpu0 gpu1 gpu2 gpu3 cpu; do [[ -f $STATE/$j.done || -f $STATE/$j.failed ]] && n=$((n + 1)); done
  ((n == 5)) && break
  sleep 30
done
kill "$SMI" 2>/dev/null || true
wait "$PF" || true
bad=0
for j in gpu0 gpu1 gpu2 gpu3 cpu; do [[ -f $STATE/$j.failed ]] && { say "job $j FAILED"; bad=1; }; done

cp "$LOG" "$TESTS/run_log.txt"   # the .log itself is excluded from the push (still being written)
push_dir "$TESTS" "$PUSH_SUB" || bad=1
push_dir "$FUNC" "$PUSH_SUB/functional" || bad=1
push_dir "$EDGE" "$PUSH_SUB/overlap" || bad=1
((bad)) && { say "NOT DONE (job failure or push verify failed)"; echo "EXIT=1" >> "$LOG"; exit 1; }
say "ALL DONE"; echo "EXIT=0" >> "$LOG"
