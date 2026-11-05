#!/usr/bin/env bash
# 10-fold robustness check (HANDOFF-folds.md, Step 4) on an N-GPU box (2-4x RTX 3090).
#
#   repro gate  fold_runner.py on the ORIGINAL data/ for elicit_child must reproduce the saved
#               results_klword/elicit_child/scores.pt mean (max abs diff < 1e-4). Fail -> stop.
#   phase A     GPU queue: sanity + KL scores for 6 score sets x 10 folds (one fold_runner job per
#               set, model loaded once), hidden preference for 3 parents on the folds and on the
#               old split.
#   gate        per-fold sanity EM: children vs the old split (ec 0.953, tc 0.145), word-task
#               parents + base at ~0. Hard fail (> 0.10 off, or a word parent > 0.02) stops before
#               phase B; 0.05-0.10 off is logged as WARN (binomial SD on 256 problems is ~0.02-0.03).
#   phase B     GPU queue: per fold, cmap.py elicit/teach and functional_reuse.py for both children
#               at 0.2 / 1 / 5%. CPU lane meanwhile: fold_overlap.py (nine sizes).
#   report      fold_report.py, then push results_folds/ (no per_example.pt) to HF and verify.
#
#   bash run_box_folds.sh --confirm-cost      # full run; DRY_RUN=1 prints the plan only
# Re-run: a job with a .state/jobs/<job>.done marker AND its outputs is skipped.
# Env: MODEL_<tag>, MODELS_DIR=/workspace/models, VENV=/workspace/venv, BATCH_SIZE=32,
#      FOLDS="0 1 ... 9", NO_PUSH=1, RESULTS_REPO=mhieuuu/geode-internals,
#      PUSH_SUB=results/eapig_folds, SCORES_HF_PREFIX=results/eapig_kl_symbol, JOB_RETRIES=2.
#      Test hooks: FAKE_GPUS="24576 24576", JOB_PY, SKIP_PREFETCH=1, WAIT_POLL, MAIN_POLL.
# Progress: tail -f results_folds/run_box_folds.log  (last line EXIT=0|1)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || dirname "$(dirname "$HERE")")}
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

CONFIRM=0; WORKER=""; WGPU=""; WQUEUE=""; DRY=${DRY_RUN:-0}
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --worker) WORKER=$2; WGPU=$3; WQUEUE=$4; shift 3 ;;   # internal: slot, gpu index, queue file
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $WORKER ]] || echo "[folds] estimated ~1.2-2 h wall on 2-4x RTX 3090 (~\$0.4-0.8/h):" \
  "setup + downloads ~20 min; repro gate ~3 min; phase A ~70 GPU-min (6 score sets x 10 folds);" \
  "phase B ~170 GPU-min (20 cmap jobs ~5-8 min, 20 functional ~1.5 min); push ~10 min." \
  "~4 GPU-h, ~\$1-2 total; plan cap \$6. Box rental is the only cost."
((CONFIRM || DRY)) || { echo "[folds] refusing to run without --confirm-cost" >&2; exit 2; }

RES=$HERE/results_folds
STATE=$RES/.state
# shellcheck disable=SC1091
[[ -n $WORKER && -f $STATE/env.sh ]] && source "$STATE/env.sh"
VENV=${VENV:-/workspace/venv}
# shellcheck disable=SC1091
[[ -f $VENV/bin/activate ]] && source "$VENV/bin/activate"

declare -A MODEL=(
  [base]=${MODEL_base:-podhajskimarcin/evt-ts1b-base}
  [elicit_parent]=${MODEL_elicit_parent:-podhajskimarcin/evt-ts1b-op-bridge-mix}
  [elicit_child]=${MODEL_elicit_child:-podhajskimarcin/evt-ts1b-elicit-ft-n4000000}
  [fmt_parent]=${MODEL_fmt_parent:-podhajskimarcin/evt-ts1b-fig2ts-installer}
  [teach_child]=${MODEL_teach_child:-podhajskimarcin/evt-ts1b-teach-ft-fmt-n4000000}
)
MODEL_TAGS=(elicit_child elicit_parent teach_child fmt_parent base)
MODELS_DIR=${MODELS_DIR:-/workspace/models}
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
SCORES_HF_PREFIX=${SCORES_HF_PREFIX:-results/eapig_kl_symbol}
PUSH_SUB=${PUSH_SUB:-results/eapig_folds}
JOB_PY=${JOB_PY:-python3}
JOB_RETRIES=${JOB_RETRIES:-2}
BATCH_SIZE=${BATCH_SIZE:-32}
FOLDS=${FOLDS:-"0 1 2 3 4 5 6 7 8 9"}
SIZES_B="0.002 0.01 0.05"
FRACS_ALL="0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05"
FOLDS_ROOT=$HERE/data/folds
REF=$RES/ref
JOBLOGS=$RES/logs; JOBSTATE=$STATE/jobs
LOG=$RES/run_box_folds.log

fdir() { printf '%s/fold_%02d' "$RES" "$1"; }      # results dir of fold k
ddir() { printf '%s/fold_%02d' "$FOLDS_ROOT" "$1"; } # data dir of fold k

# ---- job lists
# score jobs are split by fold half (_a / _b) so 12 shorter jobs pack evenly onto 4 GPUs
A_JOBS=()
for s in ec tc ep_sym ep_word fp base; do A_JOBS+=("score_${s}_a" "score_${s}_b"); done
A_JOBS+=(hp_elicit_parent hp_fmt_parent hp_base hpold_elicit_parent hpold_fmt_parent hpold_base)
B_JOBS=()
for k in $FOLDS; do B_JOBS+=("cmap_elicit_$k" "cmap_teach_$k"); done
for k in $FOLDS; do B_JOBS+=("func_elicit_child_$k" "func_teach_child_$k"); done

say() { local m; m="[folds $(date +%H:%M:%S)] $*"
        if ((DRY)); then echo "$m"; else mkdir -p "$RES"; echo "$m" | tee -a "$LOG"; fi; }

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
    sleep "${WAIT_POLL:-10}"; t=$((t + ${WAIT_POLL:-10}))
  done
}
gpu_mems() {
  local out=""
  if [[ -n ${FAKE_GPUS:-} ]]; then tr ' ,' '\n\n' <<<"$FAKE_GPUS" | grep .; return; fi
  out=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || true
  if [[ -n $out && -z $(grep -v '^[0-9]*$' <<<"$out") ]]; then echo "$out"
  elif ((DRY)); then printf '24576\n24576\n'; fi
}

half_folds() {   # half_folds a|b: first / second half of $FOLDS
  local -a f; read -r -a f <<<"$FOLDS"; local n=$(( (${#f[@]} + 1) / 2 ))
  if [[ $1 == a ]]; then echo "${f[@]:0:n}"; else echo "${f[@]:n}"; fi
}
# score job score_<set>_<a|b> -> tag task subdir
score_spec() {
  case ${1%_[ab]} in
    score_ec) echo "elicit_child word klword" ;;
    score_tc) echo "teach_child word klword" ;;
    score_ep_sym) echo "elicit_parent symbol klsym" ;;
    score_ep_word) echo "elicit_parent word klword" ;;
    score_fp) echo "fmt_parent word klword" ;;
    score_base) echo "base word klword" ;;
  esac
}
job_outputs() {
  local k t
  case $1 in
    score_*) read -r t _ sub <<<"$(score_spec "$1")"
             for k in $(half_folds "${1##*_}"); do echo "$(fdir "$k")/$sub/$t/scores.pt"; echo "$(fdir "$k")/$sub/$t/sanity.json"; done ;;
    hp_*) for k in $FOLDS; do echo "$(fdir "$k")/hidden_pref/${1#hp_}.json"; done ;;
    hpold_*) echo "$RES/old_split/hidden_pref/${1#hpold_}.json" ;;
    cmap_*) local rk=${1#cmap_}; k=${rk##*_}; echo "$(fdir "$k")/cmap/${rk%_*}.json" ;;
    func_*) local ck=${1#func_}; k=${ck##*_}; echo "$(fdir "$k")/functional/${ck%_*}.json" ;;
    overlap) echo "$RES/overlap.json" ;;
  esac
}
outputs_ok() { local f; while read -r f; do [[ -f $f ]] || return 1; done < <(job_outputs "$1"); }
job_models() {
  case $1 in
    score_*) read -r t _ <<<"$(score_spec "$1")"; echo "$t" ;;
    hp_*) echo "${1#hp_}" ;;
    hpold_*) echo "${1#hpold_}" ;;
    cmap_elicit_*) echo elicit_parent elicit_child ;;
    cmap_teach_*) echo fmt_parent teach_child ;;
    func_*) local ck=${1#func_}; echo "${ck%_*}" ;;
  esac
}
CMD=()
build_cmd() {   # build_cmd <job> [bs]: fills CMD
  local job=$1 bs=${2:-$BATCH_SIZE} t task sub k c F
  local -a py; read -r -a py <<<"$JOB_PY"
  case $job in
    score_*)
      read -r t task sub <<<"$(score_spec "$job")"
      # shellcheck disable=SC2206
      CMD=("${py[@]}" "$HERE/fold_runner.py" --tag "$t" --model "$(model_path "$t")" --task "$task"
           --metric kl --stage sanity score --folds-root "$FOLDS_ROOT" --folds $(half_folds "${job##*_}")
           --results-root "$RES" --subdir "$sub" --batch-size "$bs" --device cuda) ;;
    hp_*)
      t=${job#hp_}
      # shellcheck disable=SC2206
      CMD=("${py[@]}" "$HERE/hidden_pref.py" --tag "$t" --model "$(model_path "$t")"
           --folds-root "$FOLDS_ROOT" --folds $FOLDS --results-root "$RES" --batch-size "$bs" --device cuda) ;;
    hpold_*)
      t=${job#hpold_}
      CMD=("${py[@]}" "$HERE/hidden_pref.py" --tag "$t" --model "$(model_path "$t")"
           --data-dir "$HERE/data" --out "$RES/old_split/hidden_pref/$t.json" --batch-size "$bs" --device cuda) ;;
    cmap_*)
      local rk=${job#cmap_} route pt ct; k=${rk##*_}; route=${rk%_*}; F=$(fdir "$k")
      if [[ $route == elicit ]]; then pt=elicit_parent; ct=elicit_child; else pt=fmt_parent; ct=teach_child; fi
      # shellcheck disable=SC2206
      CMD=("${py[@]}" "$HERE/cmap.py" --route "$route" --parent "$(model_path "$pt")"
           --child "$(model_path "$ct")" --child-scores "$F/klword/$ct" --sizes $SIZES_B
           --data-dir "$(ddir "$k")" --batch-size "$bs" --device cuda --out "$F/cmap/$route.json")
      [[ $route == elicit ]] && CMD+=(--extra-circuit "ep_sym=$F/klsym/elicit_parent") ;;
    func_*)
      local ck=${job#func_}; k=${ck##*_}; c=${ck%_*}; F=$(fdir "$k")
      # shellcheck disable=SC2206
      CMD=("${py[@]}" "$HERE/functional_reuse.py" --child "$c" --model "$(model_path "$c")" --metric kl
           --own "$F/klword/$c" --sizes $SIZES_B --data-dir "$(ddir "$k")" --batch-size "$bs"
           --device cuda --out "$F/functional/$c.json" --circuit)
      if [[ $c == elicit_child ]]; then CMD+=("ep_sym=$F/klsym/elicit_parent" "ep_word=$F/klword/elicit_parent")
      else CMD+=("fp_word=$F/klword/fmt_parent"); fi ;;
    overlap)
      # shellcheck disable=SC2206
      CMD=(nice -n 10 "${py[@]}" "$HERE/fold_overlap.py" --results-root "$RES" --folds $FOLDS
           --fracs $FRACS_ALL --n-null 1000 --out "$RES/overlap.json") ;;
  esac
  return 0
}
job_done() { [[ -f $JOBSTATE/$1.done ]] && outputs_ok "$1"; }

claim() {   # atomic pop of the first line of queue $1; status 1 when empty
  ( flock 9; [[ -s $1 ]] || exit 1
    head -n1 "$1"; tail -n +2 "$1" > "$1.tmp"; mv "$1.tmp" "$1" ) 9>"$1.lock"
}
run_gpu_job() {   # retried with a halved batch (OOM), floor 8
  local job=$1 bs=$2 attempt jl m f
  for m in $(job_models "$job"); do wait_model "$m" || return 1; done
  jl=$JOBLOGS/$job.log
  for ((attempt = 0; attempt <= JOB_RETRIES; attempt++)); do
    ((attempt)) && bs=$(( bs / 2 < 8 ? 8 : bs / 2 ))
    build_cmd "$job" "$bs"
    say "$WORKER gpu$WGPU run $job (attempt $((attempt + 1)), bs=$bs)"
    local t0=$SECONDS
    if "${CMD[@]}" >>"$jl" 2>&1 && outputs_ok "$job"; then
      say "$WORKER done $job in $((SECONDS - t0)) s"; return 0
    fi
    say "$WORKER FAILED $job attempt $((attempt + 1)); log tail:"; tail -n 8 "$jl" | tee -a "$LOG" || true
    # never reuse a failed single-output job's file (score jobs resume per fold instead)
    case $job in cmap_*|func_*|hpold_*) while read -r f; do [[ -f $f ]] && mv -f "$f" "$f.bad"; done < <(job_outputs "$job") ;; esac
  done
  return 1
}

# ------------------------------------------------------------------ worker
if [[ -n $WORKER ]]; then
  export CUDA_VISIBLE_DEVICES=$WGPU OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
         OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
  unset GEODE_EAPIG_DENSE
  mkdir -p "$JOBSTATE" "$JOBLOGS"
  say "$WORKER up: gpu $WGPU, queue $(basename "$WQUEUE")"
  while job=$(claim "$WQUEUE"); do
    if run_gpu_job "$job" "$BATCH_SIZE"; then touch "$JOBSTATE/$job.done"
    else touch "$JOBSTATE/$job.failed"; fi
  done
  touch "$STATE/$WORKER.$(basename "$WQUEUE").done"; exit 0
fi

# ------------------------------------------------------------------ plan
mapfile -t MEMS < <(gpu_mems)
N_GPU=${#MEMS[@]}
((N_GPU >= 1)) || { say "FATAL: no GPU found"; exit 1; }
NPROC=$(nproc)
THREADS=${THREADS:-$(( (NPROC - 2) / N_GPU < 1 ? 1 : (NPROC - 2) / N_GPU ))}

if ((DRY)); then
  say "DRY RUN: $N_GPU GPU(s) (MiB: ${MEMS[*]}), one job per GPU, $THREADS threads/job, folds: $FOLDS"
  for t in "${MODEL_TAGS[@]}"; do say "prefetch $t: ${MODEL[$t]} -> $(model_path "$t")"; done
  say "ref <- $RESULTS_REPO:$SCORES_HF_PREFIX/klword/elicit_child/scores.pt"
  say "repro: $JOB_PY $HERE/fold_runner.py --tag elicit_child ... --data-dir $HERE/data --results $RES/repro --repro-ref $REF/elicit_child_scores.pt"
  say "phase A queue:"; for j in "${A_JOBS[@]}"; do build_cmd "$j"; say "  $j: ${CMD[*]}"; done
  say "gate: children EM vs 0.953 / 0.145 (hard > 0.10, warn > 0.05), word parents + base <= 0.02"
  build_cmd overlap; say "CPU: ${CMD[*]}"
  say "phase B queue (${#B_JOBS[@]} jobs), e.g.:"
  for j in "${B_JOBS[0]}" "${B_JOBS[1]}" "func_elicit_child_${FOLDS%% *}" "func_teach_child_${FOLDS%% *}"; do
    build_cmd "$j"; say "  $j: ${CMD[*]}"; done
  say "report: $JOB_PY $HERE/fold_report.py --results-root $RES --folds $FOLDS --overlap $RES/overlap.json --old $HERE/old_split.json --out-md $RES/fold_report.md --out-json $RES/fold_report.json --fig-dir $RES/figures"
  say "push: $RES -> $RESULTS_REPO:$PUSH_SUB (excl .state, logs, per_example.pt, *.bad)"
  exit 0
fi

# ------------------------------------------------------------------ main
mkdir -p "$RES" "$STATE" "$JOBSTATE" "$JOBLOGS" "$REF" "$MODELS_DIR"
[[ -f $LOG ]] && mv "$LOG" "$LOG.prev"
rm -f "$STATE"/*.queue* "$STATE"/w*.done "$STATE/env.sh" "$JOBSTATE"/*.failed
fatal() { say "FATAL: $*"; echo "EXIT=1" >> "$LOG"; exit 1; }
say "repo $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo ?) gpus: ${MEMS[*]} MiB, $NPROC cpus, $THREADS threads/job, folds: $FOLDS"
for k in $FOLDS; do for f in pairs_seed0.pt pairs_symbol_seed0.pt copy_pairs_seed1.pt; do
  [[ -f $(ddir "$k")/$f ]] || fatal "$(ddir "$k")/$f missing (copy data/folds up)"; done; done
for f in data/pairs_seed0.pt data/copy_pairs_seed1.pt fold_runner.py hidden_pref.py fold_overlap.py fold_report.py old_split.json; do
  [[ -f $f ]] || fatal "$f missing"; done
cp "$FOLDS_ROOT/manifest.json" "$RES/data_manifest.json" 2>/dev/null || say "WARN: no data/folds/manifest.json"

{ printf 'export THREADS=%q MODELS_DIR=%q JOB_PY=%q JOB_RETRIES=%q PYTHONPATH=%q REPO_ROOT=%q BATCH_SIZE=%q FOLDS=%q\n' \
    "$THREADS" "$MODELS_DIR" "$JOB_PY" "$JOB_RETRIES" "$PYTHONPATH" "$REPO_ROOT" "$BATCH_SIZE" "$FOLDS"
  for t in "${MODEL_TAGS[@]}"; do printf 'export MODEL_%s=%q\n' "$t" "${MODEL[$t]}"; done
  for v in FAKE_GPUS WAIT_POLL; do [[ -n ${!v:-} ]] && printf 'export %s=%q\n' "$v" "${!v}"; done
  true; } > "$STATE/env.sh"

prefetch_one() {
  local tag=$1 repo id
  repo=${MODEL[$tag]}; id=$(basename "$repo")
  [[ -d $repo || -f $MODELS_DIR/$tag.done ]] && { say "model $tag present"; return 0; }
  rm -f "$MODELS_DIR/$tag.failed"
  say "prefetch $tag ($repo)"
  if HF_HUB_DISABLE_XET=1 hf download "$repo" --include "runs/$id/model/*" \
       --local-dir "$MODELS_DIR/$tag" >>"$JOBLOGS/prefetch_$tag.log" 2>&1 && [[ -f $(model_path "$tag")/config.json ]]; then
    touch "$MODELS_DIR/$tag.done"; say "prefetched $tag"
  else touch "$MODELS_DIR/$tag.failed"; say "PREFETCH FAILED $tag"; return 1; fi
}
if [[ -z ${SKIP_PREFETCH:-} ]]; then
  for t in "${MODEL_TAGS[@]}"; do prefetch_one "$t" & done
  if [[ ! -f $REF/elicit_child_scores.pt ]]; then
    python3 - "$RESULTS_REPO" "$SCORES_HF_PREFIX" "$REF" <<'PY' 2>&1 | tee -a "$LOG"
import shutil, sys
from huggingface_hub import hf_hub_download
repo, prefix, ref = sys.argv[1:4]
shutil.copy(hf_hub_download(repo, f"{prefix}/klword/elicit_child/scores.pt"), f"{ref}/elicit_child_scores.pt")
print("have ref scores")
PY
  fi || true
fi
[[ -f $REF/elicit_child_scores.pt ]] || fatal "reference scores missing"

# ---- reproduction gate (GPU 0, before any fold job)
wait_model elicit_child || fatal "elicit_child model missing"
if [[ -f $RES/repro/elicit_child/repro.json ]] && grep -q '"pass": true' "$RES/repro/elicit_child/repro.json"; then
  say "repro gate already passed"
else
  say "repro gate: elicit_child on the original data/"
  read -r -a PY_ <<<"$JOB_PY"
  CUDA_VISIBLE_DEVICES=0 "${PY_[@]}" "$HERE/fold_runner.py" --tag elicit_child --model "$(model_path elicit_child)" \
    --task word --metric kl --stage sanity score --data-dir "$HERE/data" --results "$RES/repro" \
    --batch-size "$BATCH_SIZE" --device cuda --repro-ref "$REF/elicit_child_scores.pt" >>"$JOBLOGS/repro.log" 2>&1 \
    || { tail -n 20 "$JOBLOGS/repro.log" | tee -a "$LOG"; fatal "REPRO GATE FAILED"; }
  say "repro gate PASSED: $(cat "$RES/repro/elicit_child/repro.json" | tr -d '\n ')"
fi

run_queue() {   # run_queue <name> <job ...>: fill a queue, one worker per GPU, wait
  local name=$1; shift
  local q=$STATE/$name.queue j g
  : > "$q"
  for j in "$@"; do if job_done "$j"; then say "skip $j (done)"; else rm -f "$JOBSTATE/$j.done"; echo "$j" >> "$q"; fi; done
  say "queue $name: $(wc -l < "$q") job(s) on $N_GPU GPU(s)"
  for ((g = 0; g < N_GPU; g++)); do
    bash "$HERE/run_box_folds.sh" --confirm-cost --worker "w$g" "$g" "$q" >>"$JOBLOGS/worker_w$g.out" 2>&1 &
  done
  wait_workers "$name"
}
wait_workers() {
  local g
  for ((g = 0; g < N_GPU; g++)); do
    until [[ -f $STATE/w$g.$1.queue.done ]]; do sleep "${MAIN_POLL:-15}"; done
  done
}
check_jobs() {
  local j bad=0
  for j in "$@"; do
    if [[ -f $JOBSTATE/$j.failed ]]; then say "job $j FAILED"; bad=1
    elif ! job_done "$j"; then say "job $j NOT FINISHED"; bad=1; fi
  done
  return $bad
}
( while :; do sleep 180
    u=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ';' || echo n/a)
    say "util gpu[idx,%,MiB]: $u load=$(cut -d' ' -f1 /proc/loadavg) done=$(find "$JOBSTATE" -name '*.done' | wc -l) failed=$(find "$JOBSTATE" -name '*.failed' | wc -l)"
  done ) & SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT

bad=0
run_queue A "${A_JOBS[@]}"
check_jobs "${A_JOBS[@]}" || bad=1

# ---- sanity gate
say "sanity gate"
if python3 - "$RES" "$FOLDS" <<'PY' 2>&1 | tee -a "$LOG"; then gate=ok; else gate=failed; fi
import json, sys
from pathlib import Path
res, folds = Path(sys.argv[1]), [int(x) for x in sys.argv[2].split()]
ref = {"elicit_child": 0.953, "teach_child": 0.145}
rows, hard = [], []
for k in folds:
    for sub, tag in [("klword", "elicit_child"), ("klword", "teach_child"), ("klword", "elicit_parent"),
                     ("klword", "fmt_parent"), ("klword", "base"), ("klsym", "elicit_parent")]:
        p = res / f"fold_{k:02d}" / sub / tag / "sanity.json"
        if not p.is_file():
            hard.append(f"fold {k} {sub}/{tag}: sanity.json missing"); continue
        em = json.loads(p.read_text())["exact_match"]
        status = "ok"
        if tag in ref:
            d = abs(em - ref[tag])
            status = "HARD" if d > 0.10 else "WARN" if d > 0.05 else "ok"
        elif sub == "klword" and em > 0.02:
            status = "HARD"
        rows.append({"fold": k, "set": f"{sub}/{tag}", "exact_match": em, "status": status})
        if status == "HARD":
            hard.append(f"fold {k} {sub}/{tag}: EM {em:.3f}")
for r in rows:
    if r["status"] != "ok" or r["set"] in ("klword/elicit_child", "klword/teach_child", "klsym/elicit_parent"):
        print(f"  fold {r['fold']} {r['set']:22s} EM {r['exact_match']:.3f} {r['status']}")
(res / "gate.json").write_text(json.dumps({"rows": rows, "hard": hard}, indent=1))
print("gate HARD failures:", hard or "none")
sys.exit(1 if hard else 0)
PY
[[ $gate == ok ]] || { bad=1; say "GATE FAILED: skipping phase B (check the fold data)"; }

if [[ $gate == ok ]]; then
  ( build_cmd overlap; "${CMD[@]}" >>"$JOBLOGS/overlap.log" 2>&1 && touch "$JOBSTATE/overlap.done" \
      || touch "$JOBSTATE/overlap.failed" ) & OV=$!
  say "CPU lane: fold_overlap started"
  run_queue B "${B_JOBS[@]}"
  check_jobs "${B_JOBS[@]}" || bad=1
  wait "$OV" || true
  check_jobs overlap || bad=1
  if [[ -f $RES/overlap.json ]]; then
    read -r -a PY_ <<<"$JOB_PY"
    # shellcheck disable=SC2086
    "${PY_[@]}" "$HERE/fold_report.py" --results-root "$RES" --folds $FOLDS --overlap "$RES/overlap.json" \
      --old "$HERE/old_split.json" --out-md "$RES/fold_report.md" --out-json "$RES/fold_report.json" \
      --fig-dir "$RES/figures" >>"$JOBLOGS/report.log" 2>&1 && say "report written" || { say "REPORT FAILED"; bad=1; }
  fi
fi

# ---- push (single-threaded upload + list-files verify)
cp "$LOG" "$RES/run_log.txt"
if [[ -z ${NO_PUSH:-} ]]; then
  say "push $RES -> $RESULTS_REPO:$PUSH_SUB"
  if HF_HUB_DISABLE_XET=1 python3 - "$RES" "$PUSH_SUB" "$RESULTS_REPO" <<'PY' 2>&1 | tee -a "$LOG"; then pushed=ok; else pushed=FAILED; fi
import sys
from fnmatch import fnmatch
from pathlib import Path
from huggingface_hub import HfApi
res, sub, repo = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ign = [".state/*", "*.log", "*.log.prev", "*.bad", "*.out", "*per_example.pt", "ref/*", "logs/*"]
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message="eapig 10-fold robustness check", ignore_patterns=ign)
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + r for r in (str(p.relative_to(res)) for p in res.rglob("*") if p.is_file())
         if not any(fnmatch(r, g) for g in ign)}
missing = local - remote
print("verify: missing on hub:", sorted(missing) or "none", "| files:", len(local))
sys.exit(1 if missing else 0)
PY
  say "pushed=$pushed"; [[ $pushed == ok ]] || bad=1
fi
((bad)) && { say "NOT DONE (a job, gate, report or push failed)"; echo "EXIT=1" >> "$LOG"; exit 1; }
say "ALL DONE"; echo "EXIT=0" >> "$LOG"
