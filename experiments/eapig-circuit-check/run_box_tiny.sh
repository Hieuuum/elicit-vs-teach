#!/usr/bin/env bash
# Tiny circuit sizes 0.01/0.02/0.05% (k = 20/39/98 of 195,865 edges) on a 2x RTX 3090 box.
# Same models, saved KL scores, pairs and arguments as run_box_small.sh / run_box_cmap.sh /
# run_box_base.sh; only the sizes change. Queue/worker design copied from run_box_cmap.sh.
#
#   CPU lane (background, niced, starts once the scores are fetched, runs beside the GPU jobs):
#     overlap      edge_node.py: 6 score sets, the 7 EDGE_PAIRS of run_box_small.sh, ALL NINE
#                  sizes (the six 0.1-5% sizes reproduce the earlier run), 1000-draw nulls
#     base_edge    edge_node.py with run_box_base.sh's scores/pairs, nine sizes
#     base_boot    overlap_boot.py, 200 resamples, nine sizes
#     base_report  base_control_report.py (same CLI as run_box_base.sh)
#   GPU queue (.state/queue, claimed under flock; one worker slot per 24 GB GPU, two jobs on a
#   3090 OOM; phase-1 jobs all ahead of phase-2 jobs, slowest first within a phase):
#     phase 1  functional_reuse.py elicit_child / teach_child (run_box_small.sh gpu3 args);
#              cmap.py elicit / teach x 3 sizes (run_box_cmap.sh args), merged per route
#     phase 2  run.py --stage evaluate --metric kl $EVAL_TESTS for teach_child, elicit_child,
#              elicit_parent (--task symbol, results_klsym), fmt_parent (sparse patch path)
#   Push 1 (every CPU + phase-1 job finished): results_tiny minus tests/ -> log PHASE1_DONE pushed=ok|failed
#   Push 2 (phase 2 finished): all of results_tiny -> log EXIT=0 (EXIT=1 on any failure)
#
#   bash run_box_tiny.sh --confirm-cost                  # full run; DRY_RUN=1 prints the plan only
#   bash run_box_tiny.sh --confirm-cost --prefetch-only  # 7 score sets (incl. base) + 4 models, then exit
# Re-run: a job with a .state/jobs/<job>.done marker AND its outputs is skipped.
# Env: MODEL_<tag> (HF repo id or local dir), MODELS_DIR=/workspace/models, VENV=/workspace/venv,
#      JOBS_PER_GPU (default 2 if >= 46 GB else 1), BATCH_SIZE=32, THREADS (per GPU job),
#      CPU_THREADS (CPU lane; default nproc/2), NO_PUSH=1, RESULTS_REPO=mhieuuu/geode-internals,
#      PUSH_SUB=results/eapig_tiny, SCORES_HF_PREFIX=results/eapig_kl_symbol, JOB_RETRIES=1
#      (extra attempts per GPU job, batch halved), MON_INTERVAL=120, EVAL_TESTS="--tests suff_nec".
#      Test hooks: FAKE_GPUS="24576 24576" (MiB per GPU, no nvidia-smi), JOB_PY (replaces
#      "python3" for every job script; gets <script.py> <args>), NO_TMUX=1, REPO_ROOT,
#      WAIT_POLL, MAIN_POLL.
# Progress/markers: tail -f results_tiny/run_box_tiny.log  (PHASE1_DONE ..., last line EXIT=0|1)
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
export PYTHONPATH=$REPO_ROOT:$HERE${PYTHONPATH:+:$PYTHONPATH}

CONFIRM=0; PREFETCH_ONLY=0; WORKER=""; WGPU=""; DRY=${DRY_RUN:-0}
while [[ $# -gt 0 ]]; do
  case $1 in
    --confirm-cost) CONFIRM=1 ;;
    --prefetch-only) PREFETCH_ONLY=1 ;;
    --worker) WORKER=$2; WGPU=$3; shift 2 ;;   # internal: slot name, gpu index
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $WORKER ]] || echo "[tiny] estimated ~35-45 min wall on a 2x RTX 3090 box (~\$0.4/h): prefetch ~5 min;" \
  "phase 1 ~7 min (6 cmap shards ~1.5 min each, cf. 94-181 s/shard in the 0.1-5% cmap run; 2 functional_reuse" \
  "~2 min); phase 2 ~17 min (children ~11 min, parents ~6 min: ~1 s/mask sparse path, 202/102 masks per size);" \
  "CPU overlap/base control ~3 min alongside. ~1.2 GPU-h, ~\$0.3; under ~\$0.6 incl. race/idle; box rental is the only cost."
((CONFIRM || DRY)) || { echo "[tiny] refusing to run without --confirm-cost" >&2; exit 2; }

RES=$HERE/results_tiny
STATE=$RES/.state
# worker slots inherit the main invocation's resolved settings through this file
# shellcheck disable=SC1091
[[ -n $WORKER && -f $STATE/env.sh ]] && source "$STATE/env.sh"

VENV=${VENV:-/workspace/venv}
# shellcheck disable=SC1091
[[ -f $VENV/bin/activate ]] && source "$VENV/bin/activate"

declare -A MODEL=(
  [elicit_parent]=${MODEL_elicit_parent:-podhajskimarcin/evt-ts1b-op-bridge-mix}
  [elicit_child]=${MODEL_elicit_child:-podhajskimarcin/evt-ts1b-elicit-ft-n4000000}
  [fmt_parent]=${MODEL_fmt_parent:-podhajskimarcin/evt-ts1b-fig2ts-installer}
  [teach_child]=${MODEL_teach_child:-podhajskimarcin/evt-ts1b-teach-ft-fmt-n4000000}
)
MODEL_TAGS=(elicit_child elicit_parent teach_child fmt_parent)
MODELS_DIR=${MODELS_DIR:-/workspace/models}
RESULTS_REPO=${RESULTS_REPO:-mhieuuu/geode-internals}
SCORES_HF_PREFIX=${SCORES_HF_PREFIX:-results/eapig_kl_symbol}
PUSH_SUB=${PUSH_SUB:-results/eapig_tiny}
JOB_PY=${JOB_PY:-python3}
JOB_RETRIES=${JOB_RETRIES:-1}
MON_INTERVAL=${MON_INTERVAL:-120}
BATCH_SIZE=${BATCH_SIZE:-32}
EVAL_TESTS=${EVAL_TESTS-"--tests suff_nec"}   # run.py evaluate test selection (one place)
SIZES_TINY="0.0001 0.0002 0.0005"
SIZES_TINY_DESC=(0.0005 0.0002 0.0001)          # cmap queue order, slowest first
FRACS_ALL="0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05"
ROUTES=(elicit teach)

KLWORD=$HERE/results_klword; KLSYM=$HERE/results_klsym
OVERLAP=$RES/overlap; BASEC=$RES/base_control; CMAPD=$RES/cmap; SHARDS=$CMAPD/shards
FUNC=$RES/functional; TESTS=$RES/tests
JOBLOGS=$RES/logs; JOBSTATE=$STATE/jobs; QUEUE=$STATE/queue
if ((PREFETCH_ONLY)); then LOG=$RES/prefetch.log; else LOG=$RES/run_box_tiny.log; fi
SESSION_PREFIX=tiny

# run_box_small.sh's edge_node scores + pairs
EDGE_SCORES=(ep_sym="$KLSYM/elicit_parent" ep_word="$KLWORD/elicit_parent"
             ec_sym="$KLSYM/elicit_child" ec_word="$KLWORD/elicit_child"
             fp_word="$KLWORD/fmt_parent" tc_word="$KLWORD/teach_child")
EDGE_PAIRS=(ep_sym:ec_word ep_sym:tc_word fp_word:tc_word ep_word:ec_word ep_sym:ec_sym
            ec_sym:ec_word ec_word:tc_word)
# run_box_base.sh's scores + pairs
BASE_SCORES=(base="$KLWORD/base" ec_word="$KLWORD/elicit_child" tc_word="$KLWORD/teach_child"
             fp_word="$KLWORD/fmt_parent" ep_word="$KLWORD/elicit_parent" ep_sym="$KLSYM/elicit_parent")
BASE_PAIRS=(base:ec_word base:tc_word base:fp_word base:ep_word
            ep_sym:ec_word fp_word:tc_word ep_sym:tc_word ep_word:ec_word)

# ---- job lists (queue order)
CPU_JOBS=(overlap base_edge base_boot base_report)
P1_JOBS=(func_elicit_child func_teach_child)
for z in "${SIZES_TINY_DESC[@]}"; do for r in "${ROUTES[@]}"; do P1_JOBS+=("cmap_${r}_$z"); done; done
P2_JOBS=(eval_teach_child eval_elicit_child eval_elicit_parent eval_fmt_parent)
GPU_JOBS=("${P1_JOBS[@]}" "${P2_JOBS[@]}")

say() { local m; m="[tiny $(date +%H:%M:%S)] $*"
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

# ---- GPUs: memory (MiB) per GPU, one per line
gpu_mems() {
  local out=""
  if [[ -n ${FAKE_GPUS:-} ]]; then tr ' ,' '\n\n' <<<"$FAKE_GPUS" | grep .; return; fi
  out=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ') || true
  if [[ -n $out && -z $(grep -v '^[0-9]*$' <<<"$out") ]]; then echo "$out"
  elif ((DRY)); then printf '24576\n24576\n'   # DRY plan without a usable nvidia-smi: fake 2 x 24 GB
  fi
}

# ---- per-job table: outputs, models, command
job_outputs() {   # files that must exist for the job to count as done
  case $1 in
    overlap) echo "$OVERLAP/edge_node.json" ;;
    base_edge) echo "$BASEC/edge_node/edge_node.json" ;;
    base_boot) echo "$BASEC/boot/overlap_boot.json" ;;
    base_report) echo "$BASEC/base_control_table.md" ;;
    func_*) echo "$FUNC/${1#func_}.json" ;;
    cmap_*) local rz=${1#cmap_}; echo "$SHARDS/$rz.json"; echo "$SHARDS/$rz.npz" ;;
    eval_*) echo "$TESTS/${1#eval_}/evaluate.json"; echo "$TESTS/${1#eval_}/per_example.npz" ;;
  esac
}
outputs_ok() { local f; while read -r f; do [[ -f $f ]] || return 1; done < <(job_outputs "$1"); }
job_models() {
  case $1 in
    func_*) echo "${1#func_}" ;;
    cmap_elicit_*) echo elicit_parent elicit_child ;;
    cmap_teach_*) echo fmt_parent teach_child ;;
    eval_*) echo "${1#eval_}" ;;
  esac
}
CMD=()
build_cmd() {   # build_cmd <job> [bs]: fills CMD
  local job=$1 bs=${2:-$BATCH_SIZE} rz route size tag task sdir
  local -a py; read -r -a py <<<"$JOB_PY"
  local nice_cpu=(nice -n 10 env OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS"
                  OPENBLAS_NUM_THREADS="$CPU_THREADS")
  # edge_node parallelises over --workers processes: one BLAS thread each
  local nice_edge=(nice -n 10 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1)
  case $job in
    overlap)
      # shellcheck disable=SC2206
      CMD=("${nice_edge[@]}" "${py[@]}" "$HERE/edge_node.py" --scores "${EDGE_SCORES[@]}"
           --pairs "${EDGE_PAIRS[@]}" --fracs $FRACS_ALL --n-null 1000 --workers "$CPU_THREADS"
           --out "$OVERLAP") ;;
    base_edge)
      # shellcheck disable=SC2206
      CMD=("${nice_edge[@]}" "${py[@]}" "$HERE/edge_node.py" --scores "${BASE_SCORES[@]}"
           --pairs "${BASE_PAIRS[@]}" --fracs $FRACS_ALL --n-null 1000 --workers "$CPU_THREADS"
           --out "$BASEC/edge_node") ;;
    base_boot)
      # shellcheck disable=SC2206
      CMD=("${nice_cpu[@]}" "${py[@]}" "$HERE/overlap_boot.py" --scores "${BASE_SCORES[@]}"
           --pairs "${BASE_PAIRS[@]}" --fracs $FRACS_ALL --n-boot 200 --seed 0
           --data-dir "$HERE/data" --out "$BASEC/boot") ;;
    base_report)
      CMD=("${nice_cpu[@]}" "${py[@]}" "$HERE/base_control_report.py"
           --edge-node "$BASEC/edge_node/edge_node.json" --boot "$BASEC/boot/overlap_boot.json"
           --out-md "$BASEC/base_control_table.md" --fig "$BASEC/base_control_overlap.png") ;;
    func_*)
      tag=${job#func_}
      # shellcheck disable=SC2206
      CMD=("${py[@]}" "$HERE/functional_reuse.py" --child "$tag" --metric kl
           --model "$(model_path "$tag")" --own "$KLWORD/$tag"
           --circuit ep_sym="$KLSYM/elicit_parent" ep_word="$KLWORD/elicit_parent" fp_word="$KLWORD/fmt_parent"
           --sizes $SIZES_TINY --batch-size "$bs" --device cuda --out "$FUNC/$tag.json") ;;
    cmap_*)
      rz=${job#cmap_}; route=${rz%_*}; size=${rz##*_}
      local pt ct
      if [[ $route == elicit ]]; then pt=elicit_parent; ct=elicit_child; else pt=fmt_parent; ct=teach_child; fi
      CMD=("${py[@]}" "$HERE/cmap.py" --route "$route" --parent "$(model_path "$pt")"
           --child "$(model_path "$ct")" --child-scores "$KLWORD/$ct" --sizes "$size"
           --batch-size "$bs" --device cuda --out "$SHARDS/$rz.json")
      [[ $route == elicit ]] && CMD+=(--extra-circuit "ep_sym=$KLSYM/elicit_parent") ;;
    eval_*)
      tag=${job#eval_}; task=word; sdir=$KLWORD
      [[ $tag == elicit_parent ]] && { task=symbol; sdir=$KLSYM; }
      # shellcheck disable=SC2206
      CMD=("${py[@]}" "$HERE/run.py" --tag "$tag" --model "$(model_path "$tag")" --task "$task"
           --metric kl --stage evaluate --sizes $SIZES_TINY --scores-dir "$sdir"
           --results "$TESTS" --batch-size "$bs" --device cuda $EVAL_TESTS)
      # same as run_box_small.sh gpu2 (parent_in_child is skipped under suff_nec anyway)
      [[ $tag == teach_child ]] && CMD+=(--parent-tag fmt_parent) ;;
  esac
  return 0
}
finished() { [[ -f $JOBSTATE/$1.done || -f $JOBSTATE/$1.failed ]]; }
job_done() { [[ -f $JOBSTATE/$1.done ]] && outputs_ok "$1"; }

# ---- queue
claim() {   # atomic pop of the first job line; exit status 1 when the queue is empty
  ( flock 9; [[ -s $QUEUE ]] || exit 1
    head -n1 "$QUEUE"; tail -n +2 "$QUEUE" > "$QUEUE.tmp"; mv "$QUEUE.tmp" "$QUEUE" ) 9>"$QUEUE.lock"
}

run_gpu_job() {   # run_gpu_job <job> <batch>: retried with a halved batch; returns 0 on success
  local job=$1 bs=$2 attempt jl m f
  for m in $(job_models "$job"); do wait_model "$m" || return 1; done
  jl=$JOBLOGS/$job.log
  for ((attempt = 0; attempt <= JOB_RETRIES; attempt++)); do
    # a retry halves the batch (an OOM at bs 32 on a 24 GB card), floor 8
    ((attempt)) && bs=$(( bs / 2 < 8 ? 8 : bs / 2 ))
    build_cmd "$job" "$bs"
    say "$WORKER gpu$WGPU run $job (attempt $((attempt + 1)), bs=$bs)"
    local t0=$SECONDS
    if "${CMD[@]}" >>"$jl" 2>&1 && outputs_ok "$job"; then
      say "$WORKER done $job in $((SECONDS - t0)) s"; return 0
    fi
    say "$WORKER FAILED $job attempt $((attempt + 1)); log tail:"; tail -n 8 "$jl" | tee -a "$LOG" || true
    # a failed run may leave an output (e.g. cmap's identity check): never reuse it
    while read -r f; do [[ -f $f ]] && mv -f "$f" "$f.bad"; done < <(job_outputs "$job")
  done
  return 1
}

worker() {   # one slot: pull jobs until the queue is empty
  local job mem
  mem=$(gpu_mems | sed -n "$((WGPU + 1))p")
  export CUDA_VISIBLE_DEVICES=$WGPU OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
         OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS
  unset GEODE_EAPIG_DENSE   # fast sparse patch path (checked against dense in the small-sizes run)
  mkdir -p "$JOBSTATE" "$JOBLOGS" "$SHARDS" "$FUNC" "$TESTS"
  say "$WORKER up: gpu $WGPU (${mem} MiB), bs=$BATCH_SIZE, $THREADS threads"
  while job=$(claim); do
    if run_gpu_job "$job" "$BATCH_SIZE"; then touch "$JOBSTATE/$job.done"
    else touch "$JOBSTATE/$job.failed"; fi
  done
  say "$WORKER: queue empty"
}

cpu_step() {   # cpu_step <job>: skip if done, else run once; .done/.failed marker
  local job=$1 jl=$JOBLOGS/$1.log t0=$SECONDS
  if job_done "$job"; then say "cpu skip $job (done)"; return 0; fi
  build_cmd "$job"
  say "cpu run $job"
  if "${CMD[@]}" >>"$jl" 2>&1 && outputs_ok "$job"; then
    touch "$JOBSTATE/$job.done"; say "cpu done $job in $((SECONDS - t0)) s"; return 0
  fi
  touch "$JOBSTATE/$job.failed"; say "cpu FAILED $job; log tail:"; tail -n 8 "$jl" | tee -a "$LOG" || true
  return 1
}
cpu_lane() {   # sequential CPU jobs; a failure does not stop the next one
  cpu_step overlap || true
  cpu_step base_edge || true
  cpu_step base_boot || true
  if [[ -f $JOBSTATE/base_edge.done && -f $JOBSTATE/base_boot.done ]]; then
    cpu_step base_report || true
    # the table + reference-check lines into the main log, as run_box_base.sh did
    [[ -f $JOBLOGS/base_report.log ]] && sed 's/^/  | /' "$JOBLOGS/base_report.log" >> "$LOG"
  else
    touch "$JOBSTATE/base_report.failed"; say "cpu FAILED base_report (an input job failed)"
  fi
}

# ---- inputs
fetch_scores() {  # 7 saved KL score sets (scores.pt + sanity.json) -> results_klword/, results_klsym/
  python3 - "$RESULTS_REPO" "$SCORES_HF_PREFIX" "$KLWORD" "$KLSYM" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
repo, prefix, klword, klsym = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
want = [("klword", klword, t) for t in ("elicit_parent", "elicit_child", "fmt_parent", "teach_child", "base")]
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
prefetch_one() {  # one model; .done/.failed marker (shared with run_box_cmap.sh's prefetch)
  local tag=$1 repo id
  repo=${MODEL[$tag]}; id=$(basename "$repo")
  [[ -d $repo || -f $MODELS_DIR/$tag.done ]] && { say "model $tag present"; return 0; }
  rm -f "${MODELS_DIR:?}/${tag:?}.failed"
  say "prefetch $tag ($repo)"
  if HF_HUB_DISABLE_XET=1 hf download "$repo" --include "runs/$id/model/*" \
       --local-dir "$MODELS_DIR/$tag" >>"$LOG" 2>&1 && [[ -f $(model_path "$tag")/config.json ]]; then
    touch "$MODELS_DIR/$tag.done"; say "prefetched $tag"
  else
    touch "$MODELS_DIR/$tag.failed"; say "PREFETCH FAILED $tag"; return 1
  fi
}
prefetch_all() {  # the 4 models in parallel; returns 1 if any failed
  local tag p bad=0 pids=()
  mkdir -p "$MODELS_DIR"
  for tag in "${MODEL_TAGS[@]}"; do prefetch_one "$tag" & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || bad=1; done
  return $bad
}
# push_dir <local dir> <path in repo> <commit message> [ignore glob ...]:
# single-threaded upload_folder + list-files verify (ignored files are not expected on the hub)
push_dir() {
  [[ -n ${NO_PUSH:-} ]] && return 0
  [[ -d $1 ]] || { say "push: $1 missing"; return 1; }
  say "push $1 -> $RESULTS_REPO:$2"
  HF_HUB_DISABLE_XET=1 python3 - "$@" "$RESULTS_REPO" <<'PY' 2>&1 | tee -a "$LOG"
import sys
from fnmatch import fnmatch
from pathlib import Path
from huggingface_hub import HfApi
res, sub, msg, ign, repo = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:-1], sys.argv[-1]
ign = [".state/*", "*.log", "*.log.prev", "*.bad", "*.out"] + ign
api = HfApi()
api.upload_folder(folder_path=str(res), repo_id=repo, path_in_repo=sub,
                  commit_message=msg, ignore_patterns=ign)
remote = {f for f in api.list_repo_files(repo) if f.startswith(sub + "/")}
local = {sub + "/" + r for r in (str(p.relative_to(res)) for p in res.rglob("*") if p.is_file())
         if not any(fnmatch(r, g) for g in ign)}
missing = local - remote
print("verify: missing on hub:", sorted(missing) or "none")
sys.exit(1 if missing else 0)
PY
  [[ ${PIPESTATUS[0]} == 0 ]]
}

# ------------------------------------------------------------------ worker session
if [[ -n $WORKER ]]; then
  worker || true
  touch "$STATE/$WORKER.done"; exit 0
fi

# ------------------------------------------------------------------ plan
mapfile -t MEMS < <(gpu_mems)
N_GPU=${#MEMS[@]}
((N_GPU >= 1)) || { say "FATAL: no GPU found"; exit 1; }
SLOT_GPU=()   # gpu index per slot
for ((g = 0; g < N_GPU; g++)); do
  jpg=${JOBS_PER_GPU:-$(( MEMS[g] >= 46000 ? 2 : 1 ))}
  for ((j = 0; j < jpg; j++)); do SLOT_GPU+=("$g"); done
done
N_SLOTS=${#SLOT_GPU[@]}
NPROC=$(nproc)
CPU_THREADS=${CPU_THREADS:-$(( NPROC / 2 < 1 ? 1 : NPROC / 2 ))}
THREADS=${THREADS:-$(( (NPROC - CPU_THREADS) / N_SLOTS < 1 ? 1 : (NPROC - CPU_THREADS) / N_SLOTS ))}

if ((DRY)); then
  say "DRY RUN: $N_GPU GPU(s) (MiB: ${MEMS[*]}), $N_SLOTS worker slot(s), $NPROC cpus -> $THREADS threads/GPU job, CPU lane $CPU_THREADS"
  say "scores <- $RESULTS_REPO:$SCORES_HF_PREFIX/{klword/{elicit_parent,elicit_child,fmt_parent,teach_child,base},klsym/{elicit_parent,elicit_child}}/{scores.pt,sanity.json}"
  for t in "${MODEL_TAGS[@]}"; do
    say "prefetch $t (parallel): HF_HUB_DISABLE_XET=1 hf download ${MODEL[$t]} --include 'runs/$(basename "${MODEL[$t]}")/model/*' --local-dir $MODELS_DIR/$t"
  done
  ((PREFETCH_ONLY)) && { say "then exit (--prefetch-only)"; exit 0; }
  say "CPU lane (background, sequential, starts after the score fetch):"
  for j in "${CPU_JOBS[@]}"; do build_cmd "$j"; say "  $j: ${CMD[*]}"; done
  s=0; for g in "${SLOT_GPU[@]}"; do
    say "slot w$s: tmux ${SESSION_PREFIX}_w$s, CUDA_VISIBLE_DEVICES=$g, GEODE_EAPIG_DENSE unset"; s=$((s + 1)); done
  say "GPU queue (claimed under flock, in this order; retry halves bs):"
  for j in "${GPU_JOBS[@]}"; do
    ph=2; [[ " ${P1_JOBS[*]} " == *" $j "* ]] && ph=1
    build_cmd "$j"; say "  [phase $ph] $j (waits for: $(job_models "$j")): ${CMD[*]}"
  done
  for r in "${ROUTES[@]}"; do
    say "merge: $JOB_PY $HERE/cmap.py --merge $(for z in $SIZES_TINY; do printf '%s ' "$SHARDS/${r}_$z.json"; done)--out $CMAPD/$r.json"
  done
  say "summarize: $JOB_PY $HERE/cmap.py --summarize $CMAPD/elicit.json $CMAPD/teach.json > $CMAPD/summary.md"
  say "push 1 (all CPU + phase-1 jobs finished): $RES -> $RESULTS_REPO:$PUSH_SUB (excl .state, *.log, *.out, *.bad, tests/*), log 'PHASE1_DONE pushed=ok|failed'"
  say "push 2 (queue empty): $RES -> $RESULTS_REPO:$PUSH_SUB (excl .state, *.log, *.out, *.bad), log 'EXIT=0|1' in $LOG"
  exit 0
fi

# ------------------------------------------------------------------ main
mkdir -p "$RES" "$STATE" "$MODELS_DIR"
if ((PREFETCH_ONLY)); then
  say "prefetch-only: fetch scores + models"
  fetch_scores 2>&1 | tee -a "$LOG"
  [[ ${PIPESTATUS[0]} == 0 ]] || { say "FATAL: score fetch failed"; exit 1; }
  prefetch_all || { say "FATAL: model prefetch failed"; exit 1; }
  say "PREFETCH DONE"; exit 0
fi

[[ -f $LOG ]] && mv "$LOG" "$LOG.prev"   # markers must come from this run only
# keep .state/jobs/*.done (re-run skips finished work); clear everything else
rm -f "$QUEUE" "$QUEUE.lock" "$STATE"/w*.done "$STATE/env.sh" "$JOBSTATE"/*.failed
mkdir -p "$STATE" "$JOBSTATE" "$JOBLOGS" "$OVERLAP" "$BASEC" "$SHARDS" "$FUNC" "$TESTS"
fatal() { say "FATAL: $*"; echo "EXIT=1" >> "$LOG"; exit 1; }
say "repo $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo ?) gpus: ${MEMS[*]} MiB, $N_SLOTS slot(s), $NPROC cpus -> $THREADS threads/GPU job, CPU lane $CPU_THREADS"
for f in data/pairs_seed0.pt data/copy_pairs_seed1.pt data/tinystories_seed2.pt data/pairs_symbol_seed0.pt; do
  [[ -f $f ]] || fatal "$f missing"
done
if [[ -z ${NO_TMUX:-} ]]; then command -v tmux >/dev/null || fatal "tmux not installed"; fi

say "fetch saved KL scores"; fetch_scores 2>&1 | tee -a "$LOG"
[[ ${PIPESTATUS[0]} == 0 ]] || fatal "score fetch failed (check SCORES_HF_PREFIX layout)"
for t in elicit_parent elicit_child fmt_parent teach_child base; do
  [[ -f $KLWORD/$t/scores.pt ]] || fatal "$KLWORD/$t/scores.pt missing"
done
for t in elicit_parent elicit_child; do [[ -f $KLSYM/$t/scores.pt ]] || fatal "$KLSYM/$t/scores.pt missing"; done

# queue: a job with a .done marker and its outputs is skipped
: > "$QUEUE"
for j in "${GPU_JOBS[@]}"; do
  if job_done "$j"; then say "skip $j (done)"
  else rm -f "$JOBSTATE/$j.done"; echo "$j" >> "$QUEUE"; fi
done
say "queue: $(wc -l < "$QUEUE") GPU job(s)"

# resolved settings for the worker sessions
{ printf 'export THREADS=%q MODELS_DIR=%q JOB_PY=%q JOB_RETRIES=%q PYTHONPATH=%q REPO_ROOT=%q BATCH_SIZE=%q EVAL_TESTS=%q CPU_THREADS=%q\n' \
    "$THREADS" "$MODELS_DIR" "$JOB_PY" "$JOB_RETRIES" "$PYTHONPATH" "$REPO_ROOT" "$BATCH_SIZE" "$EVAL_TESTS" "$CPU_THREADS"
  for t in "${MODEL_TAGS[@]}"; do printf 'export MODEL_%s=%q\n' "$t" "${MODEL[$t]}"; done
  for v in FAKE_GPUS WAIT_POLL; do [[ -n ${!v:-} ]] && printf 'export %s=%q\n' "$v" "${!v}"; done
  true; } > "$STATE/env.sh"

prefetch_all & PF=$!
cpu_lane >>"$JOBLOGS/cpu_lane.out" 2>&1 & CPU_PID=$!
say "CPU lane started (pid $CPU_PID, $CPU_THREADS threads, niced)"

for ((s = 0; s < N_SLOTS; s++)); do
  name=w$s
  if [[ -n ${NO_TMUX:-} ]]; then
    bash "$HERE/run_box_tiny.sh" --confirm-cost --worker "$name" "${SLOT_GPU[s]}" >>"$RES/worker_$name.out" 2>&1 &
  else
    tmux kill-session -t "${SESSION_PREFIX}_$name" 2>/dev/null || true
    # non-interactive tmux gets no venv: activate it explicitly inside the session
    tmux new -d -s "${SESSION_PREFIX}_$name" "source $VENV/bin/activate 2>/dev/null; cd $HERE && bash run_box_tiny.sh --confirm-cost --worker $name ${SLOT_GPU[s]}; exec bash"
  fi
  say "launched $name on gpu ${SLOT_GPU[s]}"
done

N_ALL=$(( ${#CPU_JOBS[@]} + ${#GPU_JOBS[@]} ))
( while :; do
    sleep "$MON_INTERVAL"
    u=n/a
    if [[ -z ${FAKE_GPUS:-} ]] && command -v nvidia-smi >/dev/null; then
      u=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ';')
    fi
    say "util gpu[idx,%,MiB]: $u load=$(cut -d' ' -f1 /proc/loadavg) jobs done=$(find "$JOBSTATE" -name '*.done' | wc -l)/$N_ALL failed=$(find "$JOBSTATE" -name '*.failed' | wc -l)"
  done ) & SMI=$!

slots_done() { local s; for ((s = 0; s < N_SLOTS; s++)); do [[ -f $STATE/w$s.done ]] || return 1; done; }
all_finished() { local j; for j in "$@"; do finished "$j" || return 1; done; }
# check_jobs <job ...>: log every job that failed or never finished; returns 1 if any
check_jobs() {
  local j bad=0
  for j in "$@"; do
    if [[ -f $JOBSTATE/$j.failed ]]; then say "job $j FAILED"; bad=1
    elif ! job_done "$j"; then say "job $j NOT FINISHED"; bad=1; fi
  done
  return $bad
}

# ---- phase 1: every CPU + phase-1 job finished (or nothing left to run them)
while ! all_finished "${CPU_JOBS[@]}" "${P1_JOBS[@]}"; do
  if slots_done && ! kill -0 "$CPU_PID" 2>/dev/null; then say "workers + CPU lane exited, phase 1 unfinished"; break; fi
  sleep "${MAIN_POLL:-15}"
done
wait "$CPU_PID" || true
bad=0
check_jobs "${CPU_JOBS[@]}" "${P1_JOBS[@]}" || bad=1

# merge cmap shards per route (size order; shards of finished jobs only), then the summary
merged=()
for r in "${ROUTES[@]}"; do
  sh=(); for z in $SIZES_TINY; do job_done "cmap_${r}_$z" && sh+=("$SHARDS/${r}_$z.json"); done
  if ((${#sh[@]})); then
    # shellcheck disable=SC2086
    if $JOB_PY "$HERE/cmap.py" --merge "${sh[@]}" --out "$CMAPD/$r.json" 2>&1 | tee -a "$LOG" && [[ ${PIPESTATUS[0]} == 0 ]]; then
      merged+=("$CMAPD/$r.json"); say "merged $r (${#sh[@]} shards)"
    else say "MERGE FAILED $r"; bad=1; fi
  else say "no shards for $r"; bad=1; fi
done
if ((${#merged[@]})); then
  # shellcheck disable=SC2086
  $JOB_PY "$HERE/cmap.py" --summarize "${merged[@]}" > "$CMAPD/summary.md" 2>>"$LOG" || { say "SUMMARIZE FAILED"; bad=1; }
fi

cp "$LOG" "$RES/run_log.txt"   # the .log itself is excluded from the push (still being written)
if [[ -n ${NO_PUSH:-} ]]; then p1=skipped
elif push_dir "$RES" "$PUSH_SUB" "eapig tiny sizes 0.01-0.05%: overlap, base control, cmap, functional" "tests/*"; then p1=ok
else p1=failed; bad=1; fi
say "phase 1 finished (bad=$bad)"
echo "PHASE1_DONE pushed=$p1" >> "$LOG"

# ---- phase 2: queue drained
until slots_done; do sleep "${MAIN_POLL:-15}"; done
kill "$SMI" 2>/dev/null || true
wait "$PF" || say "model prefetch reported a failure"
check_jobs "${P2_JOBS[@]}" || bad=1

cp "$LOG" "$RES/run_log.txt"
push_dir "$RES" "$PUSH_SUB" "eapig tiny sizes 0.01-0.05%: circuit tests (suff + partial nec)" || bad=1
((bad)) && { say "NOT DONE (job failure, merge failure or push verify failed)"; echo "EXIT=1" >> "$LOG"; exit 1; }
say "ALL DONE"; echo "EXIT=0" >> "$LOG"
