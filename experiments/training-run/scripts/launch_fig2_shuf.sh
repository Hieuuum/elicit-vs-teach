#!/usr/bin/env bash
# launch_fig2_shuf.sh — the two costs inside EDL/D, calibrated on the toy task (decisions.md
# 2026-10-15; the WMDP side is stage 5 of experiments/unlearning).
#
#   1. the ROTATED-ANSWER TWIN SWEEP: the Figure-2 arms retrained on D_algo_bare_shuf (every
#      answer moved to another item, datagen/make_shuf_sets.py) at a subset of the sizes, same
#      per-size schedule as the original overlay, run ids <prefix>-shuf-n<N>;
#   2. M21 ALONG THE SWEEP: grad_transfer.py --task arith --run-id on every existing sweep point
#      (true and rotated; adapter sidecar + the run's init_from base, merged) and on the parent
#      (n = 0), results/m21_sweep/<run_id>.json;
#   3. analysis/knowledge_sweep.py --family: EDL/D(true) - EDL/D(rotated) with its signature, and
#      M21(n), per arm.
#
# Families (the arms whose outcome is known):
#   ts   evt-ts1b-fig2ts-{noinst,inst}   TinyStories-1B base / format-installed  -> TEACH (the hump)
#   mix  evt-ts1b-mix-nl                 TinyStories-1B op-bridge parent         -> ELICIT (same base)
#   nl3  evt-llama-fig2nl3-{noinst,inst} Llama-3.2-1B base / format-installed    -> ELICIT
# Prediction: the hump of the ts arms survives the subtraction (the facts are taught there); the
# elicit arms stay decreasing; M21(n) rises with n only where the facts are being learned.
#
# Usage:  bash launch_fig2_shuf.sh --family ts[,mix,nl3] --confirm-cost
#                                  [--sizes "1000 4642 21544 100000 316228 1000000"] [--m21-only]
# Needs the family's parents and (for M21 on the true sweep) its run dirs in $GEODE_STORE with
# model/adapter.safetensors (pull the pruned endpoints with hf_checkpoint.py pull). Each rotated run
# costs what the same-size original run cost; the default 6 sizes are ~40% of a family's original
# sweep (ts: ~5-8 h GPU, nl3: ~3-5 h, mix: ~2 h). M21 scoring: ~1 min per run at 1B.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=fig2shuf

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
# conda's libstdc++ before the system one: optree (imported by torch._dynamo) needs CXXABI_1.3.15,
# which the cluster's /lib/x86_64-linux-gnu/libstdc++.so.6 lacks (slurm-187219, 2026-10-16)
[[ -n ${CONDA_PREFIX:-} && -e $CONDA_PREFIX/lib/libstdc++.so.6 ]] && \
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
CONFIGS=../configs
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
M21_DIR=$REPO_ROOT/experiments/training-run/results/m21_sweep
XFER=$REPO_ROOT/experiments/unlearning/grad_transfer.py
TOK=meta-llama/Llama-3.2-1B            # every arm's tokenizer (the ts1b dirs carry none loadable)

FAMILIES=""; SIZES="1000 4642 21544 100000 316228 1000000"; CONFIRM=0; M21_ONLY=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --family) FAMILIES=${2//,/ }; shift ;;
    --sizes) SIZES=$2; shift ;;
    --confirm-cost) CONFIRM=1 ;;
    --m21-only) M21_ONLY=1 ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac; shift
done
[[ -n $FAMILIES ]] || { echo "--family ts|mix|nl3 (comma list) required" >&2; exit 2; }
((CONFIRM)) || { echo "launch_fig2_shuf.sh: --confirm-cost required (budget rule)" >&2; exit 1; }
mkdir -p "$M21_DIR"
milestone "repo $(git log --oneline -1) store=$GEODE_STORE families=[$FAMILIES] sizes=[$SIZES] m21_only=$M21_ONLY"

# ---- the rotated twin of the frozen target set -------------------------------
[[ -f $DATA_DIR/D_algo_bare.parquet ]] || fail "no $DATA_DIR/D_algo_bare.parquet (run a Figure-2 launcher's data stage first)"
if [[ ! -f $DATA_DIR/D_algo_bare_shuf.report.json ]]; then
  python3 ../datagen/make_shuf_sets.py --out "$DATA_DIR" || fail "make_shuf_sets"
fi
SHUF_HASH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['order_hash'])" "$DATA_DIR/D_algo_bare_shuf.report.json")
milestone "rotated set D_algo_bare_shuf.parquet order_hash=$SHUF_HASH"

# arm <family> -> lines "prefix|base config|init checkpoint|overlay pattern (N substituted)"
arms_of() {
  case $1 in
    ts)  echo "evt-ts1b-fig2ts-noinst|$CONFIGS/ts1b_fig2ts_noinst.yaml|$GEODE_STORE/runs/evt-ts1b-base/model|$CONFIGS/sweeps/ts1b_fig2ts/ts1b_fig2ts_noinst_nN.yaml"
         echo "evt-ts1b-fig2ts-inst|$CONFIGS/ts1b_fig2ts_inst.yaml|$GEODE_STORE/runs/evt-ts1b-fig2ts-installer/model|$CONFIGS/sweeps/ts1b_fig2ts/ts1b_fig2ts_inst_nN.yaml" ;;
    mix) echo "evt-ts1b-mix-nl|$CONFIGS/ts1b_mix_nl.yaml|$GEODE_STORE/runs/evt-ts1b-op-bridge-mix/model|$CONFIGS/sweeps/ts1b_mix_nl/ts1b_mix_nl_nN.yaml" ;;
    nl3) echo "evt-llama-fig2nl3-noinst|$CONFIGS/llama_fig2nl3_noinst.yaml|meta-llama/Llama-3.2-1B|$CONFIGS/sweeps/llama_fig2nl3/llama_fig2nl3_noinst_nN.yaml"
         echo "evt-llama-fig2nl3-inst|$CONFIGS/llama_fig2nl3_inst.yaml|$GEODE_STORE/runs/evt-llama-fig2nl3-installer/model_merged|$CONFIGS/sweeps/llama_fig2nl3/llama_fig2nl3_inst_nN.yaml" ;;
    *) fail "unknown family $1" ;;
  esac
}

# the rotated overlay: the original size overlay with the run id, the data file and its pin
# swapped, the G7 data-order match and the endpoint snapshots dropped
shuf_overlay() {  # <orig overlay> <rid> <N> <dst>
  python3 - "$@" "$SHUF_HASH" <<'PY'
import sys, yaml
src, rid, n, dst, h = sys.argv[1:6]
cfg = yaml.safe_load(open(src)) or {}
cfg["run_id"] = rid
cfg.setdefault("data", {}).update({"n_examples": int(n), "file": "D_algo_bare_shuf.parquet", "order_hash": h,
                                   "local_path": "experiments/training-run/data/full/D_algo_bare_shuf.parquet"})
cfg.setdefault("experiment", {})["match_data_order_with"] = None
cfg.get("train", {}).pop("snapshots", None)
open(dst, "w").write(f"# rotated-answer twin of {src.split('/')[-1]} (launch_fig2_shuf.sh, generated)\n" + yaml.safe_dump(cfg, sort_keys=False))
PY
}

train_or_skip() {
  local rid=$1 status; shift
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then milestone "train_skip run=$rid status=complete"; return 0
  elif [[ $status != missing ]]; then fail "$rid exists with status '$status'; inspect it rather than overwriting"; fi
  milestone "train_start run=$rid"
  "$@" || fail "$rid training"
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_complete run=$rid"
}

m21_run() {  # <run id> <train config>: M21 on the eval block at that run's final weights
  local rid=$1 cfg=$2 out=$M21_DIR/$1
  [[ -f $out.json ]] && { milestone "m21_skip run=$rid"; return 0; }
  [[ -f $GEODE_STORE/runs/$rid/model/adapter.safetensors ]] || { milestone "m21_skip run=$rid (no adapter sidecar in the store)"; return 0; }
  python3 "$XFER" --task arith --run-id "$rid" --train-config "$cfg" --tokenizer $TOK --no-per-item \
    --out "$out" --confirm-cost || { milestone "m21_FAILED run=$rid"; return 0; }
  milestone "m21_done run=$rid"
}

for fam in $FAMILIES; do
  while IFS='|' read -r prefix cfg init pattern; do
    [[ -n $prefix ]] || continue
    milestone "arm $prefix (family $fam): config $cfg init $init"
    if ((! M21_ONLY)); then
      [[ -d $init || $init == */* && ! $init == /* ]] || fail "init checkpoint $init missing for $prefix"
      mkdir -p "$CONFIGS/sweeps/${fam}_shuf"
      for n in $SIZES; do
        orig=${pattern//nN/n$n}
        [[ -f $orig ]] || { milestone "no original overlay for n=$n ($orig): size skipped"; continue; }
        rid=$prefix-shuf-n$n
        ov=$CONFIGS/sweeps/${fam}_shuf/$rid.yaml
        shuf_overlay "$orig" "$rid" "$n" "$ov" || fail "overlay for $rid"
        train_or_skip "$rid" python3 train_target.py --config "$cfg" --override "$ov" --init-from "$init" --confirm-cost
      done
    fi
    # M21 at n = 0 (the parent) and at every sweep point in the store, true and rotated
    if [[ ! -f $M21_DIR/$prefix-n0.json ]]; then
      python3 "$XFER" --task arith --init "$init" --train-config "$cfg" --tokenizer $TOK --no-per-item \
        --out "$M21_DIR/$prefix-n0" --confirm-cost || milestone "m21_FAILED run=$prefix-n0"
    fi
    for d in "$GEODE_STORE"/runs/$prefix-n[0-9]* "$GEODE_STORE"/runs/$prefix-shuf-n[0-9]*; do
      [[ -d $d ]] || continue
      [[ $(basename "$d") =~ -n[0-9]+$ ]] || continue       # not the s/s2 snapshot re-runs
      m21_run "$(basename "$d")" "$cfg"
    done
  done < <(arms_of "$fam")
  python3 ../analysis/knowledge_sweep.py --family "$fam" || milestone "knowledge_sweep $fam FAILED"
done
milestone "done: results/knowledge_sweep_<family>.json, analysis/figures/knowledge_sweep_<family>.png"
