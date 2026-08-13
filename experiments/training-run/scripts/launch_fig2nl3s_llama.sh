#!/usr/bin/env bash
# fig2nl3s — SNAPSHOT re-run of the fig2nl3 ENDPOINT pair (n=1,000,000 only,
# owner 2026-08-13: trajectory evidence for the final run of each Fig-2
# curve, both Llama arms; the TinyStories pair follows in its own family).
# Identical training to the shipped runs (same base configs, data, schedule,
# seed 316, same installer parent) under NEW run ids, with 128 adapter
# snapshots per run. The shipped fig2nl3 runs stay immutable; this family
# exists ONLY to produce weight-trajectory evidence (gradient/update
# directions derive from snapshot deltas — the runs-7/8 protocol, which fed
# the existing alignment analyses).
#
# STORE-IN-HF-THEN-DELETE (owner): after each run completes, its whole run
# dir INCLUDING snapshots/ is pushed to a PER-RUN private HF repo
# ($HF_NAMESPACE/<run_id>), the main checkpoint is sha256-verified against
# the hub, and then snapshots + full weights are deleted locally (adapter
# sidecar, manifest, and logs kept). Footprint: ~95 GB per run (128 x
# 0.72 GB adapters) — ~190 GB on HF for this pair, peak local disk ~100 GB.
#
# NO gates, NO figure: EDL/G5 evidence lives in the shipped fig2nl3 family.
# The inst arm's parent is the EXISTING evt-llama-fig2nl3-installer (gates
# recorded); require_parent_ready enforces it as usual. train_target's G7
# check pins each inst run to the SAME-SIZE fig2nl3s noinst run.
#
# Cost: ~2 h GPU (the two n=1M runs re-train in ~50 min each + snapshot
# I/O) + ~190 GB upload (inline between runs).
#
# Token: $HF_WRITE_TOKEN or $HF_TOKEN must have WRITE scope (repo creation).
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=fig2nl3s

echo "[fig2nl3s] estimated cost: ~2 h GPU + ~190 GB upload (two per-run"
echo "[fig2nl3s] private HF repos); peak local disk ~100 GB."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_fig2nl3s_llama.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
export HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?need HF_TOKEN or HF_WRITE_TOKEN (write scope)}}
HF_NAMESPACE=${HF_NAMESPACE:-mhieuuu}

INSTALLER_RID=evt-llama-fig2nl3-installer   # REUSED from the shipped family
INSTALLER_MODEL=$GEODE_STORE/runs/$INSTALLER_RID/model_merged
# n=1,000,000 ONLY (owner 2026-08-13): snapshots for the ENDPOINT run of
# each Fig-2 curve — the two Llama curves here; the two TinyStories curves
# get the same treatment in their own family's launcher.
SIZES=(1000000)

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE namespace=$HF_NAMESPACE sizes=${#SIZES[@]}"

[[ -f $INSTALLER_MODEL/model.safetensors ]] ||
  fail "no merged fig2nl3 installer at $INSTALLER_MODEL — the snapshot re-run reuses the shipped installer; restore it (or re-run launch_fig2nl3_llama.sh through its installer stage) first"

train_or_skip() {
  local rid=$1 status
  shift
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    milestone "train_skip run=$rid status=complete"
    return 0
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  fi
  milestone "train_start run=$rid"
  "$@" || fail "$rid training"
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_complete run=$rid"
}

# Push one completed run WITH its snapshots to its own private HF repo, then
# reclaim local disk: snapshots/ and model/model.safetensors go; the adapter
# sidecar, manifest, and logs stay. Resume-safe: a run whose snapshots dir is
# already gone was pushed on a previous pass — skip silently.
push_and_prune() {
  local rid=$1 run_dir=$GEODE_STORE/runs/$1
  if [[ ! -d $run_dir/snapshots ]]; then
    milestone "push_skip run=$rid (snapshots already pushed+pruned)"
    return 0
  fi
  milestone "push_start run=$rid repo=$HF_NAMESPACE/$rid"
  python3 hf_checkpoint.py push --run-id "$rid" --with-snapshots \
    --repo-id "$HF_NAMESPACE/$rid" || fail "$rid snapshot push"
  # Deleting ~95 GB on the strength of an upload demands verification first:
  # sha256-compare the main checkpoint against its hub copy (snapshot files
  # are integrity-checked per-file by the hub's LFS upload path itself).
  python3 - "$rid" "$HF_NAMESPACE/$rid" <<'PY' || fail "$rid hub sha256 verify after push — NOT pruning; inspect before rerunning"
import os, sys
from pathlib import Path
from hf_checkpoint import verify_hub_checkpoint

verify_hub_checkpoint(Path(os.environ["GEODE_STORE"]), sys.argv[1], repo_id=sys.argv[2])
print(f"[fig2nl3s] hub sha256 verified for {sys.argv[1]}")
PY
  rm -rf "$run_dir/snapshots"
  rm -f "$run_dir/model/model.safetensors"
  milestone "pruned run=$rid (snapshots + full weights local; adapter/manifest/logs kept)"
}

for n in "${SIZES[@]}"; do
  rid=evt-llama-fig2nl3s-noinst-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config ../configs/llama_fig2nl3_noinst.yaml \
      --override "../configs/sweeps/llama_fig2nl3s/llama_fig2nl3s_noinst_n${n}.yaml" \
      --init-from meta-llama/Llama-3.2-1B --confirm-cost
  push_and_prune "$rid"

  rid=evt-llama-fig2nl3s-inst-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config ../configs/llama_fig2nl3_inst.yaml \
      --override "../configs/sweeps/llama_fig2nl3s/llama_fig2nl3s_inst_n${n}.yaml" \
      --init-from "$INSTALLER_MODEL" --confirm-cost
  push_and_prune "$rid"
done

notify "fig2nl3s snapshot re-run done"
echo "[fig2nl3s] TERMINAL_SUCCESS all runs pushed to $HF_NAMESPACE/<run_id> and pruned locally"
