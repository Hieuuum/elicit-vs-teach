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
# STREAM-TO-HF-THEN-DELETE (owner: "after every snap, not after the whole
# run"): stream_snapshots.py runs BESIDE training and uploads each snapshot
# to a PER-RUN public HF repo ($HF_NAMESPACE/<run_id>) as soon as a newer
# snapshot proves it fully written, sha256-verifies it against the hub's
# LFS record, and deletes it locally. After training the streamer drains
# the tail (incl. snapshots/base/), then the run's metadata + weights are
# pushed and the local full weights dropped (adapter/manifest/logs kept).
# Footprint: ~95 GB per run ON HF (128 x 0.72 GB adapters), ~190 GB for the
# pair; peak LOCAL disk only a few snapshots (~5-10 GB).
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
echo "[fig2nl3s] public HF repos); peak local disk ~100 GB."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_fig2nl3s_llama.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
export HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?need HF_TOKEN or HF_WRITE_TOKEN (write scope)}}
# Account split (owner 2026-08-13): mhieuuu (teammate) owns the geode-store
# relay + the arith dataset repo; bulk NEW storage (per-run snapshot repos)
# goes under the owner's own account.
HF_NAMESPACE=${HF_NAMESPACE:-podhajskimarcin}

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

# Train one run with the snapshot STREAMER beside it (owner: "delete locally
# after every snap, not after the whole run"): stream_snapshots.py uploads
# each snapshot to the per-run repo, sha256-verifies it, and deletes it
# locally as soon as a newer snapshot proves it complete. Local disk stays
# ~flat (a couple of snapshots + the model). After training: touch the DONE
# marker so the streamer drains the tail (incl. snapshots/base/), wait for
# it, then push the rest of the run dir (plain push — snapshots are skipped
# by default and are already up), verify the main checkpoint's hub sha, and
# drop the local full weights.
train_streamed() {
  local rid=$1 init=$2 cfg=$3 overlay=$4
  local run_dir=$GEODE_STORE/runs/$rid marker
  marker=$run_dir.stream-done
  rm -f "$marker"
  python3 stream_snapshots.py --run-id "$rid" --repo-id "$HF_NAMESPACE/$rid" \
    --done-marker "$marker" > "stream_${rid}.log" 2>&1 &
  local stream_pid=$!
  milestone "streamer_start run=$rid pid=$stream_pid repo=$HF_NAMESPACE/$rid"

  train_or_skip "$rid" \
    python3 train_target.py --config "$cfg" --override "$overlay" \
      --init-from "$init" --confirm-cost

  touch "$marker"
  wait "$stream_pid" || fail "$rid snapshot streamer failed — see stream_${rid}.log; snapshots NOT fully archived, nothing further pruned"
  rm -f "$marker"
  milestone "streamer_done run=$rid (see stream_${rid}.log)"

  milestone "push_start run=$rid repo=$HF_NAMESPACE/$rid (metadata + weights; snapshots already streamed)"
  python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$HF_NAMESPACE/$rid" --public ||
    fail "$rid push"
  python3 - "$rid" "$HF_NAMESPACE/$rid" <<'PY' || fail "$rid hub sha256 verify after push — NOT pruning; inspect before rerunning"
import os, sys
from pathlib import Path
from hf_checkpoint import verify_hub_checkpoint

verify_hub_checkpoint(Path(os.environ["GEODE_STORE"]), sys.argv[1], repo_id=sys.argv[2])
print(f"[fig2nl3s] hub sha256 verified for {sys.argv[1]}")
PY
  rm -f "$run_dir/model/model.safetensors"
  milestone "pruned run=$rid (full weights local; adapter/manifest/logs kept)"
}

for n in "${SIZES[@]}"; do
  train_streamed "evt-llama-fig2nl3s-noinst-n${n}" meta-llama/Llama-3.2-1B \
    ../configs/llama_fig2nl3_noinst.yaml \
    "../configs/sweeps/llama_fig2nl3s/llama_fig2nl3s_noinst_n${n}.yaml"

  train_streamed "evt-llama-fig2nl3s-inst-n${n}" "$INSTALLER_MODEL" \
    ../configs/llama_fig2nl3_inst.yaml \
    "../configs/sweeps/llama_fig2nl3s/llama_fig2nl3s_inst_n${n}.yaml"
done

notify "fig2nl3s snapshot re-run done"
echo "[fig2nl3s] TERMINAL_SUCCESS all runs pushed to $HF_NAMESPACE/<run_id> and pruned locally"
