#!/usr/bin/env bash
# Llama-3.2-1B EARLY-LEARNING PROBE — matched pair over the FIRST 100 target
# examples, differing ONLY in whether the run-9 format installer precedes the
# target stage:
#   noinst (elicit) : base Llama-3.2-1B          -> D_target[:100]
#   inst   (teach)  : run-9 installer merged     -> D_target[:100]
# n=100, batch 4 -> 25 steps, ONE pass, NO convergence stop (stop_reason=
# max_steps is intentional). A val point every 4 examples (n=4,8,...,100).
# This is a learning-CURVE probe, NOT an EDL measurement — do not report the
# result as an EDL ratio. Runs ON THE BOX only (budget rule: --confirm-cost).
#
# Token: huggingface_hub reads $HF_TOKEN directly. It must belong to an
# account that (a) accepted the Meta Llama-3.2 license (gated pull of the base
# + tokenizer) and (b) has write access to the relay for the push. If a
# separate write token is shipped as $HF_WRITE_TOKEN it is used for push only;
# otherwise push falls back to $HF_TOKEN.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_llama_probe100.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
ARM=all
STAGE=all
PUSH=0
PREV=""
for arg in "$@"; do
  [[ $PREV == --arm ]] && ARM=$arg
  [[ $PREV == --stage ]] && STAGE=$arg
  [[ $arg == --push ]] && PUSH=1
  PREV=$arg
done
case $ARM in all | noinst | inst) ;;
*) echo "--arm must be all|noinst|inst, got '$ARM'" >&2; exit 1 ;;
esac
case $STAGE in all | train | push) ;;
*) echo "--stage must be all|train|push, got '$STAGE'" >&2; exit 1 ;;
esac
[[ $STAGE != push || $PUSH == 1 ]] || {
  echo "launch_llama_probe100.sh: --stage push requires --push" >&2; exit 1; }

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
INSTALLER_RID=evt-run9-llama1b-inst-v2
MERGED=$GEODE_STORE/runs/$INSTALLER_RID/model_merged
NOINST_RID=evt-llama-probe100-noinst
INST_RID=evt-llama-probe100-inst

printf '[probe] MILESTONE repo %s\n' "$(git log --oneline -1)"
echo "[probe] MILESTONE store=$GEODE_STORE arm=$ARM stage=$STAGE push=$PUSH"

fail() { echo "[probe] TERMINAL_FAILURE $1" >&2; exit 1; }

status_of() {
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path
p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

push_run() {
  local rid=$1
  [[ $(status_of "$rid") == complete ]] || fail "cannot push $rid: not complete"
  echo "[probe] MILESTONE push_start run=$rid"
  HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?no HF_TOKEN/HF_WRITE_TOKEN in env}} \
    python3 hf_checkpoint.py push --run-id "$rid" || fail "$rid HF push"
  python3 - "$rid" <<'PY' || exit 1
import os, sys
from pathlib import Path
from huggingface_hub import HfApi
from hf_checkpoint import sha256_of, find_checkpoint

rid = sys.argv[1]
store = Path(os.environ["GEODE_STORE"])
local_ckpt = find_checkpoint(store, rid)
repo_path = f"runs/{rid}/{local_ckpt.relative_to(store / 'runs' / rid).as_posix()}"
info = HfApi().get_paths_info("mhieuuu/geode-store", [repo_path])
remote = info[0].lfs.sha256 if info and info[0].lfs else None
local = sha256_of(local_ckpt)
if remote != local:
    raise SystemExit(f"{rid}: hub sha256 {remote!r} != local {local}")
print(f"[probe] MILESTONE push_verified run={rid} sha256={local}")
PY
}

train_arm() {
  local rid=$1 status
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    echo "[probe] MILESTONE train_skip run=$rid status=complete"; return 0
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  fi
  case $rid in
    "$NOINST_RID")
      echo "[probe] MILESTONE train_start run=$rid arm=noinst init=base-Llama"
      python3 train_target.py --config ../configs/llama_probe100_noinst.yaml \
        --init-from meta-llama/Llama-3.2-1B --confirm-cost || fail "$rid training"
      ;;
    "$INST_RID")
      [[ -f $MERGED/model.safetensors ]] || fail "no merged parent at $MERGED (pull $INSTALLER_RID first)"
      echo "[probe] MILESTONE train_start run=$rid arm=inst init=$INSTALLER_RID/model_merged"
      python3 train_target.py --config ../configs/llama_probe100_inst.yaml \
        --init-from "$MERGED" --confirm-cost || fail "$rid training"
      ;;
  esac
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  echo "[probe] MILESTONE train_complete run=$rid"
}

if [[ $STAGE == all || $STAGE == train ]]; then
  # Gated-access + vocab/span guard before any GPU spend (llama-guide §1).
  python3 verify_llama_tokenizer.py || fail "tokenizer verification"

  # Arm A first (no-inst): its manifest is the G7 match anchor for Arm B.
  if [[ $ARM == all || $ARM == noinst ]]; then
    train_arm "$NOINST_RID"
  fi
  if [[ $ARM == all || $ARM == inst ]]; then
    # Reuse the run-9 installer; pull the full run dir (weights + model_merged/).
    if [[ ! -f $MERGED/model.safetensors ]]; then
      echo "[probe] MILESTONE pull_parent run=$INSTALLER_RID"
      python3 hf_checkpoint.py pull --run-id "$INSTALLER_RID" || fail "pull $INSTALLER_RID"
      [[ -f $MERGED/model.safetensors ]] || fail "pull left no $MERGED/model.safetensors"
    fi
    train_arm "$INST_RID"
  fi
fi

if ((PUSH)); then
  [[ $ARM == all || $ARM == noinst ]] && push_run "$NOINST_RID"
  [[ $ARM == all || $ARM == inst ]] && push_run "$INST_RID"
fi

echo "[probe] MILESTONE analysis_commands"
echo "[probe]   python3 ../analysis/plot_edl_per_token.py --floor test --per example --min-examples 0 --run-id $NOINST_RID --run-id $INST_RID"
[[ -n ${NTFY:-} ]] && curl -sd "llama probe100 done: stage=$STAGE arm=$ARM push=$PUSH" "$NTFY" >/dev/null
echo "[probe] TERMINAL_SUCCESS stage=$STAGE arm=$ARM pushed=$PUSH"
