#!/usr/bin/env bash
# ts1b pf-arm target-stage grid — the "Stage 3+" dataset-size sweep the
# 2026-08-19 pre-registration (decisions.md) left unauthorized, now
# re-confirmed live by the owner for the pf (pre-teach-FORMAT,
# permuted-label) arm ONLY — the matching base-arm (no pre-teach) grid was
# explicitly declined the same session (see EXPERIMENTS.md §19 and the
# decisions.md entry this launcher cross-references). Does the pf parent's
# format-only install reshape the target-task learning curve (paper App.
# E.1.2's causal-intervention question), now measured at 1.24B instead of
# 38.7M?
#
# GATED ON evt-ts1b-pf-parent STATUS==complete. That run has not started
# training as of this commit — the box is still on the parent-stage LR
# mini-sweep (scripts/launch_ts1b_stage12.sh). This script refuses to do
# anything until the parent finishes; re-run it (or just leave a `while`
# outside it, at the operator's discretion — this script itself does not
# poll) once evt-ts1b-pf-parent lands.
#
# THE FIVE RUNS   evt-ts1b-pf-target-n{1000,4642,21544,100000,316228} —
#   LoRA r512/alpha32 on D_algo_bare (scaffold-free NL add/sub), warm-
#   started from evt-ts1b-pf-parent/model (full FT, NO merge stage — see
#   configs/ts1b_pf_target.yaml's header). NO base-arm comparator exists at
#   this scale (owner declined it) — match_data_order_with stays null,
#   permanently, in every config here; the pf curve is read on its own
#   shape (rising/falling), not "above/below base", until a base grid
#   exists.
#
# *** NO LR MINI-SWEEP IN THIS SCRIPT — RETROFITTED OUT 2026-08-19. ***
# This script used to run its own independent 3-rung LR mini-sweep here.
# It no longer does: the elicit-vs-teach design requires the pp-arm and
# pf-arm target stages to be byte-identical except for run_id/
# parent_run_id (arms must differ ONLY in θ0 — same convention as
# ts38pp_pretaught.yaml vs ts38mw_pretaught.yaml at 38M,
# [[feedback-scope-check-pins-before-reuse]]); two independently-tuned
# LRs would silently make the arms differ in more than θ0 and break the
# comparison this whole track exists to make. The ONE shared target-stage
# LR mini-sweep now lives in scripts/launch_ts1b_pp_target_grid.sh (it
# runs there because evt-ts1b-pp-parent finishes training well before
# evt-ts1b-pf-parent even starts — a scheduling choice, not a claim the LR
# "belongs" to the pp arm). That script pins its winner into BOTH
# configs/ts1b_pp_target.yaml's AND this file's paired configs/
# ts1b_pf_target.yaml's train.lr field directly (two regex substitutions,
# one call). This script now just VERIFIES that pin exists (fails loudly,
# telling the operator to run the pp-arm grid first, if it's still the
# unpinned placeholder) rather than deriving its own LR — see the
# "SHARED LR VERIFICATION" gate below, right after the parent checks.
# ts1b_pf_target.yaml's own now-orphaned sweep-overlay files
# (pf_target_lrsweep_*.yaml) are left in place for the historical record
# only; nothing in this script references them any more.
#
# BATCH SIZE 128, NOT the paper's effective 1024 — the paper's own
# per-GPU batch size (Table 3: 128 x 8 GPUs via data parallelism, not
# gradient accumulation). Per [[feedback-paper-fidelity-methodology-not-
# infra-scale]], already the call made identically at the ts1b parent stage
# and every llama_fig2nl* family — do not add 8x accumulation here.
#
# HARD RULES (mirrors every prior launcher in this project):
#   (a) the pf parent is DELIBERATELY UNGATED (no G1/G8 certification —
#       see configs/ts1b_pf_parent.yaml's header); this script never runs
#       gates.py on it at all, and never records anything to its manifest.
#   (b) FULL FT parent, NO MERGE STAGE — runs/evt-ts1b-pf-parent/model IS
#       the checkpoint the target runs warm-start from.
#   (c) NEVER destroy the box — teardown is the operator's call, not taken
#       here.
set -uo pipefail

# ---- env guard: safe HF_TOKEN/NTFY extraction, NEVER source
# /etc/environment whole -- identical fix to launch_ts1b_stage12.sh (see
# that script's header for the PATH-clobber bug this avoids). box_onstart.sh
# writes exactly two vars there (HF_TOKEN, NTFY); extract only those two.
for _v in HF_TOKEN NTFY; do
  if [[ -z ${!_v:-} && -f /etc/environment ]]; then
    _val=$(sed -n "s/^${_v}=//p" /etc/environment | tail -1)
    _val=${_val%\"}
    _val=${_val#\"}
    [[ -n $_val ]] && export "$_v=$_val"
  fi
done
unset _v _val
[[ -f /workspace/venv/bin/activate ]] && . /workspace/venv/bin/activate
python3 - <<'PY' || { echo "[ts1bpftgt] FAILED: python3 lacks required modules (venv not active?)"; exit 1; }
import huggingface_hub, torch, geode  # noqa: F401
PY

cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts1bpftgt

echo "[ts1bpftgt] estimated cost (Stage 3+, owner re-confirmed 2026-08-19, pf-arm only):"
echo "[ts1bpftgt]   LR mini-sweep: NONE here -- shared with the pp-arm grid, run"
echo "[ts1bpftgt]     ONCE by launch_ts1b_pp_target_grid.sh (~\$1-2 there, not"
echo "[ts1bpftgt]     duplicated here). This script only verifies the pin exists."
echo "[ts1bpftgt]   5-size target grid: max_steps ceilings 1000/1000/5000/10000/"
echo "[ts1bpftgt]     30000 (sourced from configs/sweeps/llama_fig2nl3's own"
echo "[ts1bpftgt]     already-run ceilings at this exact model/adapter/lr class)"
echo "[ts1bpftgt]     -- LoRA forward/backward is full-1.24B-cost per step (only"
echo "[ts1bpftgt]     the trainable param count shrinks), so per-step cost is"
echo "[ts1bpftgt]     close to the parent's full-FT rate; total steps across all"
echo "[ts1bpftgt]     5 sizes (47000) is ~1.5x one pf-parent epoch (31093) --"
echo "[ts1bpftgt]     est \$8-15, extrapolated from launch_ts1b_stage12.sh's own"
echo "[ts1bpftgt]     \$5-10-per-31093-full-FT-steps figure."
echo "[ts1bpftgt]   TOTAL est \$8-15 for THIS script (the shared ~\$1-2 sweep is a"
echo "[ts1bpftgt]     one-time cost already counted under the pp-arm grid). Box spec:"
echo "[ts1bpftgt]     same A100/L40S-class already in use."
echo "[ts1bpftgt]   NOT authorized: a base-arm comparator grid (owner declined,"
echo "[ts1bpftgt]     2026-08-19) or any arm/size beyond what's listed here."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts1b_pf_target_grid.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

PARENT_RID=evt-ts1b-pf-parent
PARENT_MODEL_DIR=$GEODE_STORE/runs/$PARENT_RID/model
PARENT_DATA_HASH=731c18bd3c344c7fb12099bf142db9135fb968c3281c2d7b60469f4acd63c664
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
SIZES=(1000 4642 21544 100000 316228)

TARGET_CONFIG=../configs/ts1b_pf_target.yaml
OVERLAY_DIR=../configs/sweeps/ts1b
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_llama.yaml
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE parent=$PARENT_RID sizes=${#SIZES[@]}"

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

require_converged() {
  local rid=$1 n=$2 stop
  stop=$(stop_reason_of "$rid" target_result)
  milestone "convergence_check run=$rid n=$n stop_reason=$stop"
  [[ $stop == converged ]] || fail \
    "CONVERGENCE CHECK: $rid ended with stop_reason='$stop', not 'converged'. Standing
   policy: a max_steps stop is a BUG SIGNAL for a REAL target run (not the deliberately-
   truncated LR-sweep probes above). Inspect the overlay's max_steps and the loss trace
   before continuing; do not let the remaining sizes inherit whatever this is."
}

record_g5() {
  local rid=$1
  if gate_recorded "$rid" G5; then
    milestone "gate_skip run=$rid gate=G5"
    return 0
  fi
  python3 gates.py g5 --run "$rid" --config "$BARE_EVAL_CONFIG" ||
    fail "$rid G5 (evidence recording failed)"
  milestone "gate_recorded run=$rid gate=G5"
}

# ---- gate: the pf parent must exist and be complete -----------------------
P_STATUS=$(status_of "$PARENT_RID")
if [[ $P_STATUS != complete ]]; then
  echo "[ts1bpftgt] $PARENT_RID status='$P_STATUS' (expected complete)."
  echo "[ts1bpftgt] This grid trains on the pf parent's finished checkpoint; there is"
  echo "[ts1bpftgt] nothing to do until scripts/launch_ts1b_stage12.sh's stage 3 lands it."
  echo "[ts1bpftgt] Refusing to proceed. Re-invoke this script after the parent completes."
  exit 1
fi
milestone "parent_gate_check status=$P_STATUS -> proceeding"

# ---- parent verification (mirror launch_ts1b_stage12.sh's own pf checks,
# plus the stale-parent data-order-hash guard every prior family carries) --
PARENT_FIELDS=$(python3 - "$PARENT_RID" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
rid = sys.argv[1]
d = json.loads((store / "runs" / rid / "manifest.json").read_text())
status = d.get("status", "MISSING")
method = d.get("training", {}).get("method", "MISSING")
exp = d.get("experiment", {}) or {}
gates = exp.get("gates", {}) or {}
meta_p = store / "runs" / rid / "training_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
stop_reason = meta.get("stop_reason", "MISSING")
final_step = meta.get("final_step", "MISSING")
data_hash = exp.get("data_order_hash", "MISSING")
gate_names = ",".join(sorted(gates)) if gates else "none"
print(status, method, gate_names, stop_reason, final_step, data_hash)
PY
)
read -r P_STATUS2 P_METHOD P_GATES P_STOP P_STEP P_DATA_HASH <<<"$PARENT_FIELDS"
milestone "parent_fields status=$P_STATUS2 method=$P_METHOD gates=$P_GATES stop_reason=$P_STOP final_step=$P_STEP data_order_hash=$P_DATA_HASH"

[[ $P_METHOD == full_ft ]] || fail "$PARENT_RID training.method='$P_METHOD' (expected full_ft)"
[[ $P_GATES == none ]] || fail "$PARENT_RID has recorded gates {$P_GATES} — deliberately-ungated parent, inspect before proceeding"
[[ $P_DATA_HASH == "$PARENT_DATA_HASH" ]] || fail \
  "$PARENT_RID manifest data_order_hash=$P_DATA_HASH != pin $PARENT_DATA_HASH (configs/
   ts1b_pf_parent.yaml) — this checkpoint was built from a different D_target_4M_blockperm.
   parquet/config than this launcher expects. Do not train the target grid on a stale parent."
[[ -f $PARENT_MODEL_DIR/model.safetensors ]] || fail "$PARENT_MODEL_DIR/model.safetensors missing"
milestone "parent_verified full_ft, ungated, data_order_hash=OK, checkpoint present"

# ---- data preflight: reused frozen corpus, no new datagen -----------------
for f in D_algo_bare.parquet D_algo_eval_bare.parquet; do
  [[ -f $DATA_DIR/$f ]] || fail \
    "$DATA_DIR/$f missing — this grid does not build new data, it reuses the frozen
   D_algo_bare corpus every other family (including llama_fig2nl3) trains on. If this box
   never ran a family that regenerated it, pull it from the relay's cache/ path or run
   datagen/make_bare_sets.py against the already-present D_algo/D_algo_eval."
done
milestone "data_present D_algo_bare + D_algo_eval_bare at pinned local paths"

# ---- SHARED LR VERIFICATION -- no sweep here, just check the pp-arm
# grid's sweep already pinned this file's train.lr (see the header's ***
# NO LR MINI-SWEEP *** note). Refuses loudly rather than guessing or
# silently training on the unpinned placeholder. ----------------------------
python3 - "$TARGET_CONFIG" <<'PY' || fail "shared-LR verification"
import re
import sys
from pathlib import Path

target_config = sys.argv[1]
TAG = "ts1bpftgt"
text = Path(target_config).read_text()
m = re.search(r"(?m)^  lr: ([0-9.eE+-]+).*$", text)
if m is None:
    print(f"[{TAG}] FAILED: no 'train.lr' line found in {target_config} at all -- config is malformed.")
    sys.exit(1)
line = m.group(0)
if "PLACEHOLDER" in line:
    print(f"[{TAG}] FAILED: {target_config}'s train.lr is still the unpinned placeholder:")
    print(f"[{TAG}]   {line.strip()}")
    print(f"[{TAG}] Run scripts/launch_ts1b_pp_target_grid.sh first -- it runs the ONE shared")
    print(f"[{TAG}] target-stage LR mini-sweep for both arms and pins the winner into BOTH")
    print(f"[{TAG}] configs/ts1b_pp_target.yaml AND this file. This script does not run its own")
    print(f"[{TAG}] sweep (retrofitted out 2026-08-19) -- see this file's header.")
    sys.exit(1)
print(f"[{TAG}] MILESTONE shared_lr_verified lr={m.group(1)} config={target_config}")
PY
milestone "lrsweep_complete"

# ---- the five target runs, ascending n, push-as-you-go --------------------
for n in "${SIZES[@]}"; do
  rid=evt-ts1b-pf-target-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config "$TARGET_CONFIG" \
      --override "$OVERLAY_DIR/ts1b_pf_target_n${n}.yaml" \
      --init-from "$PARENT_MODEL_DIR" --confirm-cost
  require_converged "$rid" "$n"
  record_g5 "$rid"
  push_run "$rid" "$RELAY_REPO"
  milestone "size_complete n=$n run=$rid"
done

python3 - "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts1bpftgt] {'run_id':<28}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")
for n in sys.argv[1:]:
    rid = f"evt-ts1b-pf-target-n{n}"
    p = store / "runs" / rid / "manifest.json"
    r = (json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}) or {}
    edl = r.get("edl_per_label_token_nats")
    print(
        f"[ts1bpftgt] {rid:<28}{r.get('final_step', 'MISSING'):>12}"
        f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
        f"{edl if edl is not None else 'n/a':>26}"
    )
PY

# ---- push receiver-verify (feedback-verify-the-receiver-not-the-sender) --
run_receiver_check() {
  python3 - "$RELAY_REPO" "${SIZES[@]}" <<'PY'
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
sizes = sys.argv[2:]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True
for n in sizes:
    rid = f"evt-ts1b-pf-target-n{n}"
    # model.safetensors is the canonical self-contained checkpoint for every
    # run in this project, LoRA included (train_target.py:586, the full
    # state dict, base + adapter tensors together) -- same file the ts38pp/
    # ts1b parent receiver checks verify, not PEFT's separate adapter_model
    # naming.
    required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
    missing = [r for r in required if r not in files]
    status = "OK" if not missing else f"MISSING {missing}"
    print(f"  {rid}: {status}")
    if missing:
        print(f"MISSING {rid}")
        ok = False
sys.exit(0 if ok else 1)
PY
}

RECEIVER_OUT=$(run_receiver_check)
RECEIVER_STATUS=$?
echo "[ts1bpftgt] receiver check (hub files for all 5 target runs):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing whatever the hub is missing)"
  while read -r rid; do
    push_run "$rid" "$RELAY_REPO"
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts1bpftgt] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED -- see output above"
milestone "receiver_verified sizes=${#SIZES[@]}"

notify "ts1b pf-arm target grid DONE: 5/5 sizes converged, pushed, receiver-verified"
echo "[ts1bpftgt] TERMINAL_SUCCESS"
