#!/usr/bin/env bash
# ts38 mini — CERTIFIED-CHECKPOINT LoRA parent: selector + builder for the
# owner-pre-authorized fork (decisions.md 2026-08-15 "ts38: owner fork taken
# ... EARLIEST-CERTIFIED-CHECKPOINT parent pre-authorized" entry).
#
# What this is NOT: another sweep or ladder rung. launch_ts38_lora_probe.sh
# MEASURES the 3e-4 LoRA install's full step-by-step curve (one replay,
# snapshots every 1000 steps from 10000-24000, each scored G1+G8 --no-record
# into results/ts38_parent_lora_probe.json). This script SELECTS a step from
# that table and builds the certified parent from it — it never re-derives a
# number the probe already measured, and it moves neither bar.
#
# THE RULE (owner pre-authorized 2026-08-15, verbatim): if the probe finds a
# snapshot passing both bars, promote the EARLIEST step S with g1_pass at S
# AND at S+1000 (persistence — the val curve is noisy at 1k granularity) AND
# g8_pass at S to the parent, by replaying the same config with
# max_steps=S pinned (bit-exact replay; stop_reason=max_steps is then the
# DESIGN, not a bug signal), record G1 -> G8, merge the adapter, and let
# launch_ts38_mini.sh run the family. This bends run-until-convergence FOR
# THE PARENT ONLY (targets/arms/EDL/floors/bars unchanged). If no step
# qualifies -> HALT for the owner (they see the curve first); the chain
# stops here by design — launch_ts38_mini.sh must NOT run in that case.
#
# Selection logic lives in certified_step.py (select_certified_step), a pure
# function over the probe table, so it is unit-tested without a probe run
# (tests/experiments/scripts/test_ts38_certified_parent.py). This script
# imports it via a python heredoc the same way launch_ts38_mini.sh already
# does `from train import load_config` (resolved through cwd=scripts/, set
# by the `cd "$(dirname "$0")"` below — no sys.path surgery needed).
#
# Chain (set -o pipefail when tee'd, so a HALT mid-chain is not masked):
#   bash launch_ts38_lora_probe.sh --confirm-cost &&
#   bash launch_ts38_certified_parent.sh --confirm-cost &&
#   bash launch_ts38_mini.sh --confirm-cost
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38cert

echo "[ts38cert] estimated cost: one deterministic replay of the 3e-4 LoRA"
echo "[ts38cert] parent to the probe-selected step S <= 24000 (~S/16 s ~ 10-25"
echo "[ts38cert] min on a 4090) + G1/G8 recorded (~3 min) + merge/receiver"
echo "[ts38cert] check (~1 min) ~ \$0.15-0.20; disk ~250 MB."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38_certified_parent.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

RELAY_REPO=mhieuuu/geode-store
BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
PARENT_RID=evt-ts38-pretaught-parent
PROBE_RID=evt-ts38-parent-loraprobe-lr3e-4
PROBE_EVAL_LOG=$GEODE_STORE/runs/$PROBE_RID/eval_log.jsonl

PARENT_CONFIG=../configs/ts38_pretaught_parent_lora.yaml
OVERLAY_DIR=../configs/sweeps/ts38
RUN1_CONFIG=../configs/archive/runs/run1_pretrain.yaml
VAL_CACHE=$GEODE_STORE/cache/run1_val_stream.pt
FAILED_DIR=$GEODE_STORE/runs-failed
PROBE_JSON=$GEODE_STORE/results/ts38_parent_lora_probe.json

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE parent=$PARENT_RID probe_json=$PROBE_JSON"

# ---- inline helpers, copied VERBATIM from launch_ts38_lora_parent.sh ------
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

gate_score() { sed -n "s/.*$2 $3 \([0-9.]*\) on n=.*/\1/p" <<<"$1" | head -1; }
gate_passed() { grep -q "$2 $3 .* -> PASS" <<<"$1"; }

gate_verdict() {
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if not p.is_file():
    print("missing")
    sys.exit()
try:
    data = json.loads(p.read_text())
except FileNotFoundError:
    print("missing")
    sys.exit()
except (json.JSONDecodeError, OSError):
    print("corrupt")
    sys.exit()
gates = data.get("experiment", {}).get("gates", {})
g = gates.get(sys.argv[2])
if g is None:
    print("missing")
elif isinstance(g, dict) and g.get("pass") is True:
    print("pass")
else:
    print("fail")
PY
}

enforce_gate() {
  local rid=$1 gate=$2 metric=$3 threshold=$4 verdict out score
  shift 4
  verdict=$(gate_verdict "$rid" "$gate")
  if [[ $verdict == pass ]]; then
    milestone "gate_skip run=$rid gate=$gate status=recorded_pass"
    return 0
  elif [[ $verdict != missing ]]; then
    fail "$rid gate=$gate verdict='$verdict' (expected pass or missing) — inspect $GEODE_STORE/runs/$rid/manifest.json before rerunning"
  fi
  out=$("$@" --no-record 2>&1)
  echo "$out"
  score=$(gate_score "$out" "$gate" "$metric")
  [[ -n $score ]] || fail "$rid $gate printed no $metric score — it errored rather than scored (output above)"
  gate_passed "$out" "$gate" "$metric" || fail \
    "$rid $gate $metric=$score misses the bar $threshold — NOT recorded, so no target
   run can start from an ungated parent.
   $FALLBACK_MSG"
  "$@" --record-only-pass ||
    fail "$rid $gate DIVERGENCE: --no-record scored $score (PASS) but the recording pass recomputed a FAIL — nothing was written; rerun the gate block"
  milestone "gate_pass run=$rid gate=$gate $metric=$score"
}

# ---- stage 0: preflight (idempotent — pulls only what's missing) ----------
if [[ ! -f $BASE_MODEL/model.safetensors ]]; then
  milestone "pull_base_start run=$BASE_RID"
  python3 hf_checkpoint.py pull --run-id "$BASE_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $BASE_RID from $RELAY_REPO"
fi
[[ -f $BASE_MODEL/model.safetensors ]] ||
  fail "pull left no $BASE_MODEL/model.safetensors (snapshot_download is fail-open)"
[[ -f $GEODE_STORE/runs/$BASE_RID/manifest.json ]] ||
  fail "pull left no manifest for $BASE_RID — geode.zoo cannot load a run without it"

if [[ ! -f $VAL_CACHE ]]; then
  milestone "pull_val_cache_start"
  python3 - "$GEODE_STORE" <<'PY' || fail "pull $VAL_CACHE from the relay"
import sys

from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="mhieuuu/geode-store",
    repo_type="model",
    filename="cache/run1_val_stream.pt",
    local_dir=sys.argv[1],
)
PY
  milestone "val_cache_pulled"
fi
[[ -f $VAL_CACHE ]] || fail "pull left no $VAL_CACHE (snapshot_download is fail-open)"

[[ -f $PROBE_JSON ]] || fail \
  "no probe table at $PROBE_JSON — run launch_ts38_lora_probe.sh --confirm-cost
   first (this launcher selects, the probe measures)"

# ---- stage 1: select (prints the full probe table, then the decision) -----
SELECT_OUT=$(python3 - "$PROBE_JSON" <<'PY'
import json
import sys
from pathlib import Path

from certified_step import select_certified_step

rows = json.loads(Path(sys.argv[1]).read_text())
by_step = {r["step"]: r for r in rows}
for step in sorted(by_step):
    r = by_step[step]
    print(
        f"step={step} val={r.get('val_loss_nats')} g1={r['g1_accuracy']} "
        f"g1_pass={r['g1_pass']} g8={r['g8_val_loss_nats']} g8_pass={r['g8_pass']} "
        f"both_pass={r['both_pass']}"
    )

result = select_certified_step(rows)
s = result["step"]
first_g1 = result["first_g1_pass_step"]
last_g8 = result["last_g8_pass_step"]
both = ",".join(str(x) for x in result["both_pass_steps"]) or "none"
print(
    f"SUMMARY first_g1_pass_step={first_g1 if first_g1 is not None else 'none'} "
    f"last_g8_pass_step={last_g8 if last_g8 is not None else 'none'} "
    f"both_pass_steps={both}"
)


def _field(step, key):
    r = by_step.get(step)
    return r.get(key) if r is not None else None


fields = [
    s if s is not None else "NONE",
    _field(s, "g1_accuracy") if s is not None else None,
    _field(s + 1000, "g1_accuracy") if s is not None else None,
    _field(s, "g8_val_loss_nats") if s is not None else None,
    _field(s, "val_loss_nats") if s is not None else None,
    first_g1 if first_g1 is not None else "none",
    last_g8 if last_g8 is not None else "none",
    both,
]
print("FIELDS", *fields)
PY
)
echo "$SELECT_OUT"
FIELDS_LINE=$(grep '^FIELDS ' <<<"$SELECT_OUT")
[[ -n $FIELDS_LINE ]] || fail "select_certified_step printed no FIELDS line — inspect output above"
read -r _ S G1_AT_S G1_AT_S_NEXT G8_AT_S VAL_AT_S FIRST_G1_PASS LAST_G8_PASS BOTH_PASS_STEPS <<<"$FIELDS_LINE"

if [[ $S == NONE ]]; then
  fail "NO CERTIFIED STEP in $PROBE_JSON: first_g1_pass_step=$FIRST_G1_PASS
   last_g8_pass_step=$LAST_G8_PASS both_pass_steps=$BOTH_PASS_STEPS — HOLD for the
   owner (they asked to see the curve before any other fork; no bar moves). The
   chain stops here by design; launch_ts38_mini.sh must NOT run."
fi

milestone "certified_step S=$S g1=$G1_AT_S g1_next=$G1_AT_S_NEXT g8=$G8_AT_S val=$VAL_AT_S"

OVERLAY=$OVERLAY_DIR/parent_lora_certified_s${S}.yaml
[[ -f $OVERLAY ]] || fail \
  "no overlay $OVERLAY for the selected step S=$S — the 15 sibling overlays
   (parent_lora_certified_s10000.yaml .. s24000.yaml) must exist before this
   launcher runs"

# ---- stage 2: train (pinned, bit-exact replay to the certified step) ------
if [[ $(status_of "$PARENT_RID") == complete ]]; then
  EXIST_OUT=$(python3 - "$PARENT_RID" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "training_meta.json"
d = json.loads(p.read_text()) if p.is_file() else {}
print(d.get("stop_reason"), d.get("final_step"))
PY
)
  read -r EXIST_STOP EXIST_STEP <<<"$EXIST_OUT"
  [[ $EXIST_STOP == max_steps && $EXIST_STEP == "$S" ]] || fail \
    "runs/$PARENT_RID exists from a different horizon (stop_reason=$EXIST_STOP
   final_step=$EXIST_STEP, expected max_steps/$S) — inspect/archive it, this
   script deletes nothing."
fi

train_or_skip "$PARENT_RID" \
  python3 train_sft.py --config "$PARENT_CONFIG" \
    --override "$OVERLAY" \
    --init-from "$BASE_MODEL" --confirm-cost

POST_OUT=$(python3 - "$PARENT_RID" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "training_meta.json"
d = json.loads(p.read_text())
print(d.get("stop_reason"), d.get("final_step"), d.get("min_val_nats"))
PY
)
read -r POST_STOP POST_STEP POST_MINVAL <<<"$POST_OUT"
[[ $POST_STOP == max_steps && $POST_STEP == "$S" ]] || fail \
  "$PARENT_RID training_meta.json stop_reason=$POST_STOP final_step=$POST_STEP
   (expected max_steps/$S) — inspect before trusting this checkpoint as the
   certified parent."
milestone "certified_replay_done step=$S stop_reason=max_steps min_val=$POST_MINVAL"

# Replay-fidelity: this replay vs. the probe run it is pinned to (WARN, never
# fails — the recorded gates below are what count).
FIDELITY_OUT=$(python3 - "$PARENT_RID" "$PROBE_EVAL_LOG" "$S" <<'PY'
import json
import os
import sys
from pathlib import Path

rid, probe_log_path, s_max = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
run_dir = Path(os.environ["GEODE_STORE"]) / "runs" / rid
parent_evals = {
    r["step"]: r["val_loss_nats"]
    for r in (
        json.loads(line)
        for line in (run_dir / "eval_log.jsonl").read_text().splitlines()
        if line.strip()
    )
    if "val_loss_nats" in r
}
probe_evals = {}
if probe_log_path.is_file():
    probe_evals = {
        r["step"]: r["val_loss_nats"]
        for r in (
            json.loads(line) for line in probe_log_path.read_text().splitlines() if line.strip()
        )
        if "val_loss_nats" in r
    }
shared = sorted(s for s in (set(parent_evals) & set(probe_evals)) if s <= s_max)
max_abs = max((abs(parent_evals[s] - probe_evals[s]) for s in shared), default=0.0)
print(len(shared), f"{max_abs:.6e}")
PY
)
read -r FIDELITY_N FIDELITY_MAXABS <<<"$FIDELITY_OUT"
milestone "certified_replay_fidelity shared_steps=$FIDELITY_N max_abs_dval=$FIDELITY_MAXABS"
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 1e-6 else 1)" "$FIDELITY_MAXABS" &&
  echo "[ts38cert] WARN replay drifted from the probe run beyond 1e-6 (max |Δval|=$FIDELITY_MAXABS over $FIDELITY_N shared steps) — continuing, the recorded gates below are what count"

# ---- stage 3: gates, RECORDED, G1 then G8 ----------------------------------
FALLBACK_MSG="CERTIFIED PARENT G1 FAIL at S=$S: the probe scored this step as a
   pass — replay/GPU numerics differ or the probe table is stale; inspect
   eval_log vs the probe run. HOLD (owner). No bar moves."
enforce_gate "$PARENT_RID" G1 accuracy 0.95 \
  python3 gates.py g1 --run "$PARENT_RID" --config "$PARENT_CONFIG" --threshold 0.95

G8_OUT=$(python3 gates.py g8 --run "$PARENT_RID" --config "$RUN1_CONFIG" --bar 1.1718 \
  --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --no-record 2>&1)
echo "$G8_OUT"
G8_SCORE=$(gate_score "$G8_OUT" G8 val_loss_nats)
[[ -n $G8_SCORE ]] ||
  fail "$PARENT_RID G8 printed no val_loss_nats at S=$S — it errored rather than scored (output above)"

if gate_passed "$G8_OUT" G8 val_loss_nats; then
  python3 gates.py g8 --run "$PARENT_RID" --config "$RUN1_CONFIG" --bar 1.1718 \
    --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --record-only-pass ||
    fail "$PARENT_RID G8 DIVERGENCE: --no-record scored $G8_SCORE (PASS) but the recording pass recomputed a FAIL — nothing written; rerun the gate block"
  milestone "gate_pass run=$PARENT_RID gate=G8 val_loss_nats=$G8_SCORE"
else
  mkdir -p "$FAILED_DIR"
  mv "$GEODE_STORE/runs/$PARENT_RID" "$FAILED_DIR/${PARENT_RID}-cert-s${S}" ||
    fail "could not archive $PARENT_RID to $FAILED_DIR/${PARENT_RID}-cert-s${S}"
  fail "CERTIFIED PARENT G8 FAIL at S=$S (g8=$G8_SCORE) — probe said pass; inspect. HOLD (owner)."
fi

# Recorded values, for the notify/terminal lines below (manifest, not the
# --no-record printouts, so a resumed invocation reads the same numbers).
GATE_VALUES=$(python3 - "$PARENT_RID" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
gates = json.loads(p.read_text()).get("experiment", {}).get("gates", {})
print(gates["G1"]["accuracy"], gates["G8"]["val_loss_nats"])
PY
)
read -r FINAL_G1 FINAL_G8 <<<"$GATE_VALUES"

# ---- stage 4: merge for hand-off + receiver check (ALWAYS runs) -----------
MERGED_DIR=$GEODE_STORE/runs/$PARENT_RID/model_merged
if [[ -f $MERGED_DIR/model.safetensors ]]; then
  milestone "merge_skip $MERGED_DIR already exists"
elif [[ -d $MERGED_DIR ]]; then
  fail "$MERGED_DIR exists but holds no model.safetensors — a crashed merge
   (merge_adapter.py refuses to overwrite an existing directory). Remove $MERGED_DIR and
   rerun this launcher."
else
  python3 merge_adapter.py --run-id "$PARENT_RID" || fail "merge_adapter.py failed for $PARENT_RID"
  [[ -f $MERGED_DIR/model.safetensors ]] ||
    fail "merge_adapter.py exited 0 but left no $MERGED_DIR/model.safetensors"
  milestone "merge_complete $MERGED_DIR"
fi

# Receiver check, always (specs/00 V0.9 lineage): wrapped-vs-merged logits
# must agree to <1e-3 on seeded random token batches — the only proof that
# folding A/B into the base weights preserved the function.
RECEIVER_OUT=$(python3 - "$PARENT_RID" "$MERGED_DIR" <<'PY'
import os
import sys
from pathlib import Path

import torch
from transformers import LlamaForCausalLM

from geode.zoo import load_model

run_id, merged_dir = sys.argv[1], Path(sys.argv[2])
dev = "cuda" if torch.cuda.is_available() else "cpu"
store = Path(os.environ["GEODE_STORE"])

wrapped = load_model(run_id, store=store, device=dev)
merged = LlamaForCausalLM.from_pretrained(merged_dir).to(dev).eval()
vocab = merged.config.vocab_size

generator = torch.Generator().manual_seed(316)
max_abs_diff = 0.0
with torch.no_grad():
    for _ in range(4):
        ids = torch.randint(0, vocab, (8, 32), generator=generator).to(dev)
        w_logits = wrapped(ids).logits.float()
        m_logits = merged(ids).logits.float()
        max_abs_diff = max(max_abs_diff, (w_logits - m_logits).abs().max().item())

print(f"[ts38cert] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
sys.exit(0 if max_abs_diff < 1e-3 else 1)
PY
)
RECEIVER_STATUS=$?
echo "$RECEIVER_OUT"
[[ $RECEIVER_STATUS -eq 0 ]] || fail \
  "receiver check FAILED — merged checkpoint does not reproduce the wrapped model's logits
   within 1e-3 (see output above); do NOT hand this off to Arm B"
MAX_DIFF=$(sed -n 's/.*max_abs_logit_diff=\(.*\)/\1/p' <<<"$RECEIVER_OUT")
milestone "parent_merged max_abs_logit_diff=$MAX_DIFF"

# ---- stage 5: push (best-effort — WARN, never fail) ------------------------
if python3 hf_checkpoint.py push --run-id "$PARENT_RID" --repo-id mhieuuu/geode-store; then
  milestone "push_complete run=$PARENT_RID"
else
  echo "[ts38cert] WARN hf_checkpoint.py push failed for $PARENT_RID (best effort)"
  milestone "push_warn run=$PARENT_RID"
fi

PUSH_RECEIVER_OUT=$(python3 - "$PARENT_RID" <<'PY'
import sys

from huggingface_hub import HfApi

rid = sys.argv[1]
files = sorted(
    f
    for f in HfApi().list_repo_files("mhieuuu/geode-store", repo_type="model")
    if f.startswith(f"runs/{rid}/")
)
for f in files:
    print(f"  OK {f}")
print(f"COUNT {len(files)}")
PY
)
echo "[ts38cert] receiver check (hub files under runs/$PARENT_RID/):"
echo "$PUSH_RECEIVER_OUT"
PUSH_COUNT=$(grep '^COUNT ' <<<"$PUSH_RECEIVER_OUT" | awk '{print $2}')
milestone "parent_pushed run=$PARENT_RID hub_files=${PUSH_COUNT:-0}"

notify "ts38cert done: parent=$PARENT_RID step=$S g1=$FINAL_G1 g8=$FINAL_G8 merged+receiver-verified"
echo "[ts38cert] TERMINAL_SUCCESS parent=$PARENT_RID step=$S g1=$FINAL_G1 g8=$FINAL_G8"
