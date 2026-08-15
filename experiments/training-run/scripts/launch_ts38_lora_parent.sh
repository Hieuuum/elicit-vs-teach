#!/usr/bin/env bash
# ts38 mini — the LoRA-installed pre-taught parent (EXPERIMENTS §6.14;
# decisions.md 2026-08-15 "ts38 ladder CLOSED ... LoRA-installed parent
# pre-declared"). The full-FT ladder in launch_ts38_mini.sh stage 2 tried
# every LR in {3e-4, 1e-4, 3e-5, 1e-5} and never certified a parent: all four
# rungs passed G1 (capability) but failed G8 (TinyStories retention), and the
# bottom rung converged at G1 0.9404 FAIL / G8 1.1904 FAIL. That is a DESIGN
# RESULT — full FT on arithmetic-only data cannot produce a G1+G8-certified
# parent from the 38.7M base at any plausible LR. This launcher builds the
# pre-declared next step instead: a LoRA r128/alpha32 install of D_target on
# the FROZEN base (train_sft.py's existing own-yaml `lora:` opt-in, no new
# training code) — the frozen weights bound the forgetting that killed every
# full-FT rung.
#
# PRE-REGISTERED PROTOCOL (decisions.md 2026-08-15, LoRA-parent entry; owner
# delegated judgment, [[feedback-lr-sweep-before-full-run]] /
# [[feedback-simplest-experiment-first]] — sweeps SELECT, gates score only
# the full run):
#   1. Short sweep, one epoch each (max_steps 8000), LR descending over
#      {1e-3, 3e-4, 1e-4, 3e-5}, each scored G1 + G8 --no-record into
#      results/ts38_parent_lora_sweep.json. Selector: the HIGHEST LR whose
#      sweep-end G8 <= 1.1718 AND whose val is descending (finite; last eval
#      < first eval). stop_reason=max_steps is EXPECTED here by design (8000
#      is a fixed one-epoch horizon, not a convergence target) — it is only
#      a bug signal on the FULL run below. No qualifying rung -> a SECOND
#      design result (LoRA cannot install without crossing retention either)
#      -> HALT, replay-mixing is the remaining fix (owner call).
#   2. ONE full run (ceiling max_steps 160000, ~20.6 epochs — cost ceiling
#      only, never a planned stop, [[feedback-run-until-convergence]]) at
#      the winner LR, run id evt-ts38-pretaught-parent. G1 -> G8 RECORDED
#      (score --no-record, then --record-only-pass). G1 fail = HALT: the
#      winner is already the highest retention-safe LR the sweep qualified,
#      so no lower LR fixes an undertraining miss (design result #2) — UNLESS
#      training_meta.json's stop_reason is max_steps, which is a bug signal
#      at the 160k ceiling to inspect first, not a result. G8 fail archives
#      the attempt to runs-failed/ and retries ONE fallback at the
#      next-lower LR the sweep qualified (capped at 2 attempts total,
#      including already-archived ones on a resumed invocation — "a single
#      pre-declared fallback, not a ladder"); exhausting that (or having no
#      fallback) is a HALT.
#   3. Merge for hand-off: merge_adapter.py folds the adapter into plain
#      weights at runs/evt-ts38-pretaught-parent/model_merged/ (idempotent
#      here — skipped if it already has model.safetensors); the wrapped
#      model/ stays the gated artifact, loadable only via
#      geode.zoo.load_model
#      ([[feedback-lora-checkpoints-load-via-zoo-load-model]]). A receiver
#      check ALWAYS runs (even when the merge itself was skipped): wrapped
#      vs. merged logits must agree to <1e-3 on seeded random token batches.
#   4. Chained hand-off on the box:
#      `bash launch_ts38_lora_parent.sh --confirm-cost &&
#       bash launch_ts38_mini.sh --confirm-cost`
#      — the mini launcher sees the recorded G1+G8 pass on
#      evt-ts38-pretaught-parent (`ladder_skip parent already gated`, its
#      stage 2) and resolves its Arm B init path from this parent's
#      manifest training.method (deliverable E, the mini launcher's one
#      edit for this hand-off).
#
# NEITHER BAR MOVES (G1 >= 0.95, G8 <= 1.1718 — identical numbers to the
# full-FT ladder). The sweep searches theta_0 via LR only, never the
# measurement.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38lora

echo "[ts38lora] estimated cost: 4 sweep rungs (max_steps 8000 each, one epoch,"
echo "[ts38lora] ~8-12 min incl. G1+G8 --no-record gates) + ONE full run at the"
echo "[ts38lora] winner LR (max_steps 160000 ceiling ~3 h at ~16 steps/s on an"
echo "[ts38lora] RTX 4090 @ ~\$0.33/h; expected stop well before the ceiling, ~1-1.5 h"
echo "[ts38lora] typical) + a merge/receiver-check pass (CPU-cheap). Worst case (a"
echo "[ts38lora] second full-run fallback attempt) adds another ~3 h ceiling."
echo "[ts38lora] Well under \$2 total on rented hardware, \$0 on owned. Disk: ~800 MB-1 GB"
echo "[ts38lora] of fp32 38.7M checkpoints (4 sweep rungs + the full run + model_merged/)."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38_lora_parent.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
PARENT_RID=evt-ts38-pretaught-parent
PARENT_MODEL=$GEODE_STORE/runs/$PARENT_RID/model

PARENT_CONFIG=../configs/ts38_pretaught_parent_lora.yaml
OVERLAY_DIR=../configs/sweeps/ts38
RUN1_CONFIG=../configs/archive/runs/run1_pretrain.yaml
VAL_CACHE=$GEODE_STORE/cache/run1_val_stream.pt
FAILED_DIR=$GEODE_STORE/runs-failed
SWEEP_JSON=$GEODE_STORE/results/ts38_parent_lora_sweep.json

# Descending, mirrors the mini launcher's LADDER array convention.
RUNGS=("1e-3" "3e-4" "1e-4" "3e-5")

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE rungs=${#RUNGS[@]} base=$BASE_RID parent=$PARENT_RID"

# ---- inline helpers, copied VERBATIM from launch_ts38_mini.sh -------------
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

# lr_matches: the live parent slot must belong to the rung this attempt is
# about to score — a checkpoint left by anything else is inspected, never
# deleted or rescored as if it were the rung (mini launcher's stage-2 rule,
# needed here too because the full-run loop reuses PARENT_RID across
# fallback attempts).
lr_matches() {
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
lr = json.loads(p.read_text())["training"]["optimizer"]["lr"]
sys.exit(0 if abs(lr - float(sys.argv[2])) <= 1e-12 else 1)
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

# ---- sweep-JSON helpers ------------------------------------------------
sweep_row_scored() { # lr -> exit 0 iff a row for $1 already carries a G8 score
  python3 - "$SWEEP_JSON" "$1" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
rows = json.loads(p.read_text())
sys.exit(0 if any(r["lr"] == sys.argv[2] and r.get("g8_val_loss_nats") is not None for r in rows) else 1)
PY
}

sweep_record() { # lr run_id stop_reason final_step min_val first_val last_val val_desc(0/1) g1 g8 g8_pass(0/1)
  python3 - "$SWEEP_JSON" "$@" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
lr, run_id, stop_reason, final_step, min_val, first_val, last_val, val_desc, g1, g8, g8_pass = sys.argv[2:13]


def _f(x):
    return None if x in ("None", "nan") else float(x)


rows = [r for r in (json.loads(p.read_text()) if p.is_file() else []) if r["lr"] != lr]
rows.append(
    {
        "lr": lr,
        "run_id": run_id,
        "stop_reason": stop_reason,
        "final_step": None if final_step == "None" else int(final_step),
        "min_val_nats": _f(min_val),
        "first_val_nats": _f(first_val),
        "last_val_nats": _f(last_val),
        "val_descending": val_desc == "1",
        "g1_accuracy": float(g1),
        "g8_val_loss_nats": _f(g8),
        "g8_pass": g8_pass == "1",
        "g8_bar": 1.1718,
    }
)
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(rows, indent=2) + "\n")
PY
}

sweep_field() { # lr field -> the row's field value (or empty if no such row)
  python3 - "$SWEEP_JSON" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path

p = Path(sys.argv[1])
rows = json.loads(p.read_text()) if p.is_file() else []
for r in rows:
    if r["lr"] == sys.argv[2]:
        print(r.get(sys.argv[3], ""))
        break
PY
}

# ---- pre-flight -------------------------------------------------------
[[ -f $BASE_MODEL/model.safetensors ]] ||
  fail "no base checkpoint at $BASE_MODEL — run launch_ts38_mini.sh stage 0 (relay pull) first"
[[ $(status_of "$BASE_RID") == complete ]] ||
  fail "$BASE_RID is not status=complete — inspect before spending on the LoRA parent sweep"

[[ -f $VAL_CACHE ]] || fail \
  "no G8 validation-stream cache at $VAL_CACHE — build it via launch_ts38_mini.sh stage 1's
   G8 preflight (a ~30-60 min CPU pack that only needs to happen once):
   gates.py g8 --run $BASE_RID --config $RUN1_CONFIG --bar 1.1718 \\
     --tokenizer ../tokenizer --val-cache $VAL_CACHE --no-record"

# All nine config files this launcher will need, checked BEFORE any GPU spend
# — a missing full-run overlay discovered only after the sweep would throw
# away ~45 min of GPU for nothing.
CONFIG_FILES=("$PARENT_CONFIG")
for LR in "${RUNGS[@]}"; do
  CONFIG_FILES+=("$OVERLAY_DIR/parent_lora_sweep_lr_${LR}.yaml" "$OVERLAY_DIR/parent_lora_lr_${LR}.yaml")
done
MISSING_CFG=()
for f in "${CONFIG_FILES[@]}"; do
  [[ -f $f ]] || MISSING_CFG+=("$f")
done
((${#MISSING_CFG[@]} == 0)) || fail \
  "missing config file(s) before any GPU spend: ${MISSING_CFG[*]} — the fixed config names
   this launcher expects (see header) must exist first."

# The parent slot: missing is fine (fresh start); a complete LoRA run there is
# a legal resume point for the full-run stage below; anything else (a
# leftover full-FT run, a partial/running run) must be triaged by a human,
# never silently deleted or rescored.
PARENT_SLOT_STATUS=$(status_of "$PARENT_RID")
if [[ $PARENT_SLOT_STATUS != missing ]]; then
  PARENT_SLOT_METHOD=$(python3 - "$PARENT_RID" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
d = json.loads(p.read_text())
print(d.get("training", {}).get("method", ""))
PY
)
  if [[ $PARENT_SLOT_STATUS == complete && $PARENT_SLOT_METHOD == lora ]]; then
    milestone "parent_slot_resume run=$PARENT_RID status=complete method=lora"
  else
    fail "runs/$PARENT_RID exists (status=$PARENT_SLOT_STATUS method=$PARENT_SLOT_METHOD) —
   not a complete LoRA run. Archive it to runs-failed/ first; the ladder's slot must be
   free before this launcher runs (pre-registered protocol, decisions.md 2026-08-15
   LoRA-parent entry)."
  fi
fi

# ---- SWEEP: one epoch per rung, G1+G8 --no-record, numbers to SWEEP_JSON --
for LR in "${RUNGS[@]}"; do
  rid=evt-ts38-parent-lorasweep-lr${LR}
  train_or_skip "$rid" \
    python3 train_sft.py --config "$PARENT_CONFIG" \
      --override "$OVERLAY_DIR/parent_lora_sweep_lr_${LR}.yaml" \
      --init-from "$BASE_MODEL" --confirm-cost

  if sweep_row_scored "$LR"; then
    milestone "lora_sweep_skip lr=$LR (already scored in $SWEEP_JSON)"
    continue
  fi

  G1_OUT=$(python3 gates.py g1 --run "$rid" --config "$PARENT_CONFIG" --threshold 0.95 --no-record 2>&1)
  echo "$G1_OUT"
  G1_SCORE=$(gate_score "$G1_OUT" G1 accuracy)
  [[ -n $G1_SCORE ]] ||
    fail "$rid G1 printed no accuracy at lr=$LR — it errored rather than scored (output above)"

  G8_OUT=$(python3 gates.py g8 --run "$rid" --config "$RUN1_CONFIG" --bar 1.1718 \
    --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --no-record 2>&1)
  echo "$G8_OUT"
  G8_SCORE=$(gate_score "$G8_OUT" G8 val_loss_nats)
  [[ -n $G8_SCORE ]] ||
    fail "$rid G8 printed no val_loss_nats at lr=$LR — it errored rather than scored (output above)"
  G8_PASS=0
  gate_passed "$G8_OUT" G8 val_loss_nats && G8_PASS=1

  SWEEP_RUNG_OUT=$(python3 - "$rid" <<'PY'
import json, os, sys
from pathlib import Path

rid = sys.argv[1]
run_dir = Path(os.environ["GEODE_STORE"]) / "runs" / rid
meta = json.loads((run_dir / "training_meta.json").read_text())
lines = [json.loads(line) for line in (run_dir / "eval_log.jsonl").read_text().splitlines() if line.strip()]
vals = [line["val_loss_nats"] for line in lines if "val_loss_nats" in line]
first_val = vals[0] if vals else float("nan")
last_val = vals[-1] if vals else float("nan")
descending = bool(vals) and last_val < first_val
print(
    meta.get("stop_reason"),
    meta.get("final_step"),
    meta.get("min_val_nats"),
    first_val,
    last_val,
    "1" if descending else "0",
)
PY
)
  read -r STOP_REASON FINAL_STEP MIN_VAL FIRST_VAL LAST_VAL VAL_DESC <<<"$SWEEP_RUNG_OUT"

  sweep_record "$LR" "$rid" "$STOP_REASON" "$FINAL_STEP" "$MIN_VAL" "$FIRST_VAL" "$LAST_VAL" "$VAL_DESC" \
    "$G1_SCORE" "$G8_SCORE" "$G8_PASS"
  milestone "lora_sweep_rung lr=$LR g1=$G1_SCORE g8=$G8_SCORE val=$FIRST_VAL->$LAST_VAL descending=$VAL_DESC"
done

# ---- SELECT: highest LR (RUNGS order) with g8_pass AND val_descending -----
QUALIFYING_STR=$(python3 - "$SWEEP_JSON" "${RUNGS[@]}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
order = sys.argv[2:]
rows = {r["lr"]: r for r in (json.loads(path.read_text()) if path.is_file() else [])}
qualifying = [lr for lr in order if lr in rows and rows[lr].get("g8_pass") and rows[lr].get("val_descending")]
print(" ".join(qualifying))
PY
)
QUALIFYING=()
read -ra QUALIFYING <<<"$QUALIFYING_STR"

if ((${#QUALIFYING[@]} == 0)); then
  echo "[ts38lora] sweep table ($SWEEP_JSON):"
  cat "$SWEEP_JSON" 2>/dev/null || echo "  (no sweep data recorded)"
  fail "LORA SWEEP EXHAUSTED: no rung (${RUNGS[*]}) cleared BOTH the selector's G8 <= 1.1718
   AND a descending val curve at the one-epoch sweep horizon (max_steps 8000). This is a
   SECOND design result — LoRA cannot install without crossing retention either — per
   decisions.md 2026-08-15 (LoRA-parent entry); replay-mixing is the remaining fix. Table
   printed above; full JSON: $SWEEP_JSON. OWNER DECISION."
fi

WINNER=${QUALIFYING[0]}
W_G1=$(sweep_field "$WINNER" g1_accuracy)
W_G8=$(sweep_field "$WINNER" g8_val_loss_nats)
FALLBACKS=("${QUALIFYING[@]:1}")
milestone "lora_sweep_winner lr=$WINNER g1=$W_G1 g8=$W_G8 fallbacks=${FALLBACKS[*]:-none}"

# ---- FULL RUN: winner, then (on G8 fail) ONE fallback, capped at 2 --------
FULL_WINNER=
ATTEMPT=0
for LR in "${QUALIFYING[@]}"; do
  ATTEMPT=$((ATTEMPT + 1))
  ((ATTEMPT <= 2)) || break

  if [[ -d $FAILED_DIR/${PARENT_RID}-lora-lr${LR} ]]; then
    milestone "lora_full_skip lr=$LR (already archived at $FAILED_DIR/${PARENT_RID}-lora-lr${LR})"
    continue
  fi

  if [[ $(status_of "$PARENT_RID") != missing ]]; then
    lr_matches "$PARENT_RID" "$LR" || fail \
      "runs/$PARENT_RID exists but its manifest optimizer.lr is not this attempt's LR
   ($LR). Something outside this launcher's attempt loop occupies the parent slot —
   inspect it; this script deletes nothing."
  fi
  train_or_skip "$PARENT_RID" \
    python3 train_sft.py --config "$PARENT_CONFIG" \
      --override "$OVERLAY_DIR/parent_lora_lr_${LR}.yaml" \
      --init-from "$BASE_MODEL" --confirm-cost

  python3 - "$PARENT_RID" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
d = json.loads(p.read_text())
sys.exit(0 if d.get("training", {}).get("method") == "lora" else 1)
PY
  [[ $? -eq 0 ]] || fail \
    "$PARENT_RID (lr $LR) completed but its manifest training.method is not 'lora' — wrong
   checkpoint for this launcher (own_lora_block gate?); inspect
   $GEODE_STORE/runs/$PARENT_RID/manifest.json"
  [[ -f $PARENT_MODEL/model.safetensors ]] ||
    fail "$PARENT_RID (lr $LR) completed but left no $PARENT_MODEL/model.safetensors"

  FULL_STOP_REASON=$(python3 - "$PARENT_RID" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "training_meta.json"
print(json.loads(p.read_text()).get("stop_reason", "unknown") if p.is_file() else "missing")
PY
)
  STOP_REASON_NOTE=""
  if [[ $FULL_STOP_REASON == max_steps ]]; then
    echo "[ts38lora] WARN stop_reason=max_steps at the 160k ceiling (bug signal, inspect)"
    milestone "stop_reason_warn run=$PARENT_RID lr=$LR reason=max_steps"
    STOP_REASON_NOTE=" BUG SIGNAL at the 160k ceiling (~20.6 epochs) — inspect the eps/k
   stopping rule before reading a G1 miss as a design result."
  fi

  FALLBACK_MSG="LORA PARENT G1 FAIL at lr=$LR: this LR is already the highest retention-safe
   rung the sweep selector qualified — no lower LR fixes an undertraining G1 miss (design
   result #2; the full-FT parent already failed to certify both bars — decisions.md
   2026-08-15 LoRA-parent entry). training_meta.json stop_reason=$FULL_STOP_REASON.
   $STOP_REASON_NOTE
   OWNER DECISION. No bar moves. Sweep numbers: $SWEEP_JSON"

  enforce_gate "$PARENT_RID" G1 accuracy 0.95 \
    python3 gates.py g1 --run "$PARENT_RID" --config "$PARENT_CONFIG" --threshold 0.95

  G8_OUT=$(python3 gates.py g8 --run "$PARENT_RID" --config "$RUN1_CONFIG" --bar 1.1718 \
    --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --no-record 2>&1)
  echo "$G8_OUT"
  G8_SCORE=$(gate_score "$G8_OUT" G8 val_loss_nats)
  [[ -n $G8_SCORE ]] ||
    fail "$PARENT_RID G8 printed no val_loss_nats at lr=$LR (full run) — it errored rather than scored (output above)"

  if gate_passed "$G8_OUT" G8 val_loss_nats; then
    python3 gates.py g8 --run "$PARENT_RID" --config "$RUN1_CONFIG" --bar 1.1718 \
      --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --record-only-pass ||
      fail "$PARENT_RID G8 DIVERGENCE: --no-record scored $G8_SCORE (PASS) but the recording pass recomputed a FAIL — nothing written; rerun the gate block"
    milestone "gate_pass run=$PARENT_RID gate=G8 val_loss_nats=$G8_SCORE"
    FULL_WINNER=$LR
    break
  fi

  mkdir -p "$FAILED_DIR"
  mv "$GEODE_STORE/runs/$PARENT_RID" "$FAILED_DIR/${PARENT_RID}-lora-lr${LR}" ||
    fail "could not archive the lr=$LR full run to $FAILED_DIR/${PARENT_RID}-lora-lr${LR}"
  milestone "lora_full_g8_fail lr=$LR g8=$G8_SCORE (archived; one fallback allowed)"
done

[[ -n $FULL_WINNER ]] || fail \
  "LORA PARENT G8 FAIL after fallback / no fallback exists — HALT (design result: the
   sweep-qualified LR(s) (${QUALIFYING[*]}) cannot certify both G1 and G8 on the FULL
   160k-ceiling run, even though each cleared G8 <= 1.1718 at the one-epoch sweep horizon
   — retention drifts past the bar over the longer run). Archived attempt(s) under
   $FAILED_DIR; sweep numbers: $SWEEP_JSON. OWNER DECISION."

# ---- MERGE for hand-off + receiver check (ALWAYS runs) ---------------------
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

print(f"[ts38lora] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
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

notify "ts38lora launcher done: parent=$PARENT_RID lr=$FULL_WINNER merged+receiver-verified"
echo "[ts38lora] TERMINAL_SUCCESS parent=$PARENT_RID lr=$FULL_WINNER"
