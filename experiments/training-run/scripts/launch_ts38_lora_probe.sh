#!/usr/bin/env bash
# ts38 mini — LoRA-parent capability-vs-retention PROBE (decisions.md
# 2026-08-15 "LoRA parent HALT" entry). NOT a parent launcher: it selects
# nothing, records nothing, moves no bar.
#
# Why it exists. The pre-registered LoRA-parent protocol (launch_ts38_lora_
# parent.sh) HALTed: at 3e-4 the full run converged @24k with G1 0.9775 PASS /
# G8 1.1855 FAIL, and the one pre-declared fallback @1e-4 converged @55k with
# G1 0.9658 PASS / G8 1.1994 FAIL — every CONVERGED parent, full-FT or LoRA,
# sits past the retention bar. But the 3e-4 trajectory brackets a possible
# window: sweep-end (8k) G8 was 1.1379 with val 0.095 (G1 0.8672), and val
# enters the G1-pass band (~<=0.033 by the G1-vs-val points collected so far)
# around 15k–18k, while G8 reached 1.1855 only by 24k. Whether ANY step
# satisfies both bars is the fact the owner needs to decide between an
# earliest-certified-checkpoint parent (bends run-until-convergence for the
# parent — owner call) and replay-mixing (a build).
#
# How. train_sft.py is bit-exact on replay (the 1e-5 full-FT replay matched
# its ceil40k run at every eval), and the 24k run never plateau-stopped before
# 24k, so replaying the SAME 3e-4 config with a smaller max_steps reproduces
# that step's checkpoint exactly. Four horizons {14k,16k,18k,20k}, each scored
# G1 + G8 --no-record, numbers to results/ts38_parent_lora_probe.json.
# ~68k steps ≈ 70 min + gates on an RTX 4090 (~$0.40). stop_reason=max_steps
# is EXPECTED on every point (probe horizon), never a bug signal here.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38probe

echo "[ts38probe] estimated cost: 4 deterministic replays of the 3e-4 LoRA parent to"
echo "[ts38probe] {14k,16k,18k,20k} steps (68k steps ≈ 70 min at ~16 steps/s) + G1/G8"
echo "[ts38probe] --no-record on each (~2 min/point). ~\$0.40 on a rented 4090. Disk ~1 GB."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38_lora_probe.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
PARENT_CONFIG=../configs/ts38_pretaught_parent_lora.yaml
OVERLAY_DIR=../configs/sweeps/ts38
RUN1_CONFIG=../configs/archive/runs/run1_pretrain.yaml
VAL_CACHE=$GEODE_STORE/cache/run1_val_stream.pt
OUT=$GEODE_STORE/results/ts38_parent_lora_probe.json
HORIZONS=(14000 16000 18000 20000)

[[ -f $BASE_MODEL/model.safetensors ]] || fail "no base checkpoint at $BASE_MODEL (pull $BASE_RID from the relay first)"
[[ -f $VAL_CACHE ]] || fail "no G8 val-stream cache at $VAL_CACHE (relay: cache/run1_val_stream.pt)"
for S in "${HORIZONS[@]}"; do
  [[ -f $OVERLAY_DIR/parent_lora_probe_lr3e-4_s${S}.yaml ]] || fail "missing overlay parent_lora_probe_lr3e-4_s${S}.yaml"
done

milestone "probe_start lr=3e-4 horizons=${HORIZONS[*]} repo=$(git log --oneline -1 | cut -c1-7)"
for S in "${HORIZONS[@]}"; do
  rid=evt-ts38-parent-loraprobe-lr3e-4-s${S}
  if [[ $(status_of "$rid") == complete ]]; then
    milestone "probe_train_skip run=$rid"
  else
    [[ $(status_of "$rid") == missing ]] || fail "$rid exists but is not complete — inspect, this script deletes nothing"
    milestone "probe_train_start run=$rid"
    python3 train_sft.py --config "$PARENT_CONFIG" \
      --override "$OVERLAY_DIR/parent_lora_probe_lr3e-4_s${S}.yaml" \
      --init-from "$BASE_MODEL" --confirm-cost || fail "$rid training"
    [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  fi

  G1_OUT=$(python3 gates.py g1 --run "$rid" --config "$PARENT_CONFIG" --threshold 0.95 --no-record 2>&1)
  grep "G1 accuracy" <<<"$G1_OUT"
  G1=$(sed -n 's/.*G1 accuracy \([0-9.]*\) on n=.*/\1/p' <<<"$G1_OUT" | head -1)
  G8_OUT=$(python3 gates.py g8 --run "$rid" --config "$RUN1_CONFIG" --bar 1.1718 \
    --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --no-record 2>&1)
  grep "G8 val_loss_nats" <<<"$G8_OUT"
  G8=$(sed -n 's/.*G8 val_loss_nats \([0-9.]*\) on n=.*/\1/p' <<<"$G8_OUT" | head -1)
  [[ -n $G1 && -n $G8 ]] || {
    echo "$G1_OUT" | tail -5
    echo "$G8_OUT" | tail -5
    fail "$rid: a gate printed no score (errored rather than scored; output above)"
  }

  python3 - "$OUT" "$S" "$rid" "$G1" "$G8" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(sys.argv[1])
step, rid, g1, g8 = int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
run = Path(os.environ["GEODE_STORE"]) / "runs" / rid
meta = json.loads((run / "training_meta.json").read_text())
evals = [json.loads(l) for l in (run / "eval_log.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in (json.loads(p.read_text()) if p.is_file() else []) if r["step"] != step]
rows.append(
    {
        "lr": "3e-4",
        "step": step,
        "run_id": rid,
        "stop_reason": meta.get("stop_reason"),
        "final_step": meta.get("final_step"),
        "min_val_nats": meta.get("min_val_nats"),
        "final_val_nats": evals[-1]["val_loss_nats"] if evals else None,
        "g1_accuracy": g1,
        "g1_pass": g1 >= 0.95,
        "g8_val_loss_nats": g8,
        "g8_pass": g8 <= 1.1718,
        "g8_bar": 1.1718,
        "both_pass": bool(g1 >= 0.95 and g8 <= 1.1718),
        "note": "deterministic replay of the 3e-4 LoRA parent to a fixed horizon; --no-record; probe only",
    }
)
rows.sort(key=lambda r: r["step"])
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(rows, indent=2) + "\n")
PY
  milestone "probe_point step=$S g1=$G1 g8=$G8"
done

echo "[ts38probe] table: $OUT"
python3 -c "import json,sys; [print(r['step'], 'g1', r['g1_accuracy'], 'g8', r['g8_val_loss_nats'], 'both_pass', r['both_pass']) for r in json.load(open(sys.argv[1]))]" "$OUT"
echo "[ts38probe] TERMINAL_SUCCESS points=${#HORIZONS[@]} json=$OUT"
