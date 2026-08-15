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
# How (2026-08-15 redesign — ONE replay, not four). train_sft.py is bit-exact
# on replay (the 1e-5 full-FT replay matched its earlier ceil40k run at every
# eval), and geode.train.sft.train_sft now takes snapshot_steps (V5.77,
# specs/02 §6): a snapshot written at step S is bit-identical to the final
# checkpoint of an otherwise-identical run capped at max_steps=S. The old
# version of this script did FOUR separate replays cut at {14k,16k,18k,20k}.
# This version does ONE replay to 24000 (the archived 3e-4 run's converged
# step) with snapshot_steps every 1000 from 10000 to 24000 — same GPU cost as
# the training itself (snapshots are cheap save_pretrained writes, not extra
# forward passes), a 15-point curve instead of a 4-point one, and a
# loss-decomposition pass (analysis/g8_decompose.py, another worker's
# deliverable) on a subset of those points. stop_reason=max_steps OR
# converged at 24000 are BOTH expected on this run (probe horizon, not a
# fresh ceiling) — never a bug signal here. If eps/k fires before 24000 on a
# different GPU (numerics), snapshots past the stop simply do not
# materialize and this script scores whatever exists.
#
# Cost estimate. 1 replay to 24000 steps ≈ 25 min at ~16 steps/s, + 15 points
# × (G1 --no-record ≈ 1 min + G8 --no-record ≈ ~4 min: gates.py's run_g8
# ALWAYS re-scores the base-run anchor on the rebuilt val stream before
# scoring the target checkpoint — every call pays two forward passes over the
# stream, not one, so the naive "G8 ≈ 2 min" is doubled here for an honest
# number) + ~6 decompositions × ~3 min ≈ 25 + 75 + 18 ≈ 118 min ≈ ~2.0 h ≈
# $0.65-0.80 on a rented 4090 (~$0.33-0.40/h). Disk ≈ 3.9 GB: 15 wrapped-LoRA
# snapshot dirs (~200 MB each) + the final model/ (~200 MB) + ~15 x 46 MB
# adapter sidecars.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38probe

echo "[ts38probe] estimated cost: ONE replay of the 3e-4 LoRA parent to 24000 steps"
echo "[ts38probe] (~25 min at ~16 steps/s) writing snapshots every 1000 steps from"
echo "[ts38probe] 10000-24000, each scored G1 + G8 --no-record (~15 points x ~5 min,"
echo "[ts38probe] G8's always-on base-run anchor pays a second forward pass per call)"
echo "[ts38probe] + ~6 G8 loss-decompositions (~3 min each). ~2.0 h total, ~\$0.65-0.80"
echo "[ts38probe] on a rented 4090. Disk ~3.9 GB of snapshots + adapter sidecars."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38_lora_probe.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

RELAY_REPO=mhieuuu/geode-store
BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RID=evt-ts38-parent-loraprobe-lr3e-4
PARENT_CONFIG=../configs/ts38_pretaught_parent_lora.yaml
OVERLAY=../configs/sweeps/ts38/parent_lora_probe_lr3e-4.yaml
RUN1_CONFIG=../configs/archive/runs/run1_pretrain.yaml
VAL_CACHE=$GEODE_STORE/cache/run1_val_stream.pt
OUT=$GEODE_STORE/results/ts38_parent_lora_probe.json
DECOMPOSE_DIR=$GEODE_STORE/results/ts38_g8_decompose
REF_RID=evt-ts38-pretaught-parent-lora-lr3e-4
REF_EVAL_LOG=$GEODE_STORE/runs-failed/$REF_RID/eval_log.jsonl

# ---- stage 0: preflight (idempotent — pulls only what's missing) ----------
milestone "repo=$(git log --oneline -1 | cut -c1-7)"

if [[ ! -f $BASE_MODEL/model.safetensors ]]; then
  milestone "pull_base_start run=$BASE_RID"
  python3 hf_checkpoint.py pull --run-id "$BASE_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $BASE_RID from $RELAY_REPO"
  milestone "base_pulled run=$BASE_RID"
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

[[ -f $OVERLAY ]] || fail "missing overlay $OVERLAY"

if [[ ! -f $REF_EVAL_LOG ]]; then
  milestone "pull_ref_eval_log_start run=$REF_RID"
  python3 - "$REF_RID" "$GEODE_STORE" <<'PY' || fail "pull the archived reference eval_log.jsonl for $REF_RID from the relay"
import sys

from huggingface_hub import hf_hub_download

rid, store = sys.argv[1], sys.argv[2]
hf_hub_download(
    repo_id="mhieuuu/geode-store",
    repo_type="model",
    filename=f"runs-failed/{rid}/eval_log.jsonl",
    local_dir=store,
)
PY
  milestone "ref_eval_log_pulled"
fi
[[ -f $REF_EVAL_LOG ]] || fail "pull left no $REF_EVAL_LOG"

# ---- stage 1: train (train_or_skip helper, VERBATIM from
# launch_ts38_lora_parent.sh) -------------------------------------------
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

train_or_skip "$RID" \
  python3 train_sft.py --config "$PARENT_CONFIG" \
    --override "$OVERLAY" \
    --init-from "$BASE_MODEL" --confirm-cost

# ---- stage 2: replay-fidelity check (curve shape is the deliverable even
# on drift — this WARNs, never fails) -------------------------------------
FIDELITY_OUT=$(python3 - "$RID" "$REF_EVAL_LOG" <<'PY'
import json
import os
import sys
from pathlib import Path

rid, ref_path = sys.argv[1], Path(sys.argv[2])
run_dir = Path(os.environ["GEODE_STORE"]) / "runs" / rid
probe_evals = {
    r["step"]: r["val_loss_nats"]
    for r in (
        json.loads(line)
        for line in (run_dir / "eval_log.jsonl").read_text().splitlines()
        if line.strip()
    )
    if "val_loss_nats" in r
}
ref_evals = {
    r["step"]: r["val_loss_nats"]
    for r in (
        json.loads(line) for line in ref_path.read_text().splitlines() if line.strip()
    )
    if "val_loss_nats" in r
}
shared = sorted(set(probe_evals) & set(ref_evals))
max_abs = max((abs(probe_evals[s] - ref_evals[s]) for s in shared), default=0.0)
print(len(shared), f"{max_abs:.6e}", "1" if max_abs > 1e-3 else "0")
PY
)
read -r SHARED_STEPS MAX_ABS_DVAL DRIFT_FLAG <<<"$FIDELITY_OUT"
milestone "replay_fidelity shared_steps=$SHARED_STEPS max_abs_dval=$MAX_ABS_DVAL"
[[ $DRIFT_FLAG == 1 ]] &&
  echo "[ts38probe] WARN replay drifted from the archived reference (max |Δval|=$MAX_ABS_DVAL > 1e-3) — continuing, the curve shape is still the deliverable"

# ---- stage 3: score every snapshot (+ the final checkpoint if it isn't
# already one of them) with G1 + G8 --no-record ----------------------------
probe_row_scored() { # step -> exit 0 iff a row for step $1 already carries a G8 score
  python3 - "$OUT" "$1" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
rows = json.loads(p.read_text())
sys.exit(0 if any(r["step"] == int(sys.argv[2]) and r.get("g8_val_loss_nats") is not None for r in rows) else 1)
PY
}

probe_record() { # step run_id checkpoint val_loss g1 g8 -> prints both_pass (0/1) on stdout
  python3 - "$OUT" "$@" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
step, run_id, checkpoint, val_loss, g1, g8 = sys.argv[2:8]
step = int(step)
val_loss = None if val_loss == "None" else float(val_loss)
g1 = float(g1)
g8 = float(g8)
both_pass = bool(g1 >= 0.95 and g8 <= 1.1718)
rows = [r for r in (json.loads(p.read_text()) if p.is_file() else []) if r["step"] != step]
rows.append(
    {
        "lr": "3e-4",
        "step": step,
        "run_id": run_id,
        "checkpoint": checkpoint,
        "val_loss_nats": val_loss,
        "g1_accuracy": g1,
        "g1_pass": g1 >= 0.95,
        "g8_val_loss_nats": g8,
        "g8_pass": g8 <= 1.1718,
        "g8_bar": 1.1718,
        "both_pass": both_pass,
        "note": "single-replay probe (ONE run to 24000 with snapshot_steps, V5.77); --no-record; probe only",
    }
)
rows.sort(key=lambda r: r["step"])
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(rows, indent=2) + "\n")
print(1 if both_pass else 0)
PY
}

CHECKPOINTS=$(python3 - "$RID" <<'PY'
import json
import os
import sys
from pathlib import Path

rid = sys.argv[1]
run_dir = Path(os.environ["GEODE_STORE"]) / "runs" / rid
snap_root = run_dir / "sft_snapshots"
on_disk = sorted(
    int(p.name.split("_")[1])
    for p in (snap_root.glob("step_*") if snap_root.is_dir() else [])
    if (p / "model.safetensors").is_file()
)
meta = json.loads((run_dir / "training_meta.json").read_text())
final_step = meta.get("final_step")
points = [(s, str(snap_root / f"step_{s:07d}")) for s in on_disk]
if final_step is not None and final_step not in on_disk:
    points.append((final_step, str(run_dir / "model")))
points.sort(key=lambda x: x[0])
for s, ckpt in points:
    print(s, ckpt)
PY
)
[[ -n $CHECKPOINTS ]] ||
  fail "$RID: no scoreable checkpoint (no sft_snapshots/ and no training_meta.json final_step)"

while read -r S CKPT; do
  [[ -n $S ]] || continue
  if probe_row_scored "$S"; then
    milestone "probe_point_skip step=$S (already scored in $OUT)"
    continue
  fi

  G1_OUT=$(python3 gates.py g1 --run "$RID" --config "$PARENT_CONFIG" --threshold 0.95 \
    --checkpoint "$CKPT" --no-record 2>&1)
  grep "G1 accuracy" <<<"$G1_OUT"
  G1=$(sed -n 's/.*G1 accuracy \([0-9.]*\) on n=.*/\1/p' <<<"$G1_OUT" | head -1)

  G8_OUT=$(python3 gates.py g8 --run "$RID" --config "$RUN1_CONFIG" --bar 1.1718 \
    --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --checkpoint "$CKPT" --no-record 2>&1)
  grep "G8 val_loss_nats" <<<"$G8_OUT"
  G8=$(sed -n 's/.*G8 val_loss_nats \([0-9.]*\) on n=.*/\1/p' <<<"$G8_OUT" | head -1)

  [[ -n $G1 && -n $G8 ]] || {
    echo "$G1_OUT" | tail -5
    echo "$G8_OUT" | tail -5
    fail "$RID step=$S: a gate printed no score (errored rather than scored; output above)"
  }

  VAL=$(python3 - "$RID" "$S" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1]
step = int(sys.argv[2])
evals = [
    json.loads(line)
    for line in (run_dir / "eval_log.jsonl").read_text().splitlines()
    if line.strip()
]
match = next((e for e in evals if e["step"] == step), None)
print(match["val_loss_nats"] if match and "val_loss_nats" in match else "None")
PY
)

  BOTH=$(probe_record "$S" "$RID" "$CKPT" "$VAL" "$G1" "$G8")
  milestone "probe_point step=$S val=$VAL g1=$G1 g8=$G8 both_pass=$BOTH"
done <<<"$CHECKPOINTS"

# ---- stage 4: decompose a subset (another worker's analysis/g8_decompose.py
# — diagnostic, WARN-and-continue on failure, never a gate) ---------------
for S in ${DECOMPOSE_STEPS:-10000 14000 16000 18000 20000 24000}; do
  STEP_TAG=$(printf 'step_%07d' "$S")
  CKPT_DIR="$GEODE_STORE/runs/$RID/sft_snapshots/$STEP_TAG"
  [[ -d $CKPT_DIR ]] || {
    milestone "decompose_skip step=$S (no snapshot)"
    continue
  }
  DECOMPOSE_OUT_JSON="$DECOMPOSE_DIR/$STEP_TAG.json"
  if [[ -f $DECOMPOSE_OUT_JSON ]]; then
    milestone "decompose_skip step=$S (already exists)"
    continue
  fi

  DECOMP_OUT=$(python3 ../analysis/g8_decompose.py --run "$RID" --checkpoint "$CKPT_DIR" \
    --base-run "$BASE_RID" --config "$RUN1_CONFIG" --tokenizer ../tokenizer \
    --val-cache "$VAL_CACHE" --out "$DECOMPOSE_OUT_JSON" --label "$STEP_TAG" 2>&1)
  DECOMP_STATUS=$?
  if [[ $DECOMP_STATUS -ne 0 ]]; then
    echo "[ts38probe] WARN g8_decompose.py failed for step=$S (diagnostic, not a gate; output below)"
    echo "$DECOMP_OUT" | tail -10
    milestone "decompose_warn step=$S status=$DECOMP_STATUS"
    continue
  fi
  grep '\[g8dec\]' <<<"$DECOMP_OUT"
  milestone "decompose_done step=$S"
done

# ---- stage 5: summary + push (best effort — WARN, never fail) -------------
echo "[ts38probe] table: $OUT"
python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

rows = sorted(json.loads(Path(sys.argv[1]).read_text()), key=lambda r: r["step"])
for r in rows:
    print(
        r["step"], "val", r["val_loss_nats"], "g1", r["g1_accuracy"],
        "g8", r["g8_val_loss_nats"], "both_pass", r["both_pass"],
    )
PY

SUMMARY_FIELDS=$(python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

rows = json.loads(Path(sys.argv[1]).read_text())
both = sorted(r["step"] for r in rows if r["both_pass"])
g1_pass_steps = sorted(r["step"] for r in rows if r["g1_pass"])
g8_pass_steps = sorted(r["step"] for r in rows if r["g8_pass"])
print(
    ",".join(str(s) for s in both) or "none",
    min(g1_pass_steps) if g1_pass_steps else "none",
    max(g8_pass_steps) if g8_pass_steps else "none",
    len(rows),
    len(both),
)
PY
)
read -r BOTH_PASS_STEPS FIRST_G1_PASS LAST_G8_PASS N_POINTS N_BOTH_PASS <<<"$SUMMARY_FIELDS"
milestone "probe_summary both_pass_steps=$BOTH_PASS_STEPS first_g1_pass=$FIRST_G1_PASS last_g8_pass=$LAST_G8_PASS"

python3 hf_checkpoint.py push --run-id "$RID" --metadata-only ||
  echo "[ts38probe] WARN hf_checkpoint.py push --metadata-only failed for $RID (best effort)"

python3 - "$OUT" "$GEODE_STORE" "$RID" <<'PY' || echo "[ts38probe] WARN uploading both-pass adapter sidecars failed (best effort)"
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

out_path, store, rid = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rows = json.loads(out_path.read_text())
api = HfApi()
for r in rows:
    if not r["both_pass"]:
        continue
    sidecar = Path(r["checkpoint"]) / "adapter.safetensors"
    if not sidecar.is_file():
        print(f"[ts38probe] WARN no adapter sidecar at {sidecar} for both-pass step {r['step']} — skipping")
        continue
    repo_path = f"runs/{rid}/{sidecar.relative_to(store / 'runs' / rid).as_posix()}"
    api.upload_file(
        path_or_fileobj=str(sidecar),
        path_in_repo=repo_path,
        repo_id="mhieuuu/geode-store",
        repo_type="model",
    )
    print(f"[ts38probe] uploaded {repo_path}")
PY

python3 - "$OUT" "$GEODE_STORE" <<'PY' || echo "[ts38probe] WARN uploading $OUT failed (best effort)"
import sys
from pathlib import Path

from huggingface_hub import HfApi

out_path, store = Path(sys.argv[1]), Path(sys.argv[2])
HfApi().upload_file(
    path_or_fileobj=str(out_path),
    path_in_repo=str(out_path.relative_to(store).as_posix()),
    repo_id="mhieuuu/geode-store",
    repo_type="model",
)
print(f"[ts38probe] uploaded {out_path.relative_to(store).as_posix()}")
PY

if [[ -d $DECOMPOSE_DIR ]]; then
  python3 - "$DECOMPOSE_DIR" "$GEODE_STORE" <<'PY' || echo "[ts38probe] WARN uploading $DECOMPOSE_DIR failed (best effort)"
import sys
from pathlib import Path

from huggingface_hub import HfApi

decompose_dir, store = Path(sys.argv[1]), Path(sys.argv[2])
HfApi().upload_folder(
    folder_path=str(decompose_dir),
    path_in_repo=str(decompose_dir.relative_to(store).as_posix()),
    repo_id="mhieuuu/geode-store",
    repo_type="model",
)
print(f"[ts38probe] uploaded folder {decompose_dir.relative_to(store).as_posix()}")
PY
else
  echo "[ts38probe] WARN no $DECOMPOSE_DIR to upload (all decompositions skipped/failed)"
fi

# Receiver-side check (specs/00 "verify the receiver, not the sender"):
python3 - "$RID" "$GEODE_STORE" <<'PY' || echo "[ts38probe] WARN receiver check failed to run (best effort)"
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

rid, store = sys.argv[1], Path(sys.argv[2])
files = set(HfApi().list_repo_files("mhieuuu/geode-store", repo_type="model"))

expected = [
    f"runs/{rid}/manifest.json",
    f"runs/{rid}/eval_log.jsonl",
    f"runs/{rid}/train_log.jsonl",
    f"runs/{rid}/training_meta.json",
    "results/ts38_parent_lora_probe.json",
]
rows = json.loads((store / "results" / "ts38_parent_lora_probe.json").read_text())
for r in rows:
    if r["both_pass"]:
        ckpt = Path(r["checkpoint"])
        rel = ckpt.relative_to(store / "runs" / rid).as_posix()
        expected.append(f"runs/{rid}/{rel}/adapter.safetensors")

print("[ts38probe] receiver check:")
for path in expected:
    print(f"  {'OK ' if path in files else 'MISSING'} {path}")
PY

notify "ts38probe done: run=$RID both_pass=$N_BOTH_PASS/$N_POINTS json=$OUT"
echo "[ts38probe] TERMINAL_SUCCESS points=$N_POINTS both_pass=$N_BOTH_PASS json=$OUT"
