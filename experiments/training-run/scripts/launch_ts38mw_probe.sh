#!/usr/bin/env bash
# ts38mw Stage 1 — wrapper-diversity install FALSIFICATION PROBE (plan
# docs/plan-ts38mw-multiwrap-install.md §3.5; decisions.md 2026-08-15 "ts38mw
# Stage 1 pre-registration" entry). Clone of launch_ts38_lora_probe.sh,
# adapted: this run installs D_target_mw (8 wrapper templates around the SAME
# op-notation arithmetic, instead of ts38's single template) and scores every
# snapshot against 6 held-out probe phrasings (bare_op/sym_q/word_q/sumof/
# sumof_bare/dm_mix, each gates.py g5 --no-record) plus the base model on the
# same 6 pins, then applies the pre-registered verdict rule (mw_verdict.py)
# to decide GO-A / GO-B / WEAK / NO-GO / INCONCLUSIVE (plan §2). NOT a parent
# launcher: it selects nothing, records nothing to any manifest (every gate
# call --no-record), and pushes no weights.
#
# Why. ts38's certified parent computes op-notation arithmetic ONLY under its
# exact training template; even the symbol-bearing `What is a + b?` collapses
# to ~3%, flat across snapshots 10k-24k (docs/ts38-vs-bits-that-count.md).
# The suspected cause is a single-surface-form install set. This run installs
# 8 wrappers around the identical arithmetic instead and asks whether the
# skill fires under a wrapper it never saw, including the word-only target
# phrasing (`sumof`).
#
# Cost estimate. Training ~25-60 min at ~16 steps/s (the single-wrapper lane
# converged at 24k; 8 wrappers plausibly later; ceiling 160k never binds) +
# ~14 snapshots x (G1 --no-record ~1 min + 6 x g5 --no-record ~40s each) +
# G8 --no-record ~4 min ONLY at snapshots whose canonical EM >= 0.95 (skipped
# otherwise -- saves ~4 min per skipped snapshot) => ~1.5-2.5 h total,
# ~$0.5-1.0 on a rented 4090. Disk ~200 MB per snapshot.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38mw

echo "[ts38mw] estimated cost: ONE LoRA install of D_target_mw (8 wrapper"
echo "[ts38mw] templates) at lr=3e-4, ~25-60 min at ~16 steps/s (ceiling 160k"
echo "[ts38mw] never binds -- eps/k convergence stops the run), writing"
echo "[ts38mw] snapshots at 8k/12k/16k/20k/24k/28k/32k/36k/40k/48k/56k/64k/"
echo "[ts38mw] 80k/96k, each scored G1 (own val split) + 6 held-out probe"
echo "[ts38mw] pins (gates.py g5 --no-record, ~40s each) + G8 (~4 min, ONLY"
echo "[ts38mw] where canonical EM >= 0.95). ~1.5-2.5 h total, ~\$0.5-1.0 on a"
echo "[ts38mw] rented 4090. Disk ~200 MB per snapshot. Nothing recorded to"
echo "[ts38mw] any manifest; no weights pushed."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38mw_probe.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

RELAY_REPO=mhieuuu/geode-store
BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RID=evt-ts38mw-parent-probe-lr3e-4
PARENT_CONFIG=../configs/ts38mw_pretaught_parent_lora.yaml
OVERLAY=../configs/sweeps/ts38mw/parent_probe_lr3e-4.yaml
RUN1_CONFIG=../configs/archive/runs/run1_pretrain.yaml
VAL_CACHE=$GEODE_STORE/cache/run1_val_stream.pt
OUT=$GEODE_STORE/results/ts38mw_probe.json
# The 6 pins scored per snapshot (plan §3.2): bare_op's zero-shot EM IS
# canonical_em (the "G1-canonical" op-format number, evaluated under the
# canonical scaffold, disjoint from the held-out wrapper set). sym_imp/
# word_imp/dm_mix_bare exist (make_dm_probe_eval.py builds all 9) but are not
# scored here -- not on the pre-registered pin list.
PINS=(bare_op sym_q word_q sumof sumof_bare dm_mix)

# ---- stage 0: preflight (idempotent -- pulls/regenerates only what's
# missing) ------------------------------------------------------------------
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

# Data regen: deterministic and hash-checked by the config pins on load
# (geode.arith.load.load_frozen_parquet) -- a stale/corrupt regen refuses
# loudly there, not here.
DATA_MW=$REPO_ROOT/experiments/training-run/data/full/D_target_mw.parquet
if [[ ! -f $DATA_MW ]]; then
  milestone "datagen_start D_target_mw"
  python3 ../datagen/make_multiwrap_set.py --out ../data/full || fail "make_multiwrap_set.py"
  milestone "datagen_done D_target_mw"
fi
[[ -f $DATA_MW ]] || fail "make_multiwrap_set.py left no $DATA_MW"

DATA_SUMOF=$REPO_ROOT/experiments/training-run/data/full/D_dmprobe_sumof.parquet
if [[ ! -f $DATA_SUMOF ]]; then
  milestone "datagen_start dm_probe_eval (9 keys)"
  python3 ../datagen/make_dm_probe_eval.py || fail "make_dm_probe_eval.py"
  milestone "datagen_done dm_probe_eval"
fi
[[ -f $DATA_SUMOF ]] || fail "make_dm_probe_eval.py left no $DATA_SUMOF"

# ---- stage 1: train (train_or_skip helper, VERBATIM from
# launch_ts38_lora_probe.sh) -------------------------------------------
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

# ---- stage 2: score every snapshot (+ the final checkpoint if it isn't
# already one of them): G1 (own val split) + 6 probe pins (g5) + G8 (only
# where canonical EM >= 0.95) --------------------------------------------
probe_row_scored() { # step -> exit 0 iff a row for step $1 already exists in $OUT
  python3 - "$OUT" "$1" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
data = json.loads(p.read_text())
sys.exit(0 if any(r["step"] == int(sys.argv[2]) for r in data.get("rows", [])) else 1)
PY
}

score_pin() { # run_id checkpoint(or "") pin_key -> stdout "em0 em16 loss"; return 1 on failure
  local rid=$1 ckpt=$2 key=$3 out em0 em16 loss
  # Two literal call sites (not a conditional --checkpoint array) so
  # test_launcher_gate_args.py's static `gates.py` invocation check — which
  # substitutes bare $VAR/${VAR} tokens with a placeholder and parses the
  # result — sees a plain, parseable argv either way; it cannot expand
  # `"${arr[@]}"`.
  if [[ -n $ckpt ]]; then
    out=$(python3 gates.py g5 --run "$rid" --checkpoint "$ckpt" \
      --config "../configs/probe_dm/$key.yaml" --no-record 2>&1)
  else
    out=$(python3 gates.py g5 --run "$rid" \
      --config "../configs/probe_dm/$key.yaml" --no-record 2>&1)
  fi
  grep -E "G5 (zero-shot|16-shot|shared-set)" <<<"$out" >&2
  em0=$(sed -n 's/.*G5 zero-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  em16=$(sed -n 's/.*G5 16-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  loss=$(sed -n 's/.*G5 shared-set test loss \([0-9.]*\) nats over n=.*/\1/p' <<<"$out" | head -1)
  if [[ -z $em0 || -z $em16 || -z $loss ]]; then
    echo "$out" | tail -5 >&2
    echo "[ts38mw] score_pin: gates.py g5 run=$rid pin=$key printed no score" >&2
    return 1
  fi
  echo "$em0 $em16 $loss"
}

probe_record() { # step run_id checkpoint g1_own canonical_em g8 <18 pin values, PINS order (em0 em16 loss each)
  python3 - "$OUT" "$@" <<'PY'
import json
import sys
from pathlib import Path

PINS = ["bare_op", "sym_q", "word_q", "sumof", "sumof_bare", "dm_mix"]

p = Path(sys.argv[1])
step, run_id, checkpoint, g1_own, canonical_em, g8 = sys.argv[2:8]
vals = sys.argv[8:]
if len(vals) != len(PINS) * 3:
    raise SystemExit(f"expected {len(PINS) * 3} pin values, got {len(vals)}")

pins = {}
for i, key in enumerate(PINS):
    em0, em16, loss = vals[i * 3 : i * 3 + 3]
    pins[key] = {"em0": float(em0), "em16": float(em16), "loss": float(loss)}

data = json.loads(p.read_text()) if p.is_file() else {"rows": [], "base": None}
data["rows"] = [r for r in data["rows"] if r["step"] != int(step)]
data["rows"].append(
    {
        "step": int(step),
        "run_id": run_id,
        "checkpoint": checkpoint,
        "g1_own": float(g1_own),
        "canonical_em": float(canonical_em),
        "g8_val_loss_nats": None if g8 == "null" else float(g8),
        "pins": pins,
    }
)
data["rows"].sort(key=lambda r: r["step"])
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(data, indent=2) + "\n")
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
  G1_OWN=$(sed -n 's/.*G1 accuracy \([0-9.]*\) on n=.*/\1/p' <<<"$G1_OUT" | head -1)
  [[ -n $G1_OWN ]] || {
    echo "$G1_OUT" | tail -5
    fail "$RID step=$S: gates.py g1 printed no score"
  }

  unset EM0 EM16 LOSS
  declare -A EM0 EM16 LOSS
  for KEY in "${PINS[@]}"; do
    PIN_OUT=$(score_pin "$RID" "$CKPT" "$KEY") ||
      fail "$RID step=$S pin=$KEY: gates.py g5 failed (see stderr above)"
    read -r E0 E16 L <<<"$PIN_OUT"
    EM0[$KEY]=$E0
    EM16[$KEY]=$E16
    LOSS[$KEY]=$L
  done
  CANON_EM=${EM0[bare_op]}

  G8=null
  if python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.95 else 1)" "$CANON_EM"; then
    G8_OUT=$(python3 gates.py g8 --run "$RID" --config "$RUN1_CONFIG" --bar 1.1718 \
      --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --checkpoint "$CKPT" --no-record 2>&1)
    grep "G8 val_loss_nats" <<<"$G8_OUT"
    G8=$(sed -n 's/.*G8 val_loss_nats \([0-9.]*\) on n=.*/\1/p' <<<"$G8_OUT" | head -1)
    [[ -n $G8 ]] || {
      echo "$G8_OUT" | tail -5
      fail "$RID step=$S: gates.py g8 printed no score"
    }
  else
    milestone "g8_skip step=$S canonical_em=$CANON_EM (< 0.95)"
  fi

  probe_record "$S" "$RID" "$CKPT" "$G1_OWN" "$CANON_EM" "$G8" \
    "${EM0[bare_op]}" "${EM16[bare_op]}" "${LOSS[bare_op]}" \
    "${EM0[sym_q]}" "${EM16[sym_q]}" "${LOSS[sym_q]}" \
    "${EM0[word_q]}" "${EM16[word_q]}" "${LOSS[word_q]}" \
    "${EM0[sumof]}" "${EM16[sumof]}" "${LOSS[sumof]}" \
    "${EM0[sumof_bare]}" "${EM16[sumof_bare]}" "${LOSS[sumof_bare]}" \
    "${EM0[dm_mix]}" "${EM16[dm_mix]}" "${LOSS[dm_mix]}"

  milestone "probe_point step=$S canonical_em=$CANON_EM g1_own=$G1_OWN g8=$G8 sumof_em=${EM0[sumof]} sym_q_em=${EM0[sym_q]}"
done <<<"$CHECKPOINTS"

# ---- stage 3: score the base once, identical protocol/hardware (the loss
# guard needs a like-with-like base row) -----------------------------------
BASE_ALREADY_SCORED=$(python3 -c "
import json
from pathlib import Path

p = Path('$OUT')
print('1' if p.is_file() and json.loads(p.read_text()).get('base') else '0')
")
if [[ $BASE_ALREADY_SCORED == 0 ]]; then
  unset BEM0 BEM16 BLOSS
  declare -A BEM0 BEM16 BLOSS
  for KEY in "${PINS[@]}"; do
    PIN_OUT=$(score_pin "$BASE_RID" "" "$KEY") ||
      fail "$BASE_RID pin=$KEY: gates.py g5 failed (see stderr above)"
    read -r E0 E16 L <<<"$PIN_OUT"
    BEM0[$KEY]=$E0
    BEM16[$KEY]=$E16
    BLOSS[$KEY]=$L
  done
  python3 - "$OUT" "$BASE_RID" \
    "${BEM0[bare_op]}" "${BEM16[bare_op]}" "${BLOSS[bare_op]}" \
    "${BEM0[sym_q]}" "${BEM16[sym_q]}" "${BLOSS[sym_q]}" \
    "${BEM0[word_q]}" "${BEM16[word_q]}" "${BLOSS[word_q]}" \
    "${BEM0[sumof]}" "${BEM16[sumof]}" "${BLOSS[sumof]}" \
    "${BEM0[sumof_bare]}" "${BEM16[sumof_bare]}" "${BLOSS[sumof_bare]}" \
    "${BEM0[dm_mix]}" "${BEM16[dm_mix]}" "${BLOSS[dm_mix]}" <<'PY'
import json
import sys
from pathlib import Path

PINS = ["bare_op", "sym_q", "word_q", "sumof", "sumof_bare", "dm_mix"]
p, base_rid = Path(sys.argv[1]), sys.argv[2]
vals = sys.argv[3:]
pins = {}
for i, key in enumerate(PINS):
    em0, em16, loss = vals[i * 3 : i * 3 + 3]
    pins[key] = {"em0": float(em0), "em16": float(em16), "loss": float(loss)}
data = json.loads(p.read_text()) if p.is_file() else {"rows": [], "base": None}
data["base"] = {"run_id": base_rid, "pins": pins}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(data, indent=2) + "\n")
PY
  milestone "base_scored run=$BASE_RID"
else
  milestone "base_score_skip run=$BASE_RID (already scored in $OUT)"
fi

# ---- stage 4: table + verdict ---------------------------------------------
echo "[ts38mw] table: $OUT"
python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

PIN_ORDER = ("bare_op", "sym_q", "word_q", "sumof", "sumof_bare", "dm_mix")
data = json.loads(Path(sys.argv[1]).read_text())


def pins_str(pins: dict) -> str:
    return "  ".join(f"{k}={pins[k]['em0']:.3f}/{pins[k]['loss']:.2f}" for k in PIN_ORDER)


for r in data["rows"]:
    g8 = "null" if r["g8_val_loss_nats"] is None else f"{r['g8_val_loss_nats']:.4f}"
    print(
        f"step={r['step']:<7} canonical_em={r['canonical_em']:.4f} "
        f"g1_own={r['g1_own']:.4f} g8={g8}  {pins_str(r['pins'])}"
    )
if data.get("base"):
    print(f"BASE  {pins_str(data['base']['pins'])}")
PY

VERDICT_JSON=$(python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from mw_verdict import verdict

data = json.loads(Path(sys.argv[1]).read_text())
if not data.get("base"):
    raise SystemExit("no base block in the probe json -- base was never scored")
print(json.dumps(verdict(data["rows"], data["base"])))
PY
) || fail "mw_verdict.py failed (see output above)"
milestone "VERDICT $VERDICT_JSON"

# ---- stage 5: push metadata + the results json for durability (best
# effort -- WARN, never fail; no weights, per plan §3.5) -------------------
python3 hf_checkpoint.py push --run-id "$RID" --metadata-only ||
  echo "[ts38mw] WARN hf_checkpoint.py push --metadata-only failed for $RID (best effort)"

python3 - "$OUT" "$GEODE_STORE" <<'PY' || echo "[ts38mw] WARN uploading $OUT failed (best effort)"
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
print(f"[ts38mw] uploaded {out_path.relative_to(store).as_posix()}")
PY

# Receiver-side check (specs/00 "verify the receiver, not the sender"):
python3 - "$RID" "$GEODE_STORE" <<'PY' || echo "[ts38mw] WARN receiver check failed to run (best effort)"
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
    "results/ts38mw_probe.json",
]
print("[ts38mw] receiver check:")
for path in expected:
    print(f"  {'OK ' if path in files else 'MISSING'} {path}")
PY

BAND=$(python3 -c "import json, sys; print(json.loads(sys.argv[1])['band'])" "$VERDICT_JSON")
notify "ts38mw done: run=$RID verdict=$BAND json=$OUT"
echo "[ts38mw] TERMINAL_SUCCESS verdict=$BAND json=$OUT"
