#!/usr/bin/env bash
# fig2ts — the TinyStories-1B half of Figure 2 (EXPERIMENTS §6.14): the
# converged twin (evt-ts1b-base, 0.9855 nats) swept on scaffold-free NL
# add/sub, 19 sizes x 2 arms, mirroring the fig2nl3 protocol exactly:
#   noinst (teach)      : ts1b-base            -> D_algo_bare[:n]
#   inst (pre-teach fmt): + E.1.2 random-label op-mult installer -> same
# Predictions under test (paper Table 5 p.20): BOTH arms show the
# non-monotone teaching signature the Llama arms cannot produce — base
# peaking near n~300K, pre-teach format near ~150K.
#
# Stages: data (regen if missing; deterministic) -> ts1b-base guard ->
# PREMISE GUARD (twin must be ~0 EM on bare prompts — trivially expected,
# still measured) -> installer (full-FT, behavioral stop; G4-on-bare >= 0.90
# AND G3 <= 0.02 both ENFORCED; fallback rung = ts1b_fig2ts_installer_bare
# .yaml if G4 misses — no bar moves) -> both arms ascending n with per-run
# G5 + prune. The two n=1,000,000 ENDPOINT runs carry 128 adapter snapshots
# each, STREAMED to per-run public HF repos ($HF_NAMESPACE/<run_id>) during
# training and deleted locally after per-file sha verify (the fig2nl3s
# machinery, reused verbatim). Figure: dataset_size_sweep.py --family ts.
#
# No merge stage anywhere: the installer is FULL-FT, so its checkpoint is a
# plain state dict the inst arm --init-from's directly.
#
# Cost: ~12-20 h GPU (the teaching arm converges slower than any Llama
# family; ceilings carry 8-10x headroom at large n) + ~190 GB endpoint
# snapshot upload. $0 on owner hardware. Token: $HF_WRITE_TOKEN/$HF_TOKEN
# with write scope on the owner's account (creates the snapshot repos).
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=fig2ts

echo "[fig2ts] estimated cost: ~12-20 h GPU + ~190 GB endpoint-snapshot upload"
echo "[fig2ts] (two per-run public HF repos); peak local disk ~15 GB."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_fig2ts_llama.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
PRUNE=0
for arg in "$@"; do
  [[ $arg == --prune ]] && PRUNE=1
done

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
export HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?need HF_TOKEN or HF_WRITE_TOKEN (write scope)}}
HF_NAMESPACE=${HF_NAMESPACE:-podhajskimarcin}

TS_BASE_RID=evt-ts1b-base
TS_BASE_MODEL=$GEODE_STORE/runs/$TS_BASE_RID/model
INSTALLER_RID=evt-ts1b-fig2ts-installer
INSTALLER_MODEL=$GEODE_STORE/runs/$INSTALLER_RID/model
SIZES=(1000 1468 2154 3162 4642 6813 10000 14678 21544 31623 46416 68129 100000 146780 215443 316228 464159 681292 1000000)
BIG_N=1000000

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE namespace=$HF_NAMESPACE prune=$PRUNE sizes=${#SIZES[@]}"

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

# Score-then-record one ENFORCED installer gate (the fig2nl2/3 launchers'
# helper, verbatim): --no-record first, --record-only-pass on a pass.
enforce_installer_gate() {
  local gate=$1 metric=$2 threshold=$3 verdict out score
  shift 3
  verdict=$(gate_verdict "$INSTALLER_RID" "$gate")
  if [[ $verdict == pass ]]; then
    milestone "gate_skip run=$INSTALLER_RID gate=$gate status=recorded_pass"
    return 0
  elif [[ $verdict != missing ]]; then
    fail "$INSTALLER_RID gate=$gate verdict='$verdict' (expected pass or missing) — inspect $GEODE_STORE/runs/$INSTALLER_RID/manifest.json before rerunning"
  fi
  out=$("$@" --no-record 2>&1)
  echo "$out"
  score=$(gate_score "$out" "$gate" "$metric")
  [[ -n $score ]] || fail "$INSTALLER_RID $gate printed no $metric score (output above)"
  gate_passed "$out" "$gate" "$metric" ||
    fail "$INSTALLER_RID $gate $metric $score failed its bar ($threshold) — NOT recorded, no targets from an ungated parent; fallback rung: ts1b_fig2ts_installer_bare.yaml (file header)"
  "$@" --record-only-pass ||
    fail "$INSTALLER_RID $gate DIVERGENCE: --no-record scored $score (PASS) but the recording pass recomputed a FAIL — nothing was written; rerun the gate block"
  milestone "gate_pass run=$INSTALLER_RID gate=$gate $metric=$score"
}

record_g5() {
  local rid=$1
  if gate_recorded "$rid" G5; then
    milestone "gate_skip run=$rid gate=G5"
  else
    python3 gates.py g5 --run "$rid" --config ../configs/eval_bare_target_data_llama.yaml ||
      fail "$rid G5 (evidence recording failed)"
    milestone "gate_recorded run=$rid gate=G5"
  fi
}

# Endpoint runs (n=BIG_N): train with the snapshot streamer beside them
# (stream_snapshots.py: upload each snapshot, sha-verify, delete locally),
# then G5 BEFORE any pruning, drain, push metadata+weights, verify, prune.
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
  record_g5 "$rid"

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
print(f"[fig2ts] hub sha256 verified for {sys.argv[1]}")
PY
  rm -f "$run_dir/model/model.safetensors"
  milestone "pruned run=$rid (full weights local; adapter/manifest/logs kept)"
}

# Non-endpoint runs: train, G5, optional local weight prune.
train_plain() {
  local rid=$1 init=$2 cfg=$3 overlay=$4
  train_or_skip "$rid" \
    python3 train_target.py --config "$cfg" --override "$overlay" \
      --init-from "$init" --confirm-cost
  record_g5 "$rid"
  if ((PRUNE)); then
    local model_file=$GEODE_STORE/runs/$rid/model/model.safetensors
    if [[ -f $model_file ]]; then
      rm -f "$model_file"
      milestone "pruned run=$rid file=$model_file (adapter sidecar kept)"
    fi
  fi
}

# ---- stage 1: data (regenerate whatever is missing; deterministic) ---------
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
BASE_NEEDED=(D_algo.parquet D_algo_eval.parquet D_inst.parquet report.json)
MISSING=0
for f in "${BASE_NEEDED[@]}"; do
  [[ -f $DATA_DIR/$f ]] || MISSING=1
done
if ((MISSING)); then
  milestone "datagen_start (regenerating the frozen artifacts from seed 20260717)"
  mkdir -p "$DATA_DIR"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 ||
    fail "datagen base"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --eval-set ||
    fail "datagen --eval-set"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --nl-eval-set ||
    fail "datagen --nl-eval-set"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --installer-set ||
    fail "datagen --installer-set"
  milestone "datagen_complete"
else
  milestone "datagen_skip (frozen base artifacts present)"
fi

BARE_NEEDED=(D_algo_bare.parquet D_algo_eval_bare.parquet D_inst_bare.parquet)
BARE_MISSING=0
for f in "${BARE_NEEDED[@]}"; do
  [[ -f $DATA_DIR/$f ]] || BARE_MISSING=1
done
if ((BARE_MISSING)); then
  milestone "bare_datagen_start"
  python3 ../datagen/make_bare_sets.py --out "$DATA_DIR" || fail "make_bare_sets"
  milestone "bare_datagen_complete"
else
  milestone "bare_datagen_skip (all bare artifacts present)"
fi

# ---- stage 1b: parent + premise guards -------------------------------------
[[ -f $TS_BASE_MODEL/model.safetensors ]] ||
  fail "no ts1b-base checkpoint at $TS_BASE_MODEL — the twin must be pretrained (ts1b_pretrain.yaml) or pulled from podhajskimarcin/evt-ts1b-base first"
[[ $(status_of "$TS_BASE_RID") == complete ]] ||
  fail "$TS_BASE_RID manifest is not status=complete"

# The twin must be ~0 EM on bare prompts (trivially expected for a model
# that has never seen arithmetic — measured anyway, premise-guard policy).
python3 check_bare_baseline.py --model "$TS_BASE_MODEL" --n 256 --max-em 0.05 ||
  fail "PREMISE: the ts1b twin answers bare arithmetic prompts — inspect the pretrain corpus/checkpoint before sweeping"
milestone "premise_guard ts1b_bare_zero_shot<=0.05 -> PASS"

# ---- stage 2: pre-teach-format installer + gates ---------------------------
train_or_skip "$INSTALLER_RID" \
  python3 train_sft.py --config ../configs/ts1b_fig2ts_installer.yaml \
    --init-from "$TS_BASE_MODEL" --confirm-cost

# G4 >= 0.90 on BARE prompts (does the scaffolded-op dose's output convention
# transfer? the paper's input-format-irrelevance claim, tested on a blank
# model) AND G3 <= 0.02 zero-shot EM on bare add/sub (random labels taught
# no arithmetic). Both enforced; parent_required_gates: [G4, G3].
enforce_installer_gate G4 format_validity ">=0.90" \
  python3 gates.py g4 --run "$INSTALLER_RID" \
    --config ../configs/ts1b_fig2ts_installer.yaml \
    --prompt-config ../configs/eval_bare_target_data_llama.yaml \
    --threshold 0.90
enforce_installer_gate G3 accuracy "<=0.02" \
  python3 gates.py g3 --run "$INSTALLER_RID" \
    --config ../configs/eval_bare_algo_data_ts.yaml --threshold 0.02
record_g5 "$INSTALLER_RID"

# ---- stages 3+4: both arms, ascending n (arm-serial; G7 via overlays) ------
for n in "${SIZES[@]}"; do
  rid=evt-ts1b-fig2ts-noinst-n${n}
  overlay="../configs/sweeps/ts1b_fig2ts/ts1b_fig2ts_noinst_n${n}.yaml"
  if [[ $n == "$BIG_N" ]]; then
    train_streamed "$rid" "$TS_BASE_MODEL" ../configs/ts1b_fig2ts_noinst.yaml "$overlay"
  else
    train_plain "$rid" "$TS_BASE_MODEL" ../configs/ts1b_fig2ts_noinst.yaml "$overlay"
  fi
done

[[ -f $INSTALLER_MODEL/model.safetensors ]] ||
  fail "no installer checkpoint at $INSTALLER_MODEL (installer stage must complete before the inst arm)"
for n in "${SIZES[@]}"; do
  rid=evt-ts1b-fig2ts-inst-n${n}
  overlay="../configs/sweeps/ts1b_fig2ts/ts1b_fig2ts_inst_n${n}.yaml"
  if [[ $n == "$BIG_N" ]]; then
    train_streamed "$rid" "$INSTALLER_MODEL" ../configs/ts1b_fig2ts_inst.yaml "$overlay"
  else
    train_plain "$rid" "$INSTALLER_MODEL" ../configs/ts1b_fig2ts_inst.yaml "$overlay"
  fi
done

echo "[fig2ts] MILESTONE analysis_commands"
echo "[fig2ts]   the deliverable figure (EDL/D vs n, one curve per arm):"
echo "[fig2ts]     python3 ../analysis/dataset_size_sweep.py --family ts"
echo "[fig2ts]   Writes results/dataset_size_sweep_ts.parquet + analysis/figures/dataset_size_sweep_ts.png"
notify "fig2ts launcher done: prune=$PRUNE"
echo "[fig2ts] TERMINAL_SUCCESS prune=$PRUNE"
