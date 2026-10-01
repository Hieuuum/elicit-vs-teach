#!/usr/bin/env bash
# ts38tr — "truncated-adapter positive control" family. Two new parents are
# built by KEEPING only the first K of 8 transformer blocks' LoRA delta from
# the already-converged evt-ts38mt-base-n316228 target (a full r128 LoRA
# target on evt-run1-base-v3-ext): evt-ts38tr-k6-parent (blocks 0-5) and
# evt-ts38tr-k7-parent (blocks 0-6). Each parent is a DELIBERATE partial
# install -- a truncated adapter should either read as HIDDEN (zero-shot em0
# still ~0 on the bare-NL eval, same as base) or LEAKED (some capability
# already visible without further training), which is the reason this family
# exists as a mechanistic-test positive control alongside ts38mt/ts38tr_mech.
# HIDDEN parents get 3 small LoRA targets each (n in SIZES) trained the same
# way ts38mt's arms are; a LEAKED parent's targets are skipped (loudly), not
# a fatal error for the whole family.
#
# Mirrors launch_ts38mt_family.sh's score_g5_no_record / train_or_skip /
# require_converged / record_g5 / push_run / push_internals idiom, but at
# ts38tr's much smaller scale (2 parents, <=6 targets) this script is named
# run_ (not launch_) and lives alongside its analysis-only run_ts38mt_*.sh
# siblings; it still requires its own --confirm-cost gate (budget rule,
# CLAUDE.md) because unlike those siblings it trains real LoRA targets.
#
# SIZES / KS overridable via env, space-separated:
#     SIZES="1000 4642" KS="6" bash run_ts38tr_family.sh --confirm-cost
set -euo pipefail
source /workspace/venv/bin/activate
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
source lib/launch_common.sh
TAG=ts38tr

S=$GEODE_STORE/runs
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
INTERNALS_REPO=${INTERNALS_REPO:-mhieuuu/geode-internals}
SIZES=${SIZES:-"1000 2154 4642"}
KS=${KS:-"6 7"}

SOURCE_RID=evt-ts38mt-base-n316228
BASE_RID=evt-run1-base-v3-ext
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
THETA0_JSON=$GEODE_STORE/results/ts38tr_family_theta0.json

echo "[ts38tr] estimated cost: 2 LoRA-block-truncated parents (tensor-slice of"
echo "[ts38tr] an already-converged adapter -- no new training) plus up to 6"
echo "[ts38tr] small LoRA target runs (2 K's x 3 sizes by default, r128 LoRA on"
echo "[ts38tr] D_algo_bare, identical recipe to ts38mt's target arms) on one"
echo "[ts38tr] RTX 4090. ts38mt's own n=1000/2154/4642 runs measured ~1.7/2.1/"
echo "[ts38tr] 6.4 min each -> ~6 runs x ~3-6 min ~= 25-35 min wall, plus G5"
echo "[ts38tr] (~0.5 min x 6) plus dual-repo pushes -- ~= 30-40 min wall, well"
echo "[ts38tr] under \$1 at \$0.35-0.45/h. Fewer targets run if a parent is"
echo "[ts38tr] LEAKED (skipped, not trained)."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "run_ts38tr_family.sh: --confirm-cost required (budget rule) -- invoke as" >&2
  echo "  bash run_ts38tr_family.sh --confirm-cost" >&2
  exit 1
}

# ---- helpers ----------------------------------------------------------------

weights_present() {
  # Covers both a full-FT run (model.safetensors) and a LoRA run (adapter
  # .safetensors, e.g. evt-ts38mt-base-n316228) -- checking only
  # model.safetensors would report a LoRA run as perpetually missing and
  # re-trigger a pull on every invocation.
  local rid=$1
  [[ -f $S/$rid/model/model.safetensors || -f $S/$rid/model/adapter.safetensors ]]
}

pull_with_weights() {
  local rid=$1
  if weights_present "$rid"; then
    milestone "pull_skip run=$rid (weights present)"
    return 0
  fi
  milestone "pull_start run=$rid"
  python3 hf_checkpoint.py pull --run-id "$rid" --repo-id "$RELAY_REPO" ||
    fail "pull $rid from $RELAY_REPO"
  weights_present "$rid" || fail "pull left no weights for $rid (snapshot_download is fail-open)"
  milestone "pull_complete run=$rid"
}

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
  milestone "convergence_check run=$rid stop_reason=$stop"
  [[ $stop == converged ]] || fail \
    "CONVERGENCE CHECK: $rid ended with stop_reason='$stop', not 'converged'. Standing
   policy: a max_steps stop is a BUG SIGNAL, not a result. Inspect the overlay's max_steps
   and the loss trace before continuing."
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

# --no-record G5 evidence for the leak gate's theta0 probe, parsed the way
# launch_ts38mt_family.sh's score_g5_no_record parses gates.py g5's printed
# output. Full gate output goes to stderr so the $(...) capture is clean;
# stdout is "em0 em16 loss" on success.
score_g5_no_record() {
  local rid=$1 out em0 em16 loss
  out=$(python3 gates.py g5 --run "$rid" --config "$BARE_EVAL_CONFIG" --no-record 2>&1)
  grep -E "G5 (zero-shot|16-shot|shared-set)" <<<"$out" >&2 || true   # diagnostic only
  em0=$(sed -n 's/.*G5 zero-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  em16=$(sed -n 's/.*G5 16-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  loss=$(sed -n 's/.*G5 shared-set test loss \([0-9.]*\) nats over n=.*/\1/p' <<<"$out" | head -1)
  if [[ -z $em0 || -z $em16 || -z $loss ]]; then
    echo "$out" | tail -5 >&2
    return 1
  fi
  echo "$em0 $em16 $loss"
}

# Read a parent's verdict back OUT OF THE JSON FILE, never out of an
# in-process bash variable -- on a resumed run (theta0.json already complete)
# stage 2 never re-scores, so any variable set only on the fresh-scoring path
# would be empty here and silently starve stage 3.
verdict_of_k() {
  local k=$1
  python3 - "$THETA0_JSON" "$k" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.is_file():
    print("MISSING")
    raise SystemExit
d = json.loads(p.read_text())
print(d.get("parents", {}).get(sys.argv[2], {}).get("verdict", "MISSING"))
PY
}

theta0_complete() {
  python3 - "$THETA0_JSON" $KS <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
d = json.loads(p.read_text())
ks = sys.argv[2:]
parents = d.get("parents", {})
ok = bool(d.get("base")) and all(k in parents and "verdict" in parents[k] for k in ks)
sys.exit(0 if ok else 1)
PY
}

# Best-effort push to mhieuuu/geode-internals WITH snapshots, public -- the
# second repo/flag combination lib/launch_common.sh::push_run cannot express.
# Copied verbatim (in logic) from launch_ts38mt_family.sh's own push_internals.
push_internals() {
  local rid=$1
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$INTERNALS_REPO" --with-snapshots --public; then
    milestone "push_complete run=$rid repo=$INTERNALS_REPO with_snapshots=1"
  else
    echo "[ts38tr] WARN hf_checkpoint.py push --with-snapshots failed for $rid -> $INTERNALS_REPO (best effort)"
    milestone "push_warn run=$rid repo=$INTERNALS_REPO"
  fi
}

# ---- stage 0: prep -----------------------------------------------------------
echo "[ts38tr] ===== stage 0: prep -- pull weights + G7 anchors ====="
pull_with_weights "$SOURCE_RID"
pull_with_weights "$BASE_RID"

for n in $SIZES; do
  rid=evt-ts38-base-n${n}
  if [[ ! -f $S/$rid/manifest.json ]]; then
    milestone "pull_anchor_start run=$rid"
    python3 hf_checkpoint.py pull --run-id "$rid" --repo-id "$RELAY_REPO" --no-weights ||
      fail "pull $rid (G7 anchor) from $RELAY_REPO"
  fi
done
milestone "prep_complete sizes=\"$SIZES\" ks=\"$KS\""

# ---- stage 1: truncated parents ---------------------------------------------
echo "[ts38tr] ===== stage 1: truncated parents (K in {$KS}) ====="
for K in $KS; do
  PARENT_RID="evt-ts38tr-k${K}-parent"
  if [[ -f $S/$PARENT_RID/manifest.json ]]; then
    milestone "parent_skip run=$PARENT_RID (exists)"
    continue
  fi
  milestone "parent_start run=$PARENT_RID keep_blocks=$K"
  python3 truncate_lora_parent.py --source-run-id "$SOURCE_RID" \
    --keep-blocks "$K" --out-run-id "$PARENT_RID" --device cuda ||
    fail "$PARENT_RID truncation"
  [[ -f $S/$PARENT_RID/manifest.json ]] ||
    fail "$PARENT_RID: truncate_lora_parent.py did not write a manifest"
  milestone "parent_complete run=$PARENT_RID"
done

# ---- stage 2: pre-registered leak gate ---------------------------------------
echo "[ts38tr] ===== stage 2: leak gate (--no-record G5 on each parent + base) ====="
if theta0_complete; then
  milestone "theta0_skip json=$THETA0_JSON (base + verdicts for k in {$KS} present)"
else
  BASE_G5=$(score_g5_no_record "$BASE_RID") || fail "$BASE_RID G5 --no-record failed (see stderr above)"
  read -r BASE_EM0 BASE_EM16 BASE_LOSS <<<"$BASE_G5"

  declare -A P_EM0 P_EM16 P_LOSS
  for K in $KS; do
    PARENT_RID="evt-ts38tr-k${K}-parent"
    G5=$(score_g5_no_record "$PARENT_RID") || fail "$PARENT_RID G5 --no-record failed (see stderr above)"
    read -r em0 em16 loss <<<"$G5"
    P_EM0[$K]=$em0
    P_EM16[$K]=$em16
    P_LOSS[$K]=$loss
  done

  PY_ARGS=("$THETA0_JSON" "$BASE_RID" "$BASE_EM0" "$BASE_EM16" "$BASE_LOSS")
  for K in $KS; do
    PY_ARGS+=("$K" "evt-ts38tr-k${K}-parent" "${P_EM0[$K]}" "${P_EM16[$K]}" "${P_LOSS[$K]}")
  done
  python3 - "${PY_ARGS[@]}" <<'PY' || fail "writing $THETA0_JSON failed"
import json
import sys
from pathlib import Path

args = sys.argv[1:]
out_path, base_rid, base_em0, base_em16, base_loss = args[0:5]
rest = args[5:]
parents = {}
for i in range(0, len(rest), 5):
    k, rid, em0, em16, loss = rest[i : i + 5]
    em0f = float(em0)
    # em0 > 0.05: a readout is present -- the truncated adapter is NOT a
    # hidden-capability model. Same 5% margin as ts38mt/ts38pf's own
    # format-acquisition checks.
    verdict = "LEAKED" if em0f > 0.05 else "HIDDEN"
    parents[k] = {
        "run_id": rid,
        "em0": em0f,
        "em16": float(em16),
        "loss": float(loss),
        "verdict": verdict,
    }

data = {
    "base": {
        "run_id": base_rid,
        "em0": float(base_em0),
        "em16": float(base_em16),
        "loss": float(base_loss),
    },
    "parents": parents,
    "eval_config": "eval_bare_target_data_ts38.yaml",
}
out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2) + "\n")
PY
  milestone "theta0_scored base_em0=$BASE_EM0 base_loss=$BASE_LOSS"
fi

HIDDEN_KS=()
for K in $KS; do
  V=$(verdict_of_k "$K")
  [[ $V == MISSING ]] && fail "no verdict recorded for k=$K in $THETA0_JSON -- inspect stage 2"
  milestone "parent_verdict k=$K verdict=$V"
  if [[ $V == LEAKED ]]; then
    echo "[ts38tr] *** WARNING: evt-ts38tr-k${K}-parent is LEAKED (zero-shot em0 > 0.05 on"
    echo "[ts38tr] *** the bare-NL eval) -- a readout is present, this is NOT a"
    echo "[ts38tr] *** hidden-capability model. SKIPPING target training for k=$K. ***"
  elif [[ $V == HIDDEN ]]; then
    HIDDEN_KS+=("$K")
  else
    fail "unexpected verdict '$V' for k=$K in $THETA0_JSON"
  fi
done

# Push both parents, both repos, regardless of verdict -- a LEAKED parent is
# still useful evidence for the mechanistic-test scripts.
for K in $KS; do
  PARENT_RID="evt-ts38tr-k${K}-parent"
  push_run "$PARENT_RID"
  push_internals "$PARENT_RID"
done

# ---- stage 3: targets (HIDDEN parents only) ----------------------------------
echo "[ts38tr] ===== stage 3: targets (n in {$SIZES} x k in HIDDEN {${HIDDEN_KS[*]:-}}) ====="
if ((${#HIDDEN_KS[@]} == 0)); then
  echo "[ts38tr] *** every parent is LEAKED -- zero targets to train; skipping stage 3 ***"
fi
for n in $SIZES; do
  for K in "${HIDDEN_KS[@]}"; do
    RID="evt-ts38tr-k${K}-n${n}"
    PARENT_MODEL_DIR=$S/evt-ts38tr-k${K}-parent/model
    train_or_skip "$RID" \
      python3 train_target.py --config "../configs/ts38tr_k${K}.yaml" \
        --override "../configs/sweeps/ts38/ts38tr_k${K}_n${n}.yaml" \
        --init-from "$PARENT_MODEL_DIR" --confirm-cost
    require_converged "$RID" "$n"
    record_g5 "$RID"
    push_run "$RID"
    push_internals "$RID"
  done
done

# ---- stage 4 (final): summary table, then the EDL fold (non-fatal: ts38tr is
# not yet a registered --family in edl_converged_val_floor.py, mirror ts38mt's
# own registration, commit a88f22d, to make this fold actually run). ---------
echo "[ts38tr] ===== summary: parents + targets ====="
SUMMARY_ARGS=()
for K in $KS; do
  SUMMARY_ARGS+=("parent:evt-ts38tr-k${K}-parent")
done
for n in $SIZES; do
  for K in $KS; do
    SUMMARY_ARGS+=("target:evt-ts38tr-k${K}-n${n}")
  done
done
python3 - "${SUMMARY_ARGS[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts38tr] {'run_id':<30}{'final_step':>12}{'stop_reason':>14}{'g5_em0':>10}")
for item in sys.argv[1:]:
    kind, rid = item.split(":", 1)
    p = store / "runs" / rid / "manifest.json"
    if not p.is_file():
        print(f"[ts38tr] {rid:<30}{'MISSING':>12}{'':>14}{'':>10}")
        continue
    d = json.loads(p.read_text())
    exp = d.get("experiment", {}) or {}
    result_key = "sft_result" if kind == "parent" else "target_result"
    r = exp.get(result_key, {}) or {}
    g5 = (exp.get("gates", {}) or {}).get("G5", {}) or {}
    em0 = g5.get("zero_shot_accuracy")
    print(
        f"[ts38tr] {rid:<30}{r.get('final_step', 'n/a'):>12}"
        f"{r.get('stop_reason', 'n/a'):>14}"
        f"{em0 if em0 is not None else 'n/a':>10}"
    )
PY

echo "[ts38tr] ===== EDL fold (--family ts38tr) ====="
pushd ../analysis >/dev/null
if python3 edl_converged_val_floor.py --family ts38tr; then
  milestone "edl_fold_complete family=ts38tr"
else
  echo "[ts38tr] WARN: 'python3 edl_converged_val_floor.py --family ts38tr' failed." >&2
  echo "[ts38tr] WARN: ts38tr is very likely not yet registered in that script's" >&2
  echo "[ts38tr] WARN: FAMILIES dict / --family choices tuple (mirror the ts38mt" >&2
  echo "[ts38tr] WARN: registration, commit a88f22d). Fold skipped; every run's" >&2
  echo "[ts38tr] WARN: G5/manifest data above is still pushed and intact." >&2
  milestone "edl_fold_skipped family=ts38tr (see WARN above)"
fi
EDL_CSV=edl_converged_val_floor_ts38tr.csv
if [[ -f $EDL_CSV ]]; then
  echo "[ts38tr] ----- $EDL_CSV -----"
  cat "$EDL_CSV"
else
  echo "[ts38tr] $EDL_CSV not written (fold skipped or produced no rows)"
fi
popd >/dev/null

milestone "ts38tr_family_complete ks=\"$KS\" sizes=\"$SIZES\" hidden_ks=\"${HIDDEN_KS[*]:-none}\""
notify "ts38tr family done: hidden_ks=${HIDDEN_KS[*]:-none} sizes=$SIZES"
echo "[ts38tr] TERMINAL_SUCCESS"
