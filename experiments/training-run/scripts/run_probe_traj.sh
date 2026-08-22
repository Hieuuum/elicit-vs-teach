#!/usr/bin/env bash
# Probe trajectory at ONE dataset size (default n=46416): the carry-subset +
# cross-format-transfer probe (analysis/probe_routing_control.py) on every
# snapshot of the three ts38mt arms at that size (base / pp / fmt), the
# three theta0 parents, and a k7 positive-control target trained HERE at
# the same size (scripts/run_ts38tr_family.sh, KS=7 SIZES=$N) plus its
# Tier-1/2 tables for the dynamics half of the analysis. Pre-registration:
# decisions.md 2026-08-22 "probe trajectory"; EXPERIMENTS.md §6.24.
#
# Stages (each idempotent -- relaunch the same command to resume):
#   prep      pull theta0 parents (geode-store) + the 3 ts38mt targets WITH
#             snapshots (geode-internals) + the k7 parent
#   k7        train evt-ts38tr-k7-n$N (converged, G5, pushed both repos)
#   probe     probe_routing_control.py --run-id per arm (+ --model theta0s)
#   mech      grad_dynamics / weight_diff / resid_shift / jacobian_lens /
#             cross_patch on the k7 target (the 3 ts38mt arms' tables at
#             this n already exist in results/ts38mt_mech/)
#   upload    results/ts38mt_probe_traj -> geode-internals
#
# Env: N (default 46416), LIMIT (default 6000), SKIP_K7=1 to run without the
# control arm, NTFY/NTFY_AUTO as in lib/launch_common.sh.
#
#   tmux new-session -d -s probetraj 'bash run_probe_traj.sh --confirm-cost \
#       > /workspace/probe_traj.log 2>&1'
set -euo pipefail
source /workspace/venv/bin/activate
export GEODE_STORE=/workspace/elicit-vs-teach/geode-store
cd /workspace/elicit-vs-teach/experiments/training-run/scripts
source lib/launch_common.sh
TAG=ptraj

N=${N:-46416}
LIMIT=${LIMIT:-6000}
SKIP_K7=${SKIP_K7:-0}
S=$GEODE_STORE/runs
O=$GEODE_STORE/results/ts38mt_probe_traj
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
INTERNALS_REPO=${INTERNALS_REPO:-mhieuuu/geode-internals}
TASK=../data/full/D_algo_eval_bare.parquet
GENERIC=/workspace/tinystories_val_2000.txt
K7_RID=evt-ts38tr-k7-n${N}
mkdir -p "$O"

echo "[ptraj] estimated cost: one k7 LoRA target at n=$N (ts38mt's n=46416 runs"
echo "[ptraj] took ~15-25 min on a 4090) + 4 snapshot sweeps x ~28 snapshots x"
echo "[ptraj] probe at --limit $LIMIT (~8 min each) + Tier-1/2 on the k7 target"
echo "[ptraj] (~10 min) -- ~1.5 h wall, ~\$0.6 at \$0.36/h. SKIP_K7=1 removes the"
echo "[ptraj] training (~40 min wall, no new weights)."
[[ " $* " == *" --confirm-cost "* ]] || {
  echo "run_probe_traj.sh: --confirm-cost required (budget rule) -- invoke as" >&2
  echo "  bash run_probe_traj.sh --confirm-cost" >&2
  exit 1
}

weights_present() {
  local rid=$1
  [[ -f $S/$rid/model/model.safetensors || -f $S/$rid/model/adapter.safetensors ]]
}

pull_with_weights() {
  local rid=$1 repo=${2:-$RELAY_REPO}
  if weights_present "$rid"; then
    milestone "pull_skip run=$rid (weights present)"
    return 0
  fi
  milestone "pull_start run=$rid repo=$repo"
  python3 hf_checkpoint.py pull --run-id "$rid" --repo-id "$repo" || fail "pull $rid from $repo"
  weights_present "$rid" || fail "pull left no weights for $rid"
  milestone "pull_complete run=$rid"
}

snapshots_present() {
  local rid=$1
  compgen -G "$S/$rid/snapshots/step_*/adapter.safetensors" >/dev/null
}

pull_with_snapshots() {
  local rid=$1
  if weights_present "$rid" && snapshots_present "$rid"; then
    milestone "pull_skip run=$rid (weights + snapshots present)"
    return 0
  fi
  milestone "pull_start run=$rid repo=$INTERNALS_REPO with_snapshots=1"
  python3 hf_checkpoint.py pull --run-id "$rid" --repo-id "$INTERNALS_REPO" --with-snapshots ||
    fail "pull $rid (--with-snapshots) from $INTERNALS_REPO"
  weights_present "$rid" || fail "pull left no weights for $rid"
  snapshots_present "$rid" || fail "pull left no snapshots/step_*/adapter.safetensors for $rid"
  milestone "pull_complete run=$rid n_snapshots=$(ls -d "$S/$rid"/snapshots/step_* | wc -l)"
}

# ---- prep --------------------------------------------------------------------
echo "[ptraj] ===== prep: data + weights ====="
[[ -f $TASK ]] || fail "$TASK missing -- regenerate (make_data.py + make_bare_sets.py) first"
for rid in evt-run1-base-v3-ext evt-ts38pp-parent evt-ts38mt-fmt-parent; do
  pull_with_weights "$rid"
done
for arm in base pp fmt; do
  pull_with_snapshots "evt-ts38mt-${arm}-n${N}"
done
if [[ $SKIP_K7 != 1 ]]; then
  pull_with_weights evt-ts38tr-k7-parent
fi
if [[ ! -f $GENERIC ]]; then
  python3 - <<'PY'
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="validation")
with open("/workspace/tinystories_val_2000.txt", "w") as f:
    for r in ds.select(range(2000)):
        f.write(r["text"].replace("\n", " ").strip() + "\n")
PY
fi
milestone "prep_complete n=$N"

# ---- k7 control at this size ---------------------------------------------------
if [[ $SKIP_K7 != 1 ]]; then
  echo "[ptraj] ===== k7: train $K7_RID (run_ts38tr_family.sh KS=7 SIZES=$N) ====="
  if [[ $(status_of "$K7_RID") == complete ]]; then
    milestone "k7_skip run=$K7_RID status=complete"
  else
    KS=7 SIZES=$N bash run_ts38tr_family.sh --confirm-cost || fail "run_ts38tr_family.sh (k7 at n=$N)"
    [[ $(status_of "$K7_RID") == complete ]] || fail "$K7_RID did not complete"
  fi
  snapshots_present "$K7_RID" || fail "$K7_RID has no snapshots -- ts38tr_k7.yaml pins snapshots on"
  milestone "k7_complete run=$K7_RID"
fi

# ---- probe sweeps ----------------------------------------------------------------
echo "[ptraj] ===== probe: snapshot sweeps (--limit $LIMIT, transfer on) ====="
pushd ../analysis >/dev/null
probe_run() {
  local arm=$1 rid=$2 out="$O/probe_traj_${arm}.csv"
  if [[ -f $out ]]; then
    milestone "probe_skip arm=$arm (exists)"
    return 0
  fi
  local t0=$SECONDS
  python3 probe_routing_control.py --run-id "$rid" --model-name "$arm" \
    --prompt-parquet "$TASK" --set-name task --transfer-set task \
    --limit "$LIMIT" --seed 0 --device cuda --out "$out" || fail "probe sweep $arm ($rid)"
  milestone "probe_done arm=$arm run=$rid elapsed_s=$((SECONDS - t0))"
}
for arm in base pp fmt; do
  probe_run "$arm" "evt-ts38mt-${arm}-n${N}"
done
[[ $SKIP_K7 == 1 ]] || probe_run k7 "$K7_RID"

THETA0_OUT="$O/probe_traj_theta0.csv"
if [[ -f $THETA0_OUT ]]; then
  milestone "probe_skip theta0 (exists)"
else
  THETA0_ARGS=(
    --model "base=dir:$S/evt-run1-base-v3-ext/model"
    --model "pp=dir:$S/evt-ts38pp-parent/model"
    --model "fmt=dir:$S/evt-ts38mt-fmt-parent/model"
  )
  [[ $SKIP_K7 == 1 ]] || THETA0_ARGS+=(--model "k7=dir:$S/evt-ts38tr-k7-parent/model")
  python3 probe_routing_control.py "${THETA0_ARGS[@]}" \
    --prompt-parquet "$TASK" --set-name task --transfer-set task \
    --limit "$LIMIT" --seed 0 --device cuda --out "$THETA0_OUT" || fail "theta0 probe"
  milestone "probe_done theta0 models=$((${#THETA0_ARGS[@]} / 2))"
fi
milestone "probe_complete"

# ---- Tier-1/2 on the k7 target (the dynamics half's control row) -----------------
if [[ $SKIP_K7 != 1 ]]; then
  echo "[ptraj] ===== mech: Tier-1/2 on $K7_RID ====="
  P="dir:$S/evt-ts38tr-k7-parent/model"
  R=$K7_RID
  t0=$SECONDS
  out="$O/grad_dynamics_$R.csv"
  [[ -f $out ]] || python3 grad_dynamics.py --run-id "$R" --out "$out" || fail "grad_dynamics $R"
  out="$O/weight_diff_$R.parquet"
  [[ -f $out ]] || python3 weight_diff.py --model-a "$P" --model-b "run:$R" --device cuda --out "$out" || fail "weight_diff $R"
  out="$O/resid_shift_$R.csv"
  [[ -f $out ]] || python3 resid_shift.py --model-a "$P" --model-b "run:$R" --task-parquet "$TASK" --generic-local "$GENERIC" --device cuda --limit 2000 --out "$out" || fail "resid_shift $R"
  out="$O/jacobian_lens_$R.csv"
  [[ -f $out ]] || python3 jacobian_lens.py --model-a "$P" --model-b "run:$R" --prompt-parquet "$TASK" --set-name task --device cuda --limit 2000 --out "$out" || fail "jacobian_lens $R"
  out="$O/cross_patch_$R.csv"
  [[ -f $out ]] || python3 cross_patch.py --model-a "$P" --model-b "run:$R" --prompt-parquet "$TASK" --device cuda --limit 1000 --out "$out" || fail "cross_patch $R"
  milestone "mech_complete run=$R elapsed_s=$((SECONDS - t0))"
fi
popd >/dev/null

# ---- upload ------------------------------------------------------------------------
echo "[ptraj] ===== upload results/ts38mt_probe_traj -> $INTERNALS_REPO ====="
python3 - "$O" "$INTERNALS_REPO" <<'PY'
import sys
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path=sys.argv[1],
    path_in_repo="results/ts38mt_probe_traj",
    repo_id=sys.argv[2],
    repo_type="model",
)
print("[ptraj] uploaded results/ts38mt_probe_traj")
PY
milestone "upload_complete"
milestone "ALL_DONE n=$N limit=$LIMIT skip_k7=$SKIP_K7"
notify "probe_traj done: n=$N (k7 skipped=$SKIP_K7)"
echo "[ptraj] TERMINAL_SUCCESS"
