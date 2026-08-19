#!/usr/bin/env bash
# ts1b (fig2ts) Stages 0-2 — the paper-scale staged redo at TinyStories-1B
# (decisions.md 2026-08-19 "ts1b (fig2ts) staged redo: PRE-REGISTRATION";
# EXPERIMENTS.md §19). Clone of launch_ts38pp_family.sh's stage structure
# and safety patterns (train_or_skip, push-as-you-go, receiver-verify),
# generalized for the ts1b TWIN-PARENT design (pp/pf differ only in
# labels) and carrying the 2026-08-19 sed-based env-guard fix (see below).
#
# STAGES
#   0  base checkpoint pull (from the collaborator's repo) + ONE-TIME
#      mirror-push insurance copy to an mhieuuu-owned repo
#   1  LR mini-sweep (3 rungs, 2000 steps each) -- STOPS for pin
#      confirmation; the script never auto-pins a hyperparameter
#   2  evt-ts1b-pp-parent full run (App. E.2 literal: full FT, one epoch)
#   2b HALT gate: pp op EM (block render, k=0) >= 0.90 -- the ONE thing
#      that can stop pf from spending any GPU time (pre-registration's own
#      criterion; read off the theta0 diagnostic's "block" condition, NOT
#      gates.py g1, per the pre-registration's explicit "(block, k=0)"
#      phrasing)
#   3  evt-ts1b-pf-parent full run (App. E.1.2: permuted labels, trained
#      UNTIL CONVERGENCE, 3-epoch ceiling) -- only runs if 2b passes
#   4  theta0 few-shot diagnostic on base/pf/pp (Table-11 replication read
#      + M1 position-lock read, per the pre-registration's decision table)
#   5  push everything to the relay, VERIFY THE RECEIVER
#      ([[feedback-verify-the-receiver-not-the-sender]])
#
# Stage 3+ (target-stage grids) is NOT built here — the pre-registration
# authorizes Stages 0-2 only (~$25); grid spend needs owner re-confirmation
# after this script's Stage-2 deliverables land.
#
# SCOPE NOTE — the two parallel builds this launcher depends on LANDED and
# were RECONCILED 2026-08-19 (orchestrator review):
#   B0.1 (datagen): D_target_4M_block.parquet (pp) + D_target_4M_blockperm
#     .parquet (pf — the actual shipped name; the config's original
#     "_block_perm" guess is fixed). Both parents' data.order_hash pins are
#     filled from data/full/report.json; the TODO_* refusal guard below is
#     kept as permanent cheap insurance.
#   B0.3 (theta0_fewshot_diag.py): generalized via --model-family ts1b,
#     which selects the Llama eval configs (TASK_CONFIGS_TS1B), block-native
#     story_prefix/k1_position, and the hf-text story source. BOTH diag
#     invocations below MUST pass --model-family ts1b: the ts38 default
#     would score 1B checkpoints against 10K-custom-BPE-encoded eval data —
#     every id a "valid" but wrong index into the 128,256-vocab embedding,
#     silently garbage EM. The diag itself also refuses on a vocab/manifest
#     tokenizer mismatch (defense in depth, added same review). JSON schema
#     confirmed: top-level results[<run_id>][<condition>][<task>][<k>]["em"],
#     dm_mixture cells keyed by template_key with k in {0,16}.
#
# HARD RULES (mirrors launch_ts38pp_family.sh's own):
#   (a) both parents are DELIBERATELY UNGATED in the certification sense —
#       no gates.py verdict is ever recorded against either; only the 2b
#       HALT check (a bash comparison, not a gates.py record) can stop the
#       family.
#   (b) FULL FT, NO MERGE STAGE for either parent — runs/<rid>/model IS
#       the checkpoint the next stage warm-starts from.
#   (c) NEVER destroy the box — teardown is the operator's call, not taken
#       here.
set -uo pipefail

# ---- env guard: safe HF_TOKEN/NTFY extraction, NEVER source
# /etc/environment whole -- 2026-08-19 fix, supersedes launch_ts38pp_
# family.sh's `[[ -f /etc/environment ]] && { set -a; . /etc/environment;
# set +a; }`. That form only worked by strict ordering (commit 88cf240:
# source BEFORE venv activate) because it pulls EVERY var out of the file
# into this shell, including /etc/environment's own
# PATH="/usr/local/sbin:..." line — reintroducing the exact PATH-clobber
# bug the ordering fix was chasing if activation ever moved first again.
# box_onstart.sh writes exactly two vars there
# (`for var in HF_TOKEN NTFY`, box_onstart.sh:179) — extract only those
# two, via sed, and never source the file at all. Order stops mattering
# once nothing here touches PATH.
for _v in HF_TOKEN NTFY; do
  if [[ -z ${!_v:-} && -f /etc/environment ]]; then
    _val=$(sed -n "s/^${_v}=//p" /etc/environment | tail -1)
    _val=${_val%\"}
    _val=${_val#\"}   # tolerate optional surrounding quotes (seen on some
                       # vast.ai images even though box_onstart.sh itself
                       # never quotes the value it writes)
    [[ -n $_val ]] && export "$_v=$_val"
  fi
done
unset _v _val
[[ -f /workspace/venv/bin/activate ]] && . /workspace/venv/bin/activate
python3 - <<'PY' || { echo "[ts1b] FAILED: python3 lacks required modules (venv not active?)"; exit 1; }
import huggingface_hub, torch, geode  # noqa: F401
PY

cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts1b

echo "[ts1b] estimated cost (pre-registration, decisions.md 2026-08-19):"
echo "[ts1b]   stage 0 base pull + mirror-push  ~\$0 GPU (network only)"
echo "[ts1b]   stage 1 LR sweep (3 rungs x 2000 steps)             ~\$3-5"
echo "[ts1b]   stage 2 pp parent (1 epoch, 31,093 steps pinned)    ~\$5-10"
echo "[ts1b]     (0.3-0.6 s/step, one-per-row overhead-bound, short rows;"
echo "[ts1b]     2.5-5.5h estimate, throughput anchor: base pretrain"
echo "[ts1b]     measured 15.6K tok/s on the collaborator's box)"
echo "[ts1b]   stage 3 pf parent (until convergence, ceiling 93,279"
echo "[ts1b]     steps = 3 epochs)                                 ~\$5-8"
echo "[ts1b]   stage 4 theta0 diag (base/pf/pp, n=1024, 5 conditions"
echo "[ts1b]     incl. dm_mixture)                                 ~\$1-3"
echo "[ts1b]   stages 2-4 combined ~\$10-18 (THE gate); all of 0-5 ~\$15-22"
echo "[ts1b]   -- inside the pre-registration's \$25 Stage 0-2 authorization."
echo "[ts1b]   Box spec: >=48GB VRAM (L40S/A6000-Ada/A100 class),"
echo "[ts1b]   cuda_vers>=12.8, race-5 provisioning. Stage 3+ (target"
echo "[ts1b]   grids, ~\$55-105) is NOT authorized by this script."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts1b_stage12.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

# SOURCE_REPO: the collaborator's repo, anonymous read confirmed working
# 2026-08-19 (decisions.md). MIRROR_REPO: a NEW, dedicated mhieuuu-owned
# repo for the insurance copy -- deliberately NOT geode-store, which was
# shrunk 333GB->155MB 2026-08-14 and keeps weights for exactly one run;
# 4.7GB of base-model weights would undo that shrink.
SOURCE_REPO=${SOURCE_REPO:-podhajskimarcin/evt-ts1b-base}
MIRROR_REPO=${MIRROR_REPO:-mhieuuu/evt-ts1b-base}
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}

BASE_RID=evt-ts1b-base
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
PP_RID=evt-ts1b-pp-parent
PP_MODEL_DIR=$GEODE_STORE/runs/$PP_RID/model   # full FT -> no _merged stage
PF_RID=evt-ts1b-pf-parent
PF_MODEL_DIR=$GEODE_STORE/runs/$PF_RID/model

PP_CONFIG=../configs/ts1b_pp_parent.yaml
PF_CONFIG=../configs/ts1b_pf_parent.yaml
SWEEP_DIR=../configs/sweeps/ts1b
LRSWEEP_RUNGS=(1e-4 3e-5 1e-5)

HALT_DIAG_JSON=$GEODE_STORE/results/ts1b_pp_halt_diag.json
FULL_DIAG_JSON=$GEODE_STORE/results/ts1b_theta0_fewshot_diag.json

# Build-time pins: the two parents' data.order_hash values. A launch with
# either still unfilled would make the stale-parent guard vacuous, so
# refuse outright (mirrors launch_ts38pp_family.sh's PRETEACH_ORDER_HASH
# guard) rather than silently proceed against B0.1's not-yet-landed data.
PP_ORDER_HASH=$(python3 -c "
import sys
import yaml
print(yaml.safe_load(open(sys.argv[1]))['data']['order_hash'])
" "$PP_CONFIG")
PF_ORDER_HASH=$(python3 -c "
import sys
import yaml
print(yaml.safe_load(open(sys.argv[1]))['data']['order_hash'])
" "$PF_CONFIG")
[[ $PP_ORDER_HASH != TODO_* ]] || fail \
  "$PP_CONFIG data.order_hash is still '$PP_ORDER_HASH' -- B0.1's block-render datagen (a
   parallel worker's build, same pre-registration) has not landed a real
   D_target_4M_block.parquet/hash yet. Refusing to run rather than silently skip the
   stale-data guard."
[[ $PF_ORDER_HASH != TODO_* ]] || fail \
  "$PF_CONFIG data.order_hash is still '$PF_ORDER_HASH' -- same guard, for the permuted-label
   set (D_target_4M_blockperm.parquet)."

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE base=$BASE_RID pp=$PP_RID pf=$PF_RID"

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

# Best-effort push, push-as-you-go: a failed upload warns and the family
# keeps going; stage 5's receiver check is what can actually fail the run.
push_run() {
  local rid=$1 repo=${2:-$RELAY_REPO}
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$repo"; then
    milestone "push_complete run=$rid repo=$repo"
  else
    echo "[ts1b] WARN hf_checkpoint.py push failed for $rid -> $repo (best effort)"
    milestone "push_warn run=$rid repo=$repo"
  fi
}

# gate_score-style parser for the manifest/training_meta fields shared by
# both parent verification blocks (stage 2 and stage 3).
read_parent_fields() {
  python3 - "$1" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
rid = sys.argv[1]
p = store / "runs" / rid / "manifest.json"
d = json.loads(p.read_text())
status = d.get("status", "MISSING")
method = d.get("training", {}).get("method", "MISSING")
exp = d.get("experiment", {}) or {}
gates = exp.get("gates", {}) or {}
meta_p = store / "runs" / rid / "training_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
stop_reason = meta.get("stop_reason", "MISSING")
final_step = meta.get("final_step", "MISSING")
gate_names = ",".join(sorted(gates)) if gates else "none"
data_hash = exp.get("data_order_hash", "MISSING")
print(status, method, gate_names, stop_reason, final_step, data_hash)
PY
}

# ---- stage 0: base checkpoint pull + ONE-TIME mirror-push insurance ------
milestone "hf_auth_check"
python3 -c "
from huggingface_hub import HfApi
print('[ts1b] hf auth whoami:', HfApi().whoami().get('name', '<unknown>'))
" 2>&1 || echo "[ts1b] WARN: hf auth whoami failed -- if the mirror-push below fails with a
   401/403, run 'hf auth login --force' with an mhieuuu WRITE-scoped token and re-run this stage
   ([[reference-hf-token-actually-write-scoped]]: the box's ambient HF_TOKEN may already be
   write-scoped, or may not -- check, don't assume either way)."

if [[ -f $GEODE_STORE/runs/$BASE_RID/manifest.json ]]; then
  milestone "pull_base_skip run=$BASE_RID (manifest already present)"
else
  milestone "pull_base_start run=$BASE_RID repo=$SOURCE_REPO"
  python3 hf_checkpoint.py pull --run-id "$BASE_RID" --repo-id "$SOURCE_REPO" ||
    fail "pull $BASE_RID from $SOURCE_REPO (collaborator's repo, anonymous read -- verified
     working 2026-08-19 per decisions.md; token=False, no credential needed)"
fi
[[ -f $BASE_MODEL/model.safetensors ]] ||
  fail "pull left no $BASE_MODEL/model.safetensors (snapshot_download is fail-open)"

python3 - "$BASE_RID" <<'PY' || fail "$BASE_RID did not load as the expected Llama-3.2-1B TinyStories base -- do not train against it"
import sys

from transformers import AutoTokenizer

from geode.zoo import load_model, load_run, tokenizer_hash

rid = sys.argv[1]
manifest = load_run(rid).data
model = load_model(rid, device="cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

want = {"hidden_size": 2048, "num_hidden_layers": 16, "vocab_size": 128256}
got = {
    "hidden_size": model.config.hidden_size,
    "num_hidden_layers": model.config.num_hidden_layers,
    "vocab_size": len(tokenizer),
}
tok_sha = tokenizer_hash(tokenizer)
want_sha = manifest["experiment"]["tokenizer"]["sha256"]
print(f"[ts1b] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts1b] tokenizer meta-llama/Llama-3.2-1B: sha256 {tok_sha[:16]}... (manifest {want_sha[:16]}...)")
ok = got == want and tok_sha == want_sha
if not ok:
    print(f"[ts1b] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "base_verified $BASE_RID d2048/L16/vocab128256 -> PASS"

MIRROR_STATUS=$(python3 - "$BASE_RID" "$MIRROR_REPO" <<'PY'
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
try:
    files = HfApi().list_repo_files(repo)
except Exception:
    print("MISSING")
else:
    required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
    print("OK" if all(f in files for f in required) else "MISSING")
PY
)
if [[ $MIRROR_STATUS == OK ]]; then
  milestone "mirror_skip repo=$MIRROR_REPO (already holds $BASE_RID)"
else
  milestone "mirror_push_start run=$BASE_RID repo=$MIRROR_REPO"
  # NOT best-effort: the collaborator's repo is currently the ONLY copy of
  # this checkpoint (decisions.md 2026-08-19 "Stage-0 insurance" note) --
  # retry rather than warn-and-continue on failure.
  python3 hf_checkpoint.py push --run-id "$BASE_RID" --repo-id "$MIRROR_REPO" ||
    fail "mirror-push $BASE_RID to $MIRROR_REPO failed -- this is insurance against the
     collaborator's repo being the only copy; do not proceed until it succeeds"
  milestone "mirror_push_complete run=$BASE_RID repo=$MIRROR_REPO"
fi

# ---- stage 0.5: frozen data artifacts -- regenerate-if-missing ------------
# Clone of launch_ts38pp_family.sh's stage-2 pattern: every artifact is
# deterministic from its seed, so a fresh box REGENERATES rather than pulls
# (the D_algo_eval*/bare eval sets the theta0 diag needs are local-only by
# design, never hub-published -- regeneration is required for them anyway,
# so one mechanism covers everything). Determinism verified 2026-08-19 on
# the generating laptop: D_target_4M_block/_blockperm sample byte-identical
# (a, op, b, shown_answer) triples to the seed-20260816 D_target_4M (whose
# hash matches launch_ts38pp_family.sh's own pin), and train_sft.py's phase
# 2 recomputes each file's order_hash against the config pin and refuses on
# mismatch before any GPU spend -- a wrong regen can never train.
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
BASE_NEEDED=(D_target.parquet D_target_eval.parquet D_algo.parquet D_algo_eval.parquet report.json)
MISSING=0
for f in "${BASE_NEEDED[@]}"; do
  [[ -f $DATA_DIR/$f ]] || MISSING=1
done
if ((MISSING)); then
  milestone "datagen_start (regenerating the frozen artifacts from seed 20260717)"
  mkdir -p "$DATA_DIR"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 ||
    fail "datagen base (D_algo/D_inst/D_target/probe)"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --eval-set ||
    fail "datagen --eval-set (D_target_eval)"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260717 --nl-eval-set ||
    fail "datagen --nl-eval-set (D_algo_eval)"
  milestone "datagen_complete"
else
  milestone "datagen_skip (frozen base artifacts present)"
fi

BARE_NEEDED=(D_algo_bare.parquet D_algo_eval_bare.parquet)
BARE_MISSING=0
for f in "${BARE_NEEDED[@]}"; do
  [[ -f $DATA_DIR/$f ]] || BARE_MISSING=1
done
if ((BARE_MISSING)); then
  milestone "bare_datagen_start (deriving scaffold-free sets from the frozen artifacts)"
  python3 ../datagen/make_bare_sets.py --out "$DATA_DIR" --skip-dose-mult || fail "make_bare_sets (bare derivations)"
  milestone "bare_datagen_complete"
else
  milestone "bare_datagen_skip (all bare artifacts present)"
fi

if [[ -f $DATA_DIR/D_target_4M_block.parquet ]]; then
  milestone "preteach4m_block_datagen_skip (D_target_4M_block.parquet present)"
else
  milestone "preteach4m_block_datagen_start (n=4,000,000, block render, correct labels)"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260816 \
    --preteach-4m --preteach-4m-render block ||
    fail "make_data.py --preteach-4m --preteach-4m-render block (D_target_4M_block)"
  milestone "preteach4m_block_datagen_complete"
fi

if [[ -f $DATA_DIR/D_target_4M_blockperm.parquet ]]; then
  milestone "preteach4m_blockperm_datagen_skip (D_target_4M_blockperm.parquet present)"
else
  milestone "preteach4m_blockperm_datagen_start (n=4,000,000, block render, PERMUTED labels)"
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260816 \
    --preteach-4m --preteach-4m-render block --preteach-4m-labels permuted ||
    fail "make_data.py --preteach-4m ... --preteach-4m-labels permuted (D_target_4M_blockperm)"
  milestone "preteach4m_blockperm_datagen_complete"
fi
milestone "data_present D_target_4M_block + D_target_4M_blockperm + eval sets at pinned local paths"

# ---- stage 1: LR mini-sweep -- STOPS for pin confirmation -----------------
milestone "lrsweep_start rungs=${LRSWEEP_RUNGS[*]}"
for rung in "${LRSWEEP_RUNGS[@]}"; do
  rid=evt-ts1b-pp-lrsweep-${rung}
  train_or_skip "$rid" \
    python3 train_sft.py --config "$PP_CONFIG" \
      --override "$SWEEP_DIR/pp_lrsweep_${rung}.yaml" \
      --init-from "$BASE_MODEL" --confirm-cost
done

python3 - "${LRSWEEP_RUNGS[@]}" <<'PY'
import json
import os
from pathlib import Path
import sys

store = Path(os.environ["GEODE_STORE"])
print(f"[ts1b] {'run_id':<28}{'lr':>10}{'stop_reason':>14}{'min_val_nats':>16}")
for rung in sys.argv[1:]:
    rid = f"evt-ts1b-pp-lrsweep-{rung}"
    meta_p = store / "runs" / rid / "training_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    lr = meta.get("config", {}).get("lr", "MISSING")
    stop = meta.get("stop_reason", "MISSING")
    minval = meta.get("min_val_nats", "MISSING")
    print(f"[ts1b] {rid:<28}{lr!s:>10}{stop!s:>14}{minval!s:>16}")
PY

if [[ ${PIN_CONFIRMED:-0} != 1 ]]; then
  echo "[ts1b] STOP: review the val comparison table above. Pin the winning LR into BOTH"
  echo "[ts1b] $PP_CONFIG and $PF_CONFIG (identical-pipeline ruling -- edit both together, never"
  echo "[ts1b] one without the other)."
  echo "[ts1b] If the winner is an endpoint rung (1e-4 or 1e-5), extend the bracket one rung"
  echo "[ts1b] further and re-run stage 1 before pinning (nulls-need-bracketing rule)."
  echo "[ts1b] Then re-invoke: PIN_CONFIRMED=1 bash launch_ts1b_stage12.sh --confirm-cost"
  echo "[ts1b] This script never auto-pins a hyperparameter."
  exit 0
fi
CUR_PP_LR=$(python3 -c "
import yaml
print(yaml.safe_load(open('$PP_CONFIG'))['train']['lr'])
")
if [[ $CUR_PP_LR == "3e-05" || $CUR_PP_LR == "3.0e-05" ]]; then
  echo "[ts1b] WARN: $PP_CONFIG train.lr is still exactly the committed placeholder (3.0e-5)."
  echo "[ts1b] If the sweep genuinely picked a different rung, this looks like an unedited"
  echo "[ts1b] placeholder rather than a deliberate pin -- double check before proceeding."
fi
milestone "lr_pin_confirmed pp_lr=$CUR_PP_LR"

# ---- stage 2: evt-ts1b-pp-parent -- full FT, one epoch, App. E.2 literal --
train_or_skip "$PP_RID" \
  python3 train_sft.py --config "$PP_CONFIG" --init-from "$BASE_MODEL" --confirm-cost

read -r PP_STATUS PP_METHOD PP_GATES PP_STOP PP_STEP PP_DATA_HASH <<<"$(read_parent_fields "$PP_RID")"
milestone "pp_fields status=$PP_STATUS method=$PP_METHOD gates=$PP_GATES stop_reason=$PP_STOP final_step=$PP_STEP data_order_hash=$PP_DATA_HASH"

[[ $PP_STATUS == complete ]] || fail "$PP_RID manifest status='$PP_STATUS' (expected complete)"
[[ $PP_METHOD == full_ft ]] || fail "$PP_RID training.method='$PP_METHOD' (expected full_ft -- the
   paper-protocol parent is full FT, not LoRA)"
[[ $PP_GATES == none ]] || fail \
  "$PP_RID has recorded gates {$PP_GATES} -- this parent is deliberately ungated (scored
   --no-record only); inspect, do not proceed"
# min_steps == max_steps == 31093 is the one pre-registered exception to
# run-until-convergence, so a legitimate stop reads either 'converged' or
# 'max_steps' -- but the step count itself is never negotiable.
[[ $PP_STOP == converged || $PP_STOP == max_steps ]] || fail \
  "$PP_RID stop_reason='$PP_STOP' (expected converged or max_steps) -- neither the pinned
   one-epoch ceiling nor eps/k produced this; inspect before proceeding"
[[ $PP_STEP == 31093 ]] || fail \
  "$PP_RID final_step=$PP_STEP (expected 31093, the pinned one-epoch length -- see
   ts1b_pp_parent.yaml's header for the derivation) -- do not hand a wrong-length parent
   downstream"
[[ $PP_DATA_HASH == "$PP_ORDER_HASH" ]] || fail \
  "$PP_RID manifest data_order_hash=$PP_DATA_HASH != current pin $PP_ORDER_HASH -- this
   checkpoint was built from a different D_target_4M_block.parquet/config than the one this
   launch just verified. Remove $GEODE_STORE/runs/$PP_RID and rerun to rebuild against the
   current pin."
milestone "pp_verified stop_reason=$PP_STOP final_step=31093 gates={} method=full_ft data_order_hash=OK"

[[ -f $PP_MODEL_DIR/model.safetensors ]] ||
  fail "$PP_MODEL_DIR/model.safetensors is missing -- a full-FT run's checkpoint dir should
   already hold plain weights (no merge stage for this family, hard rule (b))"

push_run "$PP_RID"

# ---- stage 2b: HALT gate -- pp op EM (block, k=0) >= 0.90 ------------------
# RECONCILED 2026-08-19 (see the SCOPE NOTE at the top): B0.3 landed with
# the CLI shape below plus the REQUIRED --model-family ts1b flag; JSON
# schema (results[<run_id>]["block"]["op"]["0"]["em"]) confirmed against
# the landed script. The 0.90 threshold is the pre-registration's own.
if [[ -f $HALT_DIAG_JSON ]]; then
  milestone "halt_diag_skip json=$HALT_DIAG_JSON (already present)"
else
  milestone "halt_diag_start run=$PP_RID (op/block/k=0 only, n=1024)"
  python3 ../analysis/theta0_fewshot_diag.py \
    --model-family ts1b \
    --runs "$PP_RID" \
    --device cuda \
    --n-queries 1024 \
    --shots 0 \
    --out "$HALT_DIAG_JSON" ||
    fail "theta0_fewshot_diag.py halt check failed for $PP_RID (see output above) -- if this is
     a CLI-shape mismatch against B0.3's generalized script, fix the invocation above; do not
     skip the HALT gate"
  milestone "halt_diag_complete json=$HALT_DIAG_JSON"
fi

PP_BLOCK_OP_EM0=$(python3 - "$HALT_DIAG_JSON" "$PP_RID" <<'PY'
import json
import sys

path, rid = sys.argv[1:3]
data = json.loads(open(path).read())
try:
    cell = data[rid]["block"]["op"]["0"]
except KeyError as e:
    print(f"missing key {e} in {path} -- reconcile against the generalized diag script's actual output shape", file=sys.stderr)
    raise SystemExit(1)
print(cell["em"])
PY
) || fail "could not read pp op/block/k=0 EM from $HALT_DIAG_JSON (see stderr above)"
milestone "halt_check pp_op_block_em0=$PP_BLOCK_OP_EM0"

if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.90 else 1)" "$PP_BLOCK_OP_EM0"; then
  echo "[ts1b] HALT: pp op EM (block, k=0) $PP_BLOCK_OP_EM0 < 0.90 -- install failed; pf NOT launched"
  notify "ts1b HALT: pp op EM (block,k=0) $PP_BLOCK_OP_EM0 < 0.90 -- pf not launched"
  exit 1
fi
milestone "halt_check_pass pp_op_block_em0=$PP_BLOCK_OP_EM0 -> PASS (>= 0.90); proceeding to pf"

# ---- stage 3: evt-ts1b-pf-parent -- full FT, permuted labels, until
# convergence (gated on stage 2b passing) -----------------------------------
train_or_skip "$PF_RID" \
  python3 train_sft.py --config "$PF_CONFIG" --init-from "$BASE_MODEL" --confirm-cost

read -r PF_STATUS PF_METHOD PF_GATES PF_STOP PF_STEP PF_DATA_HASH <<<"$(read_parent_fields "$PF_RID")"
milestone "pf_fields status=$PF_STATUS method=$PF_METHOD gates=$PF_GATES stop_reason=$PF_STOP final_step=$PF_STEP data_order_hash=$PF_DATA_HASH"

[[ $PF_STATUS == complete ]] || fail "$PF_RID manifest status='$PF_STATUS' (expected complete)"
[[ $PF_METHOD == full_ft ]] || fail "$PF_RID training.method='$PF_METHOD' (expected full_ft)"
[[ $PF_GATES == none ]] || fail \
  "$PF_RID has recorded gates {$PF_GATES} -- this parent is deliberately ungated; inspect, do
   not proceed"
# UNLIKE pp: stop_reason=max_steps here is HALT-and-review, never silently
# accepted (the pre-registration's own words for the pf parent specifically
# — a permuted-label parent that never converged in 3 epochs/93,279 steps
# is itself a result worth stopping to inspect).
[[ $PF_STOP == converged ]] || fail \
  "$PF_RID stop_reason='$PF_STOP' -- expected 'converged'. Per the pre-registration,
   stop_reason=max_steps on the pf parent is HALT-and-review, NEVER silently accepted.
   Investigate the loss trace (format acquisition on permuted labels should plateau well
   before the 93,279-step ceiling) before re-running."
[[ $PF_DATA_HASH == "$PF_ORDER_HASH" ]] || fail \
  "$PF_RID manifest data_order_hash=$PF_DATA_HASH != current pin $PF_ORDER_HASH -- stale
   checkpoint; remove $GEODE_STORE/runs/$PF_RID and rebuild against the current pin."
milestone "pf_verified stop_reason=converged gates={} method=full_ft data_order_hash=OK"

[[ -f $PF_MODEL_DIR/model.safetensors ]] ||
  fail "$PF_MODEL_DIR/model.safetensors is missing -- a full-FT run's checkpoint dir should
   already hold plain weights"

push_run "$PF_RID"

# ---- stage 4: theta0 few-shot diagnostic on base/pf/pp (evidence pass) ----
# RECONCILED (same review as stage 2b). Conditions per the pre-registration:
# block (native), single-line (render control), story_prefix, k1_position,
# dm_mixture; k in {0,1,2,4,8,16}; n=1024/cell. --no-record is deliberately
# NOT passed: theta0_fewshot_diag.py has no such CLI argument (it never
# touches gates.py's own recording path itself — verified against its
# argparse) -- passing it would crash the invocation, not add safety.
if [[ -f $FULL_DIAG_JSON ]]; then
  milestone "full_diag_skip json=$FULL_DIAG_JSON (already present)"
else
  milestone "full_diag_start runs=$BASE_RID,$PF_RID,$PP_RID"
  python3 ../analysis/theta0_fewshot_diag.py \
    --model-family ts1b \
    --runs "$PP_RID" "$PF_RID" "$BASE_RID" \
    --device cuda \
    --n-queries 1024 \
    --shots 0 1 2 4 8 16 \
    --dm-mixture \
    --out "$FULL_DIAG_JSON" ||
    fail "theta0_fewshot_diag.py full evidence pass failed (see output above)"
  milestone "full_diag_complete json=$FULL_DIAG_JSON"
fi

python3 - "$FULL_DIAG_JSON" "$BASE_RID" "$PF_RID" "$PP_RID" <<'PY'
import json
import sys

path, base_rid, pf_rid, pp_rid = sys.argv[1:5]
data = json.loads(open(path).read())


def dm_mixture_k(rid, k):
    try:
        return data[rid]["dm_mixture"]["mixture"][str(k)]["em"]
    except KeyError:
        return None


def block_k(rid, task, k):
    try:
        return data[rid]["block"][task][str(k)]["em"]
    except KeyError:
        return None


def story_prefix_k0(rid, task):
    try:
        return data[rid]["story_prefix"][task]["0"]["em"]
    except KeyError:
        return None


print("[ts1b] Table-11 read (DM-mixture, scaffolded op-eval triples; paper row: base/pf 0.0/0.0,")
print("[ts1b] pp 2.0 -> 11.9; pre-registered primary criterion: pp k16-k0 >= +5 pts, base/pf <= 1%):")
print(f"[ts1b] {'run_id':<24}{'k=0':>10}{'k=16':>10}{'k16-k0':>10}")
for rid in (base_rid, pf_rid, pp_rid):
    k0 = dm_mixture_k(rid, 0)
    k16 = dm_mixture_k(rid, 16)
    delta = (k16 - k0) if (k0 is not None and k16 is not None) else None
    print(f"[ts1b] {rid:<24}{k0!s:>10}{k16!s:>10}{delta!s:>10}")

print("[ts1b] M1-lock read (does the position lock reproduce at 1B under one-per-row?):")
print(f"[ts1b] {'run_id':<24}{'block_k0':>10}{'story_prefix_k0':>18}")
for rid in (base_rid, pf_rid, pp_rid):
    b0 = block_k(rid, "op", 0)
    sp0 = story_prefix_k0(rid, "op")
    print(f"[ts1b] {rid:<24}{b0!s:>10}{sp0!s:>18}")
PY
milestone "diag_evidence_printed"

# ---- stage 5: push everything to the relay, VERIFY THE RECEIVER -----------
# push_run already ran for both parents (stages 2/3, push-as-you-go); this
# re-pushes only whatever the hub is actually missing, plus the two results
# JSONs (no hf_checkpoint.py support for arbitrary files under results/ —
# mirrors launch_ts38mw_probe.sh's own HfApi().upload_file() pattern for
# its results/ts38mw_probe.json).
push_run "$PP_RID" "$RELAY_REPO"
push_run "$PF_RID" "$RELAY_REPO"

for f in "$HALT_DIAG_JSON" "$FULL_DIAG_JSON"; do
  [[ -f $f ]] || continue
  python3 - "$f" "$GEODE_STORE" "$RELAY_REPO" <<'PY' || echo "[ts1b] WARN uploading $f failed (best effort)"
import sys
from pathlib import Path

from huggingface_hub import HfApi

out_path, store, repo = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
rel = out_path.relative_to(store).as_posix()
HfApi().upload_file(
    path_or_fileobj=str(out_path),
    path_in_repo=rel,
    repo_id=repo,
    repo_type="model",
)
print(f"[ts1b] uploaded {rel} -> {repo}")
PY
done

run_receiver_check() {
  python3 - "$RELAY_REPO" "$PP_RID" "$PF_RID" <<'PY'
import sys

from huggingface_hub import HfApi

repo, pp_rid, pf_rid = sys.argv[1:4]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True

for rid in (pp_rid, pf_rid):
    required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
    missing = [r for r in required if r not in files]
    status = "OK" if not missing else f"MISSING {missing}"
    print(f"  {rid}: {status}")
    if missing:
        print(f"MISSING {rid}")
        ok = False

for rel in ("results/ts1b_pp_halt_diag.json", "results/ts1b_theta0_fewshot_diag.json"):
    # Non-fatal on their own: not added to `ok`/the MISSING-<rid> retry list
    # below, since a results file that was never produced locally this run
    # (e.g. a re-run that skipped stage 4 because FULL_DIAG_JSON pre-existed
    # under a different store) is not the same failure as a missing run.
    status = "OK" if rel in files else "MISSING (see note above script output)"
    print(f"  {rel}: {status}")
sys.exit(0 if ok else 1)
PY
}

RECEIVER_OUT=$(run_receiver_check)
RECEIVER_STATUS=$?
echo "[ts1b] receiver check (hub files for pp + pf + results):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing whatever the hub is missing)"
  while read -r rid; do
    push_run "$rid" "$RELAY_REPO"
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts1b] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED -- see output above (at least
   one required run file is still missing on the relay after a retry; the results/*.json
   MISSING lines are non-fatal on their own, see the check's own annotation)"
milestone "receiver_verified pp=$PP_RID pf=$PF_RID"

notify "ts1b Stages 0-5 done: base verified, pp+pf parents trained, theta0 diag evidence collected"
echo "[ts1b] TERMINAL_SUCCESS stages=0-5"
