#!/usr/bin/env bash
# ts38pp — PAPER-PROTOCOL pre-teach: a full-FT parent trained ONE epoch over
# a NEW 4,000,000-unique op-notation add/sub set (D_target_4M.parquet),
# matching Donoway et al. App. E's literal recipe (single epoch, 4M unique
# examples, full FT) rather than the multi-epoch-over-1M-unique recipe every
# earlier ts38 parent attempt used. Clone of launch_ts38pf_family.sh's
# stage structure/helpers; see EXPERIMENTS.md §6.14/§6.16 for the prior
# parent attempts this deliberately departs from and decisions.md 2026-08-16
# "ts38pp pre-registration" for the design rationale.
#
# THE SIX RUNS
#   evt-ts38pp-parent            NEW, built here: FULL FT (not LoRA) on
#                                 D_target_4M.parquet (4,000,000 rows,
#                                 operator-notation, CORRECT labels — a real
#                                 teaching install, unlike ts38pf's permuted-
#                                 label format-only control), from
#                                 evt-run1-base-v3-ext, via train_sft.py, for
#                                 EXACTLY one epoch (min_steps == max_steps ==
#                                 31093 == floor((4,000,000-20,000)/128), the
#                                 config's own pin — see ts38pp_parent.yaml).
#   evt-ts38pp-pretaught-n{N}    theta0 -> LoRA r128/alpha32 @1e-3 on
#                                 D_algo_bare, SAME data order as the matching
#                                 evt-ts38-base-n{N} run (G7 anchor). for N in
#                                 1000 4642 21544 100000 316228.
#
# BASE ARM REUSED, NOT RETRAINED: evt-ts38-base-n{N} (launch_ts38_mini.sh /
# launch_ts38grid_family.sh) already trains + measures the teaching arm on
# this exact data/recipe. This launcher only READS its manifest
# (data_order_hash, n_examples) as the G7 anchor per size — it is never
# (re-)trained, gated, or pushed here.
#
# SCORING, NOT CERTIFICATION: unlike the failed ts38-pretaught-parent ladder
# (§6.14, every LR failed G8), this parent is NEVER run through gates.py's
# recording pass/fail path — G1/G8 are scored --no-record for EVIDENCE only,
# alongside a theta0 probe (parent vs. base, three renderings: op/scaffolded-
# NL/bare-NL). The only thing that can stop the family from launching is the
# op-notation install itself failing (PARENT_OP_EM < 0.90) — there is no
# retention gate; the paper's own protocol has none (EXPERIMENTS.md §6.16's
# "paper has no retention gate" note, carried over here).
#
# HARD RULES
#   (a) this parent is DELIBERATELY UNGATED in the certification sense — no
#       G1/G8 pass/fail is ever recorded against it, only --no-record
#       evidence (op-EM HALT excepted, which is this launcher's own bash
#       comparison, not a gates.py record). Nothing is ever written to its
#       manifest except train_sft.py's own run fields.
#   (b) FULL FT, NO MERGE STAGE: unlike ts38pf/ts38mw/ts38grid's LoRA
#       parents, evt-ts38pp-parent is plain full-FT weights the moment
#       training completes — runs/evt-ts38pp-parent/model/ IS the checkpoint
#       the target family warm-starts from (same shape as
#       evt-run1-base-v3-ext being handed to train_sft.py's own --init-from);
#       there is nothing to fold an adapter into.
#   (c) NEVER destroy the box — teardown is the operator's/owner's call, not
#       taken here.
set -uo pipefail

# env guard: detached tmux / non-interactive shells never source ~/.bashrc,
# so /workspace/venv/bin is not on PATH and bare python3 is the system
# interpreter (the 2026-08-16 ts38grid relaunch crash — ModuleNotFoundError:
# huggingface_hub at the first heredoc).
#
# ORDER MATTERS: /etc/environment carries its own PATH="/usr/local/sbin:…"
# line (no venv dir in it); sourcing it AFTER venv activation clobbers venv
# back off PATH (caught live on the ts38pp launch attempt, 2026-08-16 —
# activate-then-source silently reintroduced the exact bug this guard exists
# to fix). Source /etc/environment FIRST for HF_TOKEN etc., THEN activate
# the venv so its PATH prepend is the one that survives.
[[ -f /etc/environment ]] && { set -a; . /etc/environment; set +a; }   # HF_TOKEN etc. written by box_onstart.sh
[[ -f /workspace/venv/bin/activate ]] && . /workspace/venv/bin/activate
python3 - <<'PY' || { echo "[ts38pp] FAILED: python3 lacks required modules (venv not active?)"; exit 1; }
import huggingface_hub, torch, geode  # noqa: F401
PY

cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38pp

echo "[ts38pp] estimated cost: datagen ~5 min + tokenizing the 4M-row parent"
echo "[ts38pp] set ~3 min + one full-FT parent run, 31,093 steps @ ~16"
echo "[ts38pp] steps/s (measured ts38 ladder, 4090) ~= 32 min + scoring"
echo "[ts38pp] (G1/G8/theta0 probes, all --no-record) ~8 min + 5 LoRA target"
echo "[ts38pp] runs at 1.7/2.1/6.4/10.5/23.6 min (measured ts38pf, same"
echo "[ts38pp] recipe/sizes) plus G5/pushes ~= 55 min + setup ~10 min =>"
echo "[ts38pp] ~= 1h50m ~= \$0.7-1.0 at \$0.35-0.45/h; disk < 4 GB."
echo "[ts38pp] stop_reason=max_steps is a bug signal for every CHILD target"
echo "[ts38pp] run; for the PARENT it is the pinned one-epoch end (min_steps"
echo "[ts38pp] == max_steps == 31093 in the config) -- the one pre-registered"
echo "[ts38pp] exception, see ts38pp_parent.yaml."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38pp_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
PARENT_RID=evt-ts38pp-parent
PARENT_MODEL_DIR=$GEODE_STORE/runs/$PARENT_RID/model   # full FT -> no _merged stage

DEFAULT_SIZES=(1000 4642 21544 100000 316228)
if [[ -n ${SIZES:-} ]]; then
  read -r -a SIZES <<<"$SIZES"
else
  SIZES=("${DEFAULT_SIZES[@]}")
fi
# Ascending order is load-bearing (G7 anchors + the family loop both assume
# it); a descending/unsorted SIZES override would silently reorder the sweep.
for i in "${!SIZES[@]}"; do
  ((i == 0)) && continue
  ((SIZES[i] > SIZES[i - 1])) || fail \
    "SIZES must be strictly ascending -- got '${SIZES[*]}' (index $i: ${SIZES[i]} <= ${SIZES[i - 1]})"
done

PARENT_CONFIG=../configs/ts38pp_parent.yaml
TARGET_CONFIG=../configs/ts38pp_pretaught.yaml
OVERLAY_DIR=../configs/sweeps/ts38
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
RUN1_CONFIG=../configs/archive/runs/run1_pretrain.yaml
VAL_CACHE=$GEODE_STORE/cache/run1_val_stream.pt
THETA0_JSON=$GEODE_STORE/results/ts38pp_family_theta0.json
G8_BAR=1.1718

# Flag W1 is adding to make_data.py for the paper-protocol 4M-unique
# pre-teach set (App. E). CONFIRM AGAINST `make_data.py --help` before
# launch -- fix the spelling here, in one place, if it shipped differently.
PRETEACH_DATAGEN_ARGS="--preteach-4m"

# Build-time pin: D_target_4M.parquet's order_hash, computed by W1's datagen
# commit. A launch with this still unfilled would make the stale-parent
# guard below vacuous, so refuse outright rather than silently skip it.
PRETEACH_ORDER_HASH="ba2d6efdd939f63e6da75420a93362fcf86a6adeaa66bf5b5cce01532fbec54c"
[[ $PRETEACH_ORDER_HASH != TODO_* ]] || fail \
  "PRETEACH_ORDER_HASH is still unpinned ('$PRETEACH_ORDER_HASH') -- fill in the real
   D_target_4M.parquet order_hash from W1's datagen commit before launching. Refusing to
   run against an unpinned hash rather than silently skip the stale-parent guard."

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE sizes=${#SIZES[@]} base=$BASE_RID parent=$PARENT_RID"

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

# Standing policy (feedback-run-until-convergence): a target run stops
# because eps/k said so, never because it ran out of steps. Checked for
# EVERY size (ts38grid's rule, not ts38pf's single-size spot-check).
require_converged() {
  local rid=$1 n=$2 stop
  stop=$(stop_reason_of "$rid" target_result)
  milestone "convergence_check run=$rid stop_reason=$stop"
  [[ $stop == converged ]] || fail \
    "CONVERGENCE CHECK: $rid ended with stop_reason='$stop', not 'converged'. Standing
   policy: a max_steps stop is a BUG SIGNAL, not a result. Inspect the overlay's max_steps
   and the loss trace before continuing; do not let the remaining sizes inherit whatever
   this is."
}

# G5 evidence on the bare target eval (zero/16-shot EM + shared-set test loss
# on identical data for every run). No pass bar by design.
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

# Best-effort push, push-as-you-go: a failed upload warns and the family
# keeps going; the final receiver check is the thing that can actually fail
# the run.
push_run() {
  local rid=$1
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$RELAY_REPO"; then
    milestone "push_complete run=$rid"
  else
    echo "[ts38pp] WARN hf_checkpoint.py push failed for $rid (best effort)"
    milestone "push_warn run=$rid"
  fi
}

# gate_score "<gates.py stdout>" <GATE> <metric name> -> the parsed numeric
# value, or "" if the printed line was never emitted (an argparse error, a
# crash before the score line).
gate_score() { sed -n "s/.*$2 $3 \([0-9.]*\) on n=.*/\1/p" <<<"$1" | head -1; }

# --no-record G5 evidence, parsed the way ts38pf/ts38mw's score_g5_no_record
# parses gates.py g5's printed output. Full gate output goes to stderr so
# the $(...) capture is clean; stdout is "em0 em16 loss" on success.
score_g5_no_record() {
  local rid=$1 cfg=$2 out em0 em16 loss
  out=$(python3 gates.py g5 --run "$rid" --config "$cfg" --no-record 2>&1)
  grep -E "G5 (zero-shot|16-shot|shared-set)" <<<"$out" >&2
  em0=$(sed -n 's/.*G5 zero-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  em16=$(sed -n 's/.*G5 16-shot exact_match \([0-9.]*\) on n=.*/\1/p' <<<"$out" | head -1)
  loss=$(sed -n 's/.*G5 shared-set test loss \([0-9.]*\) nats over n=.*/\1/p' <<<"$out" | head -1)
  if [[ -z $em0 || -z $em16 || -z $loss ]]; then
    echo "$out" | tail -5 >&2
    return 1
  fi
  echo "$em0 $em16 $loss"
}

# ---- stage 1: the base checkpoint, receiver-verified (ts38pf's stage 1,
# verbatim — the new full-FT parent's build starts from this same base) ----
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38pp family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38pp] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38pp] MISSING on the hub: {missing}")
    sys.exit(1)
PY

if [[ ! -f $BASE_MODEL/model.safetensors ]]; then
  milestone "pull_base run=$BASE_RID"
  python3 hf_checkpoint.py pull --run-id "$BASE_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $BASE_RID from $RELAY_REPO"
fi
[[ -f $BASE_MODEL/model.safetensors ]] ||
  fail "pull left no $BASE_MODEL/model.safetensors (snapshot_download is fail-open)"
[[ -f $GEODE_STORE/runs/$BASE_RID/manifest.json ]] ||
  fail "pull left no manifest for $BASE_RID — geode.zoo cannot load a run without it"

python3 - "$BASE_RID" <<'PY' || fail "$BASE_RID did not load as the expected 38.7M TinyStories base — do not train against it"
import sys
from pathlib import Path

from transformers import AutoTokenizer

from geode.zoo import load_model, load_run, tokenizer_hash

rid = sys.argv[1]
manifest = load_run(rid).data
model = load_model(rid, device="cpu")
tok_dir = Path.cwd().parent / "tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tok_dir)

want = {"hidden_size": 512, "num_hidden_layers": 8, "vocab_size": 10000}
got = {
    "hidden_size": model.config.hidden_size,
    "num_hidden_layers": model.config.num_hidden_layers,
    "vocab_size": len(tokenizer),
}
tok_sha = tokenizer_hash(tokenizer)
want_sha = manifest["experiment"]["tokenizer"]["sha256"]
print(f"[ts38pp] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38pp] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38pp] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 2: data artifacts, then the paper-protocol 4M parent set -------
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

if [[ -f $DATA_DIR/D_target_4M.parquet ]]; then
  milestone "preteach4m_datagen_skip (D_target_4M.parquet present)"
else
  milestone "preteach4m_datagen_start (n=4,000,000, operator-notation, correct labels)"
  # shellcheck disable=SC2086  # PRETEACH_DATAGEN_ARGS is a deliberately
  # word-split flag string (mirrors --eval-set/--nl-eval-set's shape above).
  python3 ../datagen/make_data.py --scale full --out "$DATA_DIR" --seed 20260816 $PRETEACH_DATAGEN_ARGS ||
    fail "make_data.py $PRETEACH_DATAGEN_ARGS (D_target_4M derivation)"
  milestone "preteach4m_datagen_complete"
fi

python3 - "$TARGET_CONFIG" "$BARE_EVAL_CONFIG" "$PARENT_CONFIG" "$PRETEACH_ORDER_HASH" <<'PY' || fail "data/config pre-flight"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

target_cfg_path, bare_eval, parent_cfg_path, preteach_hash = sys.argv[1:5]
for path in (Path(target_cfg_path), Path(bare_eval), Path(parent_cfg_path)):
    if not path.is_file():
        print(f"[ts38pp] MISSING config {path}")
        sys.exit(1)

for label, path in (
    ("D_algo_bare", Path(target_cfg_path)),
    ("D_algo_eval_bare", Path(bare_eval)),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38pp] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")

parent_cfg = load_config(Path(parent_cfg_path), None)
if parent_cfg["data"]["order_hash"] != preteach_hash:
    print(
        f"[ts38pp] {parent_cfg_path}: data.order_hash "
        f"{parent_cfg['data']['order_hash']} != build-time pin {preteach_hash} — "
        "the committed config and the file on disk disagree; regenerate or re-pin"
    )
    sys.exit(1)
df = load_frozen_parquet(parent_cfg)
print(f"[ts38pp] D_target_4M: {len(df)} rows, order_hash verified ({parent_cfg['data']['file']})")
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare/D_target_4M order hashes OK"

# ---- stage 3: G7 anchors — the reused evt-ts38-base-n<size> runs ----------
for n in "${SIZES[@]}"; do
  rid=evt-ts38-base-n${n}
  if [[ ! -f $GEODE_STORE/runs/$rid/manifest.json ]]; then
    milestone "pull_anchor_start run=$rid"
    python3 hf_checkpoint.py pull --run-id "$rid" --repo-id "$RELAY_REPO" --no-weights ||
      fail "pull $rid (G7 anchor) from $RELAY_REPO"
  fi
done

python3 - "${SIZES[@]}" <<'PY' || fail "G7 anchor preflight — see output above (a reused evt-ts38-base-n<size> run is missing, incomplete, or its data_order_hash/n_examples are unset; inspect before proceeding)"
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
ok = True
for n in sys.argv[1:]:
    rid = f"evt-ts38-base-n{n}"
    p = store / "runs" / rid / "manifest.json"
    if not p.is_file():
        print(f"[ts38pp] MISSING manifest for {rid}")
        ok = False
        continue
    d = json.loads(p.read_text())
    status = d.get("status")
    exp = d.get("experiment", {})
    n_examples = exp.get("n_examples")
    order_hash = exp.get("data_order_hash")
    good = status == "complete" and n_examples == int(n) and bool(order_hash)
    print(
        f"[ts38pp] {rid}: status={status} n_examples={n_examples} "
        f"order_hash={'set' if order_hash else 'MISSING'} -> {'OK' if good else 'BAD'}"
    )
    ok = ok and good
sys.exit(0 if ok else 1)
PY
milestone "g7_anchors_ready sizes=${#SIZES[@]}"

# ---- stage 4: the paper-protocol full-FT parent, one epoch over 4M -------
train_or_skip "$PARENT_RID" \
  python3 train_sft.py --config "$PARENT_CONFIG" --init-from "$BASE_MODEL" --confirm-cost

PARENT_FIELDS=$(python3 - "$PARENT_RID" <<'PY'
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
sft_result = exp.get("sft_result", {}) or {}
meta_p = store / "runs" / rid / "training_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
stop_reason = meta.get("stop_reason", sft_result.get("stop_reason", "MISSING"))
final_step = meta.get("final_step", sft_result.get("final_step", "MISSING"))
gate_names = ",".join(sorted(gates)) if gates else "none"
data_hash = exp.get("data_order_hash", "MISSING")
print(status, method, gate_names, stop_reason, final_step, data_hash)
PY
)
read -r P_STATUS P_METHOD P_GATES P_STOP P_STEP P_DATA_HASH <<<"$PARENT_FIELDS"
milestone "parent_fields status=$P_STATUS method=$P_METHOD gates=$P_GATES stop_reason=$P_STOP final_step=$P_STEP data_order_hash=$P_DATA_HASH"

[[ $P_STATUS == complete ]] || fail "$PARENT_RID manifest status='$P_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$PARENT_RID/manifest.json"
[[ $P_METHOD == full_ft ]] || fail "$PARENT_RID training.method='$P_METHOD' (expected full_ft — the paper-protocol parent is full FT, not LoRA)"
[[ $P_GATES == none ]] || fail \
  "$PARENT_RID has recorded gates {$P_GATES} — this family was designed against an
   ungated parent (scored --no-record only); inspect, do not proceed"
# The ONE pre-registered exception to run-until-convergence: this config
# pins min_steps == max_steps == 31093 (one full epoch under train_sft.py's
# own step counting), so a legitimate stop can read either 'converged' or
# 'max_steps' — but the step count itself is never negotiable.
[[ $P_STOP == converged || $P_STOP == max_steps ]] || fail \
  "$PARENT_RID stop_reason='$P_STOP' (expected converged or max_steps) — neither the
   pinned one-epoch ceiling nor eps/k produced this; inspect before proceeding"
[[ $P_STEP == 31093 ]] || fail \
  "$PARENT_RID final_step=$P_STEP (expected 31093 = floor((4,000,000-20,000)/128), the
   pinned one-epoch length) — inspect $PARENT_CONFIG's min_steps/max_steps pins and the
   manifest; do not hand a wrong-length parent to the target sweep"
# Stale-parent guard: train_or_skip reuses ANY complete-status checkpoint at
# this run_id, including one built from a since-changed config/dataset on an
# earlier invocation. The manifest's own data_order_hash (recorded from
# cfg["data"]["order_hash"] at build time, train_sft.py:190) is the cheapest
# fingerprint tying the checkpoint back to the exact D_target_4M.parquet
# this launch run verified in stage 2 — a mismatch means "delete
# runs/$PARENT_RID and rebuild", not "keep training on top of it".
[[ $P_DATA_HASH == "$PRETEACH_ORDER_HASH" ]] || fail \
  "$PARENT_RID manifest data_order_hash=$P_DATA_HASH != current pin $PRETEACH_ORDER_HASH —
   this checkpoint was built from a different D_target_4M.parquet/config than the one this
   launch just verified. Remove $GEODE_STORE/runs/$PARENT_RID and rerun to rebuild against
   the current pin; do not train the target sweep on a stale parent."
milestone "parent_verified stop_reason=$P_STOP final_step=31093 gates={} method=full_ft data_order_hash=OK"

[[ -f $PARENT_MODEL_DIR/model.safetensors ]] ||
  fail "$PARENT_MODEL_DIR/model.safetensors is missing — a full-FT run's checkpoint dir
   should already hold plain weights (no merge stage for this family, hard rule (b))"

# Push-as-you-go: the expensive full-FT parent's weights survive a box death
# even if the scoring stage below HALTs the sweep. Best-effort — the final
# receiver check is what can actually fail the run.
if python3 hf_checkpoint.py push --run-id "$PARENT_RID" --repo-id "$RELAY_REPO" --with-snapshots; then
  milestone "push_complete run=$PARENT_RID (full weights + snapshots)"
else
  echo "[ts38pp] WARN hf_checkpoint.py push failed for $PARENT_RID (best effort)"
  milestone "push_warn run=$PARENT_RID"
fi

# ---- stage 5: paper-protocol scoring (G1/G8/theta0, ALL --no-record) plus
# the op-EM HALT gate -------------------------------------------------------
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

G1_OUT=$(python3 gates.py g1 --run "$PARENT_RID" --config "$PARENT_CONFIG" --threshold 0.90 --no-record 2>&1)
echo "$G1_OUT"
PARENT_OP_EM=$(gate_score "$G1_OUT" G1 accuracy)
[[ -n $PARENT_OP_EM ]] || fail "$PARENT_RID G1 --no-record printed no accuracy (see output above)"
milestone "theta0 g1_op_em=$PARENT_OP_EM"

G8_OUT=$(python3 gates.py g8 --run "$PARENT_RID" --config "$RUN1_CONFIG" --bar 1.1718 \
  --tokenizer ../tokenizer --val-cache "$VAL_CACHE" --no-record 2>&1)
echo "$G8_OUT"
PARENT_G8_NATS=$(gate_score "$G8_OUT" G8 val_loss_nats)
[[ -n $PARENT_G8_NATS ]] || fail "$PARENT_RID G8 --no-record printed no val_loss_nats (see output above)"
milestone "theta0 g8_val_loss_nats=$PARENT_G8_NATS bar=$G8_BAR"

PROBE_LABELS=(op nl_scaffolded bare)
PROBE_CONFIGS=(../configs/eval_target_data.yaml ../configs/eval_nl_target_data_ts38.yaml ../configs/eval_bare_target_data_ts38.yaml)

if [[ -f $THETA0_JSON ]]; then
  milestone "theta0_skip json=$THETA0_JSON (already present)"
else
  declare -A P_EM0 P_EM16 P_LOSS B_EM0 B_EM16 B_LOSS
  for i in "${!PROBE_LABELS[@]}"; do
    label=${PROBE_LABELS[$i]}
    cfg=${PROBE_CONFIGS[$i]}
    OUT=$(score_g5_no_record "$PARENT_RID" "$cfg") ||
      fail "$PARENT_RID G5 --no-record (theta0 probe=$label) failed (see stderr above)"
    read -r "P_EM0[$label]" "P_EM16[$label]" "P_LOSS[$label]" <<<"$OUT"
    OUT=$(score_g5_no_record "$BASE_RID" "$cfg") ||
      fail "$BASE_RID G5 --no-record (theta0 probe=$label) failed (see stderr above)"
    read -r "B_EM0[$label]" "B_EM16[$label]" "B_LOSS[$label]" <<<"$OUT"
    milestone "theta0 probe=$label parent_em0=${P_EM0[$label]} parent_loss=${P_LOSS[$label]} base_em0=${B_EM0[$label]} base_loss=${B_LOSS[$label]}"
  done

  VALS=()
  for label in "${PROBE_LABELS[@]}"; do
    VALS+=("${P_EM0[$label]}" "${P_EM16[$label]}" "${P_LOSS[$label]}" "${B_EM0[$label]}" "${B_EM16[$label]}" "${B_LOSS[$label]}")
  done

  python3 - "$THETA0_JSON" "$PARENT_RID" "$BASE_RID" "$PARENT_OP_EM" "$PARENT_G8_NATS" "$G8_BAR" "${VALS[@]}" <<'PY' || fail "writing $THETA0_JSON failed"
import json
import sys
from pathlib import Path

out_path, parent_rid, base_rid, g1_em, g8_nats, g8_bar = sys.argv[1:7]
vals = sys.argv[7:]
LABELS = ["op", "nl_scaffolded", "bare"]
CONFIGS = {
    "op": "eval_target_data.yaml",
    "nl_scaffolded": "eval_nl_target_data_ts38.yaml",
    "bare": "eval_bare_target_data_ts38.yaml",
}
if len(vals) != len(LABELS) * 6:
    raise SystemExit(f"expected {len(LABELS) * 6} probe values, got {len(vals)}")

parent_probes, base_probes = {}, {}
for i, label in enumerate(LABELS):
    p_em0, p_em16, p_loss, b_em0, b_em16, b_loss = vals[i * 6 : i * 6 + 6]
    parent_probes[label] = {
        "config": CONFIGS[label],
        "zero_shot_em": float(p_em0),
        "sixteen_shot_em": float(p_em16),
        "label_loss_nats": float(p_loss),
    }
    base_probes[label] = {
        "config": CONFIGS[label],
        "zero_shot_em": float(b_em0),
        "sixteen_shot_em": float(b_em16),
        "label_loss_nats": float(b_loss),
    }

data = {
    "parent": {
        "run_id": parent_rid,
        "g1_em": float(g1_em),
        "g8_nats": float(g8_nats),
        "g8_bar": float(g8_bar),
        "probes": parent_probes,
    },
    "base": {"run_id": base_rid, "probes": base_probes},
}
out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2) + "\n")
PY
  milestone "theta0_written json=$THETA0_JSON"
fi

# ---- the HALT gate: op-notation install success is the only bar ----------
# "Otherwise continue regardless of G8/NL-probe values (paper has no
# retention gate)" — G8/theta0 above are evidence, printed as MILESTONEs;
# only PARENT_OP_EM can stop the family from launching.
if ! python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= 0.90 else 1)" "$PARENT_OP_EM"; then
  echo "[ts38pp] HALT: parent op EM $PARENT_OP_EM < 0.90 — install failed; family NOT launched"
  notify "ts38pp HALT: parent op EM $PARENT_OP_EM < 0.90 — install failed; family NOT launched"
  exit 1
fi
milestone "parent_op_em_check em=$PARENT_OP_EM -> PASS (>= 0.90); proceeding regardless of G8/NL-probe values (paper has no retention gate)"

# ---- stage 6: the five target runs, ascending n, push-as-you-go ----------
for n in "${SIZES[@]}"; do
  rid=evt-ts38pp-pretaught-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config "$TARGET_CONFIG" \
      --override "$OVERLAY_DIR/ts38pp_pretaught_n${n}.yaml" \
      --init-from "$PARENT_MODEL_DIR" --confirm-cost
  require_converged "$rid" "$n"
  record_g5 "$rid"
  push_run "$rid"
  milestone "size_complete n=$n run=$rid"
done

python3 - "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts38pp] {'run_id':<30}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")
for n in sys.argv[1:]:
    rid = f"evt-ts38pp-pretaught-n{n}"
    p = store / "runs" / rid / "manifest.json"
    r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}
    r = r or {}
    edl = r.get("edl_per_label_token_nats")
    print(
        f"[ts38pp] {rid:<30}{r.get('final_step', 'MISSING'):>12}"
        f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
        f"{edl if edl is not None else 'n/a':>26}"
    )
PY
milestone "family_complete runs=${#SIZES[@]}"

# ---- stage 7: receiver-verify parent + every child on the hub ------------
# Verify the RECEIVER, not the sender ([[feedback-verify-the-receiver-not-
# the-sender]]): each run was already pushed as it finished (parent in stage
# 4, children in stage 6); this re-pushes only whatever the hub is actually
# missing, once, then re-checks.
run_receiver_check() {
  python3 - "$RELAY_REPO" "$PARENT_RID" "${SIZES[@]}" <<'PY'
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
parent_rid = sys.argv[2]
sizes = sys.argv[3:]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True

required = [f"runs/{parent_rid}/manifest.json", f"runs/{parent_rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
status = "OK" if not missing else f"MISSING {missing}"
print(f"  {parent_rid}: {status}")
if missing:
    print(f"MISSING {parent_rid}")
    ok = False

for n in sizes:
    rid = f"evt-ts38pp-pretaught-n{n}"
    required = [
        f"runs/{rid}/manifest.json",
        f"runs/{rid}/eval_log.jsonl",
        f"runs/{rid}/logs/prequential.jsonl",
    ]
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
echo "[ts38pp] receiver check (hub files for parent + ${#SIZES[@]} children):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing whatever the hub is missing)"
  while read -r rid; do
    if [[ $rid == "$PARENT_RID" ]]; then
      python3 hf_checkpoint.py push --run-id "$PARENT_RID" --repo-id "$RELAY_REPO" --with-snapshots ||
        echo "[ts38pp] WARN retry push failed for $PARENT_RID (best effort)"
    else
      push_run "$rid"
    fi
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts38pp] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED — see output above (at least one file is still missing on the relay after a retry)"
milestone "receiver_verified parent=$PARENT_RID children=${#SIZES[@]}"

# ---- stage 8: finish -------------------------------------------------------
echo "[ts38pp] MILESTONE analysis_commands"
echo "[ts38pp]   CPU-only, run here off \$GEODE_STORE. --family ts38pp is NOT"
echo "[ts38pp]   a default anywhere — pass it explicitly:"
echo "[ts38pp]     python3 ../analysis/edl_converged_val_floor.py --family ts38pp"
echo "[ts38pp]     python3 ../analysis/dataset_size_sweep.py --family ts38pp"
echo "[ts38pp]   (dataset_size_sweep.py may not have a ts38pp FAMILIES entry"
echo "[ts38pp]   yet — same caveat ts38pf shipped with; out of this"
echo "[ts38pp]   launcher's scope to add one.)"
echo "[ts38pp]   theta0 evidence (g1_em/g8_nats + 3-way probes, parent vs."
echo "[ts38pp]   base) at $THETA0_JSON — not a run, not pushed."
notify "ts38pp family done: 6 runs (1 full-FT parent + ${#SIZES[@]} sizes), parent_op_em=$PARENT_OP_EM"
echo "[ts38pp] TERMINAL_SUCCESS runs=${#SIZES[@]}"
