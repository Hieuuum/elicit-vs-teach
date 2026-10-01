#!/usr/bin/env bash
# ts38mw family — Donoway et al. "Bits That Count" §5/Fig-3 CAUSAL
# intervention, minimal LoRA version, on the 38.7M TinyStories base.
# Trains the NEW "pretaught-mw" target arm from the GO-B multiwrap-installed
# LoRA parent (decisions.md 2026-08-15 "ts38mw Stage 1 outcome — verdict
# GO-B": symbol-invariant, not language-invariant) onto the bare-NL target
# D_algo_bare ("What is the sum of a and b?" / "What is the difference
# between a and b?"), with the recipe IDENTICAL to the existing ts38 family
# (launch_ts38_mini.sh) — LoRA r128/alpha32 @1e-3, same data, same stopping
# rule. Arms differ ONLY in theta0.
#
# THE FIVE RUNS
#   evt-ts38mw-pretaught-n{N}  theta0 -> LoRA r128/a32 @1e-3 on D_algo_bare,
#                              SAME data order as the matching
#                              evt-ts38-base-n{N} run (G7 anchor).
#   for N in 1000 4642 21544 100000 316228.
#   theta0 = evt-ts38mw-parent-probe-lr3e-4's step-28000 checkpoint (the
#   GO-B multiwrap install measured by launch_ts38mw_probe.sh +
#   mw_verdict.py; owner-selected over 20000/24000 — decisions.md
#   "ts38mw target family PRE-REGISTRATION" entry).
#
# BASE ARM REUSED, NOT RETRAINED: evt-ts38-base-n{N} (launch_ts38_mini.sh)
# already trains + measures the teaching arm on this exact data/recipe. This
# launcher only READS its manifest (data_order_hash, n_examples) as the G7
# anchor per size — it is never (re-)trained, gated, or pushed here.
#
# PRE-REGISTERED READOUT (do not re-derive after seeing numbers):
#   pretaught-mw elicitation marker = EDL/D monotone DEcreasing (non-
#     increasing) AND below the base's EDL/D at every n, under both the OCV
#     and test floors.
#   base teaching marker = its ALREADY-MEASURED rising span 4642->21544
#     (+15% under all three floors) — not re-derived here.
#   pretaught-mw sitting ABOVE base at any n = the RETENTION-CONFOUND class
#     (fig2nl's failure shape: an installed arm entering with worse
#     retention than the base) -> report it as "arms not cleanly separated
#     at that n", NEVER as teaching.
#
# HARD RULES
#   (a) the parent is DELIBERATELY UNGATED (experiment.gates: {} in its
#       manifest; G8 is a KNOWN FAIL, 1.2694 vs bar 1.1718, measured
#       --no-record; owner accepted this 2026-08-15). This launcher NEVER
#       runs gates.py on the parent except --no-record (stage 5, evidence
#       only, no bar) and never records anything to the parent's manifest
#       except merge_adapter.py's own experiment.merged_checkpoint entry.
#   (b) LoRA checkpoints load ONLY via geode.zoo.load_model
#       ([[feedback-lora-checkpoints-load-via-zoo-load-model]]) — the target
#       arms warm-start from the parent's MERGED plain weights
#       (runs/<id>/model_merged/), never from_pretrained on the wrapped
#       runs/<id>/model/.
#   (c) NEVER destroy the box — it is the owner's own rental, not ours to
#       tear down.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38mw

echo "[ts38mw] estimated cost: 5 LoRA target runs on one RTX 4090; the base"
echo "[ts38mw] arm's same-size runs converged at 135/270/1825/5000/10875"
echo "[ts38mw] steps (~18k steps total) -> ~20-40 min training + ~5 min/run"
echo "[ts38mw] of evals+G5 + ~2 min merge/receiver check + pushes => ~1-2 h"
echo "[ts38mw] wall, well under \$1; disk < 1 GB. Ceilings (max_steps per"
echo "[ts38mw] overlay) never bind -- eps/k convergence stops each run;"
echo "[ts38mw] stop_reason=max_steps is a bug signal, never a result."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38mw_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
PARENT_RID=evt-ts38mw-parent-probe-lr3e-4
PARENT_STEP=28000
SIZES=(1000 4642 21544 100000 316228)
PIN_CHECK_N=1000

TARGET_CONFIG=../configs/ts38mw_pretaught.yaml
OVERLAY_DIR=../configs/sweeps/ts38
# Bare-NL eval pin (D_algo_eval_bare), reused verbatim from the ts38 family —
# every target run's G5 and the theta0 latency record (stage 5) read it.
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
THETA0_JSON=$GEODE_STORE/results/ts38mw_family_theta0.json

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

# G5 evidence on the bare target eval (zero/16-shot EM + shared-set test loss
# on identical data for every run). No pass bar by design, never pruned —
# the ts38 family's record_g5, verbatim.
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

# --no-record G5 evidence, parsed the way launch_ts38mw_probe.sh's score_pin
# parses gates.py g5's printed output. Full gate output goes to stderr so the
# $(...) capture is clean; stdout is "em0 em16 loss" on success.
score_g5_no_record() {
  local rid=$1 out em0 em16 loss
  out=$(python3 gates.py g5 --run "$rid" --config "$BARE_EVAL_CONFIG" --no-record 2>&1)
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

# ---- stage 1: the base checkpoint, receiver-verified (launch_ts38_mini.sh
# stage 0, verbatim — the LoRA parent's load_model needs the base run
# present in the store) ------------------------------------------------------
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38mw family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38mw] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38mw] MISSING on the hub: {missing}")
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
# The launcher cd's to scripts/, so this is experiments/training-run/tokenizer —
# the frozen artifact, NOT the pulled checkpoint dir (load_model is model-only).
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
print(f"[ts38mw] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38mw] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38mw] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 2: data artifacts, then their order hashes (launch_ts38_mini.sh
# stage 1, adapted — only D_algo_bare/D_algo_eval_bare need verifying here;
# no g1 runs in this family, so the parent-config val_fraction/seed check is
# not needed) -----------------------------------------------------------------
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

python3 - "$TARGET_CONFIG" "$BARE_EVAL_CONFIG" <<'PY' || fail "data/config pre-flight"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

target_cfg_path, bare_eval = (Path(a) for a in sys.argv[1:3])
for path in (target_cfg_path, bare_eval):
    if not path.is_file():
        print(f"[ts38mw] MISSING config {path}")
        sys.exit(1)

for label, path in (
    ("D_algo_bare", target_cfg_path),
    ("D_algo_eval_bare", bare_eval),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38mw] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare order hashes OK"

# ---- stage 3: G7 anchors — the reused evt-ts38-base-n<size> runs ----------
# Never (re-)trained here; only their manifests (data_order_hash, n_examples)
# are needed, so a metadata-only pull suffices when a size is missing.
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
        print(f"[ts38mw] MISSING manifest for {rid}")
        ok = False
        continue
    d = json.loads(p.read_text())
    status = d.get("status")
    exp = d.get("experiment", {})
    n_examples = exp.get("n_examples")
    order_hash = exp.get("data_order_hash")
    good = status == "complete" and n_examples == int(n) and bool(order_hash)
    print(
        f"[ts38mw] {rid}: status={status} n_examples={n_examples} "
        f"order_hash={'set' if order_hash else 'MISSING'} -> {'OK' if good else 'BAD'}"
    )
    ok = ok and good
sys.exit(0 if ok else 1)
PY
milestone "g7_anchors_ready sizes=${#SIZES[@]}"

# ---- stage 4: the theta0 parent — pulled, verified, merged ----------------
# The parent is DELIBERATELY UNGATED (hard rule (a) above) — this block
# reads its manifest/training_meta.json and refuses to proceed on anything
# other than the exact owner-selected checkpoint; it records nothing.
if [[ ! -f $GEODE_STORE/runs/$PARENT_RID/manifest.json ]]; then
  milestone "pull_parent_start run=$PARENT_RID"
  python3 hf_checkpoint.py pull --run-id "$PARENT_RID" --repo-id "$RELAY_REPO" ||
    fail "pull $PARENT_RID from $RELAY_REPO"
fi
[[ -f $GEODE_STORE/runs/$PARENT_RID/model/adapter.safetensors ]] ||
  fail "pull left no $GEODE_STORE/runs/$PARENT_RID/model/adapter.safetensors — the parent's LoRA adapter must be on the relay"
[[ -f $GEODE_STORE/runs/$PARENT_RID/manifest.json ]] ||
  fail "pull left no manifest for $PARENT_RID — geode.zoo cannot load a run without it"

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
gates = d.get("experiment", {}).get("gates", {}) or {}
sft_result = d.get("experiment", {}).get("sft_result", {}) or {}
meta_p = store / "runs" / rid / "training_meta.json"
meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
stop_reason = meta.get("stop_reason", sft_result.get("stop_reason", "MISSING"))
final_step = meta.get("final_step", sft_result.get("final_step", "MISSING"))
gate_names = ",".join(sorted(gates)) if gates else "none"
print(status, method, gate_names, stop_reason, final_step)
PY
)
read -r P_STATUS P_METHOD P_GATES P_STOP P_STEP <<<"$PARENT_FIELDS"
milestone "parent_fields status=$P_STATUS method=$P_METHOD gates=$P_GATES stop_reason=$P_STOP final_step=$P_STEP"

[[ $P_STATUS == complete ]] || fail "$PARENT_RID manifest status='$P_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$PARENT_RID/manifest.json"
[[ $P_METHOD == lora ]] || fail "$PARENT_RID training.method='$P_METHOD' (expected lora) — the pretaught-mw family merges a LoRA adapter; a non-LoRA checkpoint is the wrong run in this slot"
[[ $P_GATES == none ]] || fail \
  "$PARENT_RID has recorded gates {$P_GATES} — this family was designed against an
   ungated parent; inspect, do not proceed"
[[ $P_STOP == converged ]] || fail "$PARENT_RID stop_reason='$P_STOP' (expected converged) — the owner selected step $PARENT_STEP precisely because it converged; do not pick another"
[[ $P_STEP == "$PARENT_STEP" ]] || fail "$PARENT_RID final_step=$P_STEP (expected $PARENT_STEP) — the owner selected step $PARENT_STEP as theta0; do not pick another"
milestone "parent_verified step=$PARENT_STEP gates={} method=lora"

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

# Receiver check (specs/00 V0.9 lineage): wrapped-vs-merged logits must agree
# to <1e-3 on seeded random token batches — the only proof that folding A/B
# into the base weights preserved the function.
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

print(f"[ts38mw] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
sys.exit(0 if max_abs_diff < 1e-3 else 1)
PY
)
RECEIVER_STATUS=$?
echo "$RECEIVER_OUT"
[[ $RECEIVER_STATUS -eq 0 ]] || fail \
  "receiver check FAILED — merged checkpoint does not reproduce the wrapped model's logits
   within 1e-3 (see output above); do NOT hand this off to the target arms"
MAX_DIFF=$(sed -n 's/.*max_abs_logit_diff=\(.*\)/\1/p' <<<"$RECEIVER_OUT")
milestone "parent_merged max_abs_logit_diff=$MAX_DIFF"

# ---- stage 5: theta0 latency record — evidence only, --no-record, no bar --
# Scores the parent's wrapped model/ and the base on the family's own bare
# eval pin, so the head start is on record on the exact eval set the target
# runs use. Nothing is recorded to any manifest.
if python3 - "$THETA0_JSON" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
d = json.loads(p.read_text())
sys.exit(0 if d.get("parent") and d.get("base") else 1)
PY
then
  milestone "theta0_skip json=$THETA0_JSON (already has parent+base)"
else
  PARENT_G5=$(score_g5_no_record "$PARENT_RID") || fail "$PARENT_RID G5 --no-record (theta0 evidence) failed (see stderr above)"
  read -r PARENT_EM0 PARENT_EM16 PARENT_LOSS <<<"$PARENT_G5"
  BASE_G5=$(score_g5_no_record "$BASE_RID") || fail "$BASE_RID G5 --no-record (theta0 evidence) failed (see stderr above)"
  read -r BASE_EM0 BASE_EM16 BASE_LOSS <<<"$BASE_G5"

  python3 - "$THETA0_JSON" "$PARENT_RID" "$PARENT_EM0" "$PARENT_EM16" "$PARENT_LOSS" \
    "$BASE_RID" "$BASE_EM0" "$BASE_EM16" "$BASE_LOSS" <<'PY' || fail "writing $THETA0_JSON failed"
import json
import sys
from pathlib import Path

out_path, p_rid, p_em0, p_em16, p_loss, b_rid, b_em0, b_em16, b_loss = sys.argv[1:10]
data = {
    "parent": {"run_id": p_rid, "em0": float(p_em0), "em16": float(p_em16), "loss": float(p_loss)},
    "base": {"run_id": b_rid, "em0": float(b_em0), "em16": float(b_em16), "loss": float(b_loss)},
    "eval_config": "eval_bare_target_data_ts38.yaml",
}
out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2) + "\n")
PY
  milestone "theta0_latency parent_em0=$PARENT_EM0 parent_loss=$PARENT_LOSS base_em0=$BASE_EM0 base_loss=$BASE_LOSS"
fi

# ---- stage 6: the five target runs, ascending n ----------------------------
for n in "${SIZES[@]}"; do
  rid=evt-ts38mw-pretaught-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config "$TARGET_CONFIG" \
      --override "$OVERLAY_DIR/ts38mw_pretaught_n${n}.yaml" \
      --init-from "$MERGED_DIR" --confirm-cost

  if [[ $n == "$PIN_CHECK_N" ]]; then
    # Cheapest run in the family, checked before the other four queue up
    # behind it. stop_reason=max_steps is a bug signal by standing policy,
    # never a result.
    PIN_OUT=$(python3 - "$rid" <<'PY'
import json
import os
import sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if not p.is_file():
    print("MISSING - - -")
    sys.exit()
r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) or {}
print(
    r.get("stop_reason", "MISSING"),
    r.get("min_val_nats"),
    r.get("best_val_nats"),
    r.get("final_step"),
)
PY
)
    read -r PIN_STOP PIN_MIN PIN_BEST PIN_STEP <<<"$PIN_OUT"
    milestone "lr_pin_check run=$rid stop_reason=$PIN_STOP min_val_nats=$PIN_MIN best_val_nats=$PIN_BEST final_step=$PIN_STEP"
    [[ $PIN_STOP == converged ]] || fail \
      "LR PIN CHECK: $rid ended with stop_reason='$PIN_STOP' (min val $PIN_MIN, best $PIN_BEST,
   step $PIN_STEP), not 'converged'. The pretaught-mw arm's first/cheapest run did not
   converge under the shared eps/k rule inherited from the ts38 family — OWNER DECISION,
   do not widen. Four more runs would inherit whatever this is. A max_steps stop is a
   bug signal, not a result."
  fi
  record_g5 "$rid"
done

python3 - "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts38mw] {'run_id':<28}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")
for n in sys.argv[1:]:
    rid = f"evt-ts38mw-pretaught-n{n}"
    p = store / "runs" / rid / "manifest.json"
    r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}
    edl = r.get("edl_per_label_token_nats") if r else None
    print(
        f"[ts38mw] {rid:<28}{r.get('final_step', 'MISSING'):>12}"
        f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
        f"{edl if edl is not None else 'n/a':>26}"
    )
PY
milestone "family_complete runs=${#SIZES[@]}"

# ---- stage 7: push + receiver-verify ---------------------------------------
for n in "${SIZES[@]}"; do
  rid=evt-ts38mw-pretaught-n${n}
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$RELAY_REPO"; then
    milestone "push_complete run=$rid"
  else
    echo "[ts38mw] WARN hf_checkpoint.py push failed for $rid (best effort)"
    milestone "push_warn run=$rid"
  fi
done

# [[verify-the-receiver-not-the-sender]]: list what the hub actually holds,
# not what the push claimed to send.
RECEIVER_OUT=$(python3 - "$RELAY_REPO" "${SIZES[@]}" <<'PY'
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
sizes = sys.argv[2:]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True
for n in sizes:
    rid = f"evt-ts38mw-pretaught-n{n}"
    required = [
        f"runs/{rid}/manifest.json",
        f"runs/{rid}/eval_log.jsonl",
        f"runs/{rid}/logs/prequential.jsonl",
    ]
    missing = [r for r in required if r not in files]
    status = "OK" if not missing else f"MISSING {missing}"
    print(f"  {rid}: {status}")
    ok = ok and not missing
sys.exit(0 if ok else 1)
PY
)
RECEIVER_STATUS=$?
echo "[ts38mw] receiver check (hub files under runs/evt-ts38mw-pretaught-n*/):"
echo "$RECEIVER_OUT"
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED — see output above (at least one run's manifest/eval_log/prequential log is missing on the relay)"
milestone "targets_pushed sizes=${#SIZES[@]} receiver=OK"

python3 hf_checkpoint.py push --run-id "$PARENT_RID" --repo-id "$RELAY_REPO" --metadata-only ||
  echo "[ts38mw] WARN hf_checkpoint.py push --metadata-only failed for $PARENT_RID (best effort)"

echo "[ts38mw] theta0 evidence at $THETA0_JSON — not a run, not pushed; the operator scp's results/ back"

# ---- stage 8: finish --------------------------------------------------------
echo "[ts38mw] MILESTONE analysis_commands"
echo "[ts38mw]   Both are CPU-only and run here off \$GEODE_STORE. --family ts38mw is"
echo "[ts38mw]   NOT a default anywhere — pass it explicitly:"
echo "[ts38mw]     python3 ../analysis/edl_converged_val_floor.py --family ts38mw  # OCV (primary)"
echo "[ts38mw]     python3 ../analysis/dataset_size_sweep.py --family ts38mw       # test + min-val"
echo "[ts38mw]   Three floors, OCV primary, floor NAMED on every figure. Read the"
echo "[ts38mw]   pretaught-mw limb against the reused evt-ts38-base-n* teaching span"
echo "[ts38mw]   (4642->21544, already measured, not re-derived here) and against"
echo "[ts38mw]   the theta0 evidence in $THETA0_JSON. Marker rule: pretaught-mw EDL/D"
echo "[ts38mw]   monotone DEcreasing and below base at every n = elicitation; base's"
echo "[ts38mw]   rising span = teaching (already measured); pretaught-mw ABOVE base at"
echo "[ts38mw]   any n reads as the retention-confound class (fig2nl's failure shape),"
echo "[ts38mw]   reported as such, never as teaching."
notify "ts38mw family done: 5 runs, parent=$PARENT_RID step=$PARENT_STEP merged+receiver-verified"
echo "[ts38mw] TERMINAL_SUCCESS runs=5"
