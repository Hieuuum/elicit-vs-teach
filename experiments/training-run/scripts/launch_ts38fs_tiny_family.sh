#!/usr/bin/env bash
# ts38fs-tiny — format-install dose sweep, extended down to i=2/10
# (decisions.md 2026-08-20 "ts38fs sweep killed mid-run, ts38fs-tiny extends
# the dose grid down to i=2/10"). ts38fs itself tested install sizes
# 1000/4642/21544/100000 and every one came back verdict=LEARNED — this
# family probes for the NOT_LEARNED floor that grid never found, at a
# single seed (owner's own read of the ts38fs interim data: seed doesn't
# meaningfully change the curve).
#
# THE GRID: 10 (i, n) cells, seed fixed at 316 (no seed axis this time) —
#   i (install size) in {2, 10}
#   n (target size)   in {1000, 4642, 21544, 100000, 316228}
# Both install sizes are NEW parents (no reuse — ts38fs's own reuse only
# applied to its i=21544 column, out of scope here).
#
# TWO STRUCTURAL DEVIATIONS FROM ts38fs, both forced by the tiny install
# sizes (full derivation in decisions.md, same entry as above):
#   (a) LABELS: geode.arith.cyclic_shift_labels (V5.78), not permute_labels —
#       a random shuffle can't guarantee a wrong label this small.
#   (b) STOPPING: the install parents use stopping_metric="train_loss"
#       (geode/train/sft.py V5.65/V5.66 — full-batch, no val split), not
#       ts38fs's val-loss eps/k — val_fraction rounds to 0 held-out rows at
#       these sizes. eps_nats=0.0002/k=5 CALIBRATED from real pilot curves
#       (configs/sweeps/dose_cal/ts38fs_tiny_cal_n{2,10}.yaml +
#       analysis/dose_stop_calibration.py), already pinned into both parent
#       configs — this launcher does not recalibrate.
#
# LEAN ON PURPOSE — this is NOT launch_ts38fs_family.sh trimmed by removing
# a few lines; several of that launcher's stages don't apply here at all and
# are omitted rather than kept as dead weight: no reused-parent stage (both
# installs are new), no 55-overlay generator (10 overlays are committed
# files, hand-authored like every ts38(*) family except ts38fs itself), no
# seed loop (one seed). What's KEPT, because dropping it would risk exactly
# the failure modes those guards exist for: base relay verify, per-parent
# order_hash verification (load_frozen_parquet's own recompute-and-refuse),
# parent stop_reason==converged check, merge + receiver logit check, LEAKED
# hard-fail / NOT_LEARNED-continues (finding NOT_LEARNED here is the
# measurement, not a bug — same semantics as ts38fs's own stage 5), a
# first-cell pin check, push-as-you-go, and an end-of-run receiver verify.
#
# HARD RULES (same as ts38fs's own, restated for this smaller grid)
#   (a) both install parents are DELIBERATELY UNGATED (format-only
#       controls). This launcher never runs gates.py against either except
#       --no-record (theta0 stage, evidence only) and never records
#       anything to a parent manifest except merge_adapter.py's own
#       experiment.merged_checkpoint entry.
#   (b) LoRA checkpoints load ONLY via geode.zoo.load_model
#       ([[feedback-lora-checkpoints-load-via-zoo-load-model]]) — every
#       target run warm-starts from its install parent's MERGED plain
#       weights, never from_pretrained on the wrapped model/.
#   (c) NEW SIZES ONLY: train_or_skip keys on the LOCAL store, so a run
#       living only on the relay looks "missing" and would be retrained —
#       both install sizes and all 10 target run ids are new to every prior
#       family, so no existing shipped run can collide.
#   (d) PUSH AS YOU GO — every target run is pushed the moment it trains.
#   (e) NEVER destroy the box — operator/owner call, not this launcher's.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38fs-tiny

echo "[ts38fs-tiny] estimated cost: TWO new tiny LoRA format-only parents"
echo "[ts38fs-tiny] (i=2/i=10; calibration pilots already measured ~27-76"
echo "[ts38fs-tiny] steps to the pinned eps/k, ceiling 1000 never binds --"
echo "[ts38fs-tiny] seconds each) PLUS 10 LoRA target runs on one RTX 4090,"
echo "[ts38fs-tiny] same per-n timing ts38fs itself measured (theta0 does"
echo "[ts38fs-tiny] not change target-stage step cost): 1.7 / 2.1 / 6.4 /"
echo "[ts38fs-tiny] 10.5 / 23.6 min at n = 1000 / 4642 / 21544 / 100000 /"
echo "[ts38fs-tiny] 316228, x2 installs => ~2*(1.7+2.1+6.4+10.5+23.6) ="
echo "[ts38fs-tiny] ~88.6 min target-run compute, plus G5 (~0.5 min x 10)"
echo "[ts38fs-tiny] and setup/merges => roughly 1.5h wall, well under \$1.50"
echo "[ts38fs-tiny] at \$0.35-0.45/h; disk a few hundred MB."
echo "[ts38fs-tiny] Ceilings (max_steps) must never bind -- stop_reason=max_steps"
echo "[ts38fs-tiny] is a bug signal, never a result."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38fs_tiny_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}

INSTALLS=(2 10)
SIZES=(1000 4642 21544 100000 316228)
SEED=316

TARGET_CONFIG=../configs/ts38fs_target.yaml
OVERLAY_DIR=../configs/sweeps/ts38
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
FMT_JSON=$GEODE_STORE/results/ts38fs_tiny_format_acquisition.json
PIN_CHECK_RID=evt-ts38fs-tiny-i2-n1000-s316

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE installs=${#INSTALLS[@]} sizes=${#SIZES[@]} seed=$SEED base=$BASE_RID"

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

# --no-record G5 evidence, same parser as launch_ts38fs_family.sh's own.
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

# Merge a LoRA parent to plain weights and prove the merge preserved the
# function — verbatim from launch_ts38fs_family.sh's own merge_and_verify.
MERGED_DIR=
merge_and_verify() {
  local rid=$1 merged receiver_out receiver_status max_diff
  merged=$GEODE_STORE/runs/$rid/model_merged
  if [[ -f $merged/model.safetensors ]]; then
    milestone "merge_skip $merged already exists"
  elif [[ -d $merged ]]; then
    fail "$merged exists but holds no model.safetensors — a crashed merge
   (merge_adapter.py refuses to overwrite an existing directory). Remove $merged and
   rerun this launcher."
  else
    python3 merge_adapter.py --run-id "$rid" || fail "merge_adapter.py failed for $rid"
    [[ -f $merged/model.safetensors ]] ||
      fail "merge_adapter.py exited 0 but left no $merged/model.safetensors"
    milestone "merge_complete $merged"
  fi

  receiver_out=$(python3 - "$rid" "$merged" <<'PY'
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

print(f"[ts38fs-tiny] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
sys.exit(0 if max_abs_diff < 1e-3 else 1)
PY
)
  receiver_status=$?
  echo "$receiver_out"
  [[ $receiver_status -eq 0 ]] || fail \
    "receiver check FAILED for $rid — the merged checkpoint does not reproduce the
   wrapped model's logits within 1e-3 (see output above); do NOT hand this off to the
   target runs"
  max_diff=$(sed -n 's/.*max_abs_logit_diff=\(.*\)/\1/p' <<<"$receiver_out")
  milestone "parent_merged run=$rid max_abs_logit_diff=$max_diff"
  MERGED_DIR=$merged
}

# ---- stage 1: the base checkpoint, receiver-verified ----------------------
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; ts38fs-tiny has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38fs-tiny] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38fs-tiny] MISSING on the hub: {missing}")
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
print(f"[ts38fs-tiny] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38fs-tiny] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38fs-tiny] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 2: data preflight — base/bare (needed by every target run) +
# the two tiny cyclic-shift installs. load_frozen_parquet recomputes each
# order_hash and refuses on mismatch — no separate sentinel check is needed
# here (these two configs ship real, measured pins from the start; the
# PIN_AFTER_DATAGEN sentinel pattern was ts38fs's own multi-size-datagen
# convention, not used for this family's two hand-authored configs). -------
DATA_DIR=$REPO_ROOT/experiments/training-run/data/full
BASE_NEEDED=(D_algo_bare.parquet D_algo_eval_bare.parquet D_algo.parquet D_algo_eval.parquet report.json)
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

for i in "${INSTALLS[@]}"; do
  if [[ -f $DATA_DIR/D_preteachfmt_n${i}.parquet ]]; then
    milestone "preteachfmt_datagen_skip i=$i (D_preteachfmt_n${i}.parquet present)"
  else
    milestone "preteachfmt_datagen_start i=$i (operator-notation, cyclic-shift labels)"
    python3 ../datagen/make_preteach_format.py --out "$DATA_DIR" --n "$i" --cyclic-shift ||
      fail "make_preteach_format --n $i --cyclic-shift (D_preteachfmt_n$i derivation)"
    milestone "preteachfmt_datagen_complete i=$i"
  fi
done

python3 - "$TARGET_CONFIG" "$BARE_EVAL_CONFIG" "${INSTALLS[@]}" <<'PY' || fail "data/config pre-flight"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

target_cfg_path, bare_eval, *installs = sys.argv[1:]
for path in (Path(target_cfg_path), Path(bare_eval)):
    if not path.is_file():
        print(f"[ts38fs-tiny] MISSING config {path}")
        sys.exit(1)

for label, path in (
    ("D_algo_bare", Path(target_cfg_path)),
    ("D_algo_eval_bare", Path(bare_eval)),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38fs-tiny] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")

ok = True
for i in installs:
    parent_cfg_path = Path(f"../configs/ts38fs_tiny_parent_n{i}.yaml")
    if not parent_cfg_path.is_file():
        print(f"[ts38fs-tiny] MISSING config {parent_cfg_path}")
        ok = False
        continue
    parent_cfg = load_config(parent_cfg_path, None)
    df = load_frozen_parquet(parent_cfg)
    print(f"[ts38fs-tiny] D_preteachfmt_n{i}: {len(df)} rows, order_hash verified ({parent_cfg['data']['file']})")
sys.exit(0 if ok else 1)
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare/D_preteachfmt_n{2,10} order hashes OK"

# ---- stage 3: overlay presence check — 10 committed files (hand-authored,
# like every ts38(*) family except ts38fs itself; no generator here) -------
OVERLAY_COUNT=$(find "$OVERLAY_DIR" -maxdepth 1 -name 'ts38fs_tiny_i*_n*.yaml' | wc -l | tr -d ' ')
[[ $OVERLAY_COUNT -eq 10 ]] || fail \
  "found $OVERLAY_COUNT ts38fs_tiny_i*_n*.yaml overlay(s) under $OVERLAY_DIR, expected
   exactly 10 (2 installs x 5 target sizes); inspect before training anything"
milestone "overlays_verified count=$OVERLAY_COUNT dir=$OVERLAY_DIR"

# ---- stage 4: build the two tiny format-only parents + merge --------------
declare -A MERGED_BY_INSTALL
for i in "${INSTALLS[@]}"; do
  rid=evt-ts38fs-tiny-parent-n${i}
  parent_config=../configs/ts38fs_tiny_parent_n${i}.yaml
  train_or_skip "$rid" \
    python3 train_sft.py --config "$parent_config" --init-from "$BASE_MODEL" --confirm-cost

  FIELDS=$(python3 - "$rid" <<'PY'
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
  read -r P_STATUS P_METHOD P_GATES P_STOP P_STEP P_DATA_HASH <<<"$FIELDS"
  milestone "parent_fields run=$rid status=$P_STATUS method=$P_METHOD gates=$P_GATES stop_reason=$P_STOP final_step=$P_STEP data_order_hash=$P_DATA_HASH"

  [[ $P_STATUS == complete ]] || fail "$rid manifest status='$P_STATUS' (expected complete) — inspect $GEODE_STORE/runs/$rid/manifest.json"
  [[ $P_METHOD == lora ]] || fail "$rid training.method='$P_METHOD' (expected lora)"
  [[ $P_GATES == none ]] || fail \
    "$rid has recorded gates {$P_GATES} — this family was designed against an ungated
     parent; inspect, do not proceed"
  [[ $P_STOP == converged ]] || fail \
    "$rid stop_reason='$P_STOP' (expected converged) — a max_steps stop on a format-only
     parent is a bug signal (the calibrated eps/k should plateau well inside the 1000-step
     ceiling); do not proceed with an unconverged parent"

  EXPECTED_HASH=$(python3 - "$i" <<'PY'
import sys
from pathlib import Path

from train import load_config

n = sys.argv[1]
cfg = load_config(Path(f"../configs/ts38fs_tiny_parent_n{n}.yaml"), None)
print(cfg["data"]["order_hash"])
PY
)
  [[ $P_DATA_HASH == "$EXPECTED_HASH" ]] || fail \
    "$rid manifest data_order_hash=$P_DATA_HASH != current pin $EXPECTED_HASH — this
     checkpoint was built from a different D_preteachfmt_n${i}.parquet/config than the one
     this launch just verified. Remove $GEODE_STORE/runs/$rid and rerun to rebuild against
     the current pin; do not train the target sweep on a stale parent."
  milestone "parent_verified run=$rid stop_reason=converged final_step=$P_STEP gates={} method=lora data_order_hash=OK"

  merge_and_verify "$rid"
  MERGED_BY_INSTALL[$i]=$MERGED_DIR
  # LOAD-BEARING full push (adapter.safetensors included) — lets a later
  # family reuse this parent the way ts38fs itself reuses ts38pf's.
  push_run "$rid"
done

# ---- stage 5: format-acquisition theta0 check, both install parents.
# LEAKED still hard-fails the whole family; NOT_LEARNED logs and continues
# (same semantics as ts38fs's own stage 5 — at these even-smaller sizes,
# NOT_LEARNED is exactly the floor this family exists to find). --------------
BASE_G5=$(score_g5_no_record "$BASE_RID") || fail "$BASE_RID G5 --no-record (theta0 evidence) failed (see stderr above)"
read -r BASE_EM0 BASE_EM16 BASE_LOSS <<<"$BASE_G5"
milestone "theta0_base em0=$BASE_EM0 em16=$BASE_EM16 loss=$BASE_LOSS"

declare -A FMT_VERDICT_BY_INSTALL
for i in "${INSTALLS[@]}"; do
  parent_rid=evt-ts38fs-tiny-parent-n${i}

  PARENT_G5=$(score_g5_no_record "$parent_rid") || fail "$parent_rid G5 --no-record (theta0 evidence) failed (see stderr above)"
  read -r PARENT_EM0 PARENT_EM16 PARENT_LOSS <<<"$PARENT_G5"

  VERDICT_LINE=$(python3 - "$parent_rid" "$PARENT_EM0" "$PARENT_EM16" "$PARENT_LOSS" "$BASE_LOSS" <<'PY'
import sys

rid, em0, em16, loss, base_loss = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
loss_drop_frac = (base_loss - loss) / base_loss
if em0 > 0.05:
    verdict = "LEAKED"
elif loss_drop_frac < 0.10:
    verdict = "NOT_LEARNED"
else:
    verdict = "LEARNED"
print(verdict, f"{loss_drop_frac:.4f}")
PY
)
  read -r VERDICT LOSS_DROP <<<"$VERDICT_LINE"
  FMT_VERDICT_BY_INSTALL[$i]=$VERDICT
  milestone "format_acquisition_check install=$i parent=$parent_rid verdict=$VERDICT loss_drop_frac=$LOSS_DROP em0=$PARENT_EM0"

  if [[ $VERDICT == LEAKED ]]; then
    fail "FORMAT-ACQUISITION CHECK (install=$i, $parent_rid): zero-shot EM on the bare-NL
     eval is materially above 0 — the cyclic-shift label control did not act as a control
     (algorithm may have leaked). Do NOT proceed with ANY of this family's target runs;
     investigate make_preteach_format.py / cyclic_shift_labels before relaunching."
  elif [[ $VERDICT == NOT_LEARNED ]]; then
    echo "[ts38fs-tiny] NOTE install=$i ($parent_rid): loss drop vs base is $LOSS_DROP (<0.10) —" \
      "format not (fully) acquired at this install size. Logged, NOT a launch blocker —" \
      "this IS the dose-response floor this family exists to find."
  fi

  python3 - "$FMT_JSON" "$i" "$parent_rid" "$PARENT_EM0" "$PARENT_EM16" "$PARENT_LOSS" \
    "$VERDICT" "$LOSS_DROP" "$BASE_RID" "$BASE_EM0" "$BASE_EM16" "$BASE_LOSS" <<'PY' || fail "writing $FMT_JSON failed"
import json
import sys
from pathlib import Path

out_path, install, p_rid, p_em0, p_em16, p_loss, verdict, loss_drop, b_rid, b_em0, b_em16, b_loss = sys.argv[1:13]
out = Path(out_path)
data = json.loads(out.read_text()) if out.is_file() else {"eval_config": "eval_bare_target_data_ts38.yaml", "base": None, "parents": {}}
data["base"] = {"run_id": b_rid, "em0": float(b_em0), "em16": float(b_em16), "loss": float(b_loss)}
data["parents"][install] = {
    "run_id": p_rid,
    "em0": float(p_em0),
    "em16": float(p_em16),
    "loss": float(p_loss),
    "verdict": verdict,
    "loss_drop_frac": float(loss_drop),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2) + "\n")
PY
done
milestone "format_acquisition_complete json=$FMT_JSON verdicts=$(for i in "${INSTALLS[@]}"; do echo -n "i$i=${FMT_VERDICT_BY_INSTALL[$i]} "; done)"

# ---- stage 6: the target sweep — I ascending, N ascending (single seed,
# so no seed loop) -----------------------------------------------------------
FIRST_CELL_CHECKED=0
for i in "${INSTALLS[@]}"; do
  merged=${MERGED_BY_INSTALL[$i]}
  for n in "${SIZES[@]}"; do
    rid=evt-ts38fs-tiny-i${i}-n${n}-s${SEED}
    overlay=$OVERLAY_DIR/ts38fs_tiny_i${i}_n${n}.yaml
    [[ -f $overlay ]] || fail "$overlay missing — should be a committed file, not generated"

    train_or_skip "$rid" \
      python3 train_target.py --config "$TARGET_CONFIG" \
        --override "$overlay" \
        --init-from "$merged" --confirm-cost
    push_run "$rid"

    if [[ $rid == "$PIN_CHECK_RID" && $FIRST_CELL_CHECKED -eq 0 ]]; then
      FIRST_CELL_CHECKED=1
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
      milestone "pin_check run=$rid stop_reason=$PIN_STOP min_val_nats=$PIN_MIN best_val_nats=$PIN_BEST final_step=$PIN_STEP"
      [[ $PIN_STOP == converged ]] || fail \
        "PIN CHECK: $rid (the first cell trained) ended with stop_reason='$PIN_STOP' (min
   val $PIN_MIN, best $PIN_BEST, step $PIN_STEP), not 'converged'. 9 more cells would
   inherit whatever this is. A max_steps stop is a bug signal, not a result."
    fi
  done
done
milestone "target_sweep_complete cells=10"

# ---- stage 7: receiver-verify every target run on the hub -----------------
run_receiver_check() {
  python3 - "$RELAY_REPO" <<'PY'
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True
for i in (2, 10):
    for n in (1000, 4642, 21544, 100000, 316228):
        rid = f"evt-ts38fs-tiny-i{i}-n{n}-s316"
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
echo "[ts38fs-tiny] receiver check (hub files for all 10 target runs):"
echo "$RECEIVER_OUT"
if [[ $RECEIVER_STATUS -ne 0 ]]; then
  milestone "receiver_retry (re-pushing the runs the hub is missing)"
  while read -r rid; do
    push_run "$rid"
  done < <(sed -n 's/^MISSING //p' <<<"$RECEIVER_OUT")
  RECEIVER_OUT=$(run_receiver_check)
  RECEIVER_STATUS=$?
  echo "[ts38fs-tiny] receiver check (after one push retry):"
  echo "$RECEIVER_OUT"
fi
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED — see output above"
milestone "targets_pushed cells=10 receiver=OK"

for i in "${INSTALLS[@]}"; do
  python3 hf_checkpoint.py push --run-id "evt-ts38fs-tiny-parent-n${i}" --repo-id "$RELAY_REPO" --metadata-only ||
    echo "[ts38fs-tiny] WARN hf_checkpoint.py push --metadata-only failed for evt-ts38fs-tiny-parent-n${i} (best effort)"
done

echo "[ts38fs-tiny] format-acquisition evidence at $FMT_JSON — not a run, not pushed"

# ---- stage 8: finish --------------------------------------------------------
echo "[ts38fs-tiny] MILESTONE analysis_commands"
echo "[ts38fs-tiny]   CPU-only, run here off \$GEODE_STORE. Same OCV-floor"
echo "[ts38fs-tiny]   primitives as analysis/ts38fs_dose_curve.py -- that"
echo "[ts38fs-tiny]   script's TS38FS_RE regex does NOT match evt-ts38fs-tiny-*"
echo "[ts38fs-tiny]   ids on purpose (recipe mismatch vs ts38fs proper --"
echo "[ts38fs-tiny]   see decisions.md's caveat). Extend it deliberately, not"
echo "[ts38fs-tiny]   by loosening the regex to match both families silently."
echo "[ts38fs-tiny]   The box is NOT torn down here; teardown is the operator's"
echo "[ts38fs-tiny]   call once these artifacts are verified on the relay."
notify "ts38fs-tiny family done: 2 parents + 10 target runs, format_acquisition=$(for i in "${INSTALLS[@]}"; do echo -n "i$i:${FMT_VERDICT_BY_INSTALL[$i]} "; done)"
echo "[ts38fs-tiny] TERMINAL_SUCCESS runs=10"
