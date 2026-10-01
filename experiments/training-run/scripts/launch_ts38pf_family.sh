#!/usr/bin/env bash
# ts38pf family — Donoway et al. "Bits That Count" App. E.1.2 pre-teach-
# FORMAT causal intervention, on the 38.7M TinyStories base (EXPERIMENTS.md
# §6.16, decisions.md 2026-08-15 "ts38pf pre-registration"). Builds a NEW
# format-only parent (LoRA on D_preteachfmt.parquet — operator-notation,
# RANDOMLY PERMUTED/incorrect labels, teaches numeral vocab + output format
# without the input-output mapping), then trains the "pretaught-format"
# target arm from it onto the bare-NL target D_algo_bare, with the recipe
# IDENTICAL to the existing ts38/ts38mw families — LoRA r128/alpha32 @1e-3,
# same data, same stopping rule. Arms differ ONLY in theta0.
#
# WHY: the base (teach) arm's EDL/D-vs-n curve doesn't show the paper's
# "up-then-down" teaching hump on this grid; removing the format-learning
# transient before the real target run is meant to reveal it, if it exists
# here (paper App. E.1.2/Table 5: TinyStories add/sub pre-teach-format peak
# ~150K at 1B params — our grid tops out at 316K and the base's own argmax
# already sits at/below n=1000, so a flat/still-falling result here does
# NOT by itself refute the hypothesis; see decisions.md).
#
# THE SIX RUNS
#   evt-ts38pf-preteachfmt-parent   NEW, built here: LoRA r128/a32 @1e-3 on
#                                    D_preteachfmt.parquet (21,544 rows,
#                                    operator-notation, permuted labels),
#                                    from evt-run1-base-v3-ext, via
#                                    train_sft.py, to convergence (min_steps
#                                    pinned to exactly one full epoch — see
#                                    ts38_preteachfmt_parent.yaml's header).
#   evt-ts38pf-preteachfmt-n{N}     theta0 -> LoRA r128/a32 @1e-3 on
#                                    D_algo_bare, SAME data order as the
#                                    matching evt-ts38-base-n{N} run (G7
#                                    anchor). for N in 1000 4642 21544
#                                    100000 316228.
#
# BASE ARM REUSED, NOT RETRAINED: evt-ts38-base-n{N} (launch_ts38_mini.sh)
# already trains + measures the teaching arm on this exact data/recipe. This
# launcher only READS its manifest (data_order_hash, n_examples) as the G7
# anchor per size — it is never (re-)trained, gated, or pushed here.
#
# FORMAT-ACQUISITION GATE (advisor-reviewed, decisions.md pre-registration):
# after merging the new parent, its loss + zero-shot EM on the family's bare
# eval pin is measured --no-record against the base's, same mechanism as
# ts38mw's theta0 check. The 5-size sweep does NOT launch unless the parent's
# loss is materially below base's (format acquired) AND its EM stays ~0 (no
# algorithm leaked). A parent that fails this check is reported as "the
# operator-notation format lesson did not transfer to the bare-NL
# rendering" — a DIFFERENT, weaker claim than "format pre-teaching doesn't
# reshape the curve" — never conflated with the shape question below.
#
# PRE-REGISTERED READOUT for the shape question, once the sweep exists (do
# not re-derive after seeing numbers):
#   base teaching marker = its ALREADY-MEASURED rising span 4642->21544
#     (+15% under all three floors) — not re-derived here.
#   pretaught-format: does its EDL/D curve show a rising limb the base arm's
#     own curve does not (i.e. is the base's transient a format-learning
#     artifact this arm removes)? Report the shape as observed — this is
#     NOT the ts38mw elicitation-marker question and has no monotone-and-
#     below-base pass/fail bar; see decisions.md.
#
# HARD RULES
#   (a) this parent is DELIBERATELY UNGATED (no G1/G8 certification — it is
#       a format-only control, not a capability parent). This launcher
#       NEVER runs gates.py on it except --no-record (theta0 stage,
#       evidence only, no bar) and never records anything to its manifest
#       except merge_adapter.py's own experiment.merged_checkpoint entry.
#   (b) LoRA checkpoints load ONLY via geode.zoo.load_model
#       ([[feedback-lora-checkpoints-load-via-zoo-load-model]]) — the target
#       arm warm-starts from the parent's MERGED plain weights
#       (runs/<id>/model_merged/), never from_pretrained on the wrapped
#       runs/<id>/model/.
#   (c) NEVER destroy the box — it is the owner's own rental, not ours to
#       tear down.
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=ts38pf

echo "[ts38pf] estimated cost: one LoRA parent-build run (21,544 rows,"
echo "[ts38pf] 167 steps/epoch, 20-epoch ceiling 3340 steps -- eps/k should"
echo "[ts38pf] converge well before that on a low-capacity format-only"
echo "[ts38pf] objective) + 5 LoRA target runs on one RTX 4090; the base"
echo "[ts38pf] arm's same-size runs converged at 135/270/1825/5000/10875"
echo "[ts38pf] steps. Same order of magnitude as the ts38mw family launch,"
echo "[ts38pf] well under \$1; disk < 1 GB. Ceilings (max_steps) never"
echo "[ts38pf] bind -- eps/k convergence stops each run; stop_reason="
echo "[ts38pf] max_steps is a bug signal, never a result."

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_ts38pf_family.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}

BASE_RID=evt-run1-base-v3-ext
BASE_MODEL=$GEODE_STORE/runs/$BASE_RID/model
RELAY_REPO=${RELAY_REPO:-mhieuuu/geode-store}
PARENT_RID=evt-ts38pf-preteachfmt-parent
SIZES=(1000 4642 21544 100000 316228)
PIN_CHECK_N=1000

PARENT_CONFIG=../configs/ts38_preteachfmt_parent.yaml
TARGET_CONFIG=../configs/ts38pf_preteachfmt.yaml
OVERLAY_DIR=../configs/sweeps/ts38
BARE_EVAL_CONFIG=../configs/eval_bare_target_data_ts38.yaml
THETA0_JSON=$GEODE_STORE/results/ts38pf_family_theta0.json
PREFMT_ORDER_HASH=5b0b19a4c47375a4ada17cb1ee21292475b6ecaed22b2ef07aa560cf557b1bc1

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
# on identical data for every run). No pass bar by design — the ts38mw
# family's record_g5, verbatim.
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

# --no-record G5 evidence, parsed the way ts38mw's score_g5_no_record parses
# gates.py g5's printed output. Full gate output goes to stderr so the
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

# ---- stage 1: the base checkpoint, receiver-verified (ts38mw's stage 1,
# verbatim — the format-only parent's build starts from this same base) ----
milestone "relay_verify_start run=$BASE_RID repo=$RELAY_REPO"
python3 - "$BASE_RID" "$RELAY_REPO" <<'PY' || fail "$BASE_RID is not on the relay — nothing to pull; the ts38pf family has no base"
import sys

from huggingface_hub import HfApi

rid, repo = sys.argv[1], sys.argv[2]
files = [f for f in HfApi().list_repo_files(repo) if f.startswith(f"runs/{rid}/")]
required = [f"runs/{rid}/manifest.json", f"runs/{rid}/model/model.safetensors"]
missing = [r for r in required if r not in files]
print(f"[ts38pf] hub {repo} holds {len(files)} file(s) under runs/{rid}/")
if missing:
    print(f"[ts38pf] MISSING on the hub: {missing}")
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
print(f"[ts38pf] loaded {rid}: {got} status={manifest['status']}")
print(f"[ts38pf] tokenizer {tok_dir}: sha256 {tok_sha[:16]}… (manifest {want_sha[:16]}…)")
ok = got == want and tok_sha == want_sha and model.config.vocab_size == want["vocab_size"]
if not ok:
    print(f"[ts38pf] MISMATCH: wanted {want} and tokenizer sha {want_sha}")
sys.exit(0 if ok else 1)
PY
milestone "relay_verify base=$BASE_RID d512/L8/vocab10000 -> PASS"

# ---- stage 2: data artifacts, then the pre-teach-format derivation -------
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

if [[ -f $DATA_DIR/D_preteachfmt.parquet ]]; then
  milestone "preteachfmt_datagen_skip (D_preteachfmt.parquet present)"
else
  milestone "preteachfmt_datagen_start (n=21544, operator-notation, permuted labels)"
  python3 ../datagen/make_preteach_format.py --out "$DATA_DIR" --n 21544 ||
    fail "make_preteach_format (D_preteachfmt derivation)"
  milestone "preteachfmt_datagen_complete"
fi

python3 - "$TARGET_CONFIG" "$BARE_EVAL_CONFIG" "$PARENT_CONFIG" "$PREFMT_ORDER_HASH" <<'PY' || fail "data/config pre-flight"
import sys
from pathlib import Path

from train import load_config
from train_sft import load_frozen_parquet

target_cfg_path, bare_eval, parent_cfg_path, prefmt_hash = sys.argv[1:5]
for path in (Path(target_cfg_path), Path(bare_eval), Path(parent_cfg_path)):
    if not path.is_file():
        print(f"[ts38pf] MISSING config {path}")
        sys.exit(1)

for label, path in (
    ("D_algo_bare", Path(target_cfg_path)),
    ("D_algo_eval_bare", Path(bare_eval)),
):
    cfg = load_config(path, None)
    df = load_frozen_parquet(cfg)  # recomputes the order hash, refuses on mismatch
    print(f"[ts38pf] {label}: {len(df)} rows, order_hash verified ({cfg['data']['file']})")

parent_cfg = load_config(Path(parent_cfg_path), None)
if parent_cfg["data"]["order_hash"] != prefmt_hash:
    print(
        f"[ts38pf] {parent_cfg_path}: data.order_hash "
        f"{parent_cfg['data']['order_hash']} != build-time pin {prefmt_hash} — "
        "the committed config and the file on disk disagree; regenerate or re-pin"
    )
    sys.exit(1)
df = load_frozen_parquet(parent_cfg)
print(f"[ts38pf] D_preteachfmt: {len(df)} rows, order_hash verified ({parent_cfg['data']['file']})")
PY
milestone "data_verified D_algo_bare/D_algo_eval_bare/D_preteachfmt order hashes OK"

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
        print(f"[ts38pf] MISSING manifest for {rid}")
        ok = False
        continue
    d = json.loads(p.read_text())
    status = d.get("status")
    exp = d.get("experiment", {})
    n_examples = exp.get("n_examples")
    order_hash = exp.get("data_order_hash")
    good = status == "complete" and n_examples == int(n) and bool(order_hash)
    print(
        f"[ts38pf] {rid}: status={status} n_examples={n_examples} "
        f"order_hash={'set' if order_hash else 'MISSING'} -> {'OK' if good else 'BAD'}"
    )
    ok = ok and good
sys.exit(0 if ok else 1)
PY
milestone "g7_anchors_ready sizes=${#SIZES[@]}"

# ---- stage 4: build the format-only parent (NEW vs. ts38mw — that family
# pulled an already-built parent; this one trains it here) -----------------
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
[[ $P_METHOD == lora ]] || fail "$PARENT_RID training.method='$P_METHOD' (expected lora)"
[[ $P_GATES == none ]] || fail \
  "$PARENT_RID has recorded gates {$P_GATES} — this family was designed against an
   ungated parent; inspect, do not proceed"
[[ $P_STOP == converged ]] || fail \
  "$PARENT_RID stop_reason='$P_STOP' (expected converged) — a max_steps stop on the
   format-only parent is a bug signal (permuted labels should plateau well inside the
   20-epoch ceiling); do not proceed with an unconverged parent"
# Stale-parent guard: train_or_skip reuses ANY complete-status checkpoint at
# this run_id, including one built from a since-changed config/dataset on an
# earlier invocation. The manifest's own data_order_hash (recorded from
# cfg["data"]["order_hash"] at build time, train_sft.py:190) is the cheapest
# fingerprint tying the checkpoint back to the exact D_preteachfmt.parquet
# this launch run verified in stage 2 — a mismatch means "delete
# runs/$PARENT_RID and rebuild", not "keep training on top of it".
[[ $P_DATA_HASH == "$PREFMT_ORDER_HASH" ]] || fail \
  "$PARENT_RID manifest data_order_hash=$P_DATA_HASH != current pin $PREFMT_ORDER_HASH —
   this checkpoint was built from a different D_preteachfmt.parquet/config than the one
   this launch just verified. Remove $GEODE_STORE/runs/$PARENT_RID and rerun to rebuild
   against the current pin; do not train the target sweep on a stale parent."
milestone "parent_verified stop_reason=converged final_step=$P_STEP gates={} method=lora data_order_hash=OK"

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
# to <1e-3 on seeded random token batches.
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

print(f"[ts38pf] receiver check: max_abs_logit_diff={max_abs_diff:.6e}")
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

# ---- stage 5: theta0 format-acquisition check — evidence + a HALT gate ----
# Scores the parent's wrapped model/ and the base on the family's own bare
# eval pin. Unlike ts38mw's theta0 record (evidence only, no bar), THIS
# check gates whether the 5-size sweep launches at all — advisor review:
# running the sweep on an unconverted-format parent would look identical
# post hoc to "removing the format confound didn't reshape the curve", and
# those are different claims.
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

# Pre-registered format-acquisition read (decisions.md): parent loss must be
# materially below base's, EM must stay ~0 (no algorithm leaked from the
# permuted-label training). Margins: loss drop >= 10% of base's loss;
# EM threshold 5% (matches the "near-zero" bar used elsewhere in this
# family, e.g. the fig2nl base-arm ~0% zero-shot convention).
FMT_VERDICT=$(python3 - "$THETA0_JSON" <<'PY'
import json
import sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
p, b = d["parent"], d["base"]
loss_drop_frac = (b["loss"] - p["loss"]) / b["loss"]
if p["em0"] > 0.05:
    verdict = "LEAKED"
elif loss_drop_frac < 0.10:
    verdict = "NOT_LEARNED"
else:
    verdict = "LEARNED"
print(verdict, f"{loss_drop_frac:.4f}")
PY
)
read -r FMT_STATUS FMT_LOSS_DROP <<<"$FMT_VERDICT"
milestone "format_acquisition_check verdict=$FMT_STATUS loss_drop_frac=$FMT_LOSS_DROP"

if [[ $FMT_STATUS == LEAKED ]]; then
  fail "FORMAT-ACQUISITION CHECK: parent zero-shot EM on the bare-NL eval is
   materially above 0 — the label permutation did not act as a control (algorithm may
   have leaked). See $THETA0_JSON. Do NOT proceed with the 5-size sweep; investigate
   make_preteach_format.py / permute_labels before relaunching."
elif [[ $FMT_STATUS == NOT_LEARNED ]]; then
  fail "FORMAT-ACQUISITION CHECK: parent loss on the bare-NL eval is not materially
   below base's (drop=$FMT_LOSS_DROP, need >=0.10). Read this as 'the operator-notation
   format lesson did not transfer to the bare-NL rendering' (the same whole-template lock
   documented in docs/ts38-vs-bits-that-count.md) — NOT as 'format pre-teaching doesn't
   reshape the curve'. See $THETA0_JSON. Do NOT proceed with the 5-size sweep."
fi
milestone "format_acquired -> proceeding to the 5-size sweep"

# ---- stage 6: the five target runs, ascending n ----------------------------
for n in "${SIZES[@]}"; do
  rid=evt-ts38pf-preteachfmt-n${n}
  train_or_skip "$rid" \
    python3 train_target.py --config "$TARGET_CONFIG" \
      --override "$OVERLAY_DIR/ts38pf_preteachfmt_n${n}.yaml" \
      --init-from "$MERGED_DIR" --confirm-cost

  if [[ $n == "$PIN_CHECK_N" ]]; then
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
   step $PIN_STEP), not 'converged'. The pretaught-format arm's first/cheapest run did not
   converge under the shared eps/k rule inherited from the ts38 family. Four more runs would
   inherit whatever this is. A max_steps stop is a bug signal, not a result."
  fi
  record_g5 "$rid"
done

python3 - "${SIZES[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

store = Path(os.environ["GEODE_STORE"])
print(f"[ts38pf] {'run_id':<30}{'final_step':>12}{'stop_reason':>14}{'min_val_nats':>16}{'edl_per_label_token_nats':>26}")
for n in sys.argv[1:]:
    rid = f"evt-ts38pf-preteachfmt-n{n}"
    p = store / "runs" / rid / "manifest.json"
    r = json.loads(p.read_text()).get("experiment", {}).get("target_result", {}) if p.is_file() else {}
    edl = r.get("edl_per_label_token_nats") if r else None
    print(
        f"[ts38pf] {rid:<30}{r.get('final_step', 'MISSING'):>12}"
        f"{r.get('stop_reason', 'MISSING'):>14}{r.get('min_val_nats', 'MISSING'):>16}"
        f"{edl if edl is not None else 'n/a':>26}"
    )
PY
milestone "family_complete runs=${#SIZES[@]}"

# ---- stage 7: push + receiver-verify ---------------------------------------
for n in "${SIZES[@]}"; do
  rid=evt-ts38pf-preteachfmt-n${n}
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$RELAY_REPO"; then
    milestone "push_complete run=$rid"
  else
    echo "[ts38pf] WARN hf_checkpoint.py push failed for $rid (best effort)"
    milestone "push_warn run=$rid"
  fi
done

RECEIVER_OUT=$(python3 - "$RELAY_REPO" "${SIZES[@]}" <<'PY'
import sys

from huggingface_hub import HfApi

repo = sys.argv[1]
sizes = sys.argv[2:]
files = set(HfApi().list_repo_files(repo, repo_type="model"))
ok = True
for n in sizes:
    rid = f"evt-ts38pf-preteachfmt-n{n}"
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
echo "[ts38pf] receiver check (hub files under runs/evt-ts38pf-preteachfmt-n*/):"
echo "$RECEIVER_OUT"
[[ $RECEIVER_STATUS -eq 0 ]] || fail "push receiver check FAILED — see output above (at least one run's manifest/eval_log/prequential log is missing on the relay)"
milestone "targets_pushed sizes=${#SIZES[@]} receiver=OK"

python3 hf_checkpoint.py push --run-id "$PARENT_RID" --repo-id "$RELAY_REPO" --metadata-only ||
  echo "[ts38pf] WARN hf_checkpoint.py push --metadata-only failed for $PARENT_RID (best effort)"

echo "[ts38pf] theta0 evidence at $THETA0_JSON — not a run, not pushed; the operator scp's results/ back"

# ---- stage 8: finish --------------------------------------------------------
echo "[ts38pf] MILESTONE analysis_commands"
echo "[ts38pf]   CPU-only, run here off \$GEODE_STORE. --family ts38pf is NOT"
echo "[ts38pf]   a default anywhere — pass it explicitly:"
echo "[ts38pf]     python3 ../analysis/edl_converged_val_floor.py --family ts38pf  # OCV + test floor"
echo "[ts38pf]   (dataset_size_sweep.py has no ts38pf entry yet — its FAMILIES dict"
echo "[ts38pf]   + straddling-prefix special case would need extending separately,"
echo "[ts38pf]   same shape as TS38MW_PREFIX; out of this launcher's scope.)"
echo "[ts38pf]   OCV primary, floor NAMED on every figure. Read the"
echo "[ts38pf]   pretaught-format limb against the reused evt-ts38-base-n* curve"
echo "[ts38pf]   (rising span 4642->21544 already measured) and against the theta0"
echo "[ts38pf]   evidence in $THETA0_JSON. This is a SHAPE question (does removing"
echo "[ts38pf]   the format transient reveal a rising limb the base doesn't show),"
echo "[ts38pf]   not the ts38mw elicitation-marker question — no monotone-and-"
echo "[ts38pf]   below-base pass/fail bar applies here; see decisions.md."
notify "ts38pf family done: 6 runs (1 parent + 5 sizes), format_acquisition=$FMT_STATUS"
echo "[ts38pf] TERMINAL_SUCCESS runs=5"
