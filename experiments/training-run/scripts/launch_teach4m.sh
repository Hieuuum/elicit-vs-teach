#!/usr/bin/env bash
# launch_teach4m.sh — the teach-4M replication program (notes/teach4m_plan.md).
#
# The teach twin's 1M endpoint solves 0.093 of held-out problems (stopped on
# validation convergence mid-hump). This launcher builds a teach endpoint that
# gets to learn — the blank twin on 4M UNIQUE NL add/sub problems, one full
# pass minimum (the paper's recipe) — and then re-runs every teach-side
# measurement in the write-up on it with the SAME analysis scripts and flags
# as before. No new instruments.
#
# Stages (each skip-if-done; rerunning after a crash resumes):
#   0  data      datagen/make_algo_4m.py -> data/full/D_algo_bare_4m.parquet, pin-checked
#   1  train     evt-ts1b-fig2ts-noinst-n4000000 (snapshots streamed to HF unless
#                --no-stream), G5, push, verify, prune  [GPU, ~15-30 h]
#   2  behaviour dataset_size_sweep.py --family ts (+ the 4M point)  [CPU]
#   3  battery   circuits (map, split-half, faithfulness, compares), DCM roles,
#                steering donor, weight shift/travel, gradient strength, residual
#                shift, lens depth, formation curve  [GPU, ~2-3 h]
#   4  train-ft  evt-ts1b-teach-ft-n4000000: the blank twin FULLY fine-tuned on the
#                same 4M file (configs/ts1b_teach_ft.yaml; train_sft.py, lr 2e-5),
#                G5. Never pruned (no adapter). ~25 GB GPU, ~10-19 h.
#   5  battery-ft the same battery on the full-FT endpoint (no snapshots /
#                gradstats there: formation curve, weight travel and gradient
#                strength are skipped; weight shift uses the checkpoint diff)
#   6  train-elft evt-ts1b-elicit-ft-n4000000: the LATENT parent fully fine-tuned
#                on the same file (method control: both regimes under both
#                methods). ~1 h. Explicit --stage 6 only (not part of `all`).
#   7  battery-elft the battery on it (steering base / weight & residual parent =
#                the latent parent). Explicit only.
#   8  resid-v2   residual shift v2 for the four original cells + one combined
#                compare over every cell. Explicit only.
#
# Run GPU stages one after the other, not concurrently, on a 40 GB card.
#
# Usage:  bash launch_teach4m.sh --confirm-cost [--stage N] [--no-stream] [--no-prune]
#   env:  GEODE_STORE (store root), HF_WRITE_TOKEN/HF_TOKEN (write scope; only
#         needed when streaming), HF_NAMESPACE (default podhajskimarcin),
#         TS_VALID (path to TinyStoriesV2-GPT4-valid.txt; else hub download)
#         Comparator overrides (else auto-discovered from JSON sidecars in
#         analysis/):  MAP_ELICITED MAP_TAUGHT1M MAP_BASE16 DCM_ELICITED_NL DCM_PARENT_OP
set -uo pipefail
cd "$(dirname "$0")"
source lib/launch_common.sh
TAG=teach4m

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_teach4m.sh: --confirm-cost required (budget rule): stage 1 is a ~15-30 h GPU run" >&2
  exit 1
}
STAGE=all; STREAM=1; PRUNE=1
for ((i = 1; i <= $#; i++)); do
  case ${!i} in
    --stage) j=$((i + 1)); STAGE=${!j} ;;
    --no-stream) STREAM=0 ;;
    --no-prune) PRUNE=0 ;;
  esac
done
want() { [[ $STAGE == "$1" || ( $STAGE == all && $1 -le 5 ) ]]; }  # 6-8 are explicit only

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT
export PYTHONPATH=$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
HF_NAMESPACE=${HF_NAMESPACE:-podhajskimarcin}
if ((STREAM)); then
  export HF_TOKEN=${HF_WRITE_TOKEN:-${HF_TOKEN:?need HF_TOKEN or HF_WRITE_TOKEN (write scope) to stream snapshots; or pass --no-stream}}
fi

RID=evt-ts1b-fig2ts-noinst-n4000000
RID_FT=evt-ts1b-teach-ft-n4000000
CFG_FT=../configs/ts1b_teach_ft.yaml
RID_ELFT=evt-ts1b-elicit-ft-n4000000
CFG_ELFT=../configs/ts1b_elicit_ft.yaml
RID_1M=evt-ts1b-fig2ts-noinst-n1000000
BASE=evt-ts1b-base
LATENT=evt-ts1b-op-bridge-mix
ELICITED=evt-ts1b-mix-nl-n1000000
CFG=../configs/ts1b_fig2ts_noinst.yaml
OVERLAY=../configs/sweeps/ts1b_fig2ts/ts1b_fig2ts_noinst_n4000000.yaml
DATA=../data/full/D_algo_bare_4m.parquet
A=$REPO_ROOT/experiments/training-run/analysis
LOG=$A/teach4m_battery.log

milestone "repo $(git log --oneline -1)"
milestone "store=$GEODE_STORE stage=$STAGE stream=$STREAM prune=$PRUNE run=$RID"

# --------------------------------------------------------------- stage 0: data
overlay_pin() { sed -n 's/^  order_hash: *//p' "$OVERLAY" | head -1; }
if want 0; then
  pin=$(overlay_pin)
  [[ $pin != PIN_PENDING ]] || fail "overlay $OVERLAY still carries PIN_PENDING — build the data locally first and pin it"
  if [[ -f $DATA ]]; then
    milestone "data_present $DATA — verifying pin"
  else
    milestone "data_build start (deterministic, ~10 min CPU, ~6 GB RAM)"
    (cd ../datagen && python3 make_algo_4m.py --out ../data/full) || fail "make_algo_4m.py"
  fi
  python3 - "$DATA" "$pin" <<'PY' || fail "D_algo_bare_4m pin mismatch — do not train on it"
import sys
import pyarrow.parquet as pq
from geode.arith import order_hash
t = pq.read_table(sys.argv[1], columns=["a", "b", "op", "shown_answer", "format", "label_mode"])
got = order_hash(t.to_pylist())
print(f"[teach4m] {t.num_rows:,} rows, order_hash {got}")
sys.exit(0 if got == sys.argv[2] else 1)
PY
  milestone "data_ok pin=$pin"
fi

# -------------------------------------------------------------- stage 1: train
train_or_skip() {
  local rid=$1 status
  shift
  status=$(status_of "$rid")
  if [[ $status == complete ]]; then
    milestone "train_skip run=$rid status=complete"; return 0
  elif [[ $status != missing ]]; then
    fail "$rid exists with status '$status'; inspect it rather than overwriting"
  fi
  milestone "train_start run=$rid"
  "$@" || fail "$rid training"
  [[ $(status_of "$rid") == complete ]] || fail "$rid did not complete"
  milestone "train_complete run=$rid stop_reason=$(stop_reason_of "$rid" target_result)"
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
if want 1; then
  [[ -d $GEODE_STORE/runs/$BASE/model ]] || fail "parent checkpoint $GEODE_STORE/runs/$BASE/model missing"
  run_dir=$GEODE_STORE/runs/$RID
  if ((STREAM)) && [[ $(status_of "$RID") == missing ]]; then
    marker=$run_dir.stream-done; rm -f "$marker"
    python3 stream_snapshots.py --run-id "$RID" --repo-id "$HF_NAMESPACE/$RID" \
      --done-marker "$marker" > "stream_${RID}.log" 2>&1 &
    stream_pid=$!
    milestone "streamer_start run=$RID pid=$stream_pid repo=$HF_NAMESPACE/$RID"
  else
    stream_pid=""
  fi
  train_or_skip "$RID" \
    python3 train_target.py --config "$CFG" --override "$OVERLAY" \
      --init-from "$GEODE_STORE/runs/$BASE/model" --confirm-cost
  record_g5 "$RID"
  python3 - "$RID" <<'PY'
import json, os
from pathlib import Path
m = json.loads((Path(os.environ["GEODE_STORE"]) / "runs" / os.sys.argv[1] / "manifest.json").read_text())
e = m["experiment"]; r = e.get("target_result", {}); g = e.get("gates", {}).get("G5", {})
print(f"[teach4m] RESULT {os.sys.argv[1]}: steps {r.get('final_step')} stop={r.get('stop_reason')} "
      f"EDL/tok {r.get('edl_per_label_token_nats'):.4f} best_val {r.get('best_val_nats'):.4f}  "
      f"G5 EM 0-shot {g.get('zero_shot_accuracy')} 16-shot {g.get('sixteen_shot_accuracy')}")
PY
  if [[ -n $stream_pid ]]; then
    touch "$marker"; wait "$stream_pid" || fail "$RID snapshot streamer failed — see stream_${RID}.log"
    rm -f "$marker"; milestone "streamer_done run=$RID"
    python3 hf_checkpoint.py push --run-id "$RID" --repo-id "$HF_NAMESPACE/$RID" --public || fail "$RID push"
    python3 - "$RID" "$HF_NAMESPACE/$RID" <<'PY' || fail "$RID hub sha256 verify after push — NOT pruning"
import os, sys
from pathlib import Path
from hf_checkpoint import verify_hub_checkpoint
verify_hub_checkpoint(Path(os.environ["GEODE_STORE"]), sys.argv[1], repo_id=sys.argv[2])
print(f"[teach4m] hub sha256 verified for {sys.argv[1]}")
PY
    if ((PRUNE)); then
      rm -f "$run_dir/model/model.safetensors"
      milestone "pruned run=$RID (adapter/manifest/logs kept; analysis rebuilds base+sidecar)"
    fi
  fi
fi

# ---------------------------------------------------------- stage 2: behaviour
if want 2; then
  ids=()
  for n in 1000 1468 2154 3162 4642 6813 10000 14678 21544 31623 46416 68129 100000 146780 215443 316228 464159 681292 1000000 4000000; do
    ids+=(--run-id "evt-ts1b-fig2ts-noinst-n$n")
  done
  (cd "$A" && python3 dataset_size_sweep.py --family ts "${ids[@]}") 2>&1 | tee -a "$LOG" ||
    milestone "dataset_size_sweep failed (non-fatal; the manifest RESULT line above is the R1 point)"
fi

# ------------------------------------------------------------ stage 3: battery
# find_json <kind> <run-id> [surface] -> stem of an artifact in analysis/ whose
# recorded "model" is that run. kind=nodes: a circuit_nodes map (its parquet has
# a node_type column; edge maps are excluded; op-surface maps are skipped when a
# target-surface map exists). kind=dcm: a dcm_roles JSON (has "roles"), surface
# must match. Prints every candidate on stderr, picks the last by name.
find_json() {
  python3 - "$A" "$1" "$2" "${3:-}" <<'PY'
import json, sys
from pathlib import Path
import pyarrow.parquet as pq
root, kind, rid, surf = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
hits = []
for p in sorted(root.glob("*.json")):
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if not isinstance(d, dict) or d.get("model") != rid:
        continue
    if kind == "nodes":
        pf = p.with_suffix(".parquet")
        if not pf.is_file() or "node_type" not in pq.read_schema(pf).names:
            continue
        if d.get("half"):
            continue
    elif kind == "dcm":
        if "roles" not in d or (surf and d.get("surface") != surf):
            continue
    hits.append((p.stem, d))
if kind == "nodes" and any("_op" not in h for h, _ in hits):
    hits = [(h, d) for h, d in hits if "_op" not in h]
if kind == "nodes":  # a performing, 0-shot map beats a noise / few-shot one
    hits.sort(key=lambda hd: (bool(hd[1].get("performing_regime", True)),
                              int(hd[1].get("shots", 0) or 0) == 0, hd[0]))
print(f"[teach4m] find_json {kind} {rid} {surf}: candidates {[h for h, _ in hits]}", file=sys.stderr)
print(hits[-1][0] if hits else "")
PY
}
step() {  # step <done-marker-file> <cmd...>: skip if marker exists; log; mark on success
  local marker=$1; shift
  if [[ -f $marker ]]; then milestone "skip ($marker exists): $*"; return 0; fi
  milestone "run: $*"
  { "$@" 2>&1 | tee -a "$LOG"; } && [[ ${PIPESTATUS[0]} == 0 ]] || fail "step failed: $* (see $LOG)"
  [[ -f $marker ]] || touch "$marker"
}
battery() {  # battery <rid> <suffix> <lora 0|1> <parent-run-id>
  local rid=$1 sfx=$2 lora=$3 parent=$4
  cd "$A" || fail "no analysis dir"
  echo "=== teach4m battery $(date -Is) run=$rid parent=$parent ===" >> "$LOG"
  local MAP4=circ_ts_$sfx
  # stale guard: a map under this stem that belongs to ANOTHER model (stage-5 collision of
  # 2026-09-10) is moved aside together with everything derived from it
  if [[ -f $MAP4.json ]]; then
    local owner
    owner=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('model',''))" "$MAP4.json")
    if [[ $owner != "$rid" ]]; then
      mkdir -p stale
      milestone "STALE: $MAP4 belongs to '$owner', not $rid — moving map-derived outputs to $A/stale/"
      mv -f ${MAP4}*.json ${MAP4}*.parquet stale/ 2>/dev/null
      mv -f done_cmp_*_"$sfx".log done_faith_*_"$sfx".log done_steer_*_"$sfx".log steer_${MAP4}*.json stale/ 2>/dev/null
    fi
  fi
  # (a) attribution circuit + split-half ceiling
  step $MAP4.json          python3 circuit_nodes.py --run-id "$rid" --out $MAP4 --n-pairs 256
  step ${MAP4}_a.json      python3 circuit_nodes.py --run-id "$rid" --out ${MAP4}_a --half a --n-pairs 256
  step ${MAP4}_b.json      python3 circuit_nodes.py --run-id "$rid" --out ${MAP4}_b --half b --n-pairs 256
  step done_cmp_halves_$sfx.log  python3 circuit_compare.py ${MAP4}_a ${MAP4}_b
  MAP_ELICITED=${MAP_ELICITED:-$(find_json nodes "$ELICITED")}
  MAP_TAUGHT1M=${MAP_TAUGHT1M:-$(find_json nodes "$RID_1M")}
  MAP_BASE16=${MAP_BASE16:-$(find_json nodes "$GEODE_STORE/runs/$BASE/model")}
  milestone "comparators: elicited=${MAP_ELICITED:-NONE} taught1m=${MAP_TAUGHT1M:-NONE} base16=${MAP_BASE16:-NONE}"
  [[ -n $MAP_ELICITED ]] && step done_cmp_elicited_$sfx.log python3 circuit_compare.py $MAP4 "$MAP_ELICITED"
  [[ -n $MAP_TAUGHT1M ]] && step done_cmp_taught1m_$sfx.log python3 circuit_compare.py $MAP4 "$MAP_TAUGHT1M"
  [[ -n $MAP_BASE16 ]]   && step done_cmp_base16_$sfx.log   python3 circuit_compare.py $MAP4 "$MAP_BASE16"
  # cross-endpoint compares: LoRA-4M teach, FT teach, FT elicit (whichever exist and are not this run)
  local other
  for other in circ_ts_4m circ_ts_ft4m circ_ts_elft4m; do
    [[ $other != "$MAP4" && -f $other.json ]] && step done_cmp_${other#circ_ts_}_$sfx.log python3 circuit_compare.py $MAP4 $other
  done
  # (b) true activation patching
  step done_faith_suff_$sfx.log python3 circuit_faithfulness.py --map $MAP4 --run-id "$rid" --ks 8 16 32 64 128 528 --n-pairs 128 --mode sufficiency
  step done_faith_nec_$sfx.log  python3 circuit_faithfulness.py --map $MAP4 --run-id "$rid" --ks 8 16 32 64 128 528 --n-pairs 128 --mode necessity
  # (c) DCM roles (heads only, the v2 protocol)
  step dcm_taught${sfx}_nl.json python3 dcm_roles.py learn --run-id "$rid" --surface bare_nl --out dcm_taught${sfx}_nl --roles operand_a operand_b --n-pairs 128 --lam 0.02 --components heads --steps 200
  DCM_ELICITED_NL=${DCM_ELICITED_NL:-$(find_json dcm "$ELICITED" bare_nl)}
  DCM_PARENT_OP=${DCM_PARENT_OP:-$(find_json dcm "$LATENT" bare_op)}
  milestone "DCM comparators: elicited_nl=${DCM_ELICITED_NL:-NONE} parent_op=${DCM_PARENT_OP:-NONE}"
  [[ -n $DCM_ELICITED_NL ]] && step done_dcm_cmp_elicited_$sfx.log python3 dcm_roles.py compare dcm_taught${sfx}_nl "$DCM_ELICITED_NL"
  [[ -n $DCM_PARENT_OP ]]   && step done_dcm_cmp_parent_$sfx.log   python3 dcm_roles.py compare dcm_taught${sfx}_nl "$DCM_PARENT_OP"
  # (d) steering: the taught donor into the blank twin (mean vector, per-prompt states)
  step done_steer_mean_$sfx.log python3 steer_unlock.py --base "$parent" --donor-run "$rid" --map $MAP4 --k 32 --prefill-only --n-calib 64 --n-eval 256 --seed 316
  step done_steer_pp_$sfx.log   python3 steer_unlock.py --base "$parent" --donor-run "$rid" --map $MAP4 --k 32 --prefill-only --vectors per-prompt --n-eval 256 --seed 316
  # (e) weights and gradients
  step ws_teach_$sfx.parquet   python3 weight_shift.py --base-run "$parent" --ft-run "$rid" --k 64 --out ws_teach_$sfx
  if ((lora)); then
    step done_grad_$sfx.log    python3 grad_strength.py --run-id "$ELICITED" "$rid" --labels elicit teach$sfx --out grad_strength_$sfx
  else
    milestone "skip gradient strength for $rid (train_sft.py logs no gradstats)"
  fi
  # (f) residual shift (v2) + compare with the existing JSONs when present
  step resid_teach_$sfx.json   python3 resid_shift.py run --parent "$parent" --child "$rid" --out resid_teach_$sfx --n 256
  local cmp=(); local f
  for f in resid_elicit_1m resid_elicit_3162 resid_teach_1m resid_construct resid_teach_4m resid_teach_ft4m resid_teach_elft4m; do
    [[ -f $f.json && $f != resid_teach_$sfx ]] && cmp+=("${f#resid_}=$f.json")
  done
  step done_resid_cmp_$sfx.log python3 resid_shift.py compare "${cmp[@]}" teach_$sfx=resid_teach_$sfx.json --plot resid_shift_$sfx.png
  # (g) lens depth (logit / J / R) + compare + breakdown
  step lens_taught${sfx}_nl.json python3 lens_depth.py run --run-id "$rid" --surface bare_nl --out lens_taught${sfx}_nl
  cmp=()
  for f in lens_latent_nl2 lens_elicited_nl2 lens_blank_nl2 lens_taught_nl2 lens_taught4m_nl lens_taughtft4m_nl lens_taughtelft4m_nl; do
    [[ -f $f.json && $f != lens_taught${sfx}_nl ]] && cmp+=("${f#lens_}=$f.json")
  done
  step done_lens_cmp_$sfx.log  python3 lens_depth.py compare "${cmp[@]}" taught$sfx=lens_taught${sfx}_nl.json --plot lens_nl_$sfx.png
  step done_lens_bd_$sfx.log   python3 lens_breakdown.py lens_taught${sfx}_nl.json
  # (h) formation curve + weight travel from the endpoint snapshots (LoRA runs only)
  if ((lora)); then
    step traj_ts_$sfx.parquet  python3 circuit_trajectory.py --run-id "$rid" --final-map $MAP4 --repo-id "$HF_NAMESPACE/$rid" --n-snapshots 8 --n-pairs 128 --out traj_ts_$sfx
    step wt_teach_$sfx.parquet python3 weight_traj.py --run-id "$rid" --repo-id "$HF_NAMESPACE/$rid" --n-snapshots 12 --out wt_teach_$sfx
  else
    milestone "skip formation curve / weight travel for $rid (full FT: no adapter snapshots)"
  fi
  milestone "battery complete for $rid — paste $LOG (or its SUMMARY / RESULT lines) back for the write-up update"
  cd "$OLDPWD" || true
}
if want 3; then battery "$RID" 4m 1 "$BASE"; fi

# ------------------------------------------------------- stage 4: full-FT teach
if want 4; then
  [[ -d $GEODE_STORE/runs/$BASE/model ]] || fail "parent checkpoint $GEODE_STORE/runs/$BASE/model missing"
  train_or_skip "$RID_FT" \
    python3 train_sft.py --config "$CFG_FT" --init-from "$GEODE_STORE/runs/$BASE/model" --confirm-cost
  record_g5 "$RID_FT"
  python3 - "$RID_FT" <<'PY'
import json, os, sys
from pathlib import Path
m = json.loads((Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json").read_text())
e = m["experiment"]; r = e.get("sft_result", {}); g = e.get("gates", {}).get("G5", {})
print(f"[teach4m] RESULT {sys.argv[1]}: steps {r.get('final_step')} stop={r.get('stop_reason')} "
      f"best_val {r.get('best_val_nats')}  G5 EM 0-shot {g.get('zero_shot_accuracy')} "
      f"16-shot {g.get('sixteen_shot_accuracy')} test_loss {g.get('test_loss_nats')}")
PY
  milestone "train-ft done (model.safetensors kept on purpose: no adapter sidecar exists for a full-FT run)"
fi

# --------------------------------------------------- stage 5: battery on the FT run
if want 5; then
  [[ $(status_of "$RID_FT") == complete ]] || fail "$RID_FT is not complete — run stage 4 first"
  battery "$RID_FT" ft4m 0 "$BASE"
fi
# ------------------------------------ stage 6: full-FT ELICIT (the method control)
# The FT teach endpoint changes LoRA -> full FT on the teach side only. Stage 6
# fully fine-tunes the LATENT parent on the same 4M file (min_steps 0: it
# converges in a fraction of a pass) so both regimes exist under both methods.
if want 6; then
  [[ -d $GEODE_STORE/runs/$LATENT/model ]] || fail "latent parent checkpoint $GEODE_STORE/runs/$LATENT/model missing"
  train_or_skip "$RID_ELFT" \
    python3 train_sft.py --config "$CFG_ELFT" --init-from "$GEODE_STORE/runs/$LATENT/model" --confirm-cost
  record_g5 "$RID_ELFT"
  python3 - "$RID_ELFT" <<'PY'
import json, os, sys
from pathlib import Path
m = json.loads((Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json").read_text())
e = m["experiment"]; r = e.get("sft_result", {}); g = e.get("gates", {}).get("G5", {})
print(f"[teach4m] RESULT {sys.argv[1]}: steps {r.get('final_step')} stop={r.get('stop_reason')} "
      f"best_val {r.get('best_val_nats')}  G5 EM 0-shot {g.get('zero_shot_accuracy')} "
      f"16-shot {g.get('sixteen_shot_accuracy')} test_loss {g.get('test_loss_nats')}")
PY
fi
if want 7; then
  [[ $(status_of "$RID_ELFT") == complete ]] || fail "$RID_ELFT is not complete — run stage 6 first"
  battery "$RID_ELFT" elft4m 0 "$LATENT"
fi

# ---------------------- stage 8: residual-shift v2 for the four original cells
if want 8; then
  cd "$A" || fail "no analysis dir"
  mkdir -p stale
  for f in resid_elicit_1m resid_elicit_3162 resid_teach_1m resid_construct; do
    [[ -f $f.json ]] && ! grep -q kl_task_ans $f.json && mv -f $f.json stale/$f.v1.json
  done
  step resid_elicit_1m.json   python3 resid_shift.py run --parent "$LATENT" --child "$ELICITED" --out resid_elicit_1m --n 256
  step resid_elicit_3162.json python3 resid_shift.py run --parent "$LATENT" --child evt-ts1b-mix-nl-n3162 --out resid_elicit_3162 --n 256
  step resid_teach_1m.json    python3 resid_shift.py run --parent "$BASE" --child "$RID_1M" --out resid_teach_1m --n 256
  step resid_construct.json   python3 resid_shift.py run --parent "$BASE" --child "$LATENT" --out resid_construct --n 256
  cmp=(); for f in resid_elicit_1m resid_elicit_3162 resid_teach_1m resid_teach_4m resid_teach_ft4m resid_teach_elft4m resid_construct; do [[ -f $f.json ]] && cmp+=("${f#resid_}=$f.json"); done
  rm -f done_resid_cmp_all.log
  step done_resid_cmp_all.log python3 resid_shift.py compare "${cmp[@]}" --plot resid_shift_all.png
  cd "$OLDPWD" || true
fi
milestone "done stage=$STAGE"
