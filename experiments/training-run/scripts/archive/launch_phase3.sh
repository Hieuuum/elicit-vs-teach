#!/usr/bin/env bash
# PHASE 3 (owner 2026-07-27) — the elicit arm only, run in dependency order,
# unattended, skipping anything already complete. A crash, a box restart, or a
# stage split across machines all resume from where they stopped.
#
#   ./launch_phase3.sh --confirm-cost [--stage parent|target|bridge|all]
#
# Stages (default all, in this order):
#   parent   Operator-notation ADDITION pre-intervention run + G1. Needs a GPU.
#   target   The conditional format-install decision, the installer if and only
#            if it is needed, then the unchanged no-bridge NL-addition target
#            through the prequential EDL harness + G5. Needs a GPU.
#   bridge   The answer-free bidirectional translation bridge from the same
#            operator-addition parent, its G2/G4/G6 gates, then a second target
#            on the identical frozen data/order/rule as the no-bridge control.
#            Requires the no-bridge target to exist as G7's data-order anchor.
#
# THE CONDITIONAL. Before the target, the parent's format validity is scored on
# NL ADDITION prompts drawn from the frozen external eval file:
#
#   gates.py g4 --run evt-p3-elicit-parent \
#       --config ../configs/p3_elicit_parent.yaml \
#       --prompt-config ../configs/eval_p3_data.yaml \
#       --threshold 0.90 --n-prompts 512 --no-record
#
# >= 0.90 -> no installer, the target trains straight from the parent.
# <  0.90 -> the NL-multiplication installer runs first, and the target picks
#            up configs/p3/target_after_inst.yaml.
# --no-record is required: a recorded sub-threshold G4 on the shared parent
# would make require_parent_ready refuse every child of it (V0.6).
#
# The decision is parsed from the gate's PRINTED RATE, not its exit code. In
# --no-record mode gates.py returns 1 both for "below threshold" and for any
# SystemExit, so an exit-code branch would silently read a crashed gate as
# "install needed" and spend the budget on it. Absence of the rate line is a
# hard failure here.
#
# Refuses to start unless ALL of:
#   - --confirm-cost (budget rule);
#   - every phase-3 artifact a pending stage needs is present and hash-matches;
#   - the target LR equals the committed pin AND differs from the installer LR
#     (the 2026-07-25 scope leak that destroyed run 9);
#   - the parent checkpoint exists for every pending run.
# If the installer runs, its RETENTION bar is enforced before the target: this
# is the only installer in the project that acts on a parent which already has
# the capability under study, so G2 below EXACT_MATCH_THRESHOLD halts the phase
# rather than shipping a damaged parent.
# The bridge's G6 is score-first/record-second: aggregate and both directions
# must clear 0.95 on the entire frozen held-out translation file. G4/G5 refuse
# bridge configs because those gates parse integer answer slots.
# stop_reason=max_steps on ANY run in this phase is a bug signal.
#
# ntfy fires on the deliverable and on failure only (owner preference), not at
# stage boundaries. Set $NTFY to the full topic URL.
set -uo pipefail
cd "$(dirname "$0")"

[[ " $* " == *" --confirm-cost "* ]] || {
  echo "launch_phase3.sh: --confirm-cost required (budget rule)" >&2
  exit 1
}
STAGE=all
PREV=""
for i in "$@"; do
  [[ $PREV == --stage ]] && STAGE=$i
  PREV=$i
done
case $STAGE in all | parent | target | bridge) ;;
*)
  echo "launch_phase3.sh: --stage must be all|parent|target|bridge, got '$STAGE'" >&2
  exit 1
  ;;
esac

REPO_ROOT=$(git rev-parse --show-toplevel)
export REPO_ROOT                 # the guard resolves data.local_path against
                                 # THIS, never the store's parent
export GEODE_STORE=${GEODE_STORE:-$REPO_ROOT/geode-store}
PARENT_RID=evt-p3-elicit-parent
INST_RID=evt-p3-elicit-inst
TARGET_RID=evt-p3-elicit-target
BRIDGE_RID=evt-p3-elicit-bridge
BRIDGE_TARGET_RID=evt-p3-elicit-target-bridge
echo "[p3] repo  $(git log --oneline -1)"
echo "[p3] store $GEODE_STORE  stage=$STAGE"

notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "$NTFY" >/dev/null || true; }
fail() {
  notify "phase3 FAILED: $1"
  echo "[p3] FAILED: $1" >&2
  exit 1
}

status_of() { # run_id -> complete | missing | <status>
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

gate_done() { # run_id gate -> 0 if a verdict is already recorded
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
if not p.is_file():
    sys.exit(1)
gates = json.loads(p.read_text()).get("experiment", {}).get("gates", {})
sys.exit(0 if sys.argv[2] in gates else 1)
PY
}

parent_ckpt() { # run_id -> checkpoint dir, or empty
  local d=$GEODE_STORE/runs/$1/model
  [[ -f $d/model.safetensors ]] && echo "$d"
}

# ---- guards (all before any spend) ---------------------------------------

python3 phase3_guards.py --configs ../configs --repo-root "$REPO_ROOT" || exit 1

# ---- stage: pre-intervention (operator-notation addition) ----------------

if [[ $STAGE == all || $STAGE == parent ]]; then
  ST=$(status_of "$PARENT_RID")
  if [[ $ST == complete ]]; then
    echo "[p3] $PARENT_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$PARENT_RID has status '$ST' — resolve it first"
    CKPT=$(parent_ckpt evt-run1-base-v3-ext) || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for evt-run1-base-v3-ext (the phase-3 base)"
    echo "[p3] === pre-intervention: operator-notation addition ==="
    python3 train_sft.py --config ../configs/p3_elicit_parent.yaml \
      --init-from "$CKPT" --confirm-cost || fail "$PARENT_RID training"
  fi
  # G1 IS SCORED BEFORE IT IS RECORDED. The default bar is 0.95 exact match,
  # calibrated on run 2's 4-digit add/sub (which scored 0.9961). This parent
  # learns 1-8 digit addition, ~75% of it with an operand of 5+ digits — a
  # harder task, so a miss here is as likely to mean "the bar was set for a
  # different task" as "the run is bad". A recorded FAIL on this checkpoint
  # would make require_parent_ready (V0.6) refuse every child of it and could
  # only be undone by hand, so the scoring pass runs --no-record and the
  # verdict is committed only once it passes.
  if ! gate_done "$PARENT_RID" G1; then
    echo "[p3] === G1 (scoring pass, nothing recorded) ==="
    G1_OUT=$(python3 gates.py g1 --run "$PARENT_RID" \
      --config ../configs/p3_elicit_parent.yaml --no-record 2>&1) || true
    echo "$G1_OUT"
    G1_ACC=$(sed -n 's/.*G1 accuracy \([0-9.]*\) on n=.*/\1/p' <<<"$G1_OUT" | head -1)
    [[ -n $G1_ACC ]] ||
      fail "the G1 scoring pass printed no accuracy — it errored rather than scored low"
    grep -q "G1 accuracy .* -> PASS" <<<"$G1_OUT" || fail \
      "$PARENT_RID G1 scored $G1_ACC, below the bar — NOT recorded, so the checkpoint
   is still usable. Decide before re-running: (a) the LR is role-inherited from
   run 2 and never swept on this data (p3_elicit_parent.yaml header) — sweep it;
   or (b) 0.95 is run 2's 4-digit bar and this is 8-digit addition — set the
   phase's own threshold explicitly and record that choice in decisions.md.
   Do not lower the bar after seeing the number without writing down that you did."
    python3 gates.py g1 --run "$PARENT_RID" \
      --config ../configs/p3_elicit_parent.yaml || fail "$PARENT_RID G1 (recording pass)"
  fi
fi

# ---- stage: the conditional install decision, then the target -----------

if [[ $STAGE == all || $STAGE == target ]]; then
  [[ $(status_of "$PARENT_RID") == complete ]] ||
    fail "$PARENT_RID is not complete — run --stage parent first"

  # The decision. Parsed from the printed rate; see the header for why the exit
  # code is not trusted. --no-record keeps the verdict off the shared parent.
  echo "[p3] === format-install decision: G4 on NL addition prompts ==="
  # --config is REQUIRED even when --prompt-config supplies the prompts: gates.py
  # reads the tokenizer path and cfg["train"]["stopping"] from it before it ever
  # looks at the prompt source. Omitting it is an argparse error, which the rate
  # parse below correctly refuses to read as a score (2026-07-27).
  G4_OUT=$(python3 gates.py g4 --run "$PARENT_RID" \
    --config ../configs/p3_elicit_parent.yaml \
    --prompt-config ../configs/eval_p3_data.yaml \
    --threshold 0.90 --n-prompts 512 --no-record 2>&1)
  echo "$G4_OUT"
  G4_RATE=$(sed -n 's/.*G4 format_validity \([0-9.]*\) on n=.*/\1/p' <<<"$G4_OUT" | head -1)
  [[ -n $G4_RATE ]] ||
    fail "the pre-target G4 printed no format_validity rate — it errored rather than
  scored, and its exit code cannot tell those apart. Output above."

  NEED_INSTALL=$(python3 -c "import sys; sys.exit(0 if float('$G4_RATE') < 0.90 else 1)" &&
    echo yes || echo no)
  echo "[p3] parent NL format validity $G4_RATE -> install needed: $NEED_INSTALL"

  TARGET_PARENT=$PARENT_RID
  OVERRIDE=()
  if [[ $NEED_INSTALL == yes ]]; then
    ST=$(status_of "$INST_RID")
    if [[ $ST == complete ]]; then
      echo "[p3] $INST_RID already complete — skipping"
    else
      [[ $ST == missing ]] || fail "$INST_RID has status '$ST' — resolve it first"
      CKPT=$(parent_ckpt "$PARENT_RID") || true
      [[ -n ${CKPT:-} ]] || fail "no checkpoint for $PARENT_RID"
      echo "[p3] === conditional format installer: NL multiplication ==="
      python3 train_sft.py --config ../configs/p3_elicit_inst.yaml \
        --init-from "$CKPT" --confirm-cost || fail "$INST_RID training"
    fi
    gate_done "$INST_RID" G4 || python3 gates.py g4 --run "$INST_RID" \
      --config ../configs/p3_elicit_inst.yaml \
      --prompt-config ../configs/eval_p3_data.yaml --threshold 0.90 ||
      fail "$INST_RID G4 (the format never installed)"
    gate_done "$INST_RID" G2 || python3 gates.py g2 --run "$INST_RID" \
      --config ../configs/p3_elicit_parent.yaml || fail "$INST_RID G2"

    # RETENTION BAR. Unique to this phase: every other installer in the project
    # ran on a parent that knew nothing, so a retention loss was impossible.
    # This one acts on a parent that already holds the capability the phase
    # exists to elicit. A damaged parent would depress the target's EDL and
    # read as a *better* elicitation result, so the bar halts rather than warns.
    python3 - <<'PY' || fail "installer retention bar"
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p3-elicit-inst" / "manifest.json"
g2 = json.loads(p.read_text())["experiment"]["gates"].get("G2")
if g2 is None:
    sys.exit("phase3: evt-p3-elicit-inst has no recorded G2 — score it before the target")
if not g2["pass"]:
    sys.exit(
        f"phase3: installer G2 retention {g2['exact_match']:.4f} < {g2['threshold']} — the "
        "NL-mult installer damaged the parent's addition. That is the capability the phase "
        "measures, and losing it would DEFLATE the target's EDL and read as a better "
        "elicitation result. STOP (decisions.md 2026-07-25, run-9 retention)."
    )
print(f"[p3] retention bar: installer G2 {g2['exact_match']:.4f} >= {g2['threshold']}")
PY
    TARGET_PARENT=$INST_RID
    OVERRIDE=(--override ../configs/p3/target_after_inst.yaml)
  fi

  # Record the branch: which parent the EDL measurement actually ran from is a
  # fact about the result, and it is decided at run time. The pre-exposure
  # figures ride along so the run directory carries its own caveat — anyone
  # reading a low elicit EDL asks "how much of that is recall?" first, and the
  # answer should not live three files away in decisions.md.
  mkdir -p "$GEODE_STORE/runs/$TARGET_RID"
  python3 - "$G4_RATE" "$NEED_INSTALL" "$TARGET_PARENT" <<'PY'
import json, os, sys
from pathlib import Path

import yaml

repo_root = Path(os.environ["REPO_ROOT"])
data_dir = (
    repo_root
    / yaml.safe_load((Path("../configs") / "p3_elicit_target.yaml").read_text())["data"][
        "local_path"
    ]
).parent
exposure = json.loads((data_dir / "report.json").read_text())["pre_exposure"]

path = Path(os.environ["GEODE_STORE"]) / "runs" / "evt-p3-elicit-target" / "install_decision.json"
path.write_text(
    json.dumps(
        {
            "parent_nl_format_validity": float(sys.argv[1]),
            "threshold": 0.90,
            "installer_run": sys.argv[2] == "yes",
            "target_parent_run_id": sys.argv[3],
            "protocol": (
                "gates.py g4 --prompt-config eval_p3_data.yaml --n-prompts 512 --no-record; "
                "fixed rows [2048:2560] of the frozen D_p3_nl_eval, token-prefix prompts, "
                "greedy EOS-stopped, format_valid"
            ),
            "pre_exposure": {
                "frac_of_target_direct": exposure["frac_of_target_direct"],
                "frac_of_target_incl_twin": exposure["frac_of_target_incl_twin"],
                "note": (
                    "The pre-intervention set and this target are NOT disjoint (V5.1, exact "
                    "ordered triple only; commuted twins allowed). For ADDITION the twin "
                    "carries the identical answer, so frac_of_target_incl_twin is the figure "
                    "that bounds item recall. At 1-8 digit operands it is ~1 in 17, and "
                    "~85% of that is the six small digit cells the parent has necessarily "
                    "seen whole. Quote both or neither; per-cell in report.json."
                ),
            },
        },
        indent=2,
    )
)
print(
    f"[p3] branch recorded -> {path}  "
    f"(pre-exposure {exposure['frac_of_target_direct']:.2%} direct, "
    f"{exposure['frac_of_target_incl_twin']:.2%} incl twin)"
)
PY

  ST=$(status_of "$TARGET_RID")
  if [[ $ST == complete ]]; then
    echo "[p3] $TARGET_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$TARGET_RID has status '$ST' — resolve it first"
    CKPT=$(parent_ckpt "$TARGET_PARENT") || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for $TARGET_PARENT"
    echo "[p3] === target: NL addition, prequential EDL, from $TARGET_PARENT ==="
    python3 train_target.py --config ../configs/p3_elicit_target.yaml \
      "${OVERRIDE[@]}" --init-from "$CKPT" --confirm-cost || fail "$TARGET_RID training"
  fi
  gate_done "$TARGET_RID" G5 || python3 gates.py g5 --run "$TARGET_RID" \
    --config ../configs/eval_p3_data.yaml || fail "$TARGET_RID G5"

  if [[ $STAGE == target ]]; then
    notify "phase3 no-bridge control done (installer run: $NEED_INSTALL) — EDL ready to read"
  fi
  echo "[p3] read the control curve under BOTH floors:"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID --floor test"
fi

# ---- stage: answer-free bridge, required gates, second target --------------

if [[ $STAGE == all || $STAGE == bridge ]]; then
  [[ $(status_of "$PARENT_RID") == complete ]] ||
    fail "$PARENT_RID is not complete — run --stage parent first"
  [[ $(status_of "$TARGET_RID") == complete ]] ||
    fail "$TARGET_RID is not complete — run --stage target first (G7 anchor)"

  ST=$(status_of "$BRIDGE_RID")
  if [[ $ST == complete ]]; then
    echo "[p3] $BRIDGE_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$BRIDGE_RID has status '$ST' — resolve it first"
    CKPT=$(parent_ckpt "$PARENT_RID") || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for $PARENT_RID"
    echo "[p3] === answer-free bidirectional translation bridge ==="
    python3 train_sft.py --config ../configs/p3_bridge.yaml \
      --init-from "$CKPT" --confirm-cost || fail "$BRIDGE_RID training"
  fi

  # Score before recording. A recorded false verdict poisons this checkpoint for
  # every child under V0.6; absence of the printed line is a crash, not a score.
  if ! gate_done "$BRIDGE_RID" G6; then
    echo "[p3] === G6 translation exact match (scoring pass, nothing recorded) ==="
    G6_OUT=$(python3 gates.py g6 --run "$BRIDGE_RID" \
      --config ../configs/eval_p3_bridge_data.yaml --no-record 2>&1) || true
    echo "$G6_OUT"
    G6_RATE=$(sed -n 's/.*G6 translation exact_match \([0-9.]*\) on n=.*/\1/p' \
      <<<"$G6_OUT" | head -1)
    [[ -n $G6_RATE ]] ||
      fail "the G6 scoring pass printed no exact-match rate — it errored rather than scored low"
    grep -q "G6 translation exact_match .* -> PASS" <<<"$G6_OUT" ||
      fail "$BRIDGE_RID G6 scored $G6_RATE below the aggregate/per-direction bar; NOT recorded"
    python3 gates.py g6 --run "$BRIDGE_RID" \
      --config ../configs/eval_p3_bridge_data.yaml || fail "$BRIDGE_RID G6 recording pass"
  fi
  gate_done "$BRIDGE_RID" G2 || python3 gates.py g2 --run "$BRIDGE_RID" \
    --config ../configs/p3_elicit_parent.yaml || fail "$BRIDGE_RID G2 retention"
  gate_done "$BRIDGE_RID" G4 || python3 gates.py g4 --run "$BRIDGE_RID" \
    --config ../configs/p3_elicit_parent.yaml \
    --prompt-config ../configs/eval_p3_data.yaml --threshold 0.90 ||
    fail "$BRIDGE_RID G4 format validity"

  # SUPERSEDED (2026-07-27): the bridge FAILED G2 (0.3018), so this clean-path
  # bridged target — which points parent_run_id at the bridge and would be
  # refused by require_parent_ready — never ran. It now runs via the dedicated
  # launch_phase3_bridge_target.sh (parent_run_id: null + external_base bypass).
  # Do NOT re-run this stage to get it: the bridge's G2 is already recorded, so
  # the halt above is skipped and this stage would RECORD G4 on the shared
  # bridge parent, mutating its gate record. Left here as the record of the
  # original clean design.
  ST=$(status_of "$BRIDGE_TARGET_RID")
  if [[ $ST == complete ]]; then
    echo "[p3] $BRIDGE_TARGET_RID already complete — skipping"
  else
    [[ $ST == missing ]] || fail "$BRIDGE_TARGET_RID has status '$ST' — resolve it first"
    CKPT=$(parent_ckpt "$BRIDGE_RID") || true
    [[ -n ${CKPT:-} ]] || fail "no checkpoint for $BRIDGE_RID"
    echo "[p3] === bridged target: identical NL-addition stream and epsilon/k rule ==="
    python3 train_target.py --config ../configs/p3_elicit_target.yaml \
      --override ../configs/p3/target_after_bridge.yaml \
      --init-from "$CKPT" --confirm-cost || fail "$BRIDGE_TARGET_RID training"
  fi
  gate_done "$BRIDGE_TARGET_RID" G5 || python3 gates.py g5 --run "$BRIDGE_TARGET_RID" \
    --config ../configs/eval_p3_data.yaml || fail "$BRIDGE_TARGET_RID G5"

  notify "phase3 bridge comparison done — control and bridged EDL curves ready"
  echo "[p3] compare both targets under BOTH floors:"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $BRIDGE_TARGET_RID"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $TARGET_RID --floor test"
  echo "[p3]   python3 ../analysis/plot_edl_per_token.py --run-id $BRIDGE_TARGET_RID --floor test"
fi

echo "[p3] stage '$STAGE' complete"
