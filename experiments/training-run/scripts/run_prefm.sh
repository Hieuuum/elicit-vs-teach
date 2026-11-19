#!/usr/bin/env bash
# The hidden-preference and J-lens reads with digit-matched distractors (2026-11-19; pre-registration in
# decisions.md, D1-D3). The original reads compared the answer's first token with the next problem's, whatever its
# length; a collaborator's re-read with matched digit counts found the elicit parent's preference at zero.
#   prefm   final-layer preference, matched and unmatched on the same problems, five parents (forward only)
#   lens    J-lens / logit-lens by layer, matched, on the two parents, their children and the blank base
# ~0.5 GPU-h (the lens computes average Jacobians per model). Resumable.
#   bash experiments/training-run/scripts/run_prefm.sh                 # plan
#   bash experiments/training-run/scripts/run_prefm.sh --confirm-cost
set -uo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
A=$REPO/experiments/training-run/analysis
[[ ${1:-} == --confirm-cost ]] || { echo "[prefm] GPU job (~0.5 GPU-h). Re-run with --confirm-cost."; exit 2; }
: "${GEODE_STORE:?export GEODE_STORE first}"
cd "$A"
run() { echo "[prefm] $*"; "$@" || echo "[prefm] FAILED: $*"; }
declare -A PARENT=( [blank]=evt-ts1b-base [fmt]=evt-ts1b-fig2ts-installer [engine]=evt-ts1b-op-install
                    [latent]=evt-ts1b-op-bridge-mix [llama]=meta-llama/Llama-3.2-1B )
for tag in latent fmt blank engine llama; do
  run python3 prefit_metrics.py prefm --model "${PARENT[$tag]}" --tag "$tag"
done
declare -A LENS=( [latent]=evt-ts1b-op-bridge-mix [fmt]=evt-ts1b-fig2ts-installer [blank]=evt-ts1b-base
                  [elicited]=evt-ts1b-elicit-ft-n4000000 [taught]=evt-ts1b-teach-ft-fmt-n4000000 )
for tag in latent fmt blank elicited taught; do
  [[ -f lens_${tag}_nl_matched.json ]] && { echo "[prefm] lens_${tag}_nl_matched.json exists"; continue; }
  run python3 lens_depth.py run --run-id "${LENS[$tag]}" --surface bare_nl --distractor matched --out lens_${tag}_nl_matched
done
python3 - <<'PY'
import json, pathlib
print("[prefm] final-layer preference (nats, +- SE): tag  matched  |  unmatched (same problems)")
for t in ("latent", "llama", "engine", "fmt", "blank"):
    f = pathlib.Path(f"prefit_{t}.json")
    b = json.loads(f.read_text()).get("prefm") if f.is_file() else None
    if b:
        m, u = b["matched"], b["next"]
        print(f"[prefm]   {t:<7} {m['logit_diff_mean']:+.3f} +- {m['logit_diff_se']:.3f} (n {m['n']})  |  "
              f"{u['logit_diff_mean']:+.3f} +- {u['logit_diff_se']:.3f}")
PY
python3 lens_depth.py compare $(for t in latent fmt blank elicited taught; do [[ -f lens_${t}_nl_matched.json ]] && echo "$t=lens_${t}_nl_matched.json"; done) \
  --plot lens_nl_matched.png
