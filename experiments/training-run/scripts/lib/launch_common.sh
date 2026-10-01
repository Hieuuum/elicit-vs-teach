# launch_common.sh — sourceable shell glue shared by training-run launchers.
#
# The retired launchers each re-declared the same run-status / gate / notify
# helpers inline. This is the one live copy for the NEXT launcher to source
# instead of copy-pasting. Retired launchers keep their frozen inline copies.
#
# Usage (from a launcher, after `cd scripts/`):
#     source lib/launch_common.sh
# then call the functions below. Callers may set two env vars:
#     TAG    log/notify prefix for this launcher (default: "run")
#     NTFY   full ntfy URL; when unset, notify() is a no-op
#     GEODE_STORE   artifact store root (read by status_of/gate_recorded/...)
#
# NTFY_AUTO   opt-in gate for automatic pings (incl. fail()'s): default off,
#             so notify() is silent unless NTFY_AUTO=1. Per owner 2026-07-31,
#             notification is the operating agent's explicit act, not a
#             script default — a deliberate `curl -d ... "$NTFY"` by the
#             agent is unaffected.
#
# This file is SOURCED, so it declares functions only — no top-level execution
# and deliberately NO `set -euo pipefail` (those flags would leak into and
# change the behavior of the sourcing launcher's shell).

# Fire-and-forget ntfy push; a no-op (and never an error) when $NTFY is unset
# or NTFY_AUTO isn't "1".
notify() { [[ -n ${NTFY:-} && ${NTFY_AUTO:-0} == 1 ]] && curl -sd "$1" "${NTFY}" >/dev/null || true; }

# Terminal failure: one notify, one stderr line, then exit 1.
fail() {
  notify "${TAG:-run} FAILED: $1"
  echo "[${TAG:-run}] FAILED: $1" >&2
  exit 1
}

# Progress marker on stdout, matching the launchers' "MILESTONE <event>" logs.
milestone() { echo "[${TAG:-run}] MILESTONE $*"; }

# run_id -> "complete" | "missing" | "<other manifest status>"
status_of() {
  python3 - "$1" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
print(json.loads(p.read_text())["status"] if p.is_file() else "missing")
PY
}

# run_id gate -> exit 0 if a verdict for <gate> is already recorded, else 1.
gate_recorded() {
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

# run_id result_key -> the run's stop_reason string for that result (or empty).
stop_reason_of() {
  python3 - "$1" "$2" <<'PY'
import json, os, sys
from pathlib import Path

p = Path(os.environ["GEODE_STORE"]) / "runs" / sys.argv[1] / "manifest.json"
r = json.loads(p.read_text()).get("experiment", {}).get(sys.argv[2], {}) or {}
print(r.get("stop_reason", ""))
PY
}

# Best-effort push, push-as-you-go: a failed upload warns rather than
# aborting the family (the caller's own downstream checks are what can
# actually fail a run). Callers may omit $2 to use $RELAY_REPO.
push_run() {
  local rid=$1 repo=${2:-$RELAY_REPO}
  if python3 hf_checkpoint.py push --run-id "$rid" --repo-id "$repo"; then
    milestone "push_complete run=$rid repo=$repo"
  else
    echo "[${TAG:-run}] WARN hf_checkpoint.py push failed for $rid -> $repo (best effort)"
    milestone "push_warn run=$rid repo=$repo"
  fi
}
