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
# This file is SOURCED, so it declares functions only — no top-level execution
# and deliberately NO `set -euo pipefail` (those flags would leak into and
# change the behavior of the sourcing launcher's shell).

# Fire-and-forget ntfy push; a no-op (and never an error) when $NTFY is unset.
notify() { [[ -n ${NTFY:-} ]] && curl -sd "$1" "${NTFY}" >/dev/null || true; }

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
