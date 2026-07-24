#!/usr/bin/env bash
# vast.ai "On-start Script" — paste into the instance template (or run once
# by hand on a fresh box). Provisions a geode box; NEVER launches training
# (budget rule: every GPU spend needs a human --confirm-cost).
#
# Idempotent: safe on every container (re)start. Clones once and NEVER
# auto-pulls — code sync stays the manual `git pull` + laptop-hash check in
# the guides, so a mid-experiment restart can't silently advance the code.
#
# Template env vars (vast.ai -> Environment Variables):
#   HF_TOKEN     READ-only HF token; huggingface_hub reads the env var
#                directly, no login step needed. NEVER a WRITE token (relay
#                pushes are owner-gated, see run5-6-guide.md §6). For the
#                Llama box it must belong to the Meta-licensed account.
#   NTFY_TOPIC   optional ntfy.sh topic for the "box ready" ping.
#
# vast.ai FAQ notes (2026-07-24): template env vars are NOT visible inside
# SSH/Jupyter sessions unless written to /etc/environment (done below), and
# SSH-instance restarts run /root/onstart.sh — this script installs itself
# there once, so both instance types re-provision on every container start.
set -uo pipefail
exec >>/workspace/onstart.log 2>&1
echo "=== onstart $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

BRANCH=cut-to-core
cd /workspace
[[ -d elicit-vs-teach ]] || git clone -b "$BRANCH" https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach
pip install -q -e ".[dev]"

# Every login shell / tmux window gets the exports — a stale $GEODE_STORE in
# a fresh window is the #1 guide troubleshooting item.
if ! grep -q GEODE_STORE ~/.bashrc 2>/dev/null; then
  echo "export GEODE_STORE=/workspace/elicit-vs-teach/geode-store  # store INSIDE the clone" >>~/.bashrc
  [[ -n ${NTFY_TOPIC:-} ]] && echo "export NTFY=ntfy.sh/$NTFY_TOPIC" >>~/.bashrc
fi

# Template env vars don't reach SSH sessions on vast.ai; /etc/environment is
# the documented fix (their FAQ). READ token only — the write rule stands.
for var in HF_TOKEN NTFY_TOPIC; do
  val=${!var:-}
  if [[ -n $val ]] && ! grep -q "^$var=" /etc/environment 2>/dev/null; then
    echo "$var=$val" >>/etc/environment
  fi
done

# SSH-instance restarts run /root/onstart.sh (vast.ai FAQ); the [[ -f ]]
# guard means we never overwrite whatever the template mechanism put there.
if [[ ! -f /root/onstart.sh ]]; then
  cp experiments/training-run/scripts/box_onstart.sh /root/onstart.sh
  chmod +x /root/onstart.sh
fi

# Full CPU suite once per box (~2 min); the marker skips it on restarts.
if [[ ! -f /workspace/.geode_suite_ok ]]; then
  python -m pytest -q && touch /workspace/.geode_suite_ok
fi
suite=$([[ -f /workspace/.geode_suite_ok ]] && echo ok || echo FAILED)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h /workspace | tail -1
hash=$(git rev-parse --short HEAD)
echo "ready: $hash suite=$suite (box hash must match laptop before any launch)"
[[ -n ${NTFY_TOPIC:-} ]] && curl -sd "box ready: $hash suite=$suite" "ntfy.sh/$NTFY_TOPIC" >/dev/null
echo "=== onstart done ==="
exit 0
