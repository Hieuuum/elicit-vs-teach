#!/bin/bash

if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

if python3 -c "import geode" >/dev/null 2>&1; then
  exit 0
fi

# Cloud sandbox egress is proxied and can be slow; pip's default 15s read
# timeout is what killed session startup (files.pythonhosted.org read
# timeout), and pip does not retry a read timeout mid-download itself.
export PIP_DEFAULT_TIMEOUT=120

retry() {
  local attempt
  for attempt in 1 2 3; do
    "$@" && return 0
    echo "install_pkgs.sh: attempt ${attempt}/3 failed: $*" >&2
    sleep 5
  done
  return 1
}

# CLAUDE_PROJECT_DIR only exists in hook context; fall back to the repo root
# relative to this script so it also works run manually.
cd "${CLAUDE_PROJECT_DIR:-"$(dirname "$0")/.."}" || exit 0

# CPU-only wheel: cloud sessions have no GPU, and the repo's test suite is
# CPU-only by policy (CLAUDE.md) — the default PyPI torch pulls ~2.5GB of
# unused CUDA wheels.
if retry pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu &&
   retry pip install --quiet -e ".[dev]"; then
  echo "install_pkgs.sh: geode deps installed"
else
  # stdout of a SessionStart hook lands in the agent's context.
  echo "install_pkgs.sh: dependency install FAILED after retries — run 'bash scripts/install_pkgs.sh' before using geode"
fi

# Never block session startup on a transient network failure.
exit 0
