#!/bin/bash

if [ "$CLAUDE_CODE_REMOTE" != "true" ]; then
  exit 0
fi

if python3 -c "import geode" >/dev/null 2>&1; then
  exit 0
fi

# CPU-only wheel: cloud sessions have no GPU, and the repo's test suite is
# CPU-only by policy (CLAUDE.md) — the default PyPI torch pulls ~2.5GB of
# unused CUDA wheels.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
pip install --quiet -e "${CLAUDE_PROJECT_DIR}[dev]"

exit 0
