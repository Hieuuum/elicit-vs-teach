#!/usr/bin/env bash
# vast.ai "On-start Script" — paste into the instance template (or run once
# by hand on a fresh box). Provisions a geode box; NEVER launches training
# (budget rule: every GPU spend needs a human --confirm-cost).
# Provisions a python3.11 uv venv at /workspace/venv (vast images may ship
# an older python) and runs all installs and the suite inside it.
#
# Idempotent: safe on every container (re)start. Clones once and NEVER
# auto-pulls — code sync stays the manual `git pull` + laptop-hash check in
# the guides, so a mid-experiment restart can't silently advance the code.
#
# Template env vars (vast.ai -> Environment Variables):
#   HF_TOKEN     READ-only HF token; huggingface_hub reads the env var
#                directly, no login step needed. NEVER a WRITE token (relay
#                pushes are owner-gated, see llama-guide.md §4). For the
#                Llama box it must belong to the Meta-licensed account.
#   NTFY         optional ping URL (ntfy.sh/<topic>) — the SAME variable
#                every launcher and guide uses (owner 2026-07-24: one ntfy
#                variable everywhere; NTFY_TOPIC retired).
#   NTFY_AUTO    opt-in gate for the automatic "box ready" ping below: unset
#                (default) means silent — per owner 2026-07-31, automatic
#                notification defaults off; the operating agent pings
#                manually when it decides one is warranted. Set to "1" to
#                restore the old automatic-ping behavior.
#
# vast.ai FAQ notes (2026-07-24): template env vars are NOT visible inside
# SSH/Jupyter sessions unless written to /etc/environment (done below), and
# SSH-instance restarts run /root/onstart.sh — this script installs itself
# there once, so both instance types re-provision on every container start.
set -uo pipefail

# Old-template compat: a box created before the rename may still set
# NTFY_TOPIC — feed it into the one canonical variable.
NTFY=${NTFY:-${NTFY_TOPIC:+ntfy.sh/$NTFY_TOPIC}}

# Canonical work dir (owner 2026-07-24): /workspace on EVERY box. Images
# vary — some ship /workspace, others drop the login shell in /root and
# work lands in ~/workspace. Adopt-or-create BEFORE the log redirect:
# on images without /workspace the old script died at the redirect with
# no log at all. On vast the rootfs IS the rented disk, so a created dir
# lands on the paid disk either way.
if [[ ! -e /workspace ]]; then
  if [[ -d "$HOME/workspace" ]]; then
    ln -s "$HOME/workspace" /workspace   # existing data, canonical name
  else
    mkdir -p /workspace
  fi
fi

# python shim (owner 2026-07-24): vast images ship only python3. Scripts
# call python3 and never depend on this — the shim is for hand-typed
# `python ...` in SSH sessions.
command -v python >/dev/null 2>&1 || ln -s "$(command -v python3)" /usr/local/bin/python

# Dual-sink logging (2026-07-31): tee to BOTH the durable on-box file AND
# container stdout — `vastai logs` shows only container stdout, and the old
# file-only redirect (`exec >>file`) made everything from here on (including
# the ready line) invisible remotely: a boot watcher greping `vastai logs`
# timed out while the box sat ready and idle-billing for ~30 min. One
# stream, two sinks; /workspace/onstart.log remains byte-complete.
exec > >(tee -a /workspace/onstart.log) 2>&1
echo "=== onstart $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if [[ -d "$HOME/workspace" && ! /workspace -ef "$HOME/workspace" ]]; then
  echo "WARNING: /workspace and $HOME/workspace both exist and differ — using /workspace; check where the real data lives"
fi

# 2026-08-16 incident: this was pinned to cut-to-core (an old, superseded
# branch missing ts38grid and other current work) — a box would clone code
# that didn't have the launcher it was about to be asked to run. Track
# whatever branch is currently checked out on the laptop that pastes this
# script into the vast template; update by hand when that changes.
BRANCH=ts38-mini
cd /workspace
[[ -d elicit-vs-teach ]] || git clone -b "$BRANCH" https://github.com/Hieuuum/elicit-vs-teach.git
cd elicit-vs-teach

# py>=3.11 venv (2026-07-31: the vast pytorch image shipped python 3.10, so
# the editable install + suite failed on system python; geode needs >=3.11).
# Always provision a python3.11 venv at /workspace/venv with uv and run
# everything below inside it. Marker mirrors .geode_torch_ok: create once
# per box; a container restart re-activates but never half-recreates.
venv=FAILED
if [[ ! -f /workspace/.geode_venv_ok ]]; then
  rm -rf /workspace/venv
  command -v uv >/dev/null 2>&1 || python3 -m pip install -q uv
  uv venv /workspace/venv --python 3.11 --seed && touch /workspace/.geode_venv_ok
fi
if [[ -f /workspace/.geode_venv_ok ]]; then
  set +u  # not every generator's activate script is `set -u`-clean
  source /workspace/venv/bin/activate
  set -u
  python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' && venv=ok
fi
if [[ $venv == ok ]]; then
  # Login shells / tmux windows must land in the venv too — appended BEFORE
  # the other bashrc exports below (fig2 fresh-box plan).
  if ! grep -q 'workspace/venv/bin/activate' ~/.bashrc 2>/dev/null; then
    echo "source /workspace/venv/bin/activate" >>~/.bashrc
  fi
else
  echo "WARNING: py3.11 venv provisioning FAILED — continuing on system $(python3 -V 2>&1); editable install/suite will likely fail on py<3.11 (hand-fix: uv venv /workspace/venv --python 3.11 --seed, activate, then rerun this script)"
fi

# CUDA-matched torch preflight (2026-07-29/30 incidents; see memory
# reference-vastai-box-env-fix.md): an unpinned `pip install` resolves torch
# to the newest PyPI wheel (currently cu130) regardless of the box's actual
# driver. A driver whose max supported CUDA is older still accepts the
# install, but torch.cuda.is_available() silently comes back False (a
# buried "driver too old" UserWarning, not an error) — CPU pytest still
# passes and suite=ok still prints. Read the driver's OWN max supported CUDA
# from nvidia-smi's header instead of trusting PyPI's default resolution.
cuda_tag_for() {  # nvidia-smi "CUDA Version" header string -> torch wheel tag; empty -> none
  local ver=$1 major minor num
  if [[ -z $ver ]]; then echo none; return; fi
  major=${ver%%.*}
  minor=${ver#*.}; minor=${minor%%.*}
  case ${#minor} in
    0) minor=00 ;;
    1) minor="${minor}0" ;;
    *) minor="${minor:0:2}" ;;
  esac
  num=$((10#$major * 100 + 10#$minor))
  if   ((num >= 1300)); then echo cu130
  elif ((num >= 1260)); then echo cu126
  elif ((num >= 1210)); then echo cu121
  else                       echo cu118
  fi
}

gpu=none  # default; a GPU box overwrites this below (preflight) and again near the end (real gate)
cuda_hdr=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9.]*\).*/\1/p' | head -1)
torch_tag=$(cuda_tag_for "$cuda_hdr")
if [[ $torch_tag != none ]]; then
  gpu=pending  # real value comes from the matmul gate near the end of the script
  # Marker mirrors .geode_suite_ok: install once per box — a container
  # restart must not silently advance torch mid-experiment (same rule as the
  # never-auto-pull clone above). The matmul gate below still runs every
  # start and reports gpu=FAILED loudly if the env rotted.
  if [[ ! -f /workspace/.geode_torch_ok ]]; then
    python3 -m pip install -q --upgrade torch torchvision --index-url "https://download.pytorch.org/whl/$torch_tag" \
      && touch /workspace/.geode_torch_ok
    python3 -m pip uninstall -q -y torchaudio || true  # stale torchaudio's undefined-symbol break against a fresh torch (2026-07-29 fix); geode needs no audio
  fi
  if ! command -v gcc >/dev/null 2>&1; then
    apt-get update -qq && apt-get install -y -q build-essential  # triton/torch.compile needs a C compiler at runtime (2026-07-29 fix); image ships none
  fi
  if ! grep -q '^export CC=gcc' ~/.bashrc 2>/dev/null; then
    echo "export CC=gcc" >>~/.bashrc
  fi
fi

python3 -m pip install -q -e ".[dev]"

# Every login shell / tmux window gets the exports — a stale $GEODE_STORE in
# a fresh window is the #1 guide troubleshooting item.
if ! grep -q GEODE_STORE ~/.bashrc 2>/dev/null; then
  echo "export GEODE_STORE=/workspace/elicit-vs-teach/geode-store  # store INSIDE the clone" >>~/.bashrc
  [[ -n ${NTFY:-} ]] && echo "export NTFY=$NTFY" >>~/.bashrc
  [[ -n ${NTFY_AUTO:-} ]] && echo "export NTFY_AUTO=$NTFY_AUTO" >>~/.bashrc
fi

# vast SSH often lands INSIDE a tmux session already (their default entry
# point) — nesting `tmux new` there fails. Tell every login shell which
# case it is in ($TMUX is set iff inside tmux); Ubuntu's interactive guard
# at the top of .bashrc keeps the echo out of scp/rsync shells.
if ! grep -q 'geode tmux hint' ~/.bashrc 2>/dev/null; then
  cat >>~/.bashrc <<'HINT'
# geode tmux hint
if [[ -n ${TMUX:-} ]]; then
  echo "[box] inside tmux session '$(tmux display-message -p '#S' 2>/dev/null)' — run long jobs directly; do NOT nest 'tmux new'"
else
  echo "[box] NOT in tmux — wrap long jobs: tmux new -s <name>"
fi
HINT
fi

# Template env vars don't reach SSH sessions on vast.ai; /etc/environment is
# the documented fix (their FAQ). READ token only — the write rule stands.
for var in HF_TOKEN NTFY; do
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
  python3 -m pytest -q && touch /workspace/.geode_suite_ok
fi
suite=$([[ -f /workspace/.geode_suite_ok ]] && echo ok || echo FAILED)

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h /workspace | tail -1

# Real GPU gate, every start (cheap — unlike the suite marker): a bare
# cuda.is_available() lies on a driver/CUDA mismatch (see preflight above),
# so actually run a matmul on the device instead of trusting it.
if [[ $gpu != none ]]; then
  if python3 - <<'PYEOF'
import sys
try:
    import torch
    x = torch.randn(64, 64, device="cuda")
    (x @ x).sum().item()
except Exception as e:
    print(f"[gpu-gate] FAILED: {e}")
    sys.exit(1)
PYEOF
  then
    gpu=ok
  else
    gpu=FAILED
  fi
fi

hash=$(git rev-parse --short HEAD)
echo "ready: $hash suite=$suite gpu=$gpu venv=$venv (box hash must match laptop before any launch)"
[[ -n ${NTFY:-} && ${NTFY_AUTO:-0} == 1 ]] && curl -sd "ONSTART ONLY (not a run result): box environment ready, hash=$hash suite=$suite gpu=$gpu venv=$venv — training has not started yet" "$NTFY" >/dev/null
echo "=== onstart done ==="
exit 0
