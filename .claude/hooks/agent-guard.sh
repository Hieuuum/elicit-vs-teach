#!/usr/bin/env bash
# agent-guard.sh — hard path enforcement for the four-stage protocol subagents.
# PreToolUse hook; keyed on agent_type (present only inside subagents).
# Main-loop calls and unknown agents pass through untouched.
#
# Enforced rules:
#   test-writer / test-auditor : write only tests/ (+ /tmp scratch);
#                                no Read/Grep/Glob/Bash access to geode/ impl
#   implementer                : no writes to tests/ or specs/
#   conformance-reviewer,
#   adapt-reviewer             : no writes at all (belt — allowlist already
#                                omits Edit/Write)
#   adapt-author               : write only docs/ (+ /tmp scratch)
# reference/** edit-deny for everyone lives in settings.json permissions.deny.
#
# Known soft spot: Bash redirects are not path-filtered for implementer /
# adapt-author; the writer/auditor "geode" token block covers the leak that
# matters most (reading impl pre-implementation).

set -euo pipefail

input=$(cat)
tool=$(jq -r '.tool_name // empty' <<<"$input")
agent=$(jq -r '.agent_type // empty' <<<"$input")

case "$agent" in
  test-writer|test-auditor|implementer|adapt-author|conformance-reviewer|adapt-reviewer) ;;
  *) exit 0 ;;
esac

deny() {
  jq -n --arg reason "$1" \
    '{hookSpecificOutput:{hookEventName:"PreToolUse",permissionDecision:"deny",permissionDecisionReason:$reason}}'
  exit 0
}

proj="${CLAUDE_PROJECT_DIR:-$PWD}"

# Repo-relative path; "__OUTSIDE__<abs>" when the target is outside the repo.
relpath() {
  local p="$1"
  [[ -z "$p" ]] && { echo ""; return; }
  [[ "$p" != /* ]] && p="$proj/$p"
  p=$(realpath -m "$p")
  case "$p" in
    "$proj"/*) echo "${p#"$proj"/}" ;;
    "$proj")   echo "." ;;
    *)         echo "__OUTSIDE__$p" ;;
  esac
}

is_scratch() { [[ "$1" == __OUTSIDE__/tmp/* ]]; }

target=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // .tool_input.path // empty' <<<"$input")
rel=$(relpath "$target")

case "$tool" in
  Edit|Write|NotebookEdit)
    case "$agent" in
      conformance-reviewer|adapt-reviewer)
        deny "$agent is read-only: report findings, do not edit." ;;
      test-writer|test-auditor)
        if [[ "$rel" == tests/* ]] || is_scratch "$rel"; then exit 0; fi
        deny "$agent may write only under tests/ (or /tmp scratch); attempted: $target" ;;
      implementer)
        if [[ "$rel" == tests/* || "$rel" == specs/* ]]; then
          deny "implementer must not modify tests/ or specs/ (four-stage protocol; test fixes route through test-auditor); attempted: $target"
        fi ;;
      adapt-author)
        if [[ "$rel" == docs/* ]] || is_scratch "$rel"; then exit 0; fi
        deny "adapt-author may write only under docs/ (or /tmp scratch); attempted: $target" ;;
    esac
    ;;
  Read)
    case "$agent" in
      test-writer|test-auditor)
        if [[ "$rel" == geode/* ]]; then
          deny "$agent derives tests from specs only and must not read geode/ implementation."
        fi ;;
    esac
    ;;
  Grep|Glob)
    case "$agent" in
      test-writer|test-auditor)
        if [[ -z "$target" ]]; then
          deny "$agent must scope Grep/Glob with an explicit path (tests/, specs/, PLAN.md): a repo-root search would expose geode/ implementation."
        fi
        if [[ "$rel" == geode || "$rel" == geode/* ]]; then
          deny "$agent must not search geode/ implementation."
        fi ;;
    esac
    ;;
  Bash)
    case "$agent" in
      test-writer|test-auditor)
        cmd=$(jq -r '.tool_input.command // empty' <<<"$input")
        # Package subdirs listed explicitly so the repo dir itself
        # (.../Github/geode/tests/...) never false-positives. Extend the
        # alternation when a new geode/ subpackage is added.
        if grep -qE 'geode/(zoo|edl|steering|saediff|__init__)|(import|from)[[:space:]]+geode' <<<"$cmd"; then
          deny "$agent must not touch geode/ implementation via shell (pytest/ruff on tests/ paths are fine)."
        fi ;;
    esac
    ;;
esac

exit 0
