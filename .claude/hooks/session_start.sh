#!/usr/bin/env bash
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
PROGRESS="$ROOT/PROGRESS.md"

printf '%s\n' '=== PROJECT STATE ==='
if [ -f "$PROGRESS" ]; then
  sed '/<!-- FIN-ESTADO -->/q' "$PROGRESS"
else
  printf '%s\n' 'PROGRESS.md is missing. Restore or create it before substantial work.'
fi

printf '\n%s\n' '=== TRACEABILITY FILES ==='
for rel in docs/master_plan_v3.md docs/experiment_ledger.md docs/compute_budget.md docs/claims_matrix.md docs/pilot_protocol.md; do
  if [ -f "$ROOT/$rel" ]; then
    printf 'OK %s\n' "$rel"
  else
    printf 'MISSING %s\n' "$rel"
  fi
done

printf '\n%s\n' '=== GIT STATE ==='
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$ROOT" status --short --branch 2>/dev/null | head -n 40
  upstream="$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [ -n "$upstream" ]; then
    counts="$(git -C "$ROOT" rev-list --left-right --count "$upstream"...HEAD 2>/dev/null || true)"
    [ -n "$counts" ] && printf 'upstream divergence (behind ahead): %s\n' "$counts"
  else
    printf '%s\n' 'No upstream branch configured for current branch.'
  fi
else
  printf '%s\n' 'ERROR: project root is not a Git working tree.'
fi

printf '\n%s\n' 'Start from the master plan and actual evidence. Do not infer milestone completion from this context alone.'
