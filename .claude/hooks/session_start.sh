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

printf '\n%s\n' '=== STUDIO IDLE GUARD ==='
GPY=/home/zeus/miniconda3/envs/cloudspace/bin/python
if [ -x "$GPY" ] && [ -f "$ROOT/scripts/ops/studio_idle_guard.py" ]; then
  mkdir -p "$HOME/.cache/studio_idle_guard" && date +%s > "$HOME/.cache/studio_idle_guard/last_user_activity"
  if ! kill -0 "$(cat "$HOME/.cache/studio_idle_guard/guard.pid" 2>/dev/null)" 2>/dev/null; then
    (cd "$ROOT" && setsid nohup "$GPY" scripts/ops/studio_idle_guard.py --daemon >/dev/null 2>&1 &)
    printf '%s\n' 'idle guard STARTED (stops the Studio after 45 min without user prompts, quiet CPU, no hold/protected process)'
  else
    printf '%s\n' "idle guard running (pid $(cat "$HOME/.cache/studio_idle_guard/guard.pid"))"
  fi
  printf '%s\n' 'The CPU Studio bills ~0.27 cr/h. Stop it when the session ends: scripts/ops/studio_idle_guard.py --stop-now'
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
