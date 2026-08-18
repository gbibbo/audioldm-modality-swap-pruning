#!/usr/bin/env bash
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
INPUT="$(cat)"
STOP_ACTIVE="$(python3 -c 'import json,sys
try:
    print("true" if json.load(sys.stdin).get("stop_hook_active", False) else "false")
except Exception:
    print("false")' <<<"$INPUT")"

# Avoid continuation loops. If Claude is already continuing because of this Stop hook,
# allow the turn to end even if the underlying state is still dirty.
[ "$STOP_ACTIVE" = "true" ] && exit 0

messages=()

if ! python3 "$ROOT/.claude/hooks/check_progress.py" "$ROOT/PROGRESS.md" --quiet >/dev/null 2>&1; then
  messages+=("PROGRESS.md no cumple su estructura minima")
fi

for rel in docs/experiment_ledger.md docs/compute_budget.md docs/claims_matrix.md docs/pilot_protocol.md; do
  [ -f "$ROOT/$rel" ] || messages+=("falta $rel")
done

if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  dirty="$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$dirty" -gt 0 ]; then
    messages+=("quedan ${dirty} path(s) sin commit")
  fi
  upstream="$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [ -n "$upstream" ]; then
    ahead="$(git -C "$ROOT" rev-list --count "$upstream"..HEAD 2>/dev/null || printf '0')"
    if [ "$ahead" -gt 0 ] 2>/dev/null; then
      messages+=("hay ${ahead} commit(s) locales sin push")
    fi
  fi
else
  messages+=("el proyecto no es un repositorio Git")
fi

[ "${#messages[@]}" -eq 0 ] && exit 0
joined="$(IFS='; '; echo "${messages[*]}")"
python3 - "$joined" <<'PY'
import json, sys
context = (
    "PROJECT CHECKPOINT ANTES DE CERRAR: " + sys.argv[1] + ". "
    "Revisa si el trabajo de este turno constituye una unidad validada. Si si, actualiza PROGRESS.md y los docs de trazabilidad pertinentes, valida, crea un commit coherente y haz push si hay upstream. "
    "Si no esta validado o no corresponde commitear, deja el estado de trabajo en curso explicitamente registrado. No inventes resultados ni fuerces un cierre."
)
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "Stop",
        "additionalContext": context,
    }
}, ensure_ascii=False))
PY
