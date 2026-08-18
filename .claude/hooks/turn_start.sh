#!/usr/bin/env bash
set -uo pipefail

ROOT="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
TZ_NAME="${PROJECT_TZ:-America/Montevideo}"
NOW_TIME="$(TZ="$TZ_NAME" date '+%H:%M')"
NOW_DATE="$(TZ="$TZ_NAME" date '+%Y-%m-%d')"

GIT_NOTE=""
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  dirty="$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  upstream="$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  ahead="?"
  behind="?"
  if [ -n "$upstream" ]; then
    read -r behind ahead <<EOF_COUNTS
$(git -C "$ROOT" rev-list --left-right --count "$upstream"...HEAD 2>/dev/null || printf '? ?')
EOF_COUNTS
  fi
  GIT_NOTE="Git checkpoint state: ${dirty} uncommitted path(s); ahead=${ahead}; behind=${behind}."
fi

CONTEXT="HORA DE ESTE TURNO: ${NOW_TIME} (${NOW_DATE}, ${TZ_NAME}). Empieza la respuesta visible con **${NOW_TIME}** sola en la primera linea.
Antes de terminar un turno que haya producido progreso significativo: verifica el cambio, actualiza PROGRESS.md y los docs de trazabilidad pertinentes, y crea/pushea un commit coherente si el trabajo ya constituye una unidad validada. No commitees secretos, datasets, audio generado ni checkpoints grandes. Si el trabajo no esta validado, registra el estado sin fingir cierre.
${GIT_NOTE}"

python3 - "$CONTEXT" <<'PY'
import json, sys
context = sys.argv[1]
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": context,
    }
}, ensure_ascii=False))
PY
