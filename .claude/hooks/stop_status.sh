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
# allow the turn to end even if the underlying state is still dirty / the contract stands.
[ "$STOP_ACTIVE" = "true" ] && exit 0

messages=()

if ! python3 "$ROOT/.claude/hooks/check_progress.py" "$ROOT/PROGRESS.md" --quiet >/dev/null 2>&1; then
  messages+=("PROGRESS.md no cumple su estructura minima")
fi

for rel in docs/experiment_ledger.md docs/compute_budget.md docs/claims_matrix.md docs/pilot_protocol.md; do
  [ -f "$ROOT/$rel" ] || messages+=("falta $rel")
done

# --- Final checkpoint contract: always surface HEAD / tree / push so the visible response
# --- can report the durable SHA the supervisor will independently fetch. Existing dirty-tree
# --- and unpushed-commit protections are preserved below.
head_full="(no-git)"; head_short="(no-git)"; tree_state="unknown"; push_state="unknown"
if git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  head_full="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo '(none)')"
  head_short="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '(none)')"
  dirty="$(git -C "$ROOT" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  if [ "$dirty" -gt 0 ]; then
    tree_state="dirty (${dirty} path(s))"
    messages+=("quedan ${dirty} path(s) sin commit")
  else
    tree_state="clean"
  fi
  upstream="$(git -C "$ROOT" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [ -z "$upstream" ]; then
    push_state="no-upstream"
  else
    ahead="$(git -C "$ROOT" rev-list --count "$upstream"..HEAD 2>/dev/null || printf '0')"
    behind="$(git -C "$ROOT" rev-list --count "HEAD..$upstream" 2>/dev/null || printf '0')"
    if [ "$ahead" -gt 0 ] 2>/dev/null; then
      push_state="ahead ${ahead} (unpushed)"
      messages+=("hay ${ahead} commit(s) locales sin push")
    elif [ "$behind" -gt 0 ] 2>/dev/null; then
      push_state="behind ${behind}"
    else
      push_state="synced"
    fi
  fi
else
  messages+=("el proyecto no es un repositorio Git")
fi

warn=""
[ "${#messages[@]}" -gt 0 ] && warn="$(IFS='; '; echo "${messages[*]}")"

python3 - "$head_full" "$head_short" "$tree_state" "$push_state" "$warn" <<'PY'
import json, sys
head_full, head_short, tree_state, push_state, warn = sys.argv[1:6]
parts = []
if warn:
    parts.append(
        "PROJECT CHECKPOINT ANTES DE CERRAR: " + warn + ". "
        "Revisa si el trabajo de este turno constituye una unidad validada. Si si, actualiza PROGRESS.md y los docs de trazabilidad pertinentes, valida, crea un commit coherente y haz push si hay upstream. "
        "Si no esta validado o no corresponde commitear, deja el estado de trabajo en curso explicitamente registrado. No inventes resultados ni fuerces un cierre."
    )
parts.append(
    "FINAL CHECKPOINT CONTRACT: si este turno produjo trabajo sustantivo, la respuesta visible DEBE terminar, justo antes de STOP, con un bloque `## CHECKPOINT` que reporte COMMIT/SHORT/PUSH/TREE reales; para un turno de solo-lectura reporta `NO NEW COMMIT` con HEAD/PUSH/TREE. El supervisor buscara ese commit en GitHub para verificar.\n"
    f"Current HEAD: {head_full} ({head_short})\n"
    f"Working tree: {tree_state}\n"
    f"Push status: {push_state}"
)
print(json.dumps({
    "hookSpecificOutput": {
        "hookEventName": "Stop",
        "additionalContext": "\n\n".join(parts),
    }
}, ensure_ascii=False))
PY
