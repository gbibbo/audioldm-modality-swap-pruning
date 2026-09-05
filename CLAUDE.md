# CLAUDE.md

@AGENTS.md

## Claude Code

Project hooks live in `.claude/settings.json` and `.claude/hooks/`.

Available project skills:

* `/auditar`: evidence-first technical and scientific audit.
* `/cerrar-hito`: validate, record, commit, and push a coherent unit of progress.

The `UserPromptSubmit` hook provides the Montevideo timestamp for the current turn. Start every user-facing response with that timestamp on its own first line.

After installing or changing project hooks, validate them and verify the active settings sources before relying on the behavior.

The Studio bills credits while it runs. The SessionStart hook starts `scripts/ops/studio_idle_guard.py`; stop the Studio with `--stop-now` when the session ends (see AGENTS.md, Studio uptime discipline).
