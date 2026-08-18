#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

EXPECTED_HOOK_FILES = {
    "check_progress.py",
    "session_start.sh",
    "stop_status.sh",
    "turn_start.sh",
}
EXPECTED_SKILLS = {"auditar", "cerrar-hito"}
EXPECTED_HOOK_EVENTS = {"SessionStart", "UserPromptSubmit", "Stop"}
REQUIRED_DOCS = {
    "docs/experiment_ledger.md",
    "docs/compute_budget.md",
    "docs/claims_matrix.md",
    "docs/pilot_protocol.md",
}
SUSPICIOUS = (
    "clockify",
    "meeting_transcripts",
    "check_transcripts",
    "meetings.conf",
    "edge audio labs",
    "transcripciones",
    "publicar",
    "check_brief",
    "check_claims",
)


def fail(msg: str, issues: list[str]) -> None:
    issues.append(msg)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    issues: list[str] = []

    try:
        top = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if Path(top).resolve() != root:
            fail(f"install target is not Git root: git root is {top}", issues)
    except Exception:
        fail("project is not a Git repository", issues)

    hooks_dir = root / ".claude" / "hooks"
    hook_files = {p.name for p in hooks_dir.iterdir() if p.is_file()} if hooks_dir.exists() else set()
    if hook_files != EXPECTED_HOOK_FILES:
        fail(f"unexpected project hook files: expected {sorted(EXPECTED_HOOK_FILES)}, got {sorted(hook_files)}", issues)

    skills_dir = root / ".claude" / "skills"
    skills = {p.name for p in skills_dir.iterdir() if p.is_dir()} if skills_dir.exists() else set()
    if skills != EXPECTED_SKILLS:
        fail(f"unexpected project skills: expected {sorted(EXPECTED_SKILLS)}, got {sorted(skills)}", issues)
    for skill in EXPECTED_SKILLS:
        if not (skills_dir / skill / "SKILL.md").is_file():
            fail(f"missing skill file: .claude/skills/{skill}/SKILL.md", issues)

    settings_path = root / ".claude" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        events = set(settings.get("hooks", {}))
        if events != EXPECTED_HOOK_EVENTS:
            fail(f"unexpected project hook events: expected {sorted(EXPECTED_HOOK_EVENTS)}, got {sorted(events)}", issues)
        attribution = settings.get("attribution", {})
        if attribution != {"commit": "", "pr": "", "sessionUrl": False}:
            fail(f"unexpected attribution settings: {attribution!r}", issues)
    except Exception as exc:
        fail(f"invalid .claude/settings.json: {exc}", issues)

    for rel in REQUIRED_DOCS:
        if not (root / rel).is_file():
            fail(f"missing provenance file: {rel}", issues)

    known_paths = [
        root / "CLOCKIFY_LOG.md",
        root / "meeting_transcripts",
        root / "tools" / "check_clockify.sh",
        root / "tools" / "check_transcripts.sh",
        root / "tools" / "meetings.conf",
        root / "tools" / "check_brief.py",
        root / "tools" / "check_briefs_hook.sh",
        root / "tools" / "check_claims.py",
        root / "tools" / "check_claims_hook.sh",
    ]
    for path in known_paths:
        if path.exists():
            fail(f"known EAL residual still exists: {path.relative_to(root)}", issues)

    scan_roots = [root / ".claude"]
    for scan_root in scan_roots:
        if not scan_root.exists():
            continue
        for path in scan_root.rglob("*"):
            if not path.is_file():
                continue
            rel = str(path.relative_to(root)).lower()
            if any(term in rel for term in SUSPICIOUS):
                fail(f"suspicious Claude path remains: {path.relative_to(root)}", issues)

    # Legacy commands, rules, and custom agents are also project instruction surfaces.
    # Scan their content for old EAL machinery without flagging the canonical files,
    # which intentionally say that those workflows are forbidden.
    for rel_dir in (".claude/commands", ".claude/rules", ".claude/agents"):
        base = root / rel_dir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").lower()
            except Exception:
                continue
            hits = [term for term in SUSPICIOUS if term in text]
            if hits:
                fail(f"EAL-like content in {path.relative_to(root)}: {hits}", issues)

    if issues:
        for issue in issues:
            print(f"AGENT KIT VERIFY: FAIL: {issue}")
        return 1
    print("AGENT KIT VERIFY: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
