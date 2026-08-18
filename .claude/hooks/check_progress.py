#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REQUIRED = ("## CURRENT STATE", "## OPEN ITEMS", "## RUN RECIPES")
MARKER = "<!-- FIN-ESTADO -->"


def check(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing: {path}"]
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    issues: list[str] = []
    for heading in REQUIRED:
        if heading not in text:
            issues.append(f"missing section: {heading}")
    if MARKER not in text:
        issues.append(f"missing marker: {MARKER}")
    else:
        marker_line = next(i for i, line in enumerate(lines, start=1) if MARKER in line)
        if marker_line > 80:
            issues.append(f"state block too large: {marker_line} lines before marker, limit 80")
    if len(lines) > 600:
        issues.append(f"PROGRESS.md is too large: {len(lines)} lines, rotate or compress the LOG")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="PROGRESS.md")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    issues = check(Path(args.path))
    if issues:
        if not args.quiet:
            for issue in issues:
                print(f"PROGRESS CHECK: {issue}")
        return 1
    if not args.quiet:
        print("PROGRESS CHECK: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
