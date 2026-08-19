#!/usr/bin/env python3
"""Stdlib test runner for tests/research/ (resolves audit finding F1).

`pytest` is intentionally absent from the frozen environment (it is not in
`poetry.lock`, and no pinned scientific version may be relaxed to add it). Each
research test module is directly runnable and returns exit code 0 on pass; this
runner invokes a selected set as subprocesses and aggregates the results, so the
M1 acceptance criterion "CPU tests pass" has a single reproducible command:

    .venv/bin/python scripts/research/run_research_tests.py            # M1 suite
    .venv/bin/python scripts/research/run_research_tests.py --all      # every test_*.py

Exit code 0 iff every selected module passes.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TESTS_DIR = REPO / "tests" / "research"

# M1 parameter-efficient-recovery CPU suite (the default selection).
M1_SUITE = [
    "test_lora_layers.py",
    "test_injector.py",
    "test_state_ema_optimizer.py",
    "test_peft_real_unet.py",   # F6: real pruned U-Net (present after M1-ADOPT-B)
    "test_peft_integration.py",  # F8: optimizer/lifecycle hooks (present after M1-ADOPT-C)
]


def discover_all():
    return sorted(p.name for p in TESTS_DIR.glob("test_*.py"))


def main(argv) -> int:
    if "--all" in argv:
        selection = discover_all()
    else:
        selection = [m for m in M1_SUITE if (TESTS_DIR / m).exists()]
    results = {}
    for module in selection:
        path = TESTS_DIR / module
        print(f"\n########## {module} ##########", flush=True)
        rc = subprocess.run([sys.executable, str(path)], cwd=str(REPO)).returncode
        results[module] = rc
    print("\n==================== SUMMARY ====================")
    for module in selection:
        print(f"  {module:<32} {'PASS' if results[module] == 0 else f'FAIL (rc={results[module]})'}")
    all_ok = all(rc == 0 for rc in results.values())
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}  ({len(selection)} modules)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
