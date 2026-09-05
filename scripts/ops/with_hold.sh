#!/usr/bin/env bash
# Run a long CPU command while protecting the Studio from the idle guard.
# Usage: scripts/ops/with_hold.sh <minutes> <command...>
set -uo pipefail
MIN="$1"; shift
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
G="$(dirname "$0")/studio_idle_guard.py"
"$PY" "$G" --hold "$MIN" >/dev/null 2>&1 || true
"$@"; rc=$?
"$PY" "$G" --release >/dev/null 2>&1 || true
exit $rc
