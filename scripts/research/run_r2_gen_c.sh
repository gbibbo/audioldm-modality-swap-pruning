#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP job C: E8 severity-1 P / P+FT on the first 96 severity-2 AudioCaps prompts x {3.84, 10.24} s
# (384 WAVs; CRN seeds of the severity-2 cells, so the four systems share x_T per prompt and duration).
# Protocol: docs/reviewer2_followup.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_gen_c}"; DEV="${DEV:-cpu}"; PY="${PY:-.venv/bin/python}"
G="scripts/research/reversal_xsev_gen.py"
run() { echo "=== $* ($DEV) ==="; OPENBLAS_CORETYPE=Haswell "$PY" "$G" --device "$DEV" --out "$OUT" "$@"; }
for SYS in p1_pruned p1_recovered; do
  run --system "$SYS" --context ac_short --first-n 96
  run --system "$SYS" --context ac_native --first-n 96
done
echo "R2 GEN C DONE: 384 WAVs under $OUT"
