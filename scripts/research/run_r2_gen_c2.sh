#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP job c2: complete E8. r2-gen-c generated p1_pruned (both durations) but the watchdog
# stopped it on wall-clock (queue time included) with p1_recovered only ~34/192 done. This regenerates
# p1_recovered in full (first 96 AudioCaps prompts x {3.84, 10.24} s = 192 WAVs). Protocol: docs/reviewer2_followup.md §5.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_gen_c2}"; DEV="${DEV:-cuda}"; PY="${PY:-.venv/bin/python}"
G="scripts/research/reversal_xsev_gen.py"
for CTX in ac_short ac_native; do
  echo "=== p1_recovered / $CTX ($DEV) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" "$G" --system p1_recovered --context "$CTX" --device "$DEV" --out "$OUT" --first-n 96
done
echo "R2 GEN C2 DONE: 192 p1_recovered WAVs under $OUT"
