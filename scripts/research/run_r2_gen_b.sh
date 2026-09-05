#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP job B: E5 Clotho battery (96 x {3.84, 10.24} s x {pruned2_A, recovered2, dense}) and E1c one point
# beyond the fine-tuning duration (15.36 s, latent 384; pruned2_A + recovered2, first 96 AudioCaps prompts).
# Protocol: docs/reviewer2_followup.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_gen_b}"; DEV="${DEV:-cpu}"; PY="${PY:-.venv/bin/python}"
G="scripts/research/reversal_xsev_gen.py"
run() { echo "=== $* ($DEV) ==="; OPENBLAS_CORETYPE=Haswell "$PY" "$G" --device "$DEV" --out "$OUT" "$@"; }
for SYS in pruned2_A recovered2 dense; do               # E5 (576)
  run --system "$SYS" --context clotho_short
  run --system "$SYS" --context clotho_native
done
for SYS in pruned2_A recovered2; do                     # E1c (192)
  run --system "$SYS" --context ac_d384 --first-n 96
done
echo "=== device check: pruned2_A / ac_native / indices 0-3 ==="
OPENBLAS_CORETYPE=Haswell "$PY" "$G" --system pruned2_A --context ac_native --device "$DEV" --indices 0,1,2,3 --out "$OUT/device_check"
echo "R2 GEN B DONE: 768 WAVs under $OUT (+4 device-check WAVs)"
