#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP job A (severity-2 lineage anchors): E6 dense on the frozen hip-hop battery (64 x 3 reps @3.84 s
# + 64 x 1 @10.24 s), E7 hip-hop extension (63 x 1 x {3.84, 10.24} s x {pruned2_A, recovered2, dense}), B public
# dense text-FT reference (first 96 AudioCaps prompts x {3.84, 10.24} s). Protocol: docs/reviewer2_followup.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_gen_a}"; DEV="${DEV:-cpu}"; PY="${PY:-.venv/bin/python}"
G="scripts/research/reversal_xsev_gen.py"
run() { echo "=== $* ($DEV) ==="; OPENBLAS_CORETYPE=Haswell "$PY" "$G" --device "$DEV" --out "$OUT" "$@"; }
run --system dense --context music                      # E6 (192 WAVs)
run --system dense --context music_native               # E6 (64)
for SYS in pruned2_A recovered2 dense; do               # E7 (378)
  run --system "$SYS" --context music_ext
  run --system "$SYS" --context music_ext_native
done
run --system textft --context ac_short --first-n 96     # B (96)
run --system textft --context ac_native --first-n 96    # B (96)
echo "=== device check: pruned2_A / ac_native / indices 0-3 ==="
OPENBLAS_CORETYPE=Haswell "$PY" "$G" --system pruned2_A --context ac_native --device "$DEV" --indices 0,1,2,3 --out "$OUT/device_check"
echo "R2 GEN A DONE: 826 WAVs under $OUT (+4 device-check WAVs)"
