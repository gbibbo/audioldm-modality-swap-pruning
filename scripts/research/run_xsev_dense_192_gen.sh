#!/usr/bin/env bash
# XSEV-DENSE-192-CONTROL generation (384 NEW WAVs = dense x 192 AudioCaps prompts x r0 x {3.84 s, 10.24 s}).
# Protocol: docs/xsev_dense_192_control.md (frozen before any output). NOT launched by default.
# Systems/CRN: dense EMA; x_T identical to the frozen pruned2_A / recovered2 clips of the same (context, ytid).
# Device rule: all 384 WAVs from ONE device. DEV=cpu (0 cr, ~18 h) or DEV=cuda (one T4 job, ~1.2 cr, cap 1.5).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/xsev_dense_192_gen}"
DEV="${DEV:-cpu}"
PY="${PY:-.venv/bin/python}"

for CTX in ac_native ac_short; do
  echo "=== dense / $CTX ($DEV) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system dense --context "$CTX" --device "$DEV" --out "$OUT"
  echo "=== dense / $CTX DONE ==="
done
# device-consistency check (protocol §2): 4 pruned2_A native clips regenerated in this job, separate dir
echo "=== device check: pruned2_A / ac_native / indices 0-3 ($DEV) ==="
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system pruned2_A --context ac_native --device "$DEV" --indices 0,1,2,3 --out "$OUT/device_check"
echo "XSEV-DENSE-192-GEN DONE: 384 dense WAVs under $OUT (+4 device-check WAVs)"
