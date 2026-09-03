#!/usr/bin/env bash
# DRAFT5-OPSWEEP-1 generation (E1): 1152 NEW WAVs = {dense, pruned2_A, recovered2} x 192 AudioCaps
# prompts x r0 x {5.12 s (latent 128), 7.68 s (latent 192)}, frozen recipe (DDIM 50 / guidance 2.5).
# Protocol: docs/draft5_opsweep.md (frozen + sha256 sidecar before any output).
# Device rule: ALL WAVs from ONE T4 job (same hardware class as every frozen clip).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/draft5_opsweep_gen}"
DEV="${DEV:-cpu}"
PY="${PY:-.venv/bin/python}"

for CTX in ac_d128 ac_d192; do
  for SYS in pruned2_A recovered2 dense; do
    echo "=== $SYS / $CTX ($DEV) ==="
    OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py \
      --system "$SYS" --context "$CTX" --device "$DEV" --out "$OUT"
    echo "=== $SYS / $CTX DONE ==="
  done
done
# device-consistency check (protocol section 4): 4 frozen-recipe pruned2_A native clips, separate dir
echo "=== device check: pruned2_A / ac_native / indices 0-3 ($DEV) ==="
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py \
  --system pruned2_A --context ac_native --device "$DEV" --indices 0,1,2,3 --out "$OUT/device_check"
echo "DRAFT5-OPSWEEP-1 GEN DONE: 1152 WAVs under $OUT (+4 device-check WAVs)"
