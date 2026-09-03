#!/usr/bin/env bash
# DRAFT5-PUBRECIPE-1 generation (E2b): 256 NEW WAVs = {pruned2_A, recovered2} x the FIRST 64 AudioCaps
# prompts (frozen manifest order, outcome-blind) x r0 x {3.84 s, 10.24 s} at the PUBLISHED recipe
# (DDIM 200 / guidance 3.5, single generation -- best-of-3 NOT reproduced, see protocol section 3).
# Protocol: docs/draft5_opsweep.md (frozen + sha256 sidecar before any output).
# Device rule: ALL WAVs from ONE T4 job. The _pub tag keeps these WAVs separate from the frozen ones.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/draft5_pubrecipe_gen}"
DEV="${DEV:-cpu}"
PY="${PY:-.venv/bin/python}"

for CTX in ac_short ac_native; do
  for SYS in pruned2_A recovered2; do
    echo "=== $SYS / $CTX published recipe ($DEV) ==="
    OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py \
      --system "$SYS" --context "$CTX" --recipe published --tag _pub --first-n 64 \
      --device "$DEV" --out "$OUT"
    echo "=== $SYS / $CTX DONE ==="
  done
done
# device-consistency check (protocol section 4): FROZEN recipe, so it is comparable to the frozen clips
echo "=== device check: pruned2_A / ac_native / indices 0-3, frozen recipe ($DEV) ==="
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py \
  --system pruned2_A --context ac_native --device "$DEV" --indices 0,1,2,3 --out "$OUT/device_check"
echo "DRAFT5-PUBRECIPE-1 GEN DONE: 256 WAVs under $OUT (+4 device-check WAVs)"
