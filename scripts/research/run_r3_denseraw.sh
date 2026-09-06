#!/usr/bin/env bash
# REVIEW3-DENSERAW job r3-denseraw (third review, item 1b): the RAW-weight dense baseline on the frozen 192 AudioCaps
# prompts x {3.84, 10.24} s (384 WAVs, CRN-paired with the frozen dense-EMA / denseft / P / P+FT clips) + a 4-WAV dense-EMA
# device check. Protocol: docs/review3_denseraw.md (frozen + sha256 before launch). Bounds: per-stage `timeout` (primary,
# survives a stopped Studio; 3300 + 6600 + 600 s = 175 min = the 2.6-cr cap at 0.89 cr/h) + external watchdog (insurance).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r3_denseraw}"; DEV="${DEV:-cuda}"; PY="${PY:-.venv/bin/python}"
G="scripts/research/reversal_xsev_gen.py"
run() { local tmo="$1"; shift; echo "=== $* ($DEV, timeout ${tmo}s) ==="; OPENBLAS_CORETYPE=Haswell timeout "$tmo" "$PY" "$G" --device "$DEV" "$@"; }
run 3300 --system dense_raw --context ac_short  --out "$OUT/gen"          # 192 WAVs, latent 96
run 6600 --system dense_raw --context ac_native --out "$OUT/gen"          # 192 WAVs, latent 256
run 600  --system dense --context ac_native --indices 0,1,2,3 --out "$OUT/device_check"   # dense-EMA device check vs frozen xsev_dense192
echo "R3 DENSERAW DONE: 384 WAVs under $OUT/gen (+4 device-check WAVs under $OUT/device_check)"
