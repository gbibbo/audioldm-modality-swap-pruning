#!/usr/bin/env bash
# XSEV-MUSIC-NATIVE-1 generation (128 NEW WAVs = 2 systems x 64 music prompts x replicate 0 @10.24 s).
# Frozen protocol docs/xsev_music_native_1.md; manifest configs/research/xsev_music_manifest.json
# (f5a26fbe). Completes the severity-2 domain x duration factorial (music @ native duration).
# Systems: recovered2 + pruned2_A only (A' primary; B' not generated — credit minimisation, see protocol).
# EMA convention; DDIM50/g2.5/eta0/fp32/single; CRN x_T shared across the two systems per ytid.
# DEV=cpu is the default (0 cr; the protocol's CPU-first rule); DEV=cuda for a T4 job if CPU is too slow.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/xsev_music_native_gen}"
DEV="${DEV:-cpu}"
PY="${PY:-.venv/bin/python}"

for SYS in recovered2 pruned2_A; do
  echo "=== $SYS / music_native ($DEV) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system "$SYS" --context music_native --device "$DEV" --out "$OUT"
  echo "=== $SYS / music_native DONE ==="
done
echo "XSEV-MUSIC-NATIVE-GEN DONE: 128 WAVs under $OUT"
