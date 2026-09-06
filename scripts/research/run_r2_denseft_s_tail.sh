#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP-EXT consolidation: finish the 22 denseft-s ac_native eval WAVs (prompt_index 170-191) that the
# watchdog's 4.2-cr cap cut from r2-denseft-s. Same checkpoint (denseft_short_unet.pt), same frozen GEN_SALT x_T, so
# these 22 are consistent with the 170 already generated. Protocol: docs/reviewer2_followup_ext.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_denseft_s_tail}"; DEV="${DEV:-cuda}"; PY="${PY:-.venv/bin/python}"
export DENSEFT_UNET="${DENSEFT_UNET:-data/checkpoints/denseft_short_unet.pt}"
[ -f "$DENSEFT_UNET" ] || { echo "PREFLIGHT FAIL: DENSEFT_UNET missing ($DENSEFT_UNET)"; exit 2; }
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system denseft --context ac_native \
  --indices 170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191 \
  --device "$DEV" --out "$OUT/gen"
echo "R2 DENSEFT-S TAIL DONE: 22 ac_native WAVs under $OUT/gen"
