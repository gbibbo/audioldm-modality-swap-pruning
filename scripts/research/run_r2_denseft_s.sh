#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP-EXT job r2-denseft-s (item 2, short arm): 200-step self-gate -> 20 000-step full fine-tune of
# the DENSE AudioLDM-M-Full at 3.84 s (latent 96, batch 2) -> CRN eval WAVs (192 prompts x {3.84, 10.24} s, tag _s).
# Protocol: docs/reviewer2_followup_ext.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_denseft_s}"; DEV="${DEV:-cuda}"; PY="${PY:-.venv/bin/python}"
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/e3_shortft_trainer.py --out "$OUT" \
  --backbone dense --duration 3.84 --latent-t 96 --batch 2 --accum 1 \
  --save-name denseft_short_unet.pt --cap-cr 3.2 --resume
if [ ! -f "$OUT/denseft_short_unet.pt" ]; then echo "BENCH-ONLY STOP recorded; no evaluation generated"; exit 0; fi
export DENSEFT_UNET="$OUT/denseft_short_unet.pt"
for CTX in ac_short ac_native; do
  echo "=== denseft(_s) / $CTX ($DEV) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system denseft --context "$CTX" --device "$DEV" --out "$OUT/gen"
done
echo "R2 DENSEFT-S DONE: trainer + 384 eval WAVs under $OUT"
