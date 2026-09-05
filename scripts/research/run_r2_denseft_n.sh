#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP-EXT job r2-denseft-n (item 2, native arm): 200-step self-gate -> 20 000-step full fine-tune of
# the DENSE AudioLDM-M-Full at 10.24 s (latent 256, batch 1 x accum 2 = effective batch 2, to fit a 16 GB T4)
# -> CRN eval WAVs (192 prompts x {3.84, 10.24} s, tag _n). Protocol: docs/reviewer2_followup_ext.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_denseft_n}"; DEV="${DEV:-cuda}"; PY="${PY:-.venv/bin/python}"
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/e3_shortft_trainer.py --out "$OUT" \
  --backbone dense --duration 10.24 --latent-t 256 --batch 1 --accum 2 \
  --save-name denseft_native_unet.pt --cap-cr 5.6 --mid-step 7500 --resume
if [ ! -f "$OUT/denseft_native_unet.pt" ]; then echo "BENCH-ONLY STOP recorded; no evaluation generated"; exit 0; fi
export DENSEFT_UNET="$OUT/denseft_native_unet.pt"
for CTX in ac_short ac_native; do
  echo "=== denseft(_n) / $CTX ($DEV) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system denseft --context "$CTX" --device "$DEV" --out "$OUT/gen"
done
echo "R2 DENSEFT-N DONE: trainer + 384 eval WAVs under $OUT"
