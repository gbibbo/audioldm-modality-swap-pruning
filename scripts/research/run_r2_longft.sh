#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP-EXT job r2-longft (item 1): 200-step benchmark self-gate -> 20 000-step full fine-tune of
# pruned2_A at 10.24 s (latent 256, batch 2) -> CRN evaluation WAVs (192 AudioCaps prompts x {3.84, 10.24} s).
# The symmetric control of E3 (which trained at 3.84 s). Protocol: docs/reviewer2_followup_ext.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_longft}"; DEV="${DEV:-cuda}"; PY="${PY:-.venv/bin/python}"
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/e3_shortft_trainer.py --out "$OUT" \
  --backbone pruned2_A --duration 10.24 --latent-t 256 --batch 2 --accum 1 \
  --save-name longft_unet.pt --cap-cr 4.8 --mid-step 7500 --resume
if [ ! -f "$OUT/longft_unet.pt" ]; then echo "BENCH-ONLY STOP recorded; no evaluation generated"; exit 0; fi
export LONGFT_UNET="$OUT/longft_unet.pt"
for CTX in ac_short ac_native; do
  echo "=== longft / $CTX ($DEV) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system longft --context "$CTX" --device "$DEV" --out "$OUT/gen"
done
echo "R2 LONGFT DONE: trainer + 384 eval WAVs under $OUT"
