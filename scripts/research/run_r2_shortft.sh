#!/usr/bin/env bash
# REVIEWER2-FOLLOWUP job r2-shortft (E3): 200-step benchmark self-gate -> 20 000-step full fine-tune of pruned2_A at
# 3.84 s -> CRN evaluation WAVs of the resulting checkpoint (192 AudioCaps prompts x {3.84, 10.24} s).
# Protocol: docs/reviewer2_followup.md §8.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/r2_shortft}"; DEV="${DEV:-cuda}"; PY="${PY:-.venv/bin/python}"
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/e3_shortft_trainer.py --out "$OUT" --resume
if [ ! -f "$OUT/shortft_unet.pt" ]; then echo "BENCH-ONLY STOP recorded; no evaluation generated"; exit 0; fi
export SHORTFT_UNET="$OUT/shortft_unet.pt"
for CTX in ac_short ac_native; do
  echo "=== shortft / $CTX ($DEV) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system shortft --context "$CTX" --device "$DEV" --out "$OUT/gen"
done
echo "R2 SHORTFT DONE: trainer + 384 eval WAVs under $OUT"
