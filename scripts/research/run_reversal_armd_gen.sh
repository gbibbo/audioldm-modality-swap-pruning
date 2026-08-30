#!/usr/bin/env bash
# OP-DURATION-DISCRIMINATOR-1 (Arm D) ALT generation (160 NEW WAVs = 2 systems x 80 ytids x 1 r0).
# GPU T4 job ENTRY. DO NOT run in the Studio (CPU). Frozen contract:
# docs/op_duration_discriminator_1.md (sha fe2be79f...); subset
# configs/research/op_duration_discriminator_1_subset.json (subset_sha256 ce5fad11...).
#
# ALT operating point: duration 10.24 s (latent_t 256), DDIM 50, guidance 2.5, eta 0, fp32, single,
# EMA. x_T reuses each ytid's V1.1 r0 generation seed (shape (1,8,256,16), NOT identical to the 3.84 s
# control). Same x_T across pruned/recovered per ytid (CRN). Each system's 80-WAV artifact + manifest
# is persisted on completion so an infra failure mid-run cannot erase a finished system.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OUT="${OUT:-artifacts/icassp_gate0/reversal_armd_gen}"
DEV="${DEV:-cuda}"
PY="${PY:-.venv/bin/python}"   # torch 1.13.1+cu117 (same env as V1.1 gen)

for SYSTEM in p1_pruned_ema_reconstructed p1_recovered; do
  echo "=== generating $SYSTEM (80 ALT WAVs @10.24s) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_armd_gen.py \
    --system "$SYSTEM" --device "$DEV" --out "$OUT"
  echo "=== $SYSTEM DONE + manifest persisted ==="
done
echo "REVERSAL-ARMD-GEN DONE: 160 ALT WAVs + 2 manifests under $OUT"
