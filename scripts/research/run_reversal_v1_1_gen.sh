#!/usr/bin/env bash
# RECOVERY-REVERSAL-V1.1 generation (576 NEW WAVs = 3 systems x 96 prompts x 2 reps). GPU job ENTRY.
# DO NOT run in the Studio (CPU); this is the command a Lightning T4 Job executes. Frozen contract:
# docs/recovery_reversal_v1.md + docs/recovery_reversal_v1_1.md; manifest
# configs/research/reversal_v1_1_audiocaps_manifest.json (sha e221c0e0...).
#
# Standalone (no adapter) for all three systems. Common x_T per (ytid, replicate) across systems
# (GENERATION_SALT). Each system's 192-WAV artifact + manifest is persisted on completion, so an
# infra failure mid-run cannot erase an already-finished system.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

OUT="${OUT:-artifacts/icassp_gate0/reversal_v1_1_gen}"
DEV="${DEV:-cuda}"
PY="${PY:-.venv/bin/python}"   # torch 1.13.1+cu117 (same env as the gate0 gen/phenom jobs)

for SYSTEM in dense_ema p1_pruned_ema_reconstructed p1_recovered; do
  echo "=== generating $SYSTEM (192 WAVs) ==="
  OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_v1_gen.py \
    --system "$SYSTEM" --device "$DEV" --out "$OUT" --validate
  echo "=== $SYSTEM DONE + manifest persisted ==="
done
echo "REVERSAL-V1.1-GEN DONE: 576 WAVs + 3 manifests under $OUT"
