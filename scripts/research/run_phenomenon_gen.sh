#!/usr/bin/env bash
# Phenomenon-falsifier downstream generation (768 NEW WAVs). GPU job ENTRY script.
# DO NOT run this in the Studio; it is the command a Lightning T4 Job executes. The 384 frozen
# dense Gate-0 WAVs are NOT regenerated — only the 4 compressed/recovered systems below.
#
# PRIMARY endpoint FIRST (operational risk management — NOT optional stopping; both backbones are
# pre-authorized and BOTH are generated irrespective of the result):
#   p1_recovered                 {off, on}   (sliced adapter)   = 64x3x2 = 384   [PRIMARY]
#   p1_pruned_ema_reconstructed  {off, on}   (sliced adapter)   = 64x3x2 = 384   [SECONDARY/mechanistic]
#                                                             total = 768 new WAVs
#
# Each backbone's 384-WAV artifact + manifest is persisted immediately on completion (the generator
# writes its manifest at the end of each invocation), so an infra failure during the secondary run
# cannot erase the already-completed PRIMARY endpoint.
#
# Every compressed/recovered sample reuses the SAME frozen x_T tied to (ytid, replicate) — the
# generator's make_x_T is a pure function of (ytid, replicate) (common-random-number design),
# identical across dense and every downstream system. --validate self-checks each manifest with
# the shared parametric validator; provenance (git/env/GPU/source-ckpt/dense+sliced adapter SHAs)
# is stamped by the generator.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

SLICED="artifacts/icassp_gate0/sliced_adapter/gate0_sliced_adapter_1_2_3_1.pt"
OUT="artifacts/icassp_gate0/gen_phenomenon"
DEV="${DEV:-cuda}"

for BACKBONE in p1_recovered p1_pruned_ema_reconstructed; do   # PRIMARY first, then SECONDARY
  echo "=== generating $BACKBONE (both) ==="
  OPENBLAS_CORETYPE=Haswell python scripts/research/gate0_generator.py \
    --backbone "$BACKBONE" \
    --adapter "$SLICED" \
    --adapter-mode both \
    --device "$DEV" \
    --out "$OUT" \
    --validate
  echo "=== $BACKBONE DONE + manifest persisted ==="
done
echo "PHENOMENON-GEN DONE: 768 WAVs + 2 manifests under $OUT"
