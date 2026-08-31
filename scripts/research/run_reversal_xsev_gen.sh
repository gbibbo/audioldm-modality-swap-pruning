#!/usr/bin/env bash
# RECOVERY-CROSS-SEVERITY-REP-1 generation (1808 NEW WAVs). GPU T4 job ENTRY. Do NOT run in the Studio.
# Frozen protocol docs/recovery_cross_severity_rep_1.md (sha 19c50cc3...); manifests
# xsev_audiocaps_manifest.json (4da90661) + xsev_music_manifest.json (f5a26fbe).
#
# Systems x contexts (each persists its own artifact+manifest on completion -> crash-resilient):
#   recovered2 / pruned2_A / pruned2_B  x  {ac_native(192), ac_short(192), music(64x3=192)}  = 1728
#   dense / dense_native (Arm-D 80)                                                            =   80
# Total 1808. EMA convention; DDIM50/g2.5/eta0/fp32/single. CRN x_T shared across systems per (ctx,ytid,rep).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-artifacts/icassp_gate0/reversal_xsev_gen}"
DEV="${DEV:-cuda}"
PY="${PY:-.venv/bin/python}"

for SYS in recovered2 pruned2_A pruned2_B; do
  for CTX in ac_native ac_short music; do
    echo "=== $SYS / $CTX ==="
    OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system "$SYS" --context "$CTX" --device "$DEV" --out "$OUT"
    echo "=== $SYS / $CTX DONE ==="
  done
done
echo "=== dense / dense_native ==="
OPENBLAS_CORETYPE=Haswell "$PY" scripts/research/reversal_xsev_gen.py --system dense --context dense_native --device "$DEV" --out "$OUT"
echo "REVERSAL-XSEV-GEN DONE: 1808 WAVs under $OUT"
