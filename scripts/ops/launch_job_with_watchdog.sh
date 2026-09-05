#!/usr/bin/env bash
# Launch ONE Lightning T4 job from the committed tree and attach the external cost watchdog (nohup, cloudspace python).
# Usage: scripts/ops/launch_job_with_watchdog.sh <job-name> <max-cost-cr> <max-minutes> <command...>
# Refuses to launch from a dirty tree (the job snapshots the working tree; the SHA must describe it).
set -euo pipefail
NAME="$1"; MAXCOST="$2"; MAXMIN="$3"; shift 3; CMD="$*"
cd "$(dirname "$0")/../.."
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "REFUSE: dirty tree"; git status --short | head; exit 2; fi
SHA="$(git rev-parse --short HEAD)"
echo "launching $NAME from $SHA | cap $MAXCOST cr / $MAXMIN min | $CMD"
lightning job run --name "$NAME" --machine T4 --studio gabriel-allgd-deploy-model-devbox --teamspace general \
  --org independentaudioresearch --command "cd audioldm-modality-swap-pruning && $CMD"
mkdir -p artifacts/icassp_gate0
setsid nohup /home/zeus/miniconda3/envs/cloudspace/bin/python scripts/sa3/job_watchdog.py --name "$NAME" \
  --max-cost "$MAXCOST" --max-minutes "$MAXMIN" --poll-seconds 60 > "artifacts/icassp_gate0/${NAME}_watchdog.log" 2>&1 &
echo "watchdog attached -> artifacts/icassp_gate0/${NAME}_watchdog.log"
