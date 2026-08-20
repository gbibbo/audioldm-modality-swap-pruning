#!/usr/bin/env bash
# E-BLAS fix installer (idempotent). See docs/environment_report.md (E-BLAS).
#
# numpy 1.23.5 bundles OpenBLAS 0.3.20, whose SapphireRapids/AVX-512 kernel returns WRONG
# results on this Studio's Xeon 8488C (matmul/eigh/svd/cov corrupt for n > ~64, no error
# raised). This writes a startup .pth into the venv site-packages that forces a correct
# kernel (OPENBLAS_CORETYPE=Haswell) BEFORE numpy loads, then verifies numpy is sane.
#
# .venv is gitignored, so RE-RUN THIS AFTER ANY .venv REBUILD.
#
#   bash scripts/research/install_blas_fix.sh
set -euo pipefail

VENV_PY="${VENV_PY:-.venv/bin/python}"
if [[ ! -x "$VENV_PY" ]]; then
  echo "FATAL: $VENV_PY not found. Build the env first (see docs/environment_report.md)." >&2
  exit 2
fi

SP="$($VENV_PY -c 'import site; print(site.getsitepackages()[0])')"
PTH="$SP/aaa_openblas_coretype_fix.pth"
printf "import os; os.environ.setdefault('OPENBLAS_CORETYPE', 'Haswell')\n" > "$PTH"
echo "wrote $PTH"

# Verify: with the .pth in place, a fresh interpreter must have correct numpy linalg.
"$VENV_PY" - <<'PY'
import os, sys
import numpy as np
ct = os.environ.get("OPENBLAS_CORETYPE")
rng = np.random.default_rng(0)
A = rng.standard_normal((128, 64)); B = rng.standard_normal((64, 128))
matmul_err = float(np.linalg.norm(A @ B - np.einsum("ik,kj->ij", A, B)))
M = rng.standard_normal((128, 128)); M = M @ M.T
w, V = np.linalg.eigh(M)
eig_err = float(np.linalg.norm(V.T @ V - np.eye(128)))
print(f"OPENBLAS_CORETYPE={ct}  matmul_err={matmul_err:.2e}  ||VtV-I||={eig_err:.2e}")
if matmul_err > 1e-6 or eig_err > 1e-6:
    print("FAIL: numpy BLAS still broken", file=sys.stderr); sys.exit(1)
print("OK: numpy BLAS/LAPACK is sane")
PY
echo "E-BLAS fix installed and verified."
