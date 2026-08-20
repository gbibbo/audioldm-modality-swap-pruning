"""BLAS correctness guard — MUST be imported before numpy.

Environment defect E-BLAS (found 2026-08-20): numpy 1.23.5 bundles OpenBLAS 0.3.20
(ILP64, `numpy.libs/libopenblas64_p-r0-...3.20.so`). On the Intel Xeon Platinum 8488C
(Sapphire Rapids, AVX-512) of this Studio, OpenBLAS 0.3.20 auto-selects a
SapphireRapids/AVX-512 kernel that returns WRONG results for matmul / eigh / svd / cov
on matrices larger than ~64 dims — silently, with no error. numpy's `A @ B` disagreed
with `einsum` by ~1e3, and `eigh` of a 128x128 SPD matrix gave non-orthonormal
eigenvectors (||V^T V - I|| ~ 700). scipy bundles OpenBLAS 0.3.18 and is unaffected.

Forcing `OPENBLAS_CORETYPE=Haswell` (a correct AVX2 kernel) before OpenBLAS loads fixes
numpy completely (matmul error -> 0, eigh -> 1e-14). Haswell is chosen over the older
correct kernels (Nehalem/SandyBridge/Prescott) for speed while staying correct on this
CPU.

Import this module FIRST, before `import numpy`, so the env var is read when OpenBLAS
loads. Then call `assert_numpy_linalg_sane()` at the point of use so no wrong number can
ever be produced silently — if the kernel is still bad (e.g. numpy was imported before
this guard), it RAISES rather than returning garbage.
"""
from __future__ import annotations

import os

# Must be set before numpy imports/loads OpenBLAS. setdefault so an explicit
# environment override still wins.
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")

_CHECKED = False


def assert_numpy_linalg_sane(force: bool = False) -> None:
    """Raise if numpy's LAPACK/BLAS is producing garbage. Cheap (one 128x128 eigh)."""
    global _CHECKED
    if _CHECKED and not force:
        return
    import numpy as np

    rng = np.random.default_rng(0)
    M = rng.standard_normal((128, 128))
    M = M @ M.T
    # 1) matmul sanity vs a BLAS-free contraction
    A = rng.standard_normal((96, 48))
    B = rng.standard_normal((48, 96))
    matmul_err = float(np.linalg.norm(A @ B - np.einsum("ik,kj->ij", A, B)))
    # 2) eigh orthonormality
    w, V = np.linalg.eigh(M)
    eig_err = float(np.linalg.norm(V.T @ V - np.eye(128)))
    if matmul_err > 1e-6 or eig_err > 1e-6:
        raise RuntimeError(
            "numpy BLAS/LAPACK is producing WRONG results in this environment "
            f"(matmul_err={matmul_err:.2e}, ||V^T V - I||={eig_err:.2e}). "
            "Cause: numpy's bundled OpenBLAS 0.3.20 SapphireRapids kernel is broken on "
            "this CPU. Fix: ensure OPENBLAS_CORETYPE=Haswell is set BEFORE numpy is "
            "imported (import research_pruning.eval._blas first). See "
            "docs/environment_report.md (E-BLAS)."
        )
    _CHECKED = True
