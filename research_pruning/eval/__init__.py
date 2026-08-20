"""Reproducible, in-repo evaluation helpers that do not depend on audioldm_eval's
numerically fragile Frechet path (F-eval-3).

`_blas` is imported FIRST (before numpy) so the OpenBLAS-kernel correctness guard for
environment defect E-BLAS is in force for everything under this package. See
`frechet.py` and `_blas.py`."""
from . import _blas  # noqa: F401  (must precede any numpy import)
from .frechet import frechet_distance, gaussian_frechet
from ._blas import assert_numpy_linalg_sane

__all__ = ["frechet_distance", "gaussian_frechet", "assert_numpy_linalg_sane"]
