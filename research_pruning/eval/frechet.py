"""Numerically robust Frechet distance with the standard real-part fix.

Background (project finding F-eval-3, `docs/m0_baseline_reproduction/eval_pipeline_closure.md`):
`audioldm_eval`'s two Frechet implementations (`metrics/fad.py` VGGish FAD,
`metrics/fid.py` Cnn14-FD) reproduce pytorch-fid's guard that **raises** when the
matrix square root of `sigma1 @ sigma2` carries an imaginary component above
`atol=1e-3`. On this data the imaginary component (~0.04) exceeds that tolerance, the
FAD path returns an `int` sentinel, and `eval.py` then crashes on `out.update(...)`,
so **both FAD and Cnn14-FD come back NaN** for every screening system.

This module implements the *standard* pytorch-fid Frechet distance but takes the
**real part** of the square root instead of raising (the documented fix), and reports
the imaginary magnitude and whether diagonal regularization was applied so the caller
can judge how noisy the value is. It is `audioldm_eval`-free (numpy + scipy only) and
lives in the tracked repo, so any number it produces is reproducible without patching a
pip package.

The high-dimensional-feature caveat is unchanged: a Cnn14 2048-dim Frechet estimated
from ~100 clips has a rank-deficient covariance (rank <= N << 2048); regularization
makes it *finite*, not *low-variance*. Such values are screening-only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

from ._blas import assert_numpy_linalg_sane  # sets OPENBLAS_CORETYPE before numpy loads

import numpy as np
from scipy import linalg


@dataclass
class FrechetResult:
    fd: float
    n1: int
    n2: int
    dim: int
    max_imag: float          # max |imag(eig)| of sigma1@sigma2 (should be ~0)
    n_neg_clamped: int       # # eigenvalues with real part < 0 clamped to 0 (the "real-part fix")
    min_eig_real: float      # most negative real eigenvalue (how far the clamp reached)
    negative_clamped: bool   # True if a tiny negative FD was clamped to 0.0
    # Honest low-sample warning: the covariance rank is at most min(N-1, dim). When
    # rank_deficient is True the FD is finite but high-variance (screening-only).
    rank_deficient: bool

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def _stats(features: np.ndarray):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (N, d); got shape {features.shape}")
    mu = features.mean(axis=0)
    # rowvar=False -> variables are columns (the feature dims)
    sigma = np.cov(features, rowvar=False)
    sigma = np.atleast_2d(sigma)
    return mu, sigma


def gaussian_frechet(mu1, sigma1, mu2, sigma2):
    """Frechet distance between two Gaussians, computed via eigenvalues (robust).

    The FD needs ``tr((C1 C2)^{1/2})``. ``C1 C2`` is a product of two symmetric PSD
    matrices, so it is similar to the symmetric PSD matrix ``C1^{1/2} C2 C1^{1/2}`` and
    its eigenvalues are real and non-negative. The trace of the principal square root
    is therefore ``sum(sqrt(lambda_i))`` over those eigenvalues. Computing it this way
    (instead of ``scipy.linalg.sqrtm(C1 @ C2)``) is numerically stable and gives an
    exact ``0`` self-distance; the "imaginary component" that makes pytorch-fid /
    audioldm_eval raise appears here as tiny negative or complex eigenvalues from
    rounding, which we take the real part of and clamp to 0 — the same fix, applied at
    the eigenvalue level. Returns (fd: float, info: dict).
    """
    mu1 = np.atleast_1d(np.asarray(mu1, dtype=np.float64))
    mu2 = np.atleast_1d(np.asarray(mu2, dtype=np.float64))
    sigma1 = np.atleast_2d(np.asarray(sigma1, dtype=np.float64))
    sigma2 = np.atleast_2d(np.asarray(sigma2, dtype=np.float64))

    if mu1.shape != mu2.shape:
        raise ValueError("mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("covariances have different dimensions")

    # Refuse to return a number if the BLAS/LAPACK stack is silently wrong (E-BLAS).
    assert_numpy_linalg_sane()

    diff = mu1 - mu2

    eig = linalg.eigvals(sigma1.dot(sigma2))
    max_imag = float(np.max(np.abs(eig.imag))) if eig.size else 0.0
    eig_real = eig.real
    neg_mask = eig_real < 0.0
    n_neg_clamped = int(np.count_nonzero(neg_mask))
    min_eig_real = float(eig_real.min()) if eig_real.size else 0.0
    tr_covmean = float(np.sum(np.sqrt(np.clip(eig_real, 0.0, None))))

    fd = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)

    negative_clamped = False
    # FD is >= 0 in exact arithmetic; a tiny negative is numerical. Clamp only tiny ones.
    if fd < 0.0 and fd > -1e-6:
        fd = 0.0
        negative_clamped = True

    info = {
        "max_imag": max_imag,
        "n_neg_clamped": n_neg_clamped,
        "min_eig_real": min_eig_real,
        "negative_clamped": negative_clamped,
    }
    return fd, info


def frechet_distance(features1: np.ndarray, features2: np.ndarray) -> FrechetResult:
    """Frechet distance between two sets of feature vectors (N1, d) and (N2, d).

    `features1` are typically the generated-audio embeddings and `features2` the
    reference embeddings, but the distance is symmetric. Uses float64 throughout and
    the real-part fix. Returns a `FrechetResult` with the value plus diagnostics.
    """
    mu1, sigma1 = _stats(features1)
    mu2, sigma2 = _stats(features2)
    if mu1.shape != mu2.shape:
        raise ValueError(
            f"feature dims differ: {mu1.shape[0]} vs {mu2.shape[0]}"
        )
    fd, info = gaussian_frechet(mu1, sigma1, mu2, sigma2)
    n1 = int(np.asarray(features1).shape[0])
    n2 = int(np.asarray(features2).shape[0])
    dim = int(mu1.shape[0])
    return FrechetResult(
        fd=fd,
        n1=n1,
        n2=n2,
        dim=dim,
        max_imag=info["max_imag"],
        n_neg_clamped=info["n_neg_clamped"],
        min_eig_real=info["min_eig_real"],
        negative_clamped=info["negative_clamped"],
        rank_deficient=bool(min(n1, n2) - 1 < dim),
    )
