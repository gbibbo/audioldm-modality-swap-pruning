#!/usr/bin/env python3
"""Control tests for the robust real-part Frechet distance (Q1 / F-eval-3 fix, CPU-only).

Every expected value is derived analytically, so these prove correctness, not just
"a number came back". No audio, no checkpoint, no audioldm_eval.

    T1 SELF-ZERO    FD(X, X) == 0 for well-conditioned features (the self-distance
                    control the screening comparison relies on).
    T2 ANALYTIC     For isotropic Gaussians N(mu_i, s_i^2 I_d) the closed form is
                    |mu1-mu2|^2 + d*(s1-s2)^2 (since (s1^2 I . s2^2 I)^{1/2} = s1 s2 I).
                    Checked on a large sample within Monte-Carlo tolerance.
    T3 SINGULAR     N < dim (rank-deficient covariance, the Cnn14-2048-from-100-clips
                    case) still returns a FINITE FD, and the diagnostics flag the
                    rank deficiency (rank1 <= N-1) that makes the value screening-only.
    T4 REALPART     A product sigma1 @ sigma2 with a negative eigenvalue is exactly the
                    condition that makes sqrtm complex; the eigenvalue formulation clamps
                    the negative eigenvalue to 0 (n_neg_clamped > 0) and returns a finite
                    FD, where the library guard raises "Imaginary component" and aborts.
    T5 SYMMETRY     FD(A, B) == FD(B, A).
    T6 GUARDS       dim mismatch and non-2-D input raise instead of returning a number.

Run: .venv/bin/python tests/research/test_frechet.py
"""
from __future__ import annotations

import sys

import numpy as np

from research_pruning.eval.frechet import frechet_distance, gaussian_frechet


def check_t1_self_zero() -> bool:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 16))
    r = frechet_distance(x, x)
    print(f"  FD(X,X) = {r.fd:.3e}  (finite={np.isfinite(r.fd)})")
    ok = np.isfinite(r.fd) and abs(r.fd) < 1e-6
    return bool(ok)


def check_t2_analytic() -> bool:
    rng = np.random.default_rng(1)
    d = 8
    N = 200_000
    mu1 = np.full(d, 0.0)
    mu2 = np.full(d, 0.5)
    s1, s2 = 1.0, 2.0
    x1 = mu1 + s1 * rng.standard_normal((N, d))
    x2 = mu2 + s2 * rng.standard_normal((N, d))
    r = frechet_distance(x1, x2)
    expected = float(np.dot(mu1 - mu2, mu1 - mu2) + d * (s1 - s2) ** 2)
    rel = abs(r.fd - expected) / expected
    print(f"  FD = {r.fd:.4f}  expected = {expected:.4f}  rel err = {rel:.3%}")
    return bool(rel < 0.02)


def check_t3_singular() -> bool:
    rng = np.random.default_rng(2)
    N, d = 100, 2048  # exactly the Cnn14-FD screening regime
    x1 = rng.standard_normal((N, d))
    x2 = 0.5 + rng.standard_normal((N, d))
    r = frechet_distance(x1, x2)
    print(f"  N={N} d={d}: FD={r.fd:.3f} finite={np.isfinite(r.fd)} "
          f"rank_deficient={r.rank_deficient} (N-1={N-1} < dim={d})")
    ok = np.isfinite(r.fd) and r.rank_deficient
    return bool(ok)


def check_t4_realpart() -> bool:
    # A negative eigenvalue in sigma1 @ sigma2 forces a genuinely complex sqrtm:
    # sqrtm(diag([-1, 1])) = diag([i, 1]). audioldm_eval raises here (imag 1.0 > 1e-3);
    # we take the real part and return a finite value.
    sigma1 = np.eye(2)
    sigma2 = np.diag([-1.0, 1.0])
    mu = np.zeros(2)
    fd, info = gaussian_frechet(mu, sigma1, mu, sigma2)
    print(f"  FD={fd:.4f} finite={np.isfinite(fd)} n_neg_clamped={info['n_neg_clamped']} "
          f"min_eig_real={info['min_eig_real']:.3f}")
    ok = np.isfinite(fd) and info["n_neg_clamped"] >= 1  # library would have raised here
    return bool(ok)


def check_t5_symmetry() -> bool:
    rng = np.random.default_rng(3)
    a = rng.standard_normal((500, 12))
    b = 0.3 + 1.5 * rng.standard_normal((400, 12))
    fab = frechet_distance(a, b).fd
    fba = frechet_distance(b, a).fd
    print(f"  FD(A,B)={fab:.5f}  FD(B,A)={fba:.5f}  |diff|={abs(fab-fba):.2e}")
    return bool(np.isfinite(fab) and abs(fab - fba) < 1e-6)


def check_t6_guards() -> bool:
    rng = np.random.default_rng(4)
    ok = True
    try:
        frechet_distance(rng.standard_normal((10, 5)), rng.standard_normal((10, 6)))
        print("  dim-mismatch did NOT raise")
        ok = False
    except ValueError:
        print("  dim-mismatch raised (ok)")
    try:
        frechet_distance(rng.standard_normal((10, 5, 2)), rng.standard_normal((10, 5)))
        print("  3-D input did NOT raise")
        ok = False
    except ValueError:
        print("  3-D input raised (ok)")
    return bool(ok)


def main() -> int:
    checks = [
        ("T1 SELF-ZERO", check_t1_self_zero),
        ("T2 ANALYTIC", check_t2_analytic),
        ("T3 SINGULAR", check_t3_singular),
        ("T4 REALPART", check_t4_realpart),
        ("T5 SYMMETRY", check_t5_symmetry),
        ("T6 GUARDS", check_t6_guards),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        results[name] = bool(fn())
    print("\n==== ROBUST FRECHET TESTS ====")
    for name, _ in checks:
        print(f"  {name:<14} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
