#!/usr/bin/env python3
"""Tests for the matched-null Gate A statistic (M3A, CPU-only, synthetic data).

Uses synthetic data with a KNOWN response so Gate A cannot pass by construction:

    N1 ON-CURVE      R_mod^L1 built exactly on the fitted null curve → Delta_swap ≈ 0
                     and the 95% CI contains 0.
    N2 SHIFTED +2SD  L1 shifted +2 control-SD above the curve → CI excludes 0 and the
                     standardized residual ≈ 2.
    N3 COVERAGE      bootstrap 95% CI covers the true Delta_swap on ≈95% of simulated
                     repetitions (demonstrates the CI is calibrated, not permissive).
    N4 WAV UNIT      bootstrap raises if the pool has repeated wav ids (the resampling
                     unit must be the wav, never the caption-wav entry).

Run: .venv/bin/python tests/research/test_matched_null.py
"""
from __future__ import annotations

import sys

import numpy as np

from research_pruning.diagnostics.matched_null import (
    bootstrap_delta_swap,
    fit_null_curve,
    point_estimate,
)

# Ground-truth null relationship  R_mod = A + B * D_gen. The "random-control SD"
# the Gate A statistic standardises against is the MASK-TO-MASK scatter of a mask's
# aggregate R_mod around the curve (SIGMA_MASK); per-wav noise is separate and
# small so a mask/L1 mean is precise. L1 is shifted by `l1_offset_in_sd*SIGMA_MASK`.
A_TRUE, B_TRUE = 0.30, 0.50
SIGMA_MASK = 0.02   # mask-level residual around the curve = the control SD
SIGMA_WAV = 0.005   # within-mask per-wav noise (averaged out by many wavs)
M_MASKS, W_WAVS = 20, 120


def synth(rng, l1_offset_in_sd=0.0):
    """Per-mask/per-wav random data with genuine mask-level scatter SIGMA_MASK, and
    L1 shifted by `l1_offset_in_sd` control-SD above the curve."""
    mask_dgen = rng.uniform(0.8, 2.0, size=M_MASKS)
    mask_offset = rng.normal(0, SIGMA_MASK, size=M_MASKS)   # mask-level scatter
    rand_dgen = mask_dgen[:, None] + rng.normal(0, 0.01, size=(M_MASKS, W_WAVS))
    rand_rmod = (A_TRUE + B_TRUE * rand_dgen
                 + mask_offset[:, None]
                 + rng.normal(0, SIGMA_WAV, size=(M_MASKS, W_WAVS)))
    l1_dgen = np.full(W_WAVS, 1.3) + rng.normal(0, 0.01, size=W_WAVS)
    l1_rmod = (A_TRUE + B_TRUE * l1_dgen
               + l1_offset_in_sd * SIGMA_MASK
               + rng.normal(0, SIGMA_WAV, size=W_WAVS))
    return l1_dgen, l1_rmod, rand_dgen, rand_rmod


def check_n1_on_curve():
    rng = np.random.default_rng(1)
    l1_dgen, l1_rmod, rand_dgen, rand_rmod = synth(rng, l1_offset_in_sd=0.0)
    wavs = [f"w{i}" for i in range(W_WAVS)]
    res = bootstrap_delta_swap(wavs, l1_dgen, l1_rmod, rand_dgen, rand_rmod,
                               n_boot=2000, seed=1)
    contains0 = res["ci_low"] <= 0.0 <= res["ci_high"]
    print(f"    N1 delta={res['delta_swap']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
          f"std_resid={res['standardized_residual']:.2f} fit R2={res['fit']['r2']:.3f}")
    ok = contains0 and abs(res["delta_swap"]) < 0.02
    print(f"    N1 {'ok ' if ok else 'FAIL'} on-curve → delta≈0 and CI contains 0")
    return ok


def check_n2_shifted():
    rng = np.random.default_rng(2)
    l1_dgen, l1_rmod, rand_dgen, rand_rmod = synth(rng, l1_offset_in_sd=2.0)
    wavs = [f"w{i}" for i in range(W_WAVS)]
    res = bootstrap_delta_swap(wavs, l1_dgen, l1_rmod, rand_dgen, rand_rmod,
                               n_boot=2000, seed=2)
    print(f"    N2 delta={res['delta_swap']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
          f"std_resid={res['standardized_residual']:.2f} (expect ≈2)")
    ok = res["ci_excludes_zero"] and res["ci_low"] > 0 and 1.3 < res["standardized_residual"] < 2.7
    print(f"    N2 {'ok ' if ok else 'FAIL'} +2SD shift → CI excludes 0, std residual ≈ 2")
    return ok


def check_n3_coverage():
    covered = 0
    reps = 200
    for s in range(reps):
        rng = np.random.default_rng(1000 + s)
        l1_dgen, l1_rmod, rand_dgen, rand_rmod = synth(rng, l1_offset_in_sd=0.0)
        wavs = [f"w{i}" for i in range(W_WAVS)]
        res = bootstrap_delta_swap(wavs, l1_dgen, l1_rmod, rand_dgen, rand_rmod,
                                   n_boot=400, seed=s)
        if res["ci_low"] <= 0.0 <= res["ci_high"]:  # true delta is 0
            covered += 1
    cov = covered / reps
    print(f"    N3 bootstrap 95% CI coverage of true delta=0: {cov:.3f} over {reps} reps")
    ok = 0.90 <= cov <= 0.99
    print(f"    N3 {'ok ' if ok else 'FAIL'} coverage ≈ 0.95 (CI calibrated, not permissive)")
    return ok


def check_n4_wav_unit():
    rng = np.random.default_rng(4)
    l1_dgen, l1_rmod, rand_dgen, rand_rmod = synth(rng)
    dup = [f"w{i}" for i in range(W_WAVS - 1)] + ["w0"]  # duplicate wav id
    raised = False
    try:
        bootstrap_delta_swap(dup, l1_dgen, l1_rmod, rand_dgen, rand_rmod, n_boot=10)
    except ValueError:
        raised = True
    print(f"    N4 bootstrap rejects repeated wav ids: {raised}")
    print(f"    N4 {'ok ' if raised else 'FAIL'} bootstrap unit is the wav (pseudo-replication guard)")
    return raised


def test_n1_on_curve():
    assert check_n1_on_curve()


def test_n2_shifted():
    assert check_n2_shifted()


def test_n3_coverage():
    assert check_n3_coverage()


def test_n4_wav_unit():
    assert check_n4_wav_unit()


def main() -> int:
    checks = [
        ("N1 ON-CURVE", check_n1_on_curve),
        ("N2 SHIFTED +2SD", check_n2_shifted),
        ("N3 COVERAGE", check_n3_coverage),
        ("N4 WAV UNIT", check_n4_wav_unit),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        results[name] = bool(fn())
    print("\n==== M3A MATCHED-NULL TESTS ====")
    for name, _ in checks:
        print(f"  {name:<18} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
