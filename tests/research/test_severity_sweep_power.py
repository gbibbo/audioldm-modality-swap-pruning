#!/usr/bin/env python3
"""Tests for the severity-sweep overdispersion power simulator (proposal 2026-08-21).

    W1 SHIFT-MLE     fit_common_shift recovers a known common logit shift (large n, E).
    W2 TYPE-I        at tau = 0 (homogeneous logit shift, heterogeneous p_e) the bootstrap
                     test rejects at ≈ alpha (within Monte-Carlo tolerance) — the MECHANICAL
                     heterogeneity of a bounded outcome is absorbed by the null.
    W3 MONOTONE      power increases with tau at the 50×20 mechanism-set design.
    W4 DELTA-SCALE   delta_true at tau = 0 is small but > 0 (mechanical term exists), and
                     grows with tau.

Run: .venv/bin/python tests/research/test_severity_sweep_power.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "research"))
from severity_sweep_power_sim import fit_common_shift, simulate, expit, logit, BETA_A, BETA_B  # noqa: E402


def check_w1_shift_mle():
    rng = np.random.default_rng(1)
    E, n, beta = 400, 400, 0.7
    p = np.clip(rng.beta(BETA_A, BETA_B, size=(1, E)), 0.02, 0.98)
    q = expit(logit(p) - beta)
    yb, yp = rng.binomial(n, p), rng.binomial(n, q)
    _, bhat = fit_common_shift(yb, yp, n)
    print(f"    W1 beta={beta} beta_hat={bhat[0]:.3f}")
    return abs(bhat[0] - beta) < 0.05


def check_w2_type_i():
    rng = np.random.default_rng(2)
    r = simulate(rng, n_events=50, n_prompts=20, tau=0.0, beta=0.46, a=BETA_A, b=BETA_B,
                 n_sim=600, B=99, alpha=0.05)
    print(f"    W2 FP rate at tau=0: {r['power']:.3f} (alpha=0.05)")
    return 0.02 <= r["power"] <= 0.09


def check_w3_monotone():
    rng = np.random.default_rng(3)
    pw = [simulate(rng, n_events=50, n_prompts=20, tau=t, beta=0.46, a=BETA_A, b=BETA_B,
                   n_sim=300, B=99, alpha=0.05)["power"] for t in (0.0, 0.5, 1.0)]
    print(f"    W3 power at tau=0/0.5/1.0: {pw[0]:.2f}/{pw[1]:.2f}/{pw[2]:.2f}")
    return pw[0] < pw[1] < pw[2] and pw[2] > 0.9


def check_w4_delta_scale():
    rng = np.random.default_rng(4)
    d0 = simulate(rng, n_events=50, n_prompts=20, tau=0.0, beta=1.0, a=BETA_A, b=BETA_B,
                  n_sim=200, B=19, alpha=0.05)["delta_true"]
    d1 = simulate(rng, n_events=50, n_prompts=20, tau=0.7, beta=1.0, a=BETA_A, b=BETA_B,
                  n_sim=200, B=19, alpha=0.05)["delta_true"]
    print(f"    W4 delta_true tau=0: {d0:.3f} (mechanical), tau=0.7: {d1:.3f}")
    return 0.0 < d0 < 0.10 and d1 > d0 + 0.04


def main() -> int:
    checks = [("W1", check_w1_shift_mle), ("W2", check_w2_type_i),
              ("W3", check_w3_monotone), ("W4", check_w4_delta_scale)]
    ok_all = True
    for name, fn in checks:
        ok = fn()
        ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
