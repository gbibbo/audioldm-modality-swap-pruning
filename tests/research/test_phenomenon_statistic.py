#!/usr/bin/env python3
"""Regression tests that PROTECT the corrected phenomenon decision statistic (prereg v5,
ledger PHENOM-STAT-D). The decision statistic is D(s)=ΔCLAP(0)-ΔCLAP(s), computed from
paired (prompt, seed) observations reduced per prompt BEFORE the prompt-cluster bootstrap.
The old F=D-E is deprecated and must never drive a gate.

    P1 IDENTITY      D == (A0-As) - (C0-Cs) prompt-by-prompt (machine precision).
    P2 PAIRED-FIRST  severity_verdict.D is EXACTLY cluster_percentile_ci on the per-prompt
                     paired vector d_pp (seeds reduced first, then bootstrap over prompts).
    P3 NO-GROUP-MEANS  under a strong prompt random effect the paired D CI is much tighter
                     than a "difference of separately aggregated group means" bootstrap that
                     breaks pairing; the implementation matches the PAIRED one. A refactor
                     into group-mean differences would make this FAIL.
    P4 FALSE-POSITIVE  directional case E<0: the deprecated F gate PASSES (manufactured
                     fragility) while the correct D gate FAILS. phenomenon must be False.
    P5 CLUSTER-UNIT  bootstrap unit is prompt (n==64); widening within-prompt seed spread at
                     fixed prompt means does not change D's CI (3 seeds stay clustered).

Run: .venv/bin/python tests/research/test_phenomenon_statistic.py
"""
from __future__ import annotations

import sys

import numpy as np

from research_pruning.eval.cluster_bootstrap import (
    BOOTSTRAP_B, BOOTSTRAP_SEED, CI_ALPHA, FRAGILITY_MARGIN, NONINF_MARGIN, SESOI,
    cluster_percentile_ci, per_prompt_paired_uplift, per_prompt_standalone, severity_verdict)

N, S = 64, 3


def _draw(seed, e_mean, dclap0_mean, dclaps_mean, noise=0.004, prompt_sd=0.0):
    """Build (C0,Cs,A0,As) arrays (N,S) with a controllable prompt random effect and the
    target per-prompt means: E=C0-Cs -> e_mean; ΔCLAP0 -> dclap0_mean; ΔCLAPs -> dclaps_mean.
    D = dclap0_mean - dclaps_mean; F_old = D - e_mean."""
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, prompt_sd, (N, 1))                       # shared prompt effect
    C0 = 0.50 + mu + rng.normal(0, noise, (N, S))
    Cs = C0 - e_mean + rng.normal(0, noise, (N, S))              # E = C0-Cs ~ e_mean
    A0 = C0 + dclap0_mean + rng.normal(0, noise, (N, S))         # ΔCLAP(0) ~ dclap0_mean
    As = Cs + dclaps_mean + rng.normal(0, noise, (N, S))         # ΔCLAP(s) ~ dclaps_mean
    return C0, Cs, A0, As


def check_p1_identity():
    C0, Cs, A0, As = _draw(1, 0.02, 0.06, 0.02, prompt_sd=0.05)
    d_pp = per_prompt_paired_uplift(A0, C0) - per_prompt_paired_uplift(As, Cs)
    rhs = (A0 - As).mean(1) - (C0 - Cs).mean(1)
    err = float(np.abs(d_pp - rhs).max())
    ok = err < 1e-12
    print(f"    P1 D==(A0-As)-(C0-Cs) max|Δ|={err:.2e}: {ok}")
    return ok


def check_p2_paired_first():
    C0, Cs, A0, As = _draw(2, 0.01, 0.06, 0.015, prompt_sd=0.05)
    v = severity_verdict("s", C0, Cs, A0, C0, As, Cs)
    d_pp = per_prompt_paired_uplift(A0, C0) - per_prompt_paired_uplift(As, Cs)
    ref = cluster_percentile_ci(d_pp)
    ok = (v.D.point == ref.point and v.D.lo == ref.lo and v.D.hi == ref.hi
          and v.D.n == N)   # n==64 prompts, NOT 192
    print(f"    P2 D matches paired-per-prompt bootstrap exactly, n={v.D.n}: {ok}")
    return ok


def _group_mean_diff_ci(C0, Cs, A0, As, b=BOOTSTRAP_B, seed=BOOTSTRAP_SEED, alpha=CI_ALPHA):
    """WRONG construction: bootstrap each group's prompt-mean INDEPENDENTLY (pairing broken),
    then D_boot = (mean A0 - mean As) - (mean C0 - mean Cs). This is what a refactor into a
    'difference of separately aggregated group means' would produce."""
    g = [np.asarray(x, float).mean(1) for x in (A0, As, C0, Cs)]   # per-prompt group means
    rng = np.random.default_rng(seed)
    boots = np.empty(b)
    for i in range(b):
        idx = [rng.integers(0, N, N) for _ in range(4)]            # 4 INDEPENDENT resamples
        mA0, mAs, mC0, mCs = (g[j][idx[j]].mean() for j in range(4))
        boots[i] = (mA0 - mAs) - (mC0 - mCs)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def check_p3_no_group_means():
    # Strong prompt random effect: paired D cancels it; independent-group bootstrap does not.
    C0, Cs, A0, As = _draw(3, 0.01, 0.06, 0.02, noise=0.004, prompt_sd=0.15)
    v = severity_verdict("s", C0, Cs, A0, C0, As, Cs)
    paired_w = v.D.hi - v.D.lo
    wlo, whi = _group_mean_diff_ci(C0, Cs, A0, As)
    wrong_w = whi - wlo
    d_pp = per_prompt_paired_uplift(A0, C0) - per_prompt_paired_uplift(As, Cs)
    ref = cluster_percentile_ci(d_pp)
    matches_paired = (v.D.lo == ref.lo and v.D.hi == ref.hi)
    much_tighter = paired_w < 0.5 * wrong_w
    ok = matches_paired and much_tighter
    print(f"    P3 paired width={paired_w:.4f} << group-mean-diff width={wrong_w:.4f} "
          f"(ratio {paired_w/wrong_w:.2f}); impl==paired {matches_paired}: {ok}")
    return ok


def check_p4_false_positive():
    # E<0 (compressed standalone HIGHER than dense): the deprecated F manufactures a PASS.
    # e_mean=-0.05 -> E=-0.05 ; dclap0=0.06, dclaps=0.05 -> D=0.01 (< SESOI) ; F_old=0.06.
    C0, Cs, A0, As = _draw(4, -0.05, 0.06, 0.05, noise=0.003, prompt_sd=0.0)
    v = severity_verdict("s", C0, Cs, A0, C0, As, Cs)
    d_gate = (v.D.point >= FRAGILITY_MARGIN) and (v.D.lo > 0.0)            # correct: FAIL
    f_gate = (v.F_deprecated.point >= FRAGILITY_MARGIN) and (v.F_deprecated.lo > 0.0)  # old: PASS
    e_neg = v.E.point < 0.0
    standalone_pres = v.standalone_preserved                              # E.hi <= 0.025 (true, E<0)
    ok = (v.E.point < 0 and (not d_gate) and f_gate and standalone_pres
          and (not v.phenomenon) and (not v.differential_fragility) and e_neg)
    print(f"    P4 E.pt={v.E.point:.3f}(<0) D.pt={v.D.point:.3f}(gate {d_gate}) "
          f"F_old.pt={v.F_deprecated.point:.3f}(gate {f_gate}) -> phenomenon={v.phenomenon}: {ok}")
    return ok


def check_p5_cluster_unit():
    base = np.random.default_rng(5).normal(0.02, 0.10, N)     # prompt means for D
    # narrow within-prompt seed spread
    rng = np.random.default_rng(6)
    C0 = 0.50 + rng.normal(0, 0.001, (N, S))
    A0n = C0 + base[:, None] + rng.normal(0, 0.001, (N, S))
    As = C0.copy() + rng.normal(0, 0.001, (N, S))             # ΔCLAP(s)~0 -> D~base
    v_narrow = severity_verdict("s", C0, C0, A0n, C0, As, C0)
    # widen within-prompt seed spread at the SAME prompt means
    A0w = C0 + base[:, None] + rng.normal(0, 0.05, (N, S))
    Asw = C0.copy() + rng.normal(0, 0.05, (N, S))
    # re-center each prompt's seed mean back to the narrow target so prompt means are fixed
    A0w += (A0n.mean(1) - A0w.mean(1))[:, None]
    Asw += (As.mean(1) - Asw.mean(1))[:, None]
    v_wide = severity_verdict("s", C0, C0, A0w, C0, Asw, C0)
    same_ci = (abs(v_narrow.D.lo - v_wide.D.lo) < 1e-9 and abs(v_narrow.D.hi - v_wide.D.hi) < 1e-9)
    ok = v_narrow.D.n == N and same_ci
    print(f"    P5 n={v_narrow.D.n}; D CI invariant to within-prompt seed spread "
          f"(Δlo={abs(v_narrow.D.lo-v_wide.D.lo):.2e}): {ok}")
    return ok


def main():
    checks = [check_p1_identity, check_p2_paired_first, check_p3_no_group_means,
              check_p4_false_positive, check_p5_cluster_unit]
    res = []
    for c in checks:
        print(f"  {c.__name__}")
        res.append(c())
    ok = all(res)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(res)}/{len(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
