#!/usr/bin/env python3
"""Tests for the frozen prompt-level cluster bootstrap (DECISION-V4-09).

    B1 CONSTANT     zero-variance per-prompt values -> CI collapses to the constant.
    B2 DETERMINISM  same seed -> identical CI; different seed -> different but close.
    B3 CLUSTERING   seeds are averaged within prompt, NOT treated as independent:
                    widening within-prompt seed spread (fixed prompt means) does NOT
                    change the per-prompt reduction, hence not the CI.
    B4 COVERAGE     for a known mean the 95% CI covers it; nominal coverage ~0.95 over
                    many synthetic draws (sanity band 0.90-0.98).
    B5 GATE0        PASS when uplift strong+tight; FAIL when point<SESOI; FAIL when
                    lower-CI touches 0.
    B6 SEVERITY     dual gate: phenomenon TRUE only when standalone preserved AND the
                    DECISION statistic D clears (point>=margin AND lower-CI>0);
                    standalone-broken -> descriptive only (False). D, not the deprecated F.
    B7 SHAPE-GUARDS mismatched shapes / 1-D inputs raise.

Run: .venv/bin/python tests/research/test_cluster_bootstrap.py
"""
from __future__ import annotations

import sys

import numpy as np

from research_pruning.eval.cluster_bootstrap import (
    BOOTSTRAP_SEED, FRAGILITY_MARGIN, NONINF_MARGIN, SESOI, cluster_percentile_ci,
    gate0_verdict, per_prompt_paired_uplift, severity_verdict)

N, S = 64, 3


def check_b1_constant():
    ci = cluster_percentile_ci(np.full(N, 0.041))
    ok = abs(ci.point - 0.041) < 1e-12 and abs(ci.lo - 0.041) < 1e-9 and abs(ci.hi - 0.041) < 1e-9
    print(f"    B1 constant -> point={ci.point:.4f} lo={ci.lo:.4f} hi={ci.hi:.4f}: {ok}")
    return ok


def check_b2_determinism():
    rng = np.random.default_rng(0)
    v = rng.normal(0.03, 0.05, N)
    a = cluster_percentile_ci(v, seed=BOOTSTRAP_SEED)
    b = cluster_percentile_ci(v, seed=BOOTSTRAP_SEED)
    c = cluster_percentile_ci(v, seed=BOOTSTRAP_SEED + 1)
    same = (a.lo == b.lo) and (a.hi == b.hi)
    diff = (a.lo != c.lo) or (a.hi != c.hi)
    close = abs(a.lo - c.lo) < 0.02 and abs(a.hi - c.hi) < 0.02
    print(f"    B2 same-seed identical {same}; diff-seed differs {diff} & close {close}")
    return same and diff and close


def check_b3_clustering():
    # Positive intra-cluster correlation: large between-prompt spread, tiny within-prompt
    # seed noise (3 seeds nearly equal within a prompt). Treating the 192 clips as
    # independent (flat) UNDERESTIMATES uncertainty vs the correct prompt-clustered CI.
    rng = np.random.default_rng(1)
    prompt_means = rng.normal(0.03, 0.10, N)              # strong between-prompt spread
    scores = prompt_means[:, None] + rng.normal(0, 0.002, (N, S))  # seeds ~identical within prompt
    ci_cluster = cluster_percentile_ci(scores.mean(axis=1))       # correct: n=64 clusters
    flat_wrong = cluster_percentile_ci(scores.ravel())           # misuse: n=192 "independent"
    cluster_wider = (ci_cluster.hi - ci_cluster.lo) > (flat_wrong.hi - flat_wrong.lo)
    ok = ci_cluster.n == N and flat_wrong.n == N * S and cluster_wider
    print(f"    B3 cluster n={ci_cluster.n} width={ci_cluster.hi-ci_cluster.lo:.4f} vs "
          f"flat n={flat_wrong.n} width={flat_wrong.hi-flat_wrong.lo:.4f}; "
          f"clustered correctly wider {cluster_wider}: {ok}")
    return ok


def check_b4_coverage():
    true_mean = 0.03
    hits = 0
    trials = 300
    for t in range(trials):
        rng = np.random.default_rng(1000 + t)
        v = rng.normal(true_mean, 0.06, N)
        ci = cluster_percentile_ci(v, b=2000, seed=BOOTSTRAP_SEED + t)
        if ci.lo <= true_mean <= ci.hi:
            hits += 1
    cov = hits / trials
    ok = 0.90 <= cov <= 0.985
    print(f"    B4 empirical 95% coverage = {cov:.3f} (band 0.90-0.985): {ok}")
    return ok


def check_b5_gate0():
    rng = np.random.default_rng(2)
    base = rng.normal(0.60, 0.05, (N, S))
    # strong tight uplift ~0.05 -> PASS
    strong = base + 0.05 + rng.normal(0, 0.005, (N, S))
    v_pass = gate0_verdict(strong, base)
    # weak uplift ~0.01 < SESOI -> FAIL on point
    weak = base + 0.01 + rng.normal(0, 0.005, (N, S))
    v_weak = gate0_verdict(weak, base)
    # uplift ~0.03 but very noisy so lower-CI <= 0 -> FAIL on CI
    noisy = base + 0.03 + rng.normal(0, 0.20, (N, S))
    v_noisy = gate0_verdict(noisy, base)
    ok = v_pass.passed and (not v_weak.passed) and (v_weak.delta_clap.point < SESOI) \
        and (not v_noisy.passed) and (v_noisy.delta_clap.lo <= 0)
    print(f"    B5 gate0 pass={v_pass.passed} weak={v_weak.passed}"
          f"(pt {v_weak.delta_clap.point:.3f}) noisy={v_noisy.passed}"
          f"(lo {v_noisy.delta_clap.lo:.3f}): {ok}")
    return ok


def check_b6_severity():
    rng = np.random.default_rng(3)
    base_d = rng.normal(0.60, 0.04, (N, S))
    adpt_d = base_d + 0.05 + rng.normal(0, 0.004, (N, S))    # dense uplift 0.05

    # CASE A: phenomenon — standalone preserved (E~0.01), adapter uplift collapses (D large)
    standalone_d = rng.normal(0.62, 0.03, (N, S))
    standalone_s = standalone_d - 0.01 + rng.normal(0, 0.003, (N, S))  # E ~ 0.01
    base_s = base_d.copy()
    adpt_s = base_s + 0.0 + rng.normal(0, 0.004, (N, S))     # uplift collapses -> D~0.05
    va = severity_verdict("s", standalone_d, standalone_s, adpt_d, base_d, adpt_s, base_s)

    # CASE B: generic capacity loss — standalone ALSO drops a lot (E~0.06 > margin)
    standalone_s2 = standalone_d - 0.06 + rng.normal(0, 0.003, (N, S))  # E ~ 0.06 fails non-inf
    vb = severity_verdict("s", standalone_d, standalone_s2, adpt_d, base_d, adpt_s, base_s)

    ok = (va.phenomenon and va.standalone_preserved and va.differential_fragility
          and (va.D.point >= FRAGILITY_MARGIN) and (va.D.lo > 0.0)
          and (not vb.phenomenon) and (not vb.standalone_preserved))
    print(f"    B6 A phenomenon={va.phenomenon} (E.hi={va.E.hi:.3f}<= {NONINF_MARGIN}, "
          f"D.pt={va.D.point:.3f}>= {FRAGILITY_MARGIN}, D.lo={va.D.lo:.3f}); "
          f"B phenomenon={vb.phenomenon} (E.hi={vb.E.hi:.3f}): {ok}")
    return ok


def check_b7_guards():
    bad = 0
    for fn, args in [
        (per_prompt_paired_uplift, (np.zeros((N, S)), np.zeros((N, S - 1)))),
        (per_prompt_paired_uplift, (np.zeros(N), np.zeros(N))),
        (cluster_percentile_ci, (np.zeros((N, S)),)),
    ]:
        try:
            fn(*args)
        except ValueError:
            bad += 1
    print(f"    B7 shape guards raised: {bad}/3")
    return bad == 3


def check_b8_prereg_consistency():
    # The frozen prereg YAML and the code constants must not drift apart.
    import os
    import yaml
    from research_pruning.eval import cluster_bootstrap as cb
    path = os.path.join(os.path.dirname(__file__), "..", "..",
                        "configs", "research", "icassp_gate0_prereg.yaml")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    checks = {
        "B": cfg["ci"]["B"] == cb.BOOTSTRAP_B,
        "seed": cfg["ci"]["seed"] == cb.BOOTSTRAP_SEED,
        "n_prompts": cfg["battery"]["n_prompts"] == cb.N_PROMPTS == 64,
        "n_seeds": cfg["battery"]["n_seeds"] == cb.N_SEEDS == 3,
        "SESOI": abs(cfg["gate0"]["SESOI"] - cb.SESOI) < 1e-12,
        "noninf": abs(0.025 - cb.NONINF_MARGIN) < 1e-12,
        "frag": abs(0.025 - cb.FRAGILITY_MARGIN) < 1e-12,
    }
    ok = all(checks.values())
    print(f"    B8 prereg YAML == code constants: {checks} -> {ok}")
    return ok


def main():
    checks = [check_b1_constant, check_b2_determinism, check_b3_clustering,
              check_b4_coverage, check_b5_gate0, check_b6_severity, check_b7_guards,
              check_b8_prereg_consistency]
    res = []
    for c in checks:
        print(f"  {c.__name__}")
        res.append(c())
    ok = all(res)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(res)}/{len(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
