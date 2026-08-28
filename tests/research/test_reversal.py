#!/usr/bin/env python3
"""Regression + safety tests for RECOVERY-REVERSAL-V1 CPU preflight (reversal.py).

Protects the historical music reconstruction and the 96x2 sensitivity driver:

  T1 PARSE/ORDER      wav basename -> (prompt, replicate); non-canonical order fails loudly.
  T2 MALFORMED        missing group / wrong length / duplicate-or-missing cell all raise.
  T3 RECONSTRUCTION   (real artifacts) 64x3 pairing rebuilt; R_music == frozen -0.0941,
                      CI95 [-0.1241,-0.0646]; sign fraction 0.797. Skipped if artifacts absent.
  T4 BOOTSTRAP-DET    same per-prompt vector + seed -> identical CI (twice).
  T5 VARIANCE-DECOMP  decompose_variance recovers planted between/within components.
  T6 SIM-DETERMINISM  simulate_design is bit-identical for a fixed SeedSequence.
  T7 INFLATION        2x variance -> wider mean CI half-width AND lower P(R_AC requirement).
  T8 INTERACTION      with strongly-negative R_music, P(lower_CI95(I)>0) ~ 1 (I never binds).

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python tests/research/test_reversal.py
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np  # noqa: E402

sys.path.insert(0, os.getcwd())
from research_pruning.eval.reversal import (  # noqa: E402
    R_MUSIC_TARGET_HI, R_MUSIC_TARGET_LO, R_MUSIC_TARGET_POINT,
    decompose_variance, parse_prompt_replicate, r_music,
    reconstruct_music_grids, simulate_design)
from research_pruning.eval.cluster_bootstrap import cluster_percentile_ci  # noqa: E402

A = "artifacts/icassp_gate0"
GIN, GOUT = f"{A}/_phenom_groups_in.json", f"{A}/_phenom_groups_out.json"


def _synthetic_groups(n_prompts=64, n_reps=3, shuffle=False, drop=False):
    """Canonical (or deliberately broken) phenom in/out group pair for the two OFF arms."""
    rng = np.random.default_rng(0)
    groups_in = {"groups": []}
    groups_out = {"results": []}
    for name, shift in (("p1_recovered__off", -0.02), ("p1_pruned_ema_reconstructed__off", 0.08)):
        items, cos = [], []
        order = [(p, r) for p in range(n_prompts) for r in range(n_reps)]
        if shuffle:
            order[0], order[1] = order[1], order[0]
        for (p, r) in order:
            items.append({"caption": f"cap{p}", "wav": f"sys_p{p}_r{r}.wav"})
            cos.append(0.2 + shift + rng.normal(0, 0.01))
        if drop:
            items, cos = items[:-1], cos[:-1]
        groups_in["groups"].append({"name": name, "items": items})
        groups_out["results"].append({"name": name, "n": len(cos), "cosines": cos})
    return groups_in, groups_out


def t1_parse_order():
    ok = parse_prompt_replicate("p1_recovered_noadapter_p7_r2.wav") == (7, 2)
    gin, gout = _synthetic_groups(shuffle=True)
    raised = False
    try:
        reconstruct_music_grids(gin, gout)
    except ValueError as e:
        raised = "canonical" in str(e).lower()
    print(f"  T1 parse={ok}; non-canonical order raises={raised}")
    return ok and raised


def t2_malformed():
    results = []
    # missing group
    gin, gout = _synthetic_groups()
    gin["groups"] = [gin["groups"][0]]
    try:
        reconstruct_music_grids(gin, gout); results.append(False)
    except KeyError:
        results.append(True)
    # wrong length
    gin, gout = _synthetic_groups(drop=True)
    try:
        reconstruct_music_grids(gin, gout); results.append(False)
    except ValueError:
        results.append(True)
    ok = all(results)
    print(f"  T2 malformed (missing group, bad length) all raise={ok}")
    return ok


def t3_reconstruction():
    if not (os.path.exists(GIN) and os.path.exists(GOUT)):
        print("  T3 SKIP (persisted phenom artifacts absent)")
        return True
    recon = reconstruct_music_grids(json.load(open(GIN)), json.load(open(GOUT)))
    assert recon.paired_diff.shape == (64, 3)
    ci = r_music(recon, seed=20260826)
    point_ok = abs(ci.point - R_MUSIC_TARGET_POINT) < 1e-4
    ci_ok = abs(ci.lo - R_MUSIC_TARGET_LO) < 5e-4 and abs(ci.hi - R_MUSIC_TARGET_HI) < 5e-4
    sign_ok = abs(float((recon.prompt_mean_diff < 0).mean()) - 0.797) < 0.01
    print(f"  T3 R_music={ci.point:.4f} CI[{ci.lo:.4f},{ci.hi:.4f}] "
          f"point_ok={point_ok} ci_ok={ci_ok} sign_ok={sign_ok}")
    return point_ok and ci_ok and sign_ok


def t4_bootstrap_determinism():
    v = np.linspace(-0.2, 0.1, 64)
    a, b = cluster_percentile_ci(v, seed=20260827), cluster_percentile_ci(v, seed=20260827)
    ok = a.lo == b.lo and a.hi == b.hi and a.point == b.point
    print(f"  T4 bootstrap determinism={ok}")
    return ok


def t5_variance_decomp():
    rng = np.random.default_rng(7)
    n, k = 4000, 3
    sb, sw = 0.10, 0.13
    a = rng.normal(0, sb, size=n)
    d = 0.03 + a[:, None] + rng.normal(0, sw, size=(n, k))
    vc = decompose_variance(d)
    ok = (abs(vc.sigma2_between ** 0.5 - sb) < 0.01 and abs(vc.sigma2_within ** 0.5 - sw) < 0.005
          and abs(vc.grand_mean - 0.03) < 0.01)
    print(f"  T5 decomp sigma_b={vc.sigma2_between**0.5:.4f}(~{sb}) "
          f"sigma_w={vc.sigma2_within**0.5:.4f}(~{sw}) ok={ok}")
    return ok


def _vc_and_music():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.097, size=64)
    d = -0.0941 + a[:, None] + rng.normal(0, 0.131, size=(64, 3))
    return decompose_variance(d), d.mean(axis=1)


def t6_sim_determinism():
    vc, music = _vc_and_music()
    r1 = simulate_design(vc, music, r_ac=0.05, n_sim=300, b_boot=500,
                         seed_seq=np.random.SeedSequence(20260827))
    r2 = simulate_design(vc, music, r_ac=0.05, n_sim=300, b_boot=500,
                         seed_seq=np.random.SeedSequence(20260827))
    ok = (r1["P_Rac_requirement"] == r2["P_Rac_requirement"]
          and r1["point_mean"] == r2["point_mean"]
          and r1["mean_ci_halfwidth"] == r2["mean_ci_halfwidth"])
    print(f"  T6 simulate_design determinism={ok}")
    return ok


def t7_inflation():
    vc, music = _vc_and_music()
    lo = simulate_design(vc, music, r_ac=0.05, between_scale=1.0, within_scale=1.0,
                         n_sim=500, b_boot=600, seed_seq=np.random.SeedSequence(1))
    hi = simulate_design(vc, music, r_ac=0.05, between_scale=2.0, within_scale=2.0,
                         n_sim=500, b_boot=600, seed_seq=np.random.SeedSequence(1))
    wider = hi["mean_ci_halfwidth"] > lo["mean_ci_halfwidth"]
    weaker = hi["P_Rac_requirement"] < lo["P_Rac_requirement"]
    print(f"  T7 inflation: hw {lo['mean_ci_halfwidth']:.4f}->{hi['mean_ci_halfwidth']:.4f} "
          f"wider={wider}; power {lo['P_Rac_requirement']:.3f}->{hi['P_Rac_requirement']:.3f} "
          f"weaker={weaker}")
    return wider and weaker


def t8_interaction():
    vc, music = _vc_and_music()   # music mean ~ -0.094 (strongly negative)
    res = simulate_design(vc, music, r_ac=0.025, n_sim=500, b_boot=600,
                          seed_seq=np.random.SeedSequence(2))
    ok = res["P_lowerCI_I_gt0"] > 0.99
    print(f"  T8 interaction never binds: P(lowerCI(I)>0)={res['P_lowerCI_I_gt0']:.3f} ok={ok}")
    return ok


def main():
    checks = [t1_parse_order, t2_malformed, t3_reconstruction, t4_bootstrap_determinism,
              t5_variance_decomp, t6_sim_determinism, t7_inflation, t8_interaction]
    res = []
    for c in checks:
        res.append(c())
    ok = all(res)
    print(f"{'PASS' if ok else 'FAIL'}: {sum(res)}/{len(res)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
