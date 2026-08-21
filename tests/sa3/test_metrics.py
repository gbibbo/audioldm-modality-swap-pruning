#!/usr/bin/env python3
"""Synthetic CPU tests for research_sa3.metrics / greedy (protocol section 3, 4).
Run: .venv-sa3/bin/python tests/sa3/test_metrics.py"""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import metrics as M
from research_sa3 import greedy as G


def approx(a, b, tol=1e-9): return abs(a - b) <= tol


def m1_block_damage():
    d = M.block_damage({0: 2.0, 1: 5.0}, den=10.0)
    ok = approx(d[0], 0.2) and approx(d[1], 0.5)
    print(f"    M1 D={d}")
    return ok


def m2_i_pt():
    # 2 blocks, 3 levels. den=[4,2,1], fp_sq=[100,100,0.5], eta=[1e-3,1e-3,1e-3]
    # level 2: den/fp = 1/0.5 = 2.0 >= eta -> kept. Make an excluded case: set den[2] tiny.
    num = {0: [1.0, 1.0, 1.0], 1: [2.0, 0.0, 0.0]}
    den = [4.0, 2.0, 1.0]
    fp = [100.0, 100.0, 100.0]
    eta = [1e-3, 1e-3, 0.02]   # level2: den/fp=1/100=0.01 < 0.02 -> EXCLUDED
    r = M.i_pt(num, den, fp, eta)
    exc = r["excluded_levels"] == [2]
    # pooled for block0 over kept levels {0,1}: (1+1)/(4+2)=2/6
    pooled0 = approx(r["per_block"][0]["pooled"], 2.0 / 6.0)
    # per-level block0 level0 = 1/4
    pl0 = approx(r["per_block"][0]["per_level"][0], 0.25)
    # pooled is ratio-of-sums, NOT mean-of-ratios: mean of ratios for b0 kept = (0.25+0.5)/2=0.375 != 0.3333
    not_mean = not approx(r["per_block"][0]["pooled"], 0.375, tol=1e-6)
    print(f"    M2 excluded={r['excluded_levels']} pooled0={r['per_block'][0]['pooled']:.4f} (ratio-of-sums)")
    return exc and pooled0 and pl0 and not_mean


def m3_adapt():
    a = M.adaptability(e_carry={0: 3.0}, e_full=6.0, e_int_num={0: 1.0}, e_int_den={0: 4.0}, e_tan_num={0: 2.0})
    ok = approx(a[0]["A_carry"], 0.5) and approx(a[0]["A_int"], 0.25) and approx(a[0]["A_tan"], 1.0/3.0)
    eco = M.a_eco({0: 1.0, 1: 3.0}, den=6.0)
    ok = ok and approx(eco[1], 0.5)
    print(f"    M3 A={a[0]} eco1={eco[1]:.3f}")
    return ok


def m4_lin_prec():
    ok = approx(M.linearity_ratio(2.0, 1.0), 2.0)
    ok = ok and M.precision_ok(df_sq=1.0, fp_sq=100.0, eta=1e-4, factor=10) is True     # 0.01 >= 1e-3
    ok = ok and M.precision_ok(df_sq=1.0, fp_sq=100.0, eta=1e-2, factor=10) is False    # 0.01 < 0.1
    print(f"    M4 lin/prec ok={ok}")
    return ok


def m5_greedy():
    # score(set) = sum of per-block base costs + pairwise interaction. Lower better.
    base = {0: 1.0, 1: 5.0, 2: 2.0, 3: 9.0}
    inter = {(0, 2): -0.5}
    def score(s):
        s = set(s)
        v = sum(base[g] for g in s)
        for (a, b), w in inter.items():
            if a in s and b in s:
                v += w
        return v
    r = G.greedy_path(depth=4, k_max=3, score_fn=score)
    # k=1 picks block0 (cost1). k=2 adds block2 (1+2-0.5=2.5 < 1+5=6) => {0,2}. k=3 adds block1 => {0,1,2}
    ok = r["sets"][1] == {0} and r["sets"][2] == {0, 2} and r["sets"][3] == {0, 1, 2}
    ok = ok and r["n_evals"] == 4 + 3 + 2
    gap = G.additivity_gap(frozenset({0, 2}), score)   # score({0,2}) - score({0}) - score({2}) = 2.5 - 1 - 2 = -0.5
    ok = ok and approx(gap, -0.5)
    div = G.set_divergence({0, 2, 4}, {0, 2, 5})       # sym={4,5} -> 1
    ok = ok and div == 1
    print(f"    M5 sets={r['sets']} n_evals={r['n_evals']} gap={gap} div={div}")
    return ok


def main():
    checks = [("M1", m1_block_damage), ("M2", m2_i_pt), ("M3", m3_adapt), ("M4", m4_lin_prec), ("M5", m5_greedy)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
