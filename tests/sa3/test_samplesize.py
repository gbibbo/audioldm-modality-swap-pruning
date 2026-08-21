#!/usr/bin/env python3
"""Synthetic CPU tests for research_sa3.samplesize (protocol section 2.3).
Run: .venv-sa3/bin/python tests/sa3/test_samplesize.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import samplesize as SS


def ss1_ladders():
    ok = SS.ladder(128) == [16, 32, 64, 128]
    ok = ok and SS.ladder(100) == [16, 32, 64]          # 128 > 100 dropped
    ok = ok and SS.n_u_ladder(32) == [8, 16, 32]
    ok = ok and SS.n_u_ladder(10) == [8, 16]            # stops once rung >= max
    print(f"    SS1 ladder(128)={SS.ladder(128)} ladder(100)={SS.ladder(100)} n_u(32)={SS.n_u_ladder(32)}")
    return ok


def ss2_percentile():
    ok = abs(SS.percentile([0, 0, 0, 0, 1], 95) - 0.8) < 1e-9   # numpy-linear: idx=3.8 -> 0 + (1-0)*0.8
    ok = ok and SS.percentile([0, 0, 0, 0], 95) == 0.0
    ok = ok and SS.percentile([2], 95) == 2.0
    print(f"    SS2 p95([0,0,0,0,1])={SS.percentile([0,0,0,0,1],95)}")
    return ok


def ss3_choose_rung():
    # rung16: crit D_P k2 has a p95>0 (fails). rung32: all zero (qualifies). rung64: also zero.
    dis = {
        16: {"D_P": {2: [0, 0, 1, 0, 0], 4: [0]*5, 6: [0]*5}, "A_tan": {2: [0]*5, 4: [0]*5, 6: [0]*5}},
        32: {"D_P": {2: [0]*20, 4: [0]*20, 6: [0]*20}, "A_tan": {2: [0]*20, 4: [0]*20, 6: [0]*20}},
        64: {"D_P": {2: [0]*20, 4: [0]*20, 6: [0]*20}, "A_tan": {2: [0]*20, 4: [0]*20, 6: [0]*20}},
    }
    r = SS.choose_rung(dis)
    ok = r["n_main"] == 32 and r["qualifies"] == {16: False, 32: True, 64: True}
    # p95 of [0,0,1,0,0] with 5 pts, q95 -> idx 3.8 -> 0.0 (sorted [0,0,0,0,1]) actually 0.8>0 -> fails, good
    print(f"    SS3 n_main={r['n_main']} qualifies={r['qualifies']} p95_rung16_DP_k2={r['trace'][16]['D_P'][2]:.3f}")
    # none-qualify case
    dis2 = {16: {"D_P": {2: [1]*5, 4: [1]*5, 6: [1]*5}}}
    r2 = SS.choose_rung(dis2, ks=(2, 4, 6))
    ok = ok and r2["n_main"] is None
    print(f"    SS3b none-qualify n_main={r2['n_main']}")
    return ok


def main():
    checks = [("SS1", ss1_ladders), ("SS2", ss2_percentile), ("SS3", ss3_choose_rung)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
