#!/usr/bin/env python3
"""Sufficient-statistics tests (review round 2): summing per-prompt stats reproduces aggregates,
and subsample aggregation is exact. Synthetic (no model).
Run: .venv-sa3/bin/python tests/sa3/test_sufficient_stats.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "sa3"))
from size_from_stats import crit_values, removal_set


def approx(a, b, t=1e-12): return abs(a - b) <= t


def make_pp():
    # 3 prompts, 2 blocks (5,13). D_P is ratio of sums; construct so we can check by hand.
    return {
        "5":  {"f_den": 10.0, "dp_num": {"5": 2.0, "13": 1.0},
               "fb_den": 5.0, "db_num": {"5": 0.5, "13": 0.5},
               "ipt_num": {"5": [1.0]+[0.0]*7, "13": [0.5]+[0.0]*7}, "ipt_den": [4.0]+[0.0]*7, "fp_sq": [100.0]+[0.0]*7},
        "8":  {"f_den": 20.0, "dp_num": {"5": 6.0, "13": 2.0},
               "fb_den": 5.0, "db_num": {"5": 0.5, "13": 0.5},
               "ipt_num": {"5": [1.0]+[0.0]*7, "13": [0.5]+[0.0]*7}, "ipt_den": [2.0]+[0.0]*7, "fp_sq": [100.0]+[0.0]*7},
        "2":  {"f_den": 10.0, "dp_num": {"5": 2.0, "13": 3.0},
               "fb_den": 10.0, "db_num": {"5": 1.0, "13": 0.5},
               "ipt_num": {"5": [2.0]+[0.0]*7, "13": [0.0]*8}, "ipt_den": [4.0]+[0.0]*7, "fp_sq": [100.0]+[0.0]*7},
    }


def t1_full_aggregate():
    pp = make_pp(); blocks = [5, 13]
    dp = crit_values(pp, list(pp), "D_P", blocks)
    # D_P(5) = (2+6+2)/(10+20+10) = 10/40 = 0.25 ; D_P(13) = (1+2+3)/40 = 6/40 = 0.15
    ok = approx(dp[5], 0.25) and approx(dp[13], 0.15)
    db = crit_values(pp, list(pp), "D_B", blocks)
    # D_B(5) = (0.5+0.5+1)/(5+5+10)=2/20=0.10 ; D_B(13)=(0.5+0.5+0.5)/20=1.5/20=0.075
    ok = ok and approx(db[5], 0.10) and approx(db[13], 0.075)
    ipt = crit_values(pp, list(pp), "I_PT", blocks)
    # I_PT(5) pooled = (1+1+2)/(4+2+4)=4/10=0.4 ; I_PT(13)=(0.5+0.5+0)/10=0.1
    ok = ok and approx(ipt[5], 0.4) and approx(ipt[13], 0.1)
    print(f"    T1 D_P={dp} D_B={db} I_PT={ipt}")
    return ok


def t2_subsample():
    pp = make_pp(); blocks = [5, 13]
    dp = crit_values(pp, ["5", "8"], "D_P", blocks)  # (2+6)/(10+20)=8/30 ; (1+2)/30=3/30
    ok = approx(dp[5], 8/30) and approx(dp[13], 3/30)
    print(f"    T2 subsample D_P={dp}")
    return ok


def t3_additivity():
    # summing two disjoint subsamples' numerators/denominators == full aggregate numerator/denominator
    pp = make_pp(); blocks = [5, 13]
    full = crit_values(pp, list(pp), "D_P", blocks)
    # manual: recompute via split {5,8} and {2}
    num5 = sum(pp[a]["dp_num"]["5"] for a in ["5", "8"]) + sum(pp[a]["dp_num"]["5"] for a in ["2"])
    den = sum(pp[a]["f_den"] for a in ["5", "8", "2"])
    ok = approx(full[5], num5 / den)
    print(f"    T3 additive num/den split reproduces full: {ok}")
    return ok


def t4_removal_set():
    pp = make_pp(); blocks = [5, 13]
    dp = crit_values(pp, list(pp), "D_P", blocks)  # D_P(5)=0.25 > D_P(13)=0.15 -> remove 13 first
    ok = removal_set(dp, 1) == {13}
    print(f"    T4 removal_set(k=1)={removal_set(dp,1)}")
    return ok


def main():
    checks = [("T1", t1_full_aggregate), ("T2", t2_subsample), ("T3", t3_additivity), ("T4", t4_removal_set)]
    ok_all = True
    for n, f in checks:
        ok = f(); ok_all &= ok; print(f"  {n}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
