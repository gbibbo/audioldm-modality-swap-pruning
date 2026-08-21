#!/usr/bin/env python3
"""Synthetic CPU tests for research_sa3.aeco_predict (rc1 decision core, protocol §6 + §3).
Run: .venv-sa3/bin/python tests/sa3/test_aeco_predict.py  (pure Python; no torch, no model)."""
import os, sys, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import aeco_predict as AP


def approx(a, b, tol=1e-9): return abs(a - b) <= tol


def a1_helpers():
    # lowest_k = the k least-damaging (smallest score) blocks, tie-break by id
    scores = {0: 0.9, 1: 0.1, 2: 0.5, 3: 0.1, 4: 0.7}
    ok = AP.lowest_k(scores, 2) == {1, 3}          # two smallest (0.1 ties -> ids 1,3)
    ok = ok and AP.lowest_k(scores, 3) == {1, 3, 2}
    ok = ok and AP.set_disagreement({0, 1, 2}, {0, 1, 3}) == 1
    ok = ok and AP.set_disagreement({0, 1, 2}, {3, 4, 5}) == 3
    # perfect rank agreement -> spearman 1.0; reversed -> -1.0
    ok = ok and approx(AP.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    ok = ok and approx(AP.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)
    # unequal-size set-disagreement must raise
    try:
        AP.set_disagreement({0, 1}, {0, 1, 2}); raised = False
    except ValueError:
        raised = True
    ok = ok and raised
    print(f"    A1 lowest_k/disagreement/spearman ok={ok}")
    return ok


def a2_prediction_verdict():
    ok = AP.prediction_verdict(delta_atan=0, delta_dp=2, floor=0) == "CONFIRM"     # A in, D out
    ok = ok and AP.prediction_verdict(delta_atan=2, delta_dp=0, floor=0) == "CONTRADICT"
    ok = ok and AP.prediction_verdict(delta_atan=0, delta_dp=0, floor=0) == "AMBIGUOUS"  # both in
    ok = ok and AP.prediction_verdict(delta_atan=3, delta_dp=3, floor=0) == "AMBIGUOUS"  # both out
    ok = ok and AP.prediction_verdict(delta_atan=1, delta_dp=2, floor=1) == "CONFIRM"    # floor absorbs A only
    print(f"    A2 prediction_verdict ok={ok}")
    return ok


def a3_prediction_check_confirm():
    # Construct scores so at k=6 A_tan's tail == A_eco's tail (delta 0) but D_P's tail differs (delta>0).
    # 8 blocks. A_eco least-removable tail (lowest 6) chosen; A_tan matches it exactly; D_P shifts 2.
    a_eco = {g: float(g) for g in range(8)}          # lowest-6 = {0,1,2,3,4,5}
    a_tan = {g: float(g) for g in range(8)}          # identical -> delta_atan(6)=0
    d_p = {0: 9, 1: 9, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}  # lowest-6 = {2,3,4,5,6,7} -> delta_dp(6)=2
    r = AP.prediction_check(a_eco, a_tan, d_p, floor_by_k={2: 0, 4: 0, 6: 0}, k_primary=6)
    ok = r["verdict"] == "CONFIRM"
    ok = ok and r["per_k"][6]["delta_atan_eco"] == 0 and r["per_k"][6]["delta_dp_eco"] == 2
    ok = ok and r["spearman_atan_eco"] == 1.0                      # A_tan perfectly ranks A_eco
    ok = ok and r["spearman_corroborates"] is True
    print(f"    A3 check verdict={r['verdict']} dA6={r['per_k'][6]['delta_atan_eco']} "
          f"dD6={r['per_k'][6]['delta_dp_eco']} rhoA={r['spearman_atan_eco']:.2f}")
    return ok


def a4_control_pass_even_when_b_not_top1():
    # THE rc1 CORRECTION: b passes even though it is NOT the top-1 (least removable) A_eco block.
    # Removing host block 6 deletes the adapter -> A_eco(6)~1 and uplift collapses; some external
    # removal (say block 9) still shows uplift. Another block g=9 has HIGHER A_eco than 6, so b is
    # not top-1 -- old rule would (wrongly) STOP; rc1 rule PASSES.
    r = AP.control_localization_verdict(
        block_b=6,
        a_eco_b=0.97,                                  # ~1 sanity ok
        dT_post_minus_b=0.005,                         # uplift ~0 after removing 6
        dT_external={9: 0.20, 3: 0.15, 11: 0.18},      # externals retain measurable uplift
    )
    ok = r["pass"] is True and r["verdict"] == "PASS"
    ok = ok and r["cond_sanity_A_eco_b_near_1"] and r["cond_uplift_collapse_near_0"] and r["cond_external_uplift_observable"]
    print(f"    A4 control PASS (b not top-1) verdict={r['verdict']} extmax={r['external_uplift_max']}")
    return ok


def a5_control_stop_cases():
    # STOP if the adapter does NOT vanish when its host block is removed (uplift survives at post^-b)
    r1 = AP.control_localization_verdict(6, a_eco_b=0.4, dT_post_minus_b=0.19,
                                         dT_external={9: 0.2})
    stop1 = r1["verdict"] == "STOP_RQ2" and not r1["cond_sanity_A_eco_b_near_1"] and not r1["cond_uplift_collapse_near_0"]
    # STOP if the pipeline can NEVER observe an uplift (all externals dead too) -> instrument dead
    r2 = AP.control_localization_verdict(6, a_eco_b=0.98, dT_post_minus_b=0.001,
                                         dT_external={9: 0.001, 3: 0.0})
    stop2 = r2["verdict"] == "STOP_RQ2" and not r2["cond_external_uplift_observable"]
    ok = stop1 and stop2
    print(f"    A5 STOP cases: uplift-survives={stop1} instrument-dead={stop2}")
    return ok


def main():
    checks = [("A1", a1_helpers), ("A2", a2_prediction_verdict), ("A3", a3_prediction_check_confirm),
              ("A4", a4_control_pass_even_when_b_not_top1), ("A5", a5_control_stop_cases)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
