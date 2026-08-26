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
    # pair-specific floors F_A, F_D
    ok = AP.prediction_verdict(0, 2, floor_A=0, floor_D=0) == "CONFIRM"       # A in, D out
    ok = ok and AP.prediction_verdict(2, 0, floor_A=0, floor_D=0) == "CONTRADICT"
    ok = ok and AP.prediction_verdict(0, 0, floor_A=0, floor_D=0) == "AMBIGUOUS"  # both in
    ok = ok and AP.prediction_verdict(3, 3, floor_A=0, floor_D=0) == "AMBIGUOUS"  # both out
    ok = ok and AP.prediction_verdict(1, 2, floor_A=1, floor_D=0) == "CONFIRM"    # F_A absorbs A only
    # THE patch-1 point: a noisy A_eco floor (f_aeco=2) makes δ_D=2 no longer "outside" -> AMBIGUOUS,
    # NOT the CONFIRM a single shared floor of 0 would have wrongly produced.
    ok = ok and AP.prediction_verdict(1, 2, floor_A=2, floor_D=2) == "AMBIGUOUS"
    print(f"    A2 prediction_verdict (pair-specific floors) ok={ok}")
    return ok


def a3_prediction_check_confirm():
    # Construct scores so at k=6 A_tan's tail == A_eco's tail (delta 0) but D_P's tail differs (delta>0).
    a_eco = {g: float(g) for g in range(8)}          # lowest-6 = {0,1,2,3,4,5}
    a_tan = {g: float(g) for g in range(8)}          # identical -> delta_atan(6)=0
    d_p = {0: 9, 1: 9, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}  # lowest-6 = {2,3,4,5,6,7} -> delta_dp(6)=2
    floors0 = {2: {"f_atan": 0, "f_dp": 0, "f_aeco": 0}, 4: {}, 6: {"f_atan": 0, "f_dp": 0, "f_aeco": 0}}
    r = AP.prediction_check(a_eco, a_tan, d_p, floors_by_k=floors0, k_primary=6)
    ok = r["verdict"] == "CONFIRM"
    ok = ok and r["per_k"][6]["delta_atan_eco"] == 0 and r["per_k"][6]["delta_dp_eco"] == 2
    ok = ok and r["per_k"][6]["F_A"] == 0 and r["per_k"][6]["F_D"] == 0
    ok = ok and r["spearman_atan_eco"] == 1.0 and r["spearman_corroborates"] is True
    # same data, but a noisy A_eco floor f_aeco=2 -> F_D=2 -> δ_D=2 no longer outside -> AMBIGUOUS
    floors_noisy = {2: {}, 4: {}, 6: {"f_atan": 0, "f_dp": 0, "f_aeco": 2}}
    r2 = AP.prediction_check(a_eco, a_tan, d_p, floors_by_k=floors_noisy, k_primary=6)
    ok = ok and r2["verdict"] == "AMBIGUOUS" and r2["per_k"][6]["F_D"] == 2
    print(f"    A3 clean={r['verdict']} noisy_f_aeco={r2['verdict']} (F_D={r2['per_k'][6]['F_D']})")
    return ok


def a4_control_pass_even_when_b_not_top1():
    # rc1 CORRECTION: b passes even though it is NOT the top-1 A_eco block. Decided from CIs only.
    # A_eco(6) CI contains 1 + precision ok; ΔT(post^-6) CI contains 0; an external removal has lower-CI>0.
    r = AP.control_localization_verdict(
        block_b=6,
        a_eco_b_ci=(0.92, 1.05),                       # CI contains 1
        precision_ok=True,
        dT_post_minus_b_ci=(-0.01, 0.02),              # CI contains 0
        dT_external_ci={9: (0.10, 0.28), 3: (-0.02, 0.20), 11: (0.05, 0.30)},  # 9 & 11 have lower-CI>0
    )
    ok = r["pass"] is True and r["verdict"] == "PASS"
    ok = ok and r["cond_sanity_A_eco_b_ci_contains_1"] and r["cond_uplift_collapse_ci_contains_0"]
    ok = ok and r["cond_external_uplift_lowerCI_positive"] and r["external_observable_blocks"] == [9, 11]
    print(f"    A4 control PASS (b not top-1) verdict={r['verdict']} ext_ok={r['external_observable_blocks']}")
    return ok


def a5_control_stop_cases():
    # STOP if the field precision guard fails (adapter effect not measurable) even if A_eco(b) CI ~1
    r0 = AP.control_localization_verdict(6, a_eco_b_ci=(0.9, 1.1), precision_ok=False,
                                         dT_post_minus_b_ci=(-0.01, 0.01), dT_external_ci={9: (0.1, 0.2)})
    stop0 = r0["verdict"] == "STOP_RQ2" and not r0["cond_sanity_A_eco_b_ci_contains_1"]
    # STOP if the uplift does NOT collapse at post^-b (CI strictly positive -> adapter survived removal)
    r1 = AP.control_localization_verdict(6, a_eco_b_ci=(0.2, 0.6), precision_ok=True,
                                         dT_post_minus_b_ci=(0.10, 0.25), dT_external_ci={9: (0.1, 0.2)})
    stop1 = r1["verdict"] == "STOP_RQ2" and not r1["cond_uplift_collapse_ci_contains_0"]
    # STOP if NO external removal has a positive lower-CI (instrument can never see an uplift)
    r2 = AP.control_localization_verdict(6, a_eco_b_ci=(0.95, 1.02), precision_ok=True,
                                         dT_post_minus_b_ci=(-0.005, 0.005),
                                         dT_external_ci={9: (-0.02, 0.03), 3: (-0.05, 0.01)})
    stop2 = r2["verdict"] == "STOP_RQ2" and not r2["cond_external_uplift_lowerCI_positive"]
    ok = stop0 and stop1 and stop2
    print(f"    A5 STOP: precision-fail={stop0} uplift-survives={stop1} instrument-dead={stop2}")
    return ok


def a6_paired_bootstrap_ci():
    # all-positive paired deltas -> lower CI > 0; centered-at-0 deltas -> CI contains 0
    pos = AP.paired_bootstrap_ci([0.10, 0.12, 0.08, 0.11, 0.09, 0.13, 0.10, 0.09], B=2000)
    zero = AP.paired_bootstrap_ci([0.02, -0.02, 0.01, -0.01, 0.0, 0.015, -0.015, 0.005], B=2000)
    ok = pos["lo"] > 0 and pos["mean"] > 0
    ok = ok and zero["lo"] <= 0 <= zero["hi"]
    # determinism (frozen seed)
    ok = ok and AP.paired_bootstrap_ci([0.10, 0.12, 0.08, 0.11], B=2000) == AP.paired_bootstrap_ci([0.10, 0.12, 0.08, 0.11], B=2000)
    print(f"    A6 bootstrap pos.lo={pos['lo']:.3f} zero.ci=[{zero['lo']:.3f},{zero['hi']:.3f}] deterministic")
    return ok


def a7_task_control_verdict():
    # PASS: base+post uplift lower CI>0, host CI∋0, one external (11) positive
    r = AP.task_control_verdict(6, dT_base_ci=(0.03, 0.09), dT_post_ci=(0.02, 0.08),
                                dT_host_ci=(-0.005, 0.006),
                                dT_external_ci={11: (0.01, 0.07), 12: (-0.02, 0.03), 13: (-0.01, 0.02)})
    p = r["verdict"] == "TASK_PASS" and r["external_positive_blocks"] == [11]
    # FAIL: host uplift does NOT collapse (CI strictly positive)
    r2 = AP.task_control_verdict(6, (0.03, 0.09), (0.02, 0.08), (0.02, 0.06),
                                 {11: (0.01, 0.07)})
    f1 = r2["verdict"] == "TASK_FAIL" and not r2["cond3_host_removal_collapses_to_0"]
    # FAIL: no external positive
    r3 = AP.task_control_verdict(6, (0.03, 0.09), (0.02, 0.08), (-0.005, 0.006),
                                 {11: (-0.02, 0.03), 12: (-0.03, 0.01)})
    f2 = r3["verdict"] == "TASK_FAIL" and not r3["cond4_external_uplift_positive"]
    # FAIL: post uplift not positive (adapter doesn't transfer to post)
    r4 = AP.task_control_verdict(6, (0.03, 0.09), (-0.02, 0.04), (-0.005, 0.006), {11: (0.01, 0.07)})
    f3 = r4["verdict"] == "TASK_FAIL" and not r4["cond2_post_uplift_positive"]
    ok = p and f1 and f2 and f3
    print(f"    A7 task PASS={p} host-survive-FAIL={f1} no-ext-FAIL={f2} post-FAIL={f3}")
    return ok


def a8_f1_functional_verdict():
    # RQ2b symmetric gate: both base and post need lower-CI>0 AND point>=SESOI(0.075).
    strong = {"dT_AA": 0.11, "lo": 0.06, "hi": 0.16}     # passes
    weak_pt = {"dT_AA": 0.05, "lo": 0.02, "hi": 0.09}    # positive but point < SESOI
    weak_ci = {"dT_AA": 0.09, "lo": -0.01, "hi": 0.18}   # point>=SESOI but CI contains 0
    # both strong -> F1_PASS
    r = AP.f1_functional_verdict(strong, strong)
    ok = r["verdict"] == "F1_PASS" and r["pass"] and r["base"]["pass"] and r["post"]["pass"]
    # base point below SESOI -> STOP base (even though CI>0)
    r2 = AP.f1_functional_verdict(weak_pt, strong)
    ok = ok and r2["verdict"] == "STOP_RQ2B_BASE_FAIL" and not r2["base"]["point_ge_sesoi"]
    # base ok, post CI contains 0 -> STOP post
    r3 = AP.f1_functional_verdict(strong, weak_ci)
    ok = ok and r3["verdict"] == "STOP_RQ2B_POST_FAIL" and not r3["post"]["lowerCI_gt_0"]
    # base ok, post point below SESOI (positivity alone insufficient) -> STOP post
    r4 = AP.f1_functional_verdict(strong, weak_pt)
    ok = ok and r4["verdict"] == "STOP_RQ2B_POST_FAIL" and r4["post"]["lowerCI_gt_0"] and not r4["post"]["point_ge_sesoi"]
    # exact-boundary point == SESOI passes (>=)
    edge = {"dT_AA": 0.075, "lo": 0.001, "hi": 0.15}
    r5 = AP.f1_functional_verdict(edge, edge)
    ok = ok and r5["verdict"] == "F1_PASS"
    print(f"    A8 F1 verdict PASS={r['verdict']} base<SESOI={r2['verdict']} postCI0={r3['verdict']} "
          f"post<SESOI={r4['verdict']} edge={r5['verdict']}")
    return ok


def main():
    checks = [("A1", a1_helpers), ("A2", a2_prediction_verdict), ("A3", a3_prediction_check_confirm),
              ("A4", a4_control_pass_even_when_b_not_top1), ("A5", a5_control_stop_cases),
              ("A6", a6_paired_bootstrap_ci), ("A7", a7_task_control_verdict),
              ("A8", a8_f1_functional_verdict)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
