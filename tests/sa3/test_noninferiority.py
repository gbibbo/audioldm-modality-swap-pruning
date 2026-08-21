#!/usr/bin/env python3
"""Synthetic CPU tests for research_sa3.erule (protocol section 9.2, 3.5).
Run: .venv-sa3/bin/python tests/sa3/test_noninferiority.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from research_sa3 import erule as E


def approx(a, b, tol=1e-9): return abs(a - b) <= tol


def n1_margins():
    # deterioration resolvable (delta > r) -> margin = delta; else floor r
    m = E.margins(clap_8=0.50, clap_7=0.44, kl_7v8=0.03, fd_7v8=0.5, r_clap=0.01, r_kl=0.05, r_fd=0.2)
    ok = approx(m["m_CLAP"], 0.06) and approx(m["m_KL"], 0.05) and approx(m["m_FD"], 0.5)
    # m_KL floored at r_kl=0.05 > delta_kl=0.03 ; m_FD = delta_fd 0.5 > r 0.2
    print(f"    N1 m={ {k: round(v,3) for k,v in m.items() if k.startswith('m_')} }")
    return ok


def n2_noninferior():
    m = {"m_CLAP": 0.06, "m_KL": 0.05, "m_FD": 0.5}
    v = E.noninferiority((-0.01, 0.02), (-0.01, 0.03), (-0.1, 0.4), m)   # all upper CIs within margins
    print(f"    N2 verdict={v}")
    return v == "non_inferior"


def n3_inferior_clap():
    m = {"m_CLAP": 0.06, "m_KL": 0.05, "m_FD": 0.5}
    v = E.noninferiority((0.10, 0.20), (-0.01, 0.03), (-0.1, 0.4), m)    # CLAP lower CI 0.10 > 0.06
    print(f"    N3 verdict={v}")
    return v == "inferior"


def n4_inferior_bothdrift():
    m = {"m_CLAP": 0.06, "m_KL": 0.05, "m_FD": 0.5}
    v = E.noninferiority((-0.01, 0.02), (0.10, 0.20), (0.6, 0.9), m)     # both drift lower CIs > margins
    print(f"    N4 verdict={v}")
    return v == "inferior"


def n5_indeterminate():
    m = {"m_CLAP": 0.06, "m_KL": 0.05, "m_FD": 0.5}
    v = E.noninferiority((-0.01, 0.09), (-0.01, 0.03), (-0.1, 0.4), m)   # CLAP upper 0.09>0.06 but lower<margin
    print(f"    N5 verdict={v}")
    return v == "indeterminate"


def n6_egreedy():
    m = {"m_CLAP": 0.06, "m_KL": 0.05, "m_FD": 0.5}
    # clap_dense=0.5, clap=0.44 -> 0.06/0.06=1.0 ; kl=0.10 -> 0.10/0.05=2.0 ; fd=0.25 -> 0.25/0.5=0.5 -> max=2.0
    s = E.egreedy_score(clap=0.44, kl=0.10, fd=0.25, clap_dense=0.5, m=m)
    ok = approx(s, 2.0)
    ok = ok and E.within_caps(kl=0.04, fd=0.4, m=m) is True and E.within_caps(kl=0.06, fd=0.4, m=m) is False
    print(f"    N6 egreedy_score={s} within_caps ok")
    return ok


def main():
    checks = [("N1", n1_margins), ("N2", n2_noninferior), ("N3", n3_inferior_clap),
              ("N4", n4_inferior_bothdrift), ("N5", n5_indeterminate), ("N6", n6_egreedy)]
    ok_all = True
    for name, fn in checks:
        ok = fn(); ok_all &= ok
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print("ALL PASS" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
