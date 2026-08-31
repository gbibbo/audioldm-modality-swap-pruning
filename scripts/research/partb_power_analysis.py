#!/usr/bin/env python3
"""Part B0.4 — power / decision-value analysis for the public dense text-FT duration reference.

Uses ONLY existing frozen data (op_duration_discriminator_1 raw per-ytid CLAP cosines, n=80, the SAME
Arm-D battery Part B would reuse). No download, no GPU. Determines achievable precision at n=80 for:
  J_dense_textFT = (textFT-dense)_native - (textFT-dense)_short
  Q              = J_recovery_sev1 - J_dense_textFT

Key identity: a two-system duration interaction J(A,B) = (A_nat-B_nat)-(A_short-B_short)
            = (A_nat-A_short) - (B_nat-B_short) = swing(A) - swing(B).
So the empirical Var_ytid[J_recovery] (= swing(rec)-swing(pru)) is a data-driven estimate of the
variance SCALE of ANY two-system duration interaction on this battery, incl. J_dense_textFT
(textFT vs dense). Reported with a +/-50% variance sensitivity band.

Run (CPU): OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/partb_power_analysis.py
"""
from __future__ import annotations
import json, math, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

Z95, Z90, Zpow = 1.959963985, 1.644853627, 0.841621234   # two-sided 95%, one-sided 90% (TOST), 80% power
SESOI = 0.025
N = 80


def main():
    d = json.load(open(os.path.join(ROOT, "configs/research/op_duration_discriminator_1_result.json")))
    rc = d["raw_cosines"]
    rec_ctrl = np.array(rc["recovered_ctrl"]); pru_ctrl = np.array(rc["pruned_ctrl"])     # 3.84 s
    rec_alt = np.array(rc["recovered_alt"]);   pru_alt = np.array(rc["pruned_alt"])       # 10.24 s
    assert len(rec_ctrl) == len(pru_ctrl) == len(rec_alt) == len(pru_alt) == N

    # per-ytid duration swings and the recovered-vs-pruned duration interaction J_recovery
    rec_swing = rec_alt - rec_ctrl
    pru_swing = pru_alt - pru_ctrl
    J_rec = rec_swing - pru_swing                       # = op_duration primary J_CLAP, per ytid
    sd_J_rec = float(J_rec.std(ddof=1))
    se_J_rec = sd_J_rec / math.sqrt(N)
    corr_swings = float(np.corrcoef(rec_swing, pru_swing)[0, 1])

    def half(z, se): return z * se
    def mde(se):     return (Z95 + Zpow) * se           # two-sided a=.05, 80% power

    R = {"artifact": "partb_power_analysis", "n": N, "SESOI": SESOI,
         "source": "op_duration_discriminator_1_result.json raw_cosines (existing, frozen)",
         "identity": "J(A,B)=swing(A)-swing(B); Var[J_recovery] estimates any two-system duration-interaction variance",
         "empirical": {
             "J_recovery_point": float(J_rec.mean()),
             "J_recovery_sd_ytid": sd_J_rec, "J_recovery_se": se_J_rec,
             "J_recovery_ci95": [float(J_rec.mean() - half(Z95, se_J_rec)),
                                 float(J_rec.mean() + half(Z95, se_J_rec))],
             "swing_rec_sd": float(rec_swing.std(ddof=1)),
             "swing_pru_sd": float(pru_swing.std(ddof=1)),
             "corr_swings": corr_swings}}

    # ---- J_dense_textFT precision: central sd = sd_J_rec; sensitivity band +/-50% variance
    scen = {}
    for tag, var_mult in [("low_var_0.5x", 0.5), ("central_1.0x", 1.0), ("high_var_1.5x", 1.5)]:
        sd = sd_J_rec * math.sqrt(var_mult)
        se = sd / math.sqrt(N)
        scen[tag] = {"sd_ytid": sd, "se": se, "ci95_halfwidth": half(Z95, se),
                     "mde_80pct_power": mde(se)}
    R["J_dense_textFT"] = {
        "estimator": "(textFT-dense)_nat - (textFT-dense)_short, paired ytid, n=80",
        "variance_proxy": "Var_ytid[J_recovery] (same battery/scorer/op-points)",
        "scenarios": scen}

    # ---- Q = J_recovery_sev1 - J_dense_textFT, paired per ytid.
    # Var[Q_y] = Var[J_rec_y] + Var[J_textFT_y] - 2*rho*sd(J_rec)*sd(J_textFT).
    # J_recovery_sev1 here uses the SAME Arm-D 80 (op_duration is the sev-1 native/short battery).
    q = {}
    for rho_tag, rho in [("rho_0_indep", 0.0), ("rho_0.5", 0.5), ("rho_-0.5_worstcase", -0.5)]:
        # central: sd(J_textFT)=sd_J_rec
        var_q = sd_J_rec**2 + sd_J_rec**2 - 2 * rho * sd_J_rec * sd_J_rec
        sd_q = math.sqrt(max(var_q, 0.0)); se_q = sd_q / math.sqrt(N)
        q[rho_tag] = {"sd_ytid": sd_q, "se": se_q,
                      "ci95_halfwidth": half(Z95, se_q),
                      "ci90_halfwidth_TOST": half(Z90, se_q),
                      "mde_80pct_power": mde(se_q),
                      "tost_equiv_feasible_within_0.025":
                          bool(half(Z90, se_q) < SESOI)}
    R["Q"] = {"estimator": "J_recovery_sev1 - J_dense_textFT, paired ytid, n=80",
              "note": "central sd(J_textFT)=sd(J_recovery); TOST needs 90% CI halfwidth < 0.025 at Q~0",
              "scenarios": q}

    # ---- decision summary
    central_se_Jtext = scen["central_1.0x"]["se"]
    R["decision"] = {
        "MDE_J_textFT_central": mde(central_se_Jtext),
        "detectable_effect_examples": {
            "J_recovery_sev1_native_short": float(J_rec.mean()),
            "J_recovery_sev2_CLAP": 0.159,   # xsev primary J (context: larger effect)
        },
        "TOST_Q_within_0.025_feasible_n80": all(not q[k]["tost_equiv_feasible_within_0.025"]
                                                 for k in ("rho_0_indep", "rho_0.5")) is False,
        "verdict_TOST": ("NOT attainable at n=80 unless the two duration interactions are very strongly "
                         "positively correlated (rho>0.5); remove Q-equivalence as a planned claim"),
        "asymmetric_value": {
            "positive_J_textFT": "a clearly positive J_dense_textFT would broaden the evaluation warning "
                                 "beyond pruning recovery (detectable if |J_textFT| >~ MDE)",
            "unresolved_J_textFT": "does NOT establish absence of duration dependence",
            "null_point_no_powered_equiv": "is NOT evidence of no interaction"}}

    out = os.path.join(ROOT, "configs/research/partb_power_analysis.json")
    json.dump(R, open(out, "w"), indent=2)
    print("empirical J_recovery: point=%.4f sd_ytid=%.4f se=%.4f corr(swings)=%.3f"
          % (J_rec.mean(), sd_J_rec, se_J_rec, corr_swings))
    print("J_textFT central: se=%.4f ci95_halfwidth=%.4f MDE(80%%)=%.4f"
          % (central_se_Jtext, scen["central_1.0x"]["ci95_halfwidth"], mde(central_se_Jtext)))
    for k, v in q.items():
        print("Q %-18s se=%.4f ci95_hw=%.4f ci90_hw(TOST)=%.4f MDE=%.4f TOST<0.025:%s"
              % (k, v["se"], v["ci95_halfwidth"], v["ci90_halfwidth_TOST"], v["mde_80pct_power"],
                 v["tost_equiv_feasible_within_0.025"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
