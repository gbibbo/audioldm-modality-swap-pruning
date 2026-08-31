#!/usr/bin/env python3
"""RECOVERY-CROSS-SEVERITY-REP-1 — compute the FROZEN A'/B' verdict + dense control (CPU, free).

Consumes the frozen-scorer outputs (xsev_sev2_groups_out.json, xsev_dense_groups_out.json) and the
frozen manifests. Applies the pre-registered estimands with the frozen bootstrap machinery
(cluster_percentile_ci / interaction_ci, PCG64 seed 20260831, B=10000):

  R_native_A/B = C_rec_native  - C_prunedX_native   (AudioCaps 192 ytids, cluster CI)
  R_short_A/B  = C_rec_short   - C_prunedX_short     (AudioCaps 192 ytids, cluster CI)
  R_music_A/B  = C_rec_music   - C_prunedX_music     (music 64 prompts, 3 reps->per-prompt first)
  K_A/B = R_native - R_music    (independent two-sample bootstrap) ; PASS lo95>0
  J_A/B = R_native - R_short    (paired per-ytid vector, cluster CI); gate lo95>0
  H_native: point>=+0.025 AND lo95>0 ; H_music: point<=-0.025 AND upper95<0
  short-OP equivalence: 90% CI of R_short within [-0.025,+0.025]
Dense@10.24 control (§4): C_dense/C_pruned_sev1/C_recovered_sev1 (80 ytids) + gaps G_pruned/G_recovered.
B' is SENSITIVITY ONLY and can never rescue A'. No endpoint/threshold changes.

Run: OPENBLAS_CORETYPE=Haswell .venv/bin/python scripts/research/xsev_score_verdict.py
"""
from __future__ import annotations
import json, os, sys
os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")
sys.path.insert(0, os.getcwd())
import numpy as np
from research_pruning.eval.cluster_bootstrap import cluster_percentile_ci
from research_pruning.eval.reversal import interaction_ci, _paired_prompt_diff

SEED = 20260831          # frozen xsev bootstrap seed (protocol §5)
B = 10000
SESOI = 0.025
EQUIV = 0.025
TMP = "artifacts/icassp_gate0/_score_tmp"


def cosines_by_name(path):
    d = json.load(open(path))
    for r in d["results"]:
        if len(r["cosines"]) not in (192, 80):
            raise SystemExit(f"{r['name']}: {len(r['cosines'])} cosines (expected 192 or 80)")
    return {r["name"]: np.asarray(r["cosines"], dtype=np.float64) for r in d["results"]}, d


def grid_ac(cos):      # (192,) -> (192,1)
    return cos.reshape(192, 1)


def grid_music(cos):   # (192,) canonical (pi,r) -> (64,3)
    return cos.reshape(64, 3)


def ci_d(ci):
    return {"point": ci.point, "lo": ci.lo, "hi": ci.hi, "n": ci.n}


def score_prune(sev, prune):
    """All A'/B' estimands for one pruned system ('pruned2_A' or 'pruned2_B')."""
    rec_n, pr_n = grid_ac(sev[f"recovered2__ac_native"]), grid_ac(sev[f"{prune}__ac_native"])
    rec_s, pr_s = grid_ac(sev[f"recovered2__ac_short"]), grid_ac(sev[f"{prune}__ac_short"])
    rec_m, pr_m = grid_music(sev[f"recovered2__music"]), grid_music(sev[f"{prune}__music"])
    native_diff = _paired_prompt_diff(rec_n, pr_n)     # (192,)
    short_diff = _paired_prompt_diff(rec_s, pr_s)      # (192,)
    music_diff = _paired_prompt_diff(rec_m, pr_m)      # (64,)

    R_native = cluster_percentile_ci(native_diff, b=B, seed=SEED)
    R_short = cluster_percentile_ci(short_diff, b=B, seed=SEED)
    R_music = cluster_percentile_ci(music_diff, b=B, seed=SEED)
    K = interaction_ci(native_diff, music_diff, b=B, seed=SEED)          # R_native - R_music
    J = cluster_percentile_ci(native_diff - short_diff, b=B, seed=SEED)  # paired R_native - R_short
    equiv = cluster_percentile_ci(short_diff, b=B, seed=SEED, alpha=0.10)  # 90% CI

    H_native = bool(R_native.point >= SESOI and R_native.lo > 0.0)
    H_music = bool(R_music.point <= -SESOI and R_music.hi < 0.0)
    K_pass = bool(K["lo"] > 0.0)
    J_pass = bool(J.lo > 0.0)
    equiv_pass = bool(equiv.lo > -EQUIV and equiv.hi < EQUIV)

    # descriptive means
    C = {"rec_native": float(rec_n.mean()), "pruned_native": float(pr_n.mean()),
         "rec_short": float(rec_s.mean()), "pruned_short": float(pr_s.mean()),
         "rec_music": float(rec_m.mean()), "pruned_music": float(pr_m.mean())}
    return {
        "R_native": ci_d(R_native), "R_short": ci_d(R_short), "R_music": ci_d(R_music),
        "K": {"point": K["point"], "lo": K["lo"], "hi": K["hi"], "n_ac": K["n_ac"], "n_music": K["n_music"]},
        "J": ci_d(J), "equiv_90ci": {"lo": equiv.lo, "hi": equiv.hi, "point": equiv.point},
        "gates": {"K_pass_lo95_gt0": K_pass, "H_native": H_native, "H_music": H_music,
                  "J_pass_lo95_gt0": J_pass, "short_equivalence_pass": equiv_pass},
        "means": C,
        "native_sign_frac_pos": float((native_diff > 0).mean()),
        "music_sign_frac_neg": float((music_diff < 0).mean()),
    }


def dense_control(dc):
    d = dc["dense10s__dense"]; p = dc["dense10s__pruned_sev1"]; r = dc["dense10s__recovered_sev1"]
    assert d.size == p.size == r.size == 80
    G_pruned = cluster_percentile_ci(d - p, b=B, seed=SEED)      # dense - pruned
    G_recovered = cluster_percentile_ci(d - r, b=B, seed=SEED)   # dense - recovered
    return {"C_dense_10s": float(d.mean()), "C_pruned_sev1_10s": float(p.mean()),
            "C_recovered_sev1_10s": float(r.mean()),
            "G_pruned_dense_minus_pruned": ci_d(G_pruned),
            "G_recovered_dense_minus_recovered": ci_d(G_recovered),
            "note": "severity-1 completion on the frozen Arm-D 80 ytids; SECONDARY; does not alter "
                    "the severity-2 replication verdict; no 'restored to dense' claim."}


def classify(a):
    g = a["gates"]
    if not g["K_pass_lo95_gt0"]:
        return "D", "K_A fails: principal cross-context phenomenon does not independently replicate at severity 2."
    sign = g["H_native"] and g["H_music"]
    if sign and g["J_pass_lo95_gt0"]:
        return "A", "K_A pass + sign-pattern (H_native ∧ H_music) + J_A pass: strongest independent cross-severity replication."
    if sign and not g["J_pass_lo95_gt0"]:
        return "B", "K_A + sign-pattern pass, J_A fails: cross-context sign pattern replicates; temporal interaction not independently established."
    return "C", "K_A passes, one/both sign-pattern components fail: contextual dependence replicates; specific native-positive/music-negative pattern not fully replicated."


def seam_robust(a, bp):
    """A conclusion is SEAM-ROBUST iff A' gate passes AND B' preserves it."""
    ga, gb = a["gates"], bp["gates"]
    out = {}
    out["K"] = {"A_pass": ga["K_pass_lo95_gt0"], "B_pass": gb["K_pass_lo95_gt0"],
                "seam_robust": bool(ga["K_pass_lo95_gt0"] and gb["K_pass_lo95_gt0"])}
    out["sign_pattern"] = {"A_pass": bool(ga["H_native"] and ga["H_music"]),
                           "B_pass": bool(gb["H_native"] and gb["H_music"]),
                           "seam_robust": bool(ga["H_native"] and ga["H_music"] and gb["H_native"] and gb["H_music"])}
    out["J"] = {"A_pass": ga["J_pass_lo95_gt0"], "B_pass": gb["J_pass_lo95_gt0"],
                "seam_robust": bool(ga["J_pass_lo95_gt0"] and gb["J_pass_lo95_gt0"])}
    return out


def main():
    sev, sev_raw = cosines_by_name(os.path.join(TMP, "xsev_sev2_groups_out.json"))
    dc, dc_raw = cosines_by_name(os.path.join(TMP, "xsev_dense_groups_out.json"))
    A = score_prune(sev, "pruned2_A")
    Bp = score_prune(sev, "pruned2_B")
    dense = dense_control(dc)
    case, case_desc = classify(A)
    out = {"artifact": "xsev_score_verdict", "experiment": "RECOVERY-CROSS-SEVERITY-REP-1",
           "bootstrap": {"seed": SEED, "B": B, "SESOI": SESOI, "equiv_margin": EQUIV},
           "scorer_provenance": sev_raw.get("scorer_provenance"),
           "PRIMARY_A": A, "SENSITIVITY_B": Bp, "DENSE_CONTROL": dense,
           "REPLICATION_CASE": case, "case_description": case_desc,
           "seam_robustness": seam_robust(A, Bp),
           "primary_note": "Primary inference is A' only. B' is sensitivity and can NEVER rescue a failed A'."}
    os.makedirs("artifacts/icassp_gate0", exist_ok=True)
    op = "artifacts/icassp_gate0/xsev_score_verdict.json"
    json.dump(out, open(op, "w"), indent=2)
    ga = A["gates"]
    print(f"=== PRIMARY A' ===")
    print(f"R_native_A  point={A['R_native']['point']:+.4f} [{A['R_native']['lo']:+.4f},{A['R_native']['hi']:+.4f}]  H_native={ga['H_native']}")
    print(f"R_music_A   point={A['R_music']['point']:+.4f} [{A['R_music']['lo']:+.4f},{A['R_music']['hi']:+.4f}]  H_music={ga['H_music']}")
    print(f"R_short_A   point={A['R_short']['point']:+.4f} [{A['R_short']['lo']:+.4f},{A['R_short']['hi']:+.4f}]")
    print(f"K_A         point={A['K']['point']:+.4f} [{A['K']['lo']:+.4f},{A['K']['hi']:+.4f}]  PASS={ga['K_pass_lo95_gt0']}")
    print(f"J_A         point={A['J']['point']:+.4f} [{A['J']['lo']:+.4f},{A['J']['hi']:+.4f}]  PASS={ga['J_pass_lo95_gt0']}")
    print(f"short equiv 90%CI [{A['equiv_90ci']['lo']:+.4f},{A['equiv_90ci']['hi']:+.4f}]  PASS={ga['short_equivalence_pass']}")
    print(f">>> REPLICATION CASE {case}: {case_desc}")
    print(f"=== DENSE CONTROL === C_dense={dense['C_dense_10s']:.4f} C_pruned={dense['C_pruned_sev1_10s']:.4f} C_recovered={dense['C_recovered_sev1_10s']:.4f}")
    print(f"    G_pruned={dense['G_pruned_dense_minus_pruned']['point']:+.4f} G_recovered={dense['G_recovered_dense_minus_recovered']['point']:+.4f}")
    print(f"-> {op}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
