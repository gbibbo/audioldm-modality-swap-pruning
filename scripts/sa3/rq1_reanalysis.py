#!/usr/bin/env python3
"""RQ1 re-analysis from the EXISTING pilot (no generation, no GPU) — addresses review tasks 6-7.

Reads artifacts/sa3/pilot_fields.json (D_P, D_B_common, I_PT_raw pooled, W over 20 blocks; these
are LOO single-block quantities, NOT confirmatory greedy). Reports:
  * rank correlations rho(D_P,I_PT), rho(D_P,D_B), rho(I_PT,W), rho(D_P,W), rho(D_B,I_PT);
  * per block: D_P, D_B, D_P-D_B, log(D_P/D_B) (guarded);
  * LOO-ranking-induced removal sets R_X(k)=bottom-k by X (least damage to remove) for D_P/D_B/I_PT
    at k in {2,4,6}, and disagreement counts / Jaccard between criteria -> the I_PT~=D_P collapse test.
CANNOT recompute non-normalized ||F-F^-g||^2 or ||F_P||^2/||F_B||^2 from this file (raw numerators
were not saved; the field-store was on the job's /tmp and is gone) -- those come from a separate
cheap CPU field-norm recompute (rq1_field_norms.py). Flagged explicitly here.

Run: .venv-sa3/bin/python scripts/sa3/rq1_reanalysis.py
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np

SRC = "artifacts/sa3/pilot_fields.json"
OUT = "artifacts/sa3/rq1_reanalysis.json"


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def removal_set(vals_by_block, k):
    """R_X(k) = the k blocks with the SMALLEST score (least damage to remove) -- a LOO ranking, not greedy."""
    order = sorted(vals_by_block, key=lambda g: vals_by_block[g])
    return set(order[:k])


def disagreement(sx, sy):
    return len(set(sx) ^ set(sy)) // 2


def jaccard(sx, sy):
    sx, sy = set(sx), set(sy)
    return len(sx & sy) / len(sx | sy) if (sx | sy) else 1.0


def main():
    r = json.load(open(SRC))
    gs = sorted(r["D_P"], key=lambda k: int(k))
    D_P = {g: r["D_P"][g] for g in gs}
    D_B = {g: r["D_B_common"][g] for g in gs}
    I_PT = {g: r["I_PT_raw"][g]["pooled"] for g in gs}
    W = {g: r["W"][g] for g in gs}
    v = lambda d: np.array([d[g] for g in gs])

    cors = {
        "rho_DP_IPT": spearman(v(D_P), v(I_PT)),
        "rho_DP_DB": spearman(v(D_P), v(D_B)),
        "rho_IPT_W": spearman(v(I_PT), v(W)),
        "rho_DP_W": spearman(v(D_P), v(W)),
        "rho_DB_IPT": spearman(v(D_B), v(I_PT)),
    }

    per_block = {}
    for g in gs:
        ratio = D_P[g] / D_B[g] if D_B[g] > 0 else float("inf")
        # guard log for small denominators
        logr = math.log(D_P[g] / D_B[g]) if (D_P[g] > 1e-12 and D_B[g] > 1e-12) else None
        per_block[g] = {"D_P": D_P[g], "D_B": D_B[g], "diff_DP_DB": D_P[g] - D_B[g],
                        "ratio_DP_DB": ratio, "log_ratio": logr, "I_PT": I_PT[g], "W": W[g]}

    # LOO-ranking-induced removal sets (least-damage-to-remove) + collapse test
    sets = {}
    for k in (2, 4, 6):
        sets[k] = {
            "R_DP": sorted(removal_set(D_P, k)),
            "R_DB": sorted(removal_set(D_B, k)),
            "R_IPT": sorted(removal_set(I_PT, k)),
        }
        sets[k]["disagree_DP_IPT"] = disagreement(sets[k]["R_DP"], sets[k]["R_IPT"])
        sets[k]["disagree_DP_DB"] = disagreement(sets[k]["R_DP"], sets[k]["R_DB"])
        sets[k]["jaccard_DP_IPT"] = round(jaccard(sets[k]["R_DP"], sets[k]["R_IPT"]), 3)
        sets[k]["jaccard_DP_DB"] = round(jaccard(sets[k]["R_DP"], sets[k]["R_DB"]), 3)

    collapse = all(sets[k]["disagree_DP_IPT"] == 0 for k in (2, 4, 6))
    out = {
        "source": SRC, "N": r["N"], "note": "LOO single-block rankings (NOT confirmatory greedy); "
        "no bootstrap CIs -> no floor -> disagreement counts are point estimates, not gated decisions.",
        "correlations": cors, "per_block": per_block, "loo_removal_sets": sets,
        "IPT_collapses_onto_DP": collapse,
        "cannot_from_this_file": ["non-normalized ||F-F^-g||^2 (raw numerators not saved)",
                                  "||F_P||^2/||F_B||^2 (field-store gone) -> see rq1_field_norms.py"],
    }
    json.dump(out, open(OUT, "w"), indent=2)

    print("=== rank correlations ===")
    for k, val in cors.items():
        print(f"  {k} = {val:+.3f}")
    print("\n=== LOO removal sets (least-damage-to-remove) + collapse test ===")
    for k in (2, 4, 6):
        s = sets[k]
        print(f"  k={k}: R_DP={s['R_DP']} R_IPT={s['R_IPT']}  disagree(DP,IPT)={s['disagree_DP_IPT']} "
              f"J={s['jaccard_DP_IPT']} | R_DB={s['R_DB']} disagree(DP,DB)={s['disagree_DP_DB']}")
    print(f"\nI_PT collapses onto D_P (identical removal sets at all k): {collapse}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
