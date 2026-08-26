"""RQ2 validation decision core (rc1, protocol §6 prediction check + §3 control localisation).

PURE numeric functions over already-reduced per-block scores, so they are fully unit-testable on
synthetic data with no model and no metric stack. The A_eco/ΔT_L driver produces the per-block
score dicts; this module turns them into the pre-registered discrete verdicts. Nothing here invents
a statistic: set-disagreement is §4.1, the floor is the §4.1 bootstrap floor, and the CONFIRM /
CONTRADICT / AMBIGUOUS rules were frozen in `docs/sa3/rq2_validation_protocol.md` rc1 BEFORE data.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set, Tuple
import math


# ----------------------------------------------------------------- ranking / set helpers
def lowest_k(scores: Dict[int, float], k: int) -> Set[int]:
    """The LOO removal tail: the k blocks with the SMALLEST single-block score (least damaging to
    remove). Deterministic tie-break by ascending block id. Matches the pre-gate `removal_set()`.
    These are LOO-ranking candidate tails, NOT sequential-greedy masks (§3.5)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    if k > len(scores):
        raise ValueError(f"k={k} exceeds number of blocks {len(scores)}")
    order = sorted(scores, key=lambda g: (scores[g], g))
    return set(order[:k])


def set_disagreement(set_x: Iterable[int], set_y: Iterable[int]) -> int:
    """δ_XY = |R_X △ R_Y| / 2 (protocol §4.1). Defined for equal-size sets (integer)."""
    sx, sy = set(set_x), set(set_y)
    if len(sx) != len(sy):
        raise ValueError(f"set sizes differ ({len(sx)} vs {len(sy)}); δ_XY(k) needs equal-size sets")
    return len(sx ^ sy) // 2


def spearman(x: List[float], y: List[float]) -> float:
    """Spearman rank correlation (corroborative only; never overrides the discrete gate). Average
    ranks for ties; returns nan for degenerate (constant) input."""
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    n = len(x)
    if n < 2:
        return float("nan")
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def _avg_ranks(vals: List[float]) -> List[float]:
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


# ---------------------------------------------------------- §6 prediction verdict (k=6 primary)
def prediction_verdict(delta_atan: int, delta_dp: int, floor_A: int, floor_D: int) -> str:
    """Frozen rc1.1 rule (primary k=6) with PAIR-SPECIFIC floors (§4.1: each comparison uses the
    floors of BOTH criteria it involves; A_eco's own floor never drops out):

        F_A = max(f_Atan, f_Aeco)   F_D = max(f_DP, f_Aeco)
        CONFIRM    ⇔ δ_A ≤ F_A AND δ_D > F_D   (A_tan inside A_eco's stability, D_P outside)
        CONTRADICT ⇔ δ_D ≤ F_D AND δ_A > F_A   (the inverse)
        AMBIGUOUS  ⇔ anything else.

    NOTE: this is NOT equivalent to `δ_A < δ_D`; that inequality is a secondary descriptive analysis
    only, never the gate. Corroborative rank correlations never override this discrete gate."""
    a_in = delta_atan <= floor_A
    d_in = delta_dp <= floor_D
    if a_in and not d_in:
        return "CONFIRM"
    if d_in and not a_in:
        return "CONTRADICT"
    return "AMBIGUOUS"


def prediction_check(
    a_eco: Dict[int, float],
    a_tan: Dict[int, float],
    d_p: Dict[int, float],
    floors_by_k: Dict[int, Dict[str, int]],
    k_primary: int = 6,
    ks: Tuple[int, ...] = (2, 4, 6),
) -> dict:
    """Full §6 check from per-block score dicts. Uses LOO removal tails (lowest_k) for the sets and
    the pair-specific floors of §4.1.

    `floors_by_k[k]` is a dict of the three §4.1 bootstrap floors at that k, in blocks:
    `{"f_atan": int, "f_dp": int, "f_aeco": int}` (missing keys default to 0). From these:
    `F_A = max(f_atan, f_aeco)`, `F_D = max(f_dp, f_aeco)`. Returns the primary-k verdict + full
    trace + corroborative Spearman + the descriptive-only `delta_atan < delta_dp` flag."""
    blocks = sorted(a_eco)
    per_k = {}
    for k in ks:
        R_eco = lowest_k(a_eco, k)
        R_atan = lowest_k(a_tan, k)
        R_dp = lowest_k(d_p, k)
        dA = set_disagreement(R_atan, R_eco)
        dD = set_disagreement(R_dp, R_eco)
        fk = floors_by_k.get(k, {}) or {}
        f_atan = int(fk.get("f_atan", 0)); f_dp = int(fk.get("f_dp", 0)); f_aeco = int(fk.get("f_aeco", 0))
        F_A = max(f_atan, f_aeco); F_D = max(f_dp, f_aeco)
        per_k[k] = {
            "R_eco": sorted(R_eco), "R_atan": sorted(R_atan), "R_dp": sorted(R_dp),
            "delta_atan_eco": dA, "delta_dp_eco": dD,
            "f_atan": f_atan, "f_dp": f_dp, "f_aeco": f_aeco, "F_A": F_A, "F_D": F_D,
            "verdict": prediction_verdict(dA, dD, F_A, F_D),
            "delta_A_lt_delta_D_descriptive": bool(dA < dD),
        }
    rho_atan = spearman([a_tan[g] for g in blocks], [a_eco[g] for g in blocks])
    rho_dp = spearman([d_p[g] for g in blocks], [a_eco[g] for g in blocks])
    return {
        "k_primary": k_primary,
        "verdict": per_k[k_primary]["verdict"],
        "per_k": per_k,
        "spearman_atan_eco": rho_atan,
        "spearman_dp_eco": rho_dp,
        "spearman_corroborates": (not math.isnan(rho_atan) and not math.isnan(rho_dp)
                                  and rho_atan > rho_dp),
        "note": "LOO removal tails (lowest_k); NOT sequential-greedy masks (§3.5). k=6 primary; "
                "pair-specific floors F_A=max(f_Atan,f_Aeco), F_D=max(f_DP,f_Aeco) (rc1.1).",
    }


# ------------------------------------------------- §3 control localisation verdict (rc1 re-spec)
def _ci(x):
    """Normalise a CI to an ordered (lo, hi) tuple."""
    lo, hi = float(x[0]), float(x[1])
    return (lo, hi) if lo <= hi else (hi, lo)


def paired_bootstrap_ci(deltas, seed: int = 20260824, B: int = 10000, alpha: float = 0.05):
    """Paired bootstrap 95% CI of mean(deltas) — resample the units (eval clips) with replacement.
    Deterministic given `seed`/`B` (frozen before scores, rc1.4). Pure (uses random, not numpy)."""
    import random as _r
    n = len(deltas)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0, "B": B}
    mean = sum(deltas) / n
    rng = _r.Random(seed)
    means = []
    for _ in range(B):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * (B - 1))]
    hi = means[int((1 - alpha / 2) * (B - 1))]
    return {"mean": mean, "lo": lo, "hi": hi, "n": n, "B": B, "seed": seed}


def task_control_verdict(block_b: int, dT_base_ci, dT_post_ci, dT_host_ci,
                         dT_external_ci: Dict[int, tuple]) -> dict:
    """rc1.4 TASK-LEVEL positive-control gate on the primary scalar ΔT_AA. PASS iff ALL hold:
      1. ΔT_AA(dense base)  paired lower 95% CI > 0   (adapter uplift observable on base)
      2. ΔT_AA(dense post)  paired lower 95% CI > 0   (observable on post)
      3. ΔT_AA(post^{-b})   CI contains 0             (uplift collapses when host block removed)
      4. at least one pre-frozen external g ∈ G_ext(b) has ΔT_AA(post^{-g}) lower CI > 0
    `dT_*_ci` are (lo, hi) paired-bootstrap CIs; `dT_external_ci` maps g -> (lo, hi). Only ΔT
    sustains the functional claim (§4.3); A_eco is mechanistic evidence, checked separately."""
    b_lo, b_hi = _ci(dT_base_ci)
    p_lo, p_hi = _ci(dT_post_ci)
    h_lo, h_hi = _ci(dT_host_ci)
    c1 = b_lo > 0
    c2 = p_lo > 0
    c3 = h_lo <= 0.0 <= h_hi
    ext_pos = sorted(g for g, ci in dT_external_ci.items() if _ci(ci)[0] > 0)
    c4 = len(ext_pos) > 0
    passed = c1 and c2 and c3 and c4
    return {
        "block_b": block_b, "pass": bool(passed), "verdict": "TASK_PASS" if passed else "TASK_FAIL",
        "cond1_base_uplift_positive": bool(c1),
        "cond2_post_uplift_positive": bool(c2),
        "cond3_host_removal_collapses_to_0": bool(c3),
        "cond4_external_uplift_positive": bool(c4),
        "external_positive_blocks": ext_pos,
        "dT_base_ci": [b_lo, b_hi], "dT_post_ci": [p_lo, p_hi], "dT_host_ci": [h_lo, h_hi],
        "note": "rc1.4: only ΔT_AA sustains the functional claim; primary paired audio-audio.",
    }


def control_localization_verdict(
    block_b: int,
    a_eco_b_ci,                      # (lo, hi) CI on A_eco(b; L_b)
    precision_ok: bool,              # the §6 signal/precision guard on ||δF(L_b)||^2 (from metrics.precision_ok)
    dT_post_minus_b_ci,              # (lo, hi) CI on ΔT_{L_b}(post^{-b})
    dT_external_ci: Dict[int, tuple],  # g -> (lo, hi) CI on ΔT_{L_b}(post^{-g}) for external g != b
) -> dict:
    """rc1.1 STOP gate for a single-block control L_b — decided ENTIRELY from measured intervals, with
    NO arbitrary science constants. The gate rests on δF^{-b}(L_b)=0 (removing the host block
    physically deletes the adapter), NOT on b being the top-1 of the 20 A_eco scores (descriptive —
    a single-block adapter's *function* can route through downstream blocks).

    PASS (all three):
      (1) field sanity  : precision guard passes AND the CI of A_eco(b) is COMPATIBLE WITH 1
                          (1.0 ∈ [lo, hi]);
      (2) uplift collapse: the CI of ΔT_{L_b}(post^{-b}) CONTAINS 0 (lo ≤ 0 ≤ hi);
      (3) observability : SOME external removal g has a strictly positive uplift lower-CI (lo > 0).
    Else STOP RQ2 (the chain cannot localise a known adaptation from outputs)."""
    a_lo, a_hi = _ci(a_eco_b_ci)
    d_lo, d_hi = _ci(dT_post_minus_b_ci)
    cond_sanity = bool(precision_ok) and (a_lo <= 1.0 <= a_hi)
    cond_collapse = (d_lo <= 0.0 <= d_hi)
    ext_ok_blocks = [g for g, ci in dT_external_ci.items() if _ci(ci)[0] > 0.0]
    cond_observability = len(ext_ok_blocks) > 0
    passed = cond_sanity and cond_collapse and cond_observability
    return {
        "block_b": block_b,
        "pass": bool(passed),
        "verdict": "PASS" if passed else "STOP_RQ2",
        "cond_sanity_A_eco_b_ci_contains_1": bool(cond_sanity),
        "cond_uplift_collapse_ci_contains_0": bool(cond_collapse),
        "cond_external_uplift_lowerCI_positive": bool(cond_observability),
        "precision_ok": bool(precision_ok),
        "a_eco_b_ci": [a_lo, a_hi],
        "dT_post_minus_b_ci": [d_lo, d_hi],
        "external_observable_blocks": sorted(ext_ok_blocks),
        "note": "rc1.1: decided from measured CIs only; top-1 ranking of A_eco(b) is descriptive.",
    }


def f1_functional_verdict(base_delta, post_delta, sesoi: float = 0.075):
    """RQ2b F1/F2 SYMMETRIC functional gate (Gabriel rev3.1). Each of base and post independently must
    have paired-bootstrap lower-CI > 0 AND point ΔT_AA >= SESOI. `base_delta`/`post_delta` are dicts with
    keys {"dT_AA" (point), "lo", "hi"} (as emitted by scripts/sa3/score_taa.py). MDE is a sizing quantity
    and is NEVER the PASS threshold; the bar is the preregistered SESOI.

    Terminal verdicts:
      base fails               -> STOP_RQ2B_BASE_FAIL  (task/training/measurement chain unqualified)
      base ok but post fails   -> STOP_RQ2B_POST_FAIL  (meaningful base->post transfer not qualified)
      both pass                -> F1_PASS              (F2 eligible)
    """
    def gate(d):
        lo = float(d["lo"]); pt = float(d["dT_AA"])
        return {"dT_AA": pt, "lo": lo, "hi": float(d["hi"]),
                "lowerCI_gt_0": lo > 0.0, "point_ge_sesoi": pt >= sesoi,
                "pass": (lo > 0.0) and (pt >= sesoi)}
    b = gate(base_delta); p = gate(post_delta)
    if not b["pass"]:
        verdict = "STOP_RQ2B_BASE_FAIL"
    elif not p["pass"]:
        verdict = "STOP_RQ2B_POST_FAIL"
    else:
        verdict = "F1_PASS"
    retention = (p["dT_AA"] / b["dT_AA"]) if b["dT_AA"] not in (0, 0.0) else None
    return {"sesoi": sesoi, "base": b, "post": p, "pass": b["pass"] and p["pass"],
            "verdict": verdict, "post_over_base_retention": retention}
