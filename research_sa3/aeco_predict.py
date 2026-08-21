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
def prediction_verdict(delta_atan: int, delta_dp: int, floor: int) -> str:
    """Frozen rc1 rule (primary k=6). Given the set-disagreements of A_tan and D_P against A_eco and
    the max bootstrap floor:

      CONFIRM    ⇔ δ_A ≤ floor AND δ_D > floor   (A_tan inside A_eco's stability, D_P outside)
      CONTRADICT ⇔ δ_D ≤ floor AND δ_A > floor   (the inverse)
      AMBIGUOUS  ⇔ both inside, both outside, or a tie.

    Corroborative rank correlations never override this discrete gate."""
    a_in = delta_atan <= floor
    d_in = delta_dp <= floor
    if a_in and not d_in:
        return "CONFIRM"
    if d_in and not a_in:
        return "CONTRADICT"
    return "AMBIGUOUS"


def prediction_check(
    a_eco: Dict[int, float],
    a_tan: Dict[int, float],
    d_p: Dict[int, float],
    floor_by_k: Dict[int, int],
    k_primary: int = 6,
    ks: Tuple[int, ...] = (2, 4, 6),
) -> dict:
    """Full §6 check from per-block score dicts. Uses LOO removal tails (lowest_k) for the sets and
    reports the primary verdict at k_primary plus secondary k's and corroborative Spearman.

    `floor_by_k[k]` is the §4.1 max bootstrap floor at that k (integer blocks). Returns a dict with
    `verdict` (the primary-k discrete verdict) and a full trace."""
    blocks = sorted(a_eco)
    per_k = {}
    for k in ks:
        R_eco = lowest_k(a_eco, k)
        R_atan = lowest_k(a_tan, k)
        R_dp = lowest_k(d_p, k)
        dA = set_disagreement(R_atan, R_eco)
        dD = set_disagreement(R_dp, R_eco)
        fl = int(floor_by_k.get(k, 0))
        per_k[k] = {
            "R_eco": sorted(R_eco), "R_atan": sorted(R_atan), "R_dp": sorted(R_dp),
            "delta_atan_eco": dA, "delta_dp_eco": dD, "floor": fl,
            "verdict": prediction_verdict(dA, dD, fl),
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
        "note": "LOO removal tails (lowest_k); NOT sequential-greedy masks (§3.5). k=6 primary (rc1).",
    }


# ------------------------------------------------- §3 control localisation verdict (rc1 re-spec)
def control_localization_verdict(
    block_b: int,
    a_eco_b: float,                 # A_eco(b; L_b): fraction of L_b's field effect that vanishes on removing b
    dT_post_minus_b: float,         # ΔT_{L_b}(post^{-b}): uplift left after removing the host block b
    dT_external: Dict[int, float],  # g -> ΔT_{L_b}(post^{-g}) for external removals g != b
    ci_a_eco: float = 0.10,         # CI half-width on the A_eco(b) ≈ 1 sanity
    ci_dT_zero: float = 0.0,        # uncertainty band for "ΔT ≈ 0" (0 -> use dT_measurable as the band)
    dT_measurable: float = 0.02,    # an external removal "retains measurable uplift" if ΔT > this
) -> dict:
    """rc1 STOP gate for a single-block control L_b. The gate rests on δF^{-b}(L_b)=0 (removing the
    host block physically deletes the adapter), NOT on b being the top-1 of the 20 A_eco scores
    (that is reported descriptively — a single-block adapter's *function* can route through
    downstream blocks).

    PASS (all three):
      (1) sanity        : A_eco(b) ≈ 1                    -> a_eco_b >= 1 - ci_a_eco
      (2) uplift collapse: ΔT_{L_b}(post^{-b}) ≈ 0        -> |dT_post_minus_b| <= max(ci_dT_zero, dT_measurable)
      (3) observability : some external g keeps ΔT > 0    -> max_g dT_external[g] > dT_measurable
    Else STOP RQ2 (the chain cannot localise a known adaptation from outputs)."""
    zero_band = max(ci_dT_zero, dT_measurable)
    cond_sanity = a_eco_b >= (1.0 - ci_a_eco)
    cond_collapse = abs(dT_post_minus_b) <= zero_band
    ext_max = max(dT_external.values()) if dT_external else float("-inf")
    ext_argmax = max(dT_external, key=dT_external.get) if dT_external else None
    cond_observability = ext_max > dT_measurable
    passed = cond_sanity and cond_collapse and cond_observability
    # descriptive-only: is b the least-removable (highest A_eco) block? not a gate.
    return {
        "block_b": block_b,
        "pass": bool(passed),
        "verdict": "PASS" if passed else "STOP_RQ2",
        "cond_sanity_A_eco_b_near_1": bool(cond_sanity),
        "cond_uplift_collapse_near_0": bool(cond_collapse),
        "cond_external_uplift_observable": bool(cond_observability),
        "a_eco_b": a_eco_b,
        "dT_post_minus_b": dT_post_minus_b,
        "external_uplift_max": ext_max if dT_external else None,
        "external_uplift_argmax_block": ext_argmax,
        "note": "rc1: top-1 ranking of A_eco(b) is descriptive, not a STOP criterion.",
    }
