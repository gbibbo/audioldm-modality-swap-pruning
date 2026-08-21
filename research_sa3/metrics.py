"""Executable metric definitions (protocol section 3, 4). PURE numeric functions over
already-reduced squared-norm accumulators, so they are fully unit-testable on synthetic data
with no model and no metric stack. fields.py produces the accumulators; metrics.py combines them.

Notation: ||.||^2_S = mean over a state set of the per-state squared L2 norm over the latent
(256 x T), padding-masked and divided by T. All inputs here are those reduced scalars/arrays.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import math


# ---------------------------------------------------------------- RQ1: block damage (3.1)
def block_damage(num_by_block: Dict[int, float], den: float) -> Dict[int, float]:
    """D(g) = ||F - F^{-g}||^2_S / ||F||^2_S.  num_by_block[g] = ||F - F^{-g}||^2_S ; den = ||F||^2_S."""
    if den <= 0:
        raise ValueError("den (||F||^2_S) must be > 0")
    return {g: num_by_block[g] / den for g in sorted(num_by_block)}


# ------------------------------------------- RQ1: normalized post-training-delta distortion (3.2)
def i_pt(
    num_by_level_block: Dict[int, List[float]],   # g -> [num_i(g)]_{i=0..L-1}, num_i = ||Delta_i - Delta_i^{-g}||^2
    den_by_level: List[float],                    # den_i = ||Delta_i||^2
    fp_sq_by_level: List[float],                  # ||F_{P,i}||^2  (for the eta guard denominator)
    eta_by_level: List[float],                    # eta_i measured (fp16 vs fp32) per level
) -> Dict[int, dict]:
    """Pooled ratio of sums (not mean of ratios); per-level table; eta denominator guard.

    A level i is EXCLUDED from the pooled ratio (and flagged) iff den_i / ||F_{P,i}||^2 < eta_i:
    the post-training delta at that level is below the precision floor. Per-level values are still
    reported for excluded levels."""
    L = len(den_by_level)
    assert len(fp_sq_by_level) == L and len(eta_by_level) == L
    excluded = [i for i in range(L) if fp_sq_by_level[i] > 0 and den_by_level[i] / fp_sq_by_level[i] < eta_by_level[i]]
    kept = [i for i in range(L) if i not in excluded]
    out: Dict[int, dict] = {}
    for g in sorted(num_by_level_block):
        nums = num_by_level_block[g]
        assert len(nums) == L, (g, len(nums), L)
        per_level = [nums[i] / den_by_level[i] if den_by_level[i] > 0 else float("nan") for i in range(L)]
        num_sum = sum(nums[i] for i in kept)
        den_sum = sum(den_by_level[i] for i in kept)
        pooled = num_sum / den_sum if den_sum > 0 else float("nan")
        out[g] = {"pooled": pooled, "per_level": per_level}
    return {"per_block": out, "excluded_levels": excluded, "kept_levels": kept}


# ------------------------------------------------------------- RQ1: parameter-delta covariate (3.3)
def param_delta_W(sq_delta_by_block: Dict[int, float], sq_base_by_block: Dict[int, float]) -> Dict[int, float]:
    """W(g) = sum_theta ||theta_P - theta_B||_F^2 / sum_theta ||theta_B||_F^2 over block g's params.
    Inputs are the per-block Frobenius sums. Covariate only (never an effect estimate)."""
    return {g: (sq_delta_by_block[g] / sq_base_by_block[g] if sq_base_by_block[g] > 0 else float("nan"))
            for g in sorted(sq_delta_by_block)}


# ---------------------------------------------------------------------- RQ2: adaptability (3.4)
def adaptability(
    e_carry: Dict[int, float],   # E_u ||dF(u_g)||^2         (effect living in g's own slots)
    e_full: float,               # E_u ||dF(u)||^2
    e_int_num: Dict[int, float], # E_u ||dF(u_{-g}) - dF^{-g}(u_{-g})||^2
    e_int_den: Dict[int, float], # E_u ||dF(u_{-g})||^2
    e_tan_num: Dict[int, float], # E_u ||dF(u) - dF^{-g}(u)||^2
) -> Dict[int, dict]:
    """A_carry, A_int, A_tan per block (all normalized ratios of probe-averaged squared norms)."""
    out = {}
    for g in sorted(e_carry):
        out[g] = {
            "A_carry": e_carry[g] / e_full if e_full > 0 else float("nan"),
            "A_int": e_int_num[g] / e_int_den[g] if e_int_den[g] > 0 else float("nan"),
            "A_tan": e_tan_num[g] / e_full if e_full > 0 else float("nan"),
        }
    return out


def a_eco(num_by_block: Dict[int, float], den: float) -> Dict[int, float]:
    """A_eco(g;L) = ||dF(L) - dF^{-g}(L)||^2 / ||dF(L)||^2 for a real held-out adapter L."""
    if den <= 0:
        raise ValueError("den = ||dF(L)||^2 must be > 0")
    return {g: num_by_block[g] / den for g in sorted(num_by_block)}


# ------------------------------------------------------------------ linearity / precision checks
def linearity_ratio(norm_2u: float, norm_u: float) -> float:
    """||dF(2u)|| / ||dF(u)|| ; must be in [1.9, 2.1] for the tangent regime (3.4)."""
    return norm_2u / norm_u if norm_u > 0 else float("nan")


def precision_ok(df_sq: float, fp_sq: float, eta: float, factor: float = 10.0) -> bool:
    """||dF(u)||^2 / ||F_P||^2 must exceed eta by >= factor (default 10)."""
    return fp_sq > 0 and (df_sq / fp_sq) >= factor * eta
