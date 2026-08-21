"""E(M) non-inferiority rule, margins, and E-greedy fallback score (protocol section 9.2, 3.5).

Pure decision logic over metric point estimates and CIs; the audio metrics themselves
(CLAP / KL_passt / FD_openl3) live in e2e.py. E(M) = (CLAP, KL, FD): CLAP higher-better,
KL/FD lower-better (drift from dense post model). All comparisons are seed-paired with CIs.
"""
from __future__ import annotations
from typing import Dict, Tuple


def margins(clap_8: float, clap_7: float, kl_7v8: float, fd_7v8: float,
            r_clap: float, r_kl: float, r_fd: float) -> Dict[str, float]:
    """m = max(delta_{8->7} in the deterioration direction, r_m). r_m = 95th pct of |dense-vs-dense
    seed differences| (resolution floor only; keeps margins strictly positive)."""
    d_clap = max(0.0, clap_8 - clap_7)   # CLAP drops => positive deterioration
    d_kl = max(0.0, kl_7v8)              # KL(7 vs 8) drift, already >=0 direction
    d_fd = max(0.0, fd_7v8)
    return {
        "m_CLAP": max(d_clap, r_clap), "m_KL": max(d_kl, r_kl), "m_FD": max(d_fd, r_fd),
        "delta_CLAP": d_clap, "delta_KL": d_kl, "delta_FD": d_fd,
        "r_CLAP": r_clap, "r_KL": r_kl, "r_FD": r_fd,
    }


def noninferiority(
    clap_deficit_ci: Tuple[float, float],  # CI of CLAP(Y) - CLAP(X)  (deficit of X vs Y; >0 = X worse)
    kl_deficit_ci: Tuple[float, float],    # CI of KL(X) - KL(Y)      (>0 = X drifts more)
    fd_deficit_ci: Tuple[float, float],    # CI of FD(X) - FD(Y)
    m: Dict[str, float],
) -> str:
    """Return 'non_inferior' | 'inferior' | 'indeterminate' for X relative to Y (section 9.2).

    non-inferior: upper CI of all three deficits <= their margins.
    inferior: lower CI of CLAP deficit > m_CLAP, OR both drift deficits' lower CI > their margins.
    else: indeterminate."""
    cl_lo, cl_hi = clap_deficit_ci
    kl_lo, kl_hi = kl_deficit_ci
    fd_lo, fd_hi = fd_deficit_ci
    if cl_hi <= m["m_CLAP"] and kl_hi <= m["m_KL"] and fd_hi <= m["m_FD"]:
        return "non_inferior"
    if (cl_lo > m["m_CLAP"]) or (kl_lo > m["m_KL"] and fd_lo > m["m_FD"]):
        return "inferior"
    return "indeterminate"


def egreedy_score(clap: float, kl: float, fd: float, clap_dense: float, m: Dict[str, float]) -> float:
    """Section 3.5 fallback: smallest maximum normalised deterioration over all three metrics,
    so the primary metric never drops out. ΔKL/ΔFD are drift vs dense (already >=0-directed)."""
    return max(
        max(0.0, clap_dense - clap) / m["m_CLAP"],
        max(0.0, kl) / m["m_KL"],
        max(0.0, fd) / m["m_FD"],
    )


def within_caps(kl: float, fd: float, m: Dict[str, float]) -> bool:
    """Candidate satisfies both drift caps (KL and FD within margins)."""
    return kl <= m["m_KL"] and fd <= m["m_FD"]
