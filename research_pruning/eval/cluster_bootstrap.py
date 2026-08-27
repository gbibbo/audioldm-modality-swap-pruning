"""Cluster (prompt-level) percentile bootstrap — the ONE CI definition for ICASSP Gate 0
and every phenomenon severity (frozen at DECISION-V4-09, 2026-08-26).

Frozen spec (do NOT change after seeing data):

* Resampling unit = **prompt**. The three generation seeds stay clustered inside each
  resampled prompt: we first reduce each prompt to a single per-prompt scalar (its
  within-prompt paired mean over the 3 seeds), then resample prompts with replacement.
  Seeds are never resampled independently, so n=192 clips are never treated as 192
  independent samples.
* Interval = two-sided **95 %** **percentile** bootstrap (2.5 / 97.5), **B = 10000**,
  fixed RNG **seed = 20260826**. No BCa/normal/t; no method switch after data.

Every gate calls `cluster_percentile_ci`; the verdict helpers below build the per-prompt
scalars for Gate 0 (ΔCLAP), standalone non-inferiority E(s), and differential adapter
fragility **D(s) = ΔCLAP(0) − ΔCLAP(s)**, and apply the pre-registered thresholds.

STATISTIC CORRECTION (prereg v5, 2026-08-27, PRE-PHENOMENON-DATA — ledger PHENOM-STAT-D):
D(s) already equals the excess degradation of the adapter-equipped system relative to the
standalone degradation:

    D = Δ0 − Δs = (A0 − C0) − (As − Cs) = (A0 − As) − (C0 − Cs)

(A0/As = dense/compressed + adapter, C0/Cs = dense/compressed standalone). So D IS the
"legacy adapter's uplift is disproportionately damaged" quantity the prereg strong-claim
prose already named. The old fragility statistic F = D − E = (A0−As) − 2(C0−Cs) has no
intended scientific interpretation and is BIDIRECTIONALLY unsafe (it can hide genuine
fragility when E>0 and can MANUFACTURE a fragility PASS that D denies when E<0). The
decision gate now uses **D**. F is retained ONLY as deprecated provenance, clearly invalid
for inference — never a gate input.

Score arrays are shaped (n_prompts, n_seeds). Rows are prompts (the cluster). Columns are
the paired generation seeds; the same seed index means the same generation seed across
conditions (paired). All conditions in a comparison must share prompt order and seed order.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- FROZEN at DECISION-V4-09 -------------------------------------------------
BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 20260826
CI_ALPHA = 0.05  # two-sided 95%
N_PROMPTS = 64
N_SEEDS = 3
SESOI = 0.025            # Gate-0 minimum meaningful ΔCLAP
NONINF_MARGIN = 0.025    # standalone preserved iff upper-CI95[E(s)] <= this
FRAGILITY_MARGIN = 0.025  # differential fragility iff point D(s) >= this AND lower-CI95[D]>0
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float
    n: int
    b: int

    def as_dict(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi, "n": self.n, "b": self.b}


def cluster_percentile_ci(per_prompt_values, b: int = BOOTSTRAP_B,
                          seed: int = BOOTSTRAP_SEED, alpha: float = CI_ALPHA) -> CI:
    """Percentile bootstrap CI of the mean of per-prompt scalars.

    `per_prompt_values`: 1-D array, one scalar per prompt (already reduced over seeds).
    Point estimate = mean over prompts. Bootstrap resamples PROMPTS with replacement.
    A 2-D array is rejected (not silently ravelled) to prevent the n=192 misuse; reduce
    over seeds first (`per_prompt_*`) or `.ravel()` explicitly if you truly mean flat.
    """
    v = np.asarray(per_prompt_values, dtype=np.float64)
    if v.ndim != 1 or v.size == 0:
        raise ValueError("per_prompt_values must be a non-empty 1-D array (reduce seeds first)")
    n = v.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(b, n))          # (B, n) prompt indices, with replacement
    boot_means = v[idx].mean(axis=1)               # mean over resampled prompts
    lo = float(np.percentile(boot_means, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return CI(point=float(v.mean()), lo=lo, hi=hi, n=n, b=b)


def _validate_pair(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"paired arrays must match: {a.shape} vs {b.shape}")
    if a.ndim != 2:
        raise ValueError(f"expected (n_prompts, n_seeds), got {a.shape}")


def per_prompt_paired_uplift(adapter: np.ndarray, base: np.ndarray) -> np.ndarray:
    """ΔCLAP per prompt = within-prompt mean over seeds of (adapter − base). Shapes
    (n_prompts, n_seeds), seed-paired. Returns (n_prompts,)."""
    adapter = np.asarray(adapter, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    _validate_pair(adapter, base)
    return (adapter - base).mean(axis=1)


def per_prompt_standalone(standalone: np.ndarray) -> np.ndarray:
    """C per prompt = within-prompt mean over seeds of the standalone score. (n_prompts,)."""
    s = np.asarray(standalone, dtype=np.float64)
    if s.ndim != 2:
        raise ValueError(f"expected (n_prompts, n_seeds), got {s.shape}")
    return s.mean(axis=1)


@dataclass(frozen=True)
class Gate0Verdict:
    delta_clap: CI
    passed: bool

    def as_dict(self) -> dict:
        return {"delta_clap": self.delta_clap.as_dict(), "SESOI": SESOI, "pass": self.passed}


def gate0_verdict(adapter_dense: np.ndarray, base_dense: np.ndarray) -> Gate0Verdict:
    """Gate 0 on dense: PASS iff point ΔCLAP >= SESOI AND lower-CI95 > 0."""
    dc = cluster_percentile_ci(per_prompt_paired_uplift(adapter_dense, base_dense))
    passed = (dc.point >= SESOI) and (dc.lo > 0.0)
    return Gate0Verdict(delta_clap=dc, passed=passed)


@dataclass(frozen=True)
class SeverityVerdict:
    severity: str
    E: CI            # standalone degradation C(0)-C(s)
    D: CI            # DECISION STATISTIC: adapter uplift lost dCLAP(0)-dCLAP(s)
    F_deprecated: CI  # DEPRECATED provenance ONLY (old D-E); INVALID for inference
    standalone_preserved: bool
    differential_fragility: bool
    phenomenon: bool  # BOTH conditions

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "E": self.E.as_dict(), "D": self.D.as_dict(),
            # F retained only for provenance/audit; never a gate input (prereg v5).
            "F_deprecated_invalid_for_inference": self.F_deprecated.as_dict(),
            "standalone_preserved": self.standalone_preserved,
            "differential_fragility": self.differential_fragility,
            "phenomenon": self.phenomenon,
            "decision_statistic": "D",
            "NONINF_MARGIN": NONINF_MARGIN, "FRAGILITY_MARGIN": FRAGILITY_MARGIN,
        }


def severity_verdict(severity: str,
                     standalone_dense: np.ndarray, standalone_pruned: np.ndarray,
                     adapter_dense: np.ndarray, base_dense: np.ndarray,
                     adapter_pruned: np.ndarray, base_pruned: np.ndarray) -> SeverityVerdict:
    """Dual pre-registered gate at one severity s (all arrays (n_prompts, n_seeds),
    same prompt/seed order = the same held-out battery at s=0 and at s).

    Per-prompt paired construction (reduce seeds FIRST, then bootstrap over prompts):
      E(s) = C(0) - C(s)                        # standalone degradation
      D(s) = ΔCLAP(0) - ΔCLAP(s)                # DECISION statistic (adapter uplift lost)
             = (A0 - As) - (C0 - Cs)            # == excess adapter-equipped degradation
      F_deprecated(s) = D(s) - E(s)             # NO scientific meaning; provenance only

    standalone_preserved   iff upper-CI95[E] <= NONINF_MARGIN.
    differential_fragility  iff point D >= FRAGILITY_MARGIN AND lower-CI95[D] > 0.
    phenomenon              iff BOTH.

    D is computed per prompt from paired (prompt, seed) observations BEFORE bootstrapping
    (the 3 seeds stay clustered inside each prompt), never as a difference of separately
    aggregated group means — see tests/research/test_phenomenon_statistic.py.
    """
    c0 = per_prompt_standalone(standalone_dense)
    cs = per_prompt_standalone(standalone_pruned)
    e_pp = c0 - cs                                            # E per prompt
    dclap0 = per_prompt_paired_uplift(adapter_dense, base_dense)
    dclaps = per_prompt_paired_uplift(adapter_pruned, base_pruned)
    d_pp = dclap0 - dclaps                                    # D per prompt (decision statistic)
    f_pp = d_pp - e_pp                                        # DEPRECATED (provenance only)
    for name, arr in (("standalone", c0), ("E", e_pp), ("D", d_pp), ("F", f_pp)):
        if arr.shape[0] != c0.shape[0]:
            raise ValueError(f"{name}: prompt count mismatch")
    E = cluster_percentile_ci(e_pp)
    D = cluster_percentile_ci(d_pp)
    F = cluster_percentile_ci(f_pp)
    standalone_preserved = E.hi <= NONINF_MARGIN
    differential_fragility = (D.point >= FRAGILITY_MARGIN) and (D.lo > 0.0)   # D, not F
    return SeverityVerdict(
        severity=severity, E=E, D=D, F_deprecated=F,
        standalone_preserved=standalone_preserved,
        differential_fragility=differential_fragility,
        phenomenon=standalone_preserved and differential_fragility,
    )
