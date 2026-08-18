"""Matched random-null statistic for M3A Gate A (master plan §M3A).

Random pruning may have much larger generic damage (`D_gen`) than L1, so raw
`R_mod` cannot be compared blindly. The plan instead:

  1. fits the relationship `R_mod ~ f(D_gen)` across the random controls;
  2. evaluates the expected random `R_mod` at L1's observed `D_gen`;
  3. defines `Delta_swap = R_mod^L1 - E[R_mod^random | D_gen^L1]`;
  4. bootstraps over evaluation examples AND random masks.

**Functional form of the fit: LINEAR** (`R_mod = a + b*D_gen`), fitted by ordinary
least squares over the per-mask control points. Justification: within the narrow
`D_gen` band around L1 the `R_mod`-vs-`D_gen` relationship is expected smooth and
monotone, and a line is the simplest defensible model whose residual scatter gives
the control SD used by the standardized-residual gate. `R^2` and residual SD are
returned as fit diagnostics; if a future dataset shows clear curvature the form
should be revisited (isotonic/`log`) and the change recorded in the ledger — this
is why the form is an explicit argument, not hardcoded.

**Bootstrap unit = wav** (never the caption-wav entry). Resampling captions as
independent units would be pseudo-replication and would narrow the CI, letting
Gate A pass by construction. `bootstrap_delta_swap` therefore requires unique wav
ids in its pool.
"""
from __future__ import annotations

import numpy as np


def fit_null_curve(d_gen, r_mod, form: str = "linear") -> dict:
    """Fit R_mod ~ f(D_gen) across control points. Returns coefficients + diagnostics."""
    d = np.asarray(d_gen, dtype=np.float64)
    r = np.asarray(r_mod, dtype=np.float64)
    if d.shape != r.shape or d.ndim != 1:
        raise ValueError("d_gen and r_mod must be 1-D arrays of equal length")
    if form != "linear":
        raise NotImplementedError(
            f"only 'linear' is implemented; requested {form!r}. Change the form only "
            "with a written rationale in docs/experiment_ledger.md."
        )
    b, a = np.polyfit(d, r, deg=1)  # r ~ a + b*d
    pred = a + b * d
    resid = r - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    resid_sd = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    return {"form": "linear", "a": float(a), "b": float(b),
            "resid_sd": resid_sd, "r2": r2, "n": int(d.size)}


def predict_null(fit: dict, d_gen: float) -> float:
    return float(fit["a"] + fit["b"] * float(d_gen))


def delta_swap(r_mod_l1: float, fit: dict, d_gen_l1: float) -> float:
    return float(r_mod_l1) - predict_null(fit, d_gen_l1)


def standardized_residual(delta: float, fit: dict) -> float:
    """Delta_swap expressed in random-control SD units."""
    sd = fit["resid_sd"]
    return float(delta) / sd if sd > 0 else float("nan")


def point_estimate(l1_dgen, l1_rmod, rand_dgen, rand_rmod, form: str = "linear") -> dict:
    """Delta_swap and its standardized residual from per-wav L1 data and per-mask/
    per-wav random data.

    l1_dgen, l1_rmod : [W]        per-wav L1 diagnostics
    rand_dgen, rand_rmod : [M, W] per-mask, per-wav random diagnostics
    The control points for the fit are the per-mask means.
    """
    rand_dgen = np.asarray(rand_dgen, dtype=np.float64)
    rand_rmod = np.asarray(rand_rmod, dtype=np.float64)
    ctrl_dgen = rand_dgen.mean(axis=1)   # [M]
    ctrl_rmod = rand_rmod.mean(axis=1)   # [M]
    fit = fit_null_curve(ctrl_dgen, ctrl_rmod, form=form)
    l1_dgen_m = float(np.mean(l1_dgen))
    l1_rmod_m = float(np.mean(l1_rmod))
    delta = delta_swap(l1_rmod_m, fit, l1_dgen_m)
    return {
        "fit": fit,
        "d_gen_L1": l1_dgen_m,
        "r_mod_L1": l1_rmod_m,
        "expected_random_r_mod": predict_null(fit, l1_dgen_m),
        "delta_swap": delta,
        "standardized_residual": standardized_residual(delta, fit),
    }


def bootstrap_delta_swap(
    wav_ids,
    l1_dgen, l1_rmod,
    rand_dgen, rand_rmod,
    n_boot: int = 10000,
    seed: int = 20260818,
    form: str = "linear",
    ci: float = 0.95,
) -> dict:
    """Bootstrap Delta_swap by resampling WAVS (unit) and random MASKS.

    wav_ids : [W] identifiers; MUST be unique (bootstrap unit is the wav, never the
              caption-wav entry — see module docstring). Raises otherwise.
    """
    wav_ids = list(wav_ids)
    if len(set(wav_ids)) != len(wav_ids):
        raise ValueError(
            "bootstrap pool contains repeated wav ids; the resampling unit must be "
            "the wav, not the caption-wav entry (pseudo-replication would narrow the CI)"
        )
    l1_dgen = np.asarray(l1_dgen, dtype=np.float64)
    l1_rmod = np.asarray(l1_rmod, dtype=np.float64)
    rand_dgen = np.asarray(rand_dgen, dtype=np.float64)
    rand_rmod = np.asarray(rand_rmod, dtype=np.float64)
    W = len(wav_ids)
    M = rand_dgen.shape[0]
    rng = np.random.default_rng(seed)

    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        w = rng.integers(0, W, size=W)         # resample wavs with replacement
        m = rng.integers(0, M, size=M)         # resample masks with replacement
        est = point_estimate(
            l1_dgen[w], l1_rmod[w],
            rand_dgen[np.ix_(m, w)], rand_rmod[np.ix_(m, w)],
            form=form,
        )
        deltas[i] = est["delta_swap"]

    lo = float(np.quantile(deltas, (1 - ci) / 2))
    hi = float(np.quantile(deltas, 1 - (1 - ci) / 2))
    base = point_estimate(l1_dgen, l1_rmod, rand_dgen, rand_rmod, form=form)
    return {
        "delta_swap": base["delta_swap"],
        "standardized_residual": base["standardized_residual"],
        "ci": ci,
        "ci_low": lo,
        "ci_high": hi,
        "ci_excludes_zero": (lo > 0) or (hi < 0),
        "n_boot": n_boot,
        "seed": seed,
        "fit": base["fit"],
    }
