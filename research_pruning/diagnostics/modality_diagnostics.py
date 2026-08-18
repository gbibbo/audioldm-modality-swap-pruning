"""Modality-swap diagnostics D_gen / D_mod / R_mod (master plan section 3).

These are DIAGNOSTICS, never a pruning loss. For the same example, noisy latent
z_t, timestep t and noise realisation, with epsilon predictions from the full
model (`eps_Fa`, `eps_Ft`) and a pruned/control model (`eps_Pa`, `eps_Pt`) under
audio and text conditioning, define the pruning errors and diagnostics:

    E_a  = eps_Pa - eps_Fa
    E_t  = eps_Pt - eps_Ft
    D_gen = 0.5 * (||E_a|| + ||E_t||)          generic pruning damage
    D_mod = ||E_a - E_t||                      modality-dependent damage
    R_mod = ||E_a - E_t|| / (||E_a|| + ||E_t|| + eps)

FROZEN CHOICES (change only with a written rationale in the ledger):

* **Norm = L2 (Euclidean) over the flattened per-example latent**, i.e. one scalar
  norm per example computed over all latent dimensions (C * H * W). Rationale: the
  diffusion objective is an L2 (MSE) loss on epsilon, so L2 is the norm in which
  pruning error is naturally measured; it makes R_mod scale-free and, by the
  triangle inequality, bounded in [0, 1].
* **epsilon = 1e-12** in the R_mod denominator. Rationale: it exists ONLY to keep
  R_mod finite when both errors vanish (the identity control, where it yields
  exactly 0). It is ~12 orders of magnitude below any real epsilon-error norm
  (which live at O(1e-1..1e2) on this model), so it never perturbs a real value.

AGGREGATION (master plan section 6, applied by `aggregate_over_strata`):
equal-weight mean across pre-registered timestep strata, THEN mean across
examples. Timestep-specific curves are secondary; the primary conclusion must not
depend on a favourable timestep chosen after seeing results.
"""
from __future__ import annotations

import torch

EPS_DEFAULT: float = 1e-12


def _flat_l2(x: torch.Tensor) -> torch.Tensor:
    """L2 norm per example over all non-batch dims → shape [B]."""
    return x.reshape(x.shape[0], -1).norm(p=2, dim=1)


def modality_diagnostics(
    eps_Fa: torch.Tensor,
    eps_Ft: torch.Tensor,
    eps_Pa: torch.Tensor,
    eps_Pt: torch.Tensor,
    eps: float = EPS_DEFAULT,
) -> dict:
    """Return per-example D_gen, D_mod, R_mod and the component norms.

    All four inputs share shape [B, C, H, W] and correspond to the SAME z_t, t and
    noise; only the model (full vs pruned) and modality (audio vs text) differ.
    """
    if not (eps_Fa.shape == eps_Ft.shape == eps_Pa.shape == eps_Pt.shape):
        raise ValueError("all four epsilon tensors must share shape")
    E_a = eps_Pa - eps_Fa
    E_t = eps_Pt - eps_Ft
    n_a = _flat_l2(E_a)
    n_t = _flat_l2(E_t)
    n_diff = _flat_l2(E_a - E_t)
    d_gen = 0.5 * (n_a + n_t)
    d_mod = n_diff
    r_mod = n_diff / (n_a + n_t + eps)
    return {
        "D_gen": d_gen,
        "D_mod": d_mod,
        "R_mod": r_mod,
        "norm_E_a": n_a,
        "norm_E_t": n_t,
        "norm_E_diff": n_diff,
    }


def aggregate_over_strata(per_stratum: list[dict]) -> dict:
    """Aggregate per-example diagnostics across timestep strata (master plan §6).

    `per_stratum` is a list of `modality_diagnostics` outputs, one per timestep
    stratum. Each is reduced to its across-examples mean, then the strata are
    combined with equal weight (mean of stratum means). Returns scalar floats.
    """
    if not per_stratum:
        raise ValueError("per_stratum is empty")
    keys = ["D_gen", "D_mod", "R_mod", "norm_E_a", "norm_E_t", "norm_E_diff"]
    out = {}
    for k in keys:
        stratum_means = torch.stack([s[k].mean() for s in per_stratum])
        out[k] = float(stratum_means.mean().item())
    return out
