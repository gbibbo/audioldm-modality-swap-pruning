#!/usr/bin/env python3
"""Property tests for the modality-swap diagnostics D_gen / D_mod / R_mod (CPU).

These tests validate the DIAGNOSTIC MACHINERY only, using CONTROL models and
constructed error tensors. They NEVER touch the real pruned checkpoint
`l1_audioldm-m-full_p1.ckpt`; computing any D_gen/D_mod/R_mod on it before the
pilot protocol is frozen and Compute Gate CG is resolved would contaminate the
pre-registration. All "pruned" models here are perturbed copies of a tiny,
randomly-initialised control U-Net, and the algebraic properties (D2..D5) are
checked on error tensors E_a, E_t — the exact quantities a control model
produces.

    D1 IDENTITY    pruned == exact copy of full  → D_gen = D_mod = R_mod = 0 exact
                   (end-to-end through eps_pred + the FiLM interface).
    D2 BOUNDS      R_mod ∈ [0, 1] over >= 20 perturbations of varying magnitude.
    D3 MONOTONE    perturbing ONLY the audio error → D_mod increases with magnitude.
    D4 SYMMETRY    swapping audio<->text leaves D_mod and D_gen invariant.
    D5 ISOLATION   an identical error in both modalities → D_mod ~ 0 while D_gen > 0
                   (this is the property that lets D_mod separate modal from generic
                   damage — the premise of RQ1).

Run: .venv/bin/python tests/research/test_diagnostics.py
"""
from __future__ import annotations

import copy
import sys

import torch

from research_pruning.diagnostics.modality_diagnostics import (
    EPS_DEFAULT,
    modality_diagnostics,
)
from research_pruning.diagnostics.conditioning import eps_pred
from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel

B, C, H, W = 3, 8, 16, 8
CONTROL_SEED = 7


def tiny_unet(seed: int = CONTROL_SEED) -> UNetModel:
    """A small FiLM-conditioned U-Net control model (random weights permitted:
    this is a CONTROL, not the scientific model)."""
    torch.manual_seed(seed)
    unet = UNetModel(
        image_size=W,
        in_channels=C,
        out_channels=C,
        model_channels=32,  # divisible by GroupNorm32's 32 groups
        attention_resolutions=[],
        num_res_blocks=1,
        channel_mult=[1, 2],
        num_head_channels=8,
        use_spatial_transformer=False,
        extra_film_condition_dim=512,
    )
    unet.eval()
    return unet


def fixed_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z_t = torch.randn(B, C, H, W, generator=g)
    t = torch.randint(0, 1000, (B,), generator=g, dtype=torch.long)
    cond_a = torch.randn(B, 1, 512, generator=g)
    cond_t = torch.randn(B, 1, 512, generator=g)
    return z_t, t, cond_a, cond_t


def perturb(model: UNetModel, magnitude: float, seed: int) -> UNetModel:
    g = torch.Generator().manual_seed(seed)
    pruned = copy.deepcopy(model)
    with torch.no_grad():
        for p in pruned.parameters():
            p.add_(magnitude * torch.randn(p.shape, generator=g))
    return pruned


# --------------------------------------------------------------------------- #
def check_d1_identity():
    full = tiny_unet()
    pruned = copy.deepcopy(full)  # exact copy
    z_t, t, cond_a, cond_t = fixed_inputs()
    with torch.no_grad():
        eps_Fa = eps_pred(full, z_t, t, cond_a)
        eps_Ft = eps_pred(full, z_t, t, cond_t)
        eps_Pa = eps_pred(pruned, z_t, t, cond_a)
        eps_Pt = eps_pred(pruned, z_t, t, cond_t)
    d = modality_diagnostics(eps_Fa, eps_Ft, eps_Pa, eps_Pt)
    dg = d["D_gen"].abs().max().item()
    dm = d["D_mod"].abs().max().item()
    rm = d["R_mod"].abs().max().item()
    print(f"    D1 max D_gen={dg:.3e} D_mod={dm:.3e} R_mod={rm:.3e}")
    ok = dg == 0.0 and dm == 0.0 and rm == 0.0
    print(f"    D1 {'ok ' if ok else 'FAIL'} identity → all diagnostics exactly 0")
    return ok


def check_d2_bounds():
    full = tiny_unet()
    z_t, t, cond_a, cond_t = fixed_inputs()
    with torch.no_grad():
        eps_Fa = eps_pred(full, z_t, t, cond_a)
        eps_Ft = eps_pred(full, z_t, t, cond_t)
    # 22 magnitudes spanning ~2.5 decades but kept small enough that the random
    # tiny U-Net stays finite (huge perturbations overflow to inf/NaN, which would
    # be a control-model artefact, not a diagnostic-bound violation).
    magnitudes = [0.003 * (1.25 ** i) for i in range(22)]  # 0.003 .. ~0.53
    all_ok = True
    finite = True
    rmin, rmax = 1.0, 0.0
    for i, m in enumerate(magnitudes):
        pruned = perturb(full, m, seed=100 + i)
        with torch.no_grad():
            eps_Pa = eps_pred(pruned, z_t, t, cond_a)
            eps_Pt = eps_pred(pruned, z_t, t, cond_t)
        r = modality_diagnostics(eps_Fa, eps_Ft, eps_Pa, eps_Pt)["R_mod"]
        if not torch.isfinite(r).all():
            finite = False
            all_ok = False
            continue
        rmin = min(rmin, r.min().item())
        rmax = max(rmax, r.max().item())
        if not (r.min().item() >= 0.0 and r.max().item() <= 1.0):
            all_ok = False
    print(f"    D2 over {len(magnitudes)} perturbations: R_mod ∈ [{rmin:.4f}, {rmax:.4f}], all finite={finite}")
    print(f"    D2 {'ok ' if all_ok else 'FAIL'} R_mod within [0,1] (triangle inequality)")
    return all_ok


def check_d3_monotone():
    # Perturb ONLY the audio error: E_t = 0, E_a = m * delta.  D_mod = ||E_a - E_t||.
    g = torch.Generator().manual_seed(3)
    base = torch.randn(B, C, H, W, generator=g)
    delta = torch.randn(B, C, H, W, generator=g)
    prev = -1.0
    ok = True
    vals = []
    for m in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        eps_Fa = base
        eps_Ft = base
        eps_Pa = base + m * delta   # audio error only
        eps_Pt = base               # text error zero
        dm = modality_diagnostics(eps_Fa, eps_Ft, eps_Pa, eps_Pt)["D_mod"].mean().item()
        vals.append((m, dm))
        if dm < prev - 1e-9:
            ok = False
        prev = dm
    print(f"    D3 D_mod vs magnitude: {[f'{m}:{v:.3f}' for m, v in vals]}")
    strictly_up = all(vals[i][1] < vals[i + 1][1] for i in range(1, len(vals) - 1))
    print(f"    D3 {'ok ' if ok and strictly_up else 'FAIL'} D_mod increases with audio-only perturbation")
    return ok and strictly_up


def check_d4_symmetry():
    g = torch.Generator().manual_seed(4)
    eps_Fa = torch.randn(B, C, H, W, generator=g)
    eps_Ft = torch.randn(B, C, H, W, generator=g)
    eps_Pa = torch.randn(B, C, H, W, generator=g)
    eps_Pt = torch.randn(B, C, H, W, generator=g)
    d = modality_diagnostics(eps_Fa, eps_Ft, eps_Pa, eps_Pt)
    d_swap = modality_diagnostics(eps_Ft, eps_Fa, eps_Pt, eps_Pa)  # swap a<->t
    dmod_inv = torch.allclose(d["D_mod"], d_swap["D_mod"], atol=0, rtol=0)
    dgen_inv = torch.allclose(d["D_gen"], d_swap["D_gen"], atol=0, rtol=0)
    print(f"    D4 D_mod invariant={dmod_inv} D_gen invariant={dgen_inv}")
    ok = dmod_inv and dgen_inv
    print(f"    D4 {'ok ' if ok else 'FAIL'} swapping audio<->text leaves D_mod, D_gen invariant")
    return ok


def check_d5_isolation():
    # Identical (generic) error in both modalities. Using the SAME base for audio
    # and text makes E_a and E_t bit-identical (the subtraction rounds the same
    # way), so D_mod is EXACTLY 0 — not merely small.
    g = torch.Generator().manual_seed(5)
    base = torch.randn(B, C, H, W, generator=g)
    delta = torch.randn(B, C, H, W, generator=g)
    eps_Fa, eps_Ft = base, base
    eps_Pa, eps_Pt = base + delta, base + delta  # identical additive error
    d = modality_diagnostics(eps_Fa, eps_Ft, eps_Pa, eps_Pt)
    dm = d["D_mod"].abs().max().item()
    dg = d["D_gen"].min().item()
    rm = d["R_mod"].abs().max().item()
    print(f"    D5 max D_mod={dm:.3e}  min D_gen={dg:.4f}  max R_mod={rm:.3e}")
    ok = dm == 0.0 and dg > 0.0 and rm == 0.0
    print(f"    D5 {'ok ' if ok else 'FAIL'} generic-only damage → D_mod=0 while D_gen>0")
    return ok


# --------------------------------------------------------------------------- #
def test_d1_identity():
    assert check_d1_identity()


def test_d2_bounds():
    assert check_d2_bounds()


def test_d3_monotone():
    assert check_d3_monotone()


def test_d4_symmetry():
    assert check_d4_symmetry()


def test_d5_isolation():
    assert check_d5_isolation()


def main() -> int:
    checks = [
        ("D1 IDENTITY", check_d1_identity),
        ("D2 BOUNDS", check_d2_bounds),
        ("D3 MONOTONE", check_d3_monotone),
        ("D4 SYMMETRY", check_d4_symmetry),
        ("D5 ISOLATION", check_d5_isolation),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        results[name] = bool(fn())
    print("\n==== M3A DIAGNOSTIC MACHINERY TESTS ====")
    for name, _ in checks:
        print(f"  {name:<14} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}  (epsilon={EPS_DEFAULT})")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
