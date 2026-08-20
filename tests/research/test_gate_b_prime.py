#!/usr/bin/env python3
"""Gate B' machinery tests on a CONTROL model (CPU-only, synthetic). CPU queue Q5.

No real checkpoint, no GPU. Validates per-slot saliency storage, CPU recomposition, and
the null-split kept-set-overlap distribution that Gate B' uses (plan v4 §6).

    G1 RECOMPOSE   recompose_mean(all slots) == accumulate_taylor to FP precision
                   (max|Δ| < 1e-6; the tiny residual is summation order — running sum
                   vs tree reduction — not a logic difference): storing per slot and
                   averaging recovers the same saliency.
    G2 NULL-DIST   null_split_overlaps returns n_splits overlaps in [0,1], is
                   deterministic under a seed, and its median sits well above chance
                   (two halves of the SAME data agree far more than random).
    G3 FIRE        overlap(natural, an INDEPENDENT random saliency) falls below the 5th
                   percentile of the half-half null -> gate_b_prime PASS (a real mask
                   change is detected), with a small p-value.
    G4 NO-FIRE     overlap(natural, natural) == 1.0 is NOT below the 5th percentile ->
                   gate_b_prime does NOT fire (p_value == 1.0): identical masks are not
                   a change (guards against false positives).

Run: .venv/bin/python tests/research/test_gate_b_prime.py
"""
from __future__ import annotations

import sys

import numpy as np
import torch
from torch import nn

from research_pruning.taylor import attach_gates, accumulate_taylor, keep_topk
from research_pruning.paired_modality.gate_b_prime import (
    per_slot_saliency, recompose_mean, overlap_between_saliencies,
    null_split_overlaps, gate_b_prime,
)


class ControlNet(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)

    def forward(self, x):
        return self.conv2(torch.relu(self.conv1(x)))


LAYERS = ["conv1", "conv2"]
N_PER = {"conv1": 32, "conv2": 16}
K_PER = {"conv1": 16, "conv2": 8}          # keep half of each layer


def _setup(n_slots=40, seed=0):
    m = ControlNet(seed=seed)
    gates = attach_gates(m, LAYERS)
    torch.manual_seed(seed + 1)
    slots = [torch.randn(2, 3, 8, 8) for _ in range(n_slots)]
    target = torch.randn(2, 16, 8, 8)

    def loss_fn(slot):
        return ((m(slot) - target) ** 2).mean()

    return m, gates, slots, loss_fn


def check_g1_recompose() -> bool:
    _, gates, slots, loss_fn = _setup()
    accum = accumulate_taylor(gates, loss_fn, slots)
    per = per_slot_saliency(gates, loss_fn, slots)
    recon = recompose_mean(per)
    worst = max(float((recon[k] - accum[k]).abs().max()) for k in accum)
    shapes = {k: tuple(per[k].shape) for k in per}
    print(f"    per-slot shapes {shapes}; max|recompose - accumulate| = {worst:.2e} (FP)")
    return worst < 1e-6 and all(per[k].shape == (len(slots), N_PER[k]) for k in per)


def check_g2_null_dist() -> bool:
    _, gates, slots, loss_fn = _setup()
    per = per_slot_saliency(gates, loss_fn, slots)
    null1 = null_split_overlaps(per, K_PER, N_PER, n_splits=1000, seed=20260818)
    null2 = null_split_overlaps(per, K_PER, N_PER, n_splits=1000, seed=20260818)
    in_range = bool((null1 >= 0).all() and (null1 <= 1).all())
    deterministic = bool(np.array_equal(null1, null2))
    chance = np.mean([K_PER[l] / N_PER[l] for l in LAYERS])   # ~0.5
    median = float(np.median(null1))
    print(f"    null n={null1.size} median={median:.3f} (chance≈{chance:.2f}) "
          f"in[0,1]={in_range} deterministic={deterministic}")
    return in_range and deterministic and median > chance + 0.1


def check_g3_fire() -> bool:
    _, gates, slots, loss_fn = _setup()
    per = per_slot_saliency(gates, loss_fn, slots)
    nat = recompose_mean(per)
    null = null_split_overlaps(per, K_PER, N_PER, n_splits=1000, seed=20260818)
    # an INDEPENDENT random saliency = a maximal, unrelated mask change
    rng = torch.Generator().manual_seed(999)
    variant = {l: torch.rand(N_PER[l], generator=rng) for l in LAYERS}
    obs = overlap_between_saliencies(nat, variant, K_PER, N_PER)
    res = gate_b_prime(obs, null)
    print(f"    independent variant overlap={obs:.3f} thr5%={res['threshold_pctile']:.3f} "
          f"p={res['p_value']:.4f} pass={res['pass']}")
    return res["pass"] and res["p_value"] < 0.05


def check_g4_no_fire() -> bool:
    _, gates, slots, loss_fn = _setup()
    per = per_slot_saliency(gates, loss_fn, slots)
    nat = recompose_mean(per)
    null = null_split_overlaps(per, K_PER, N_PER, n_splits=1000, seed=20260818)
    obs = overlap_between_saliencies(nat, nat, K_PER, N_PER)   # identical -> 1.0
    res = gate_b_prime(obs, null)
    print(f"    identical overlap={obs:.3f} thr5%={res['threshold_pctile']:.3f} "
          f"p={res['p_value']:.4f} pass={res['pass']}")
    return (not res["pass"]) and obs == 1.0 and res["p_value"] == 1.0


def main() -> int:
    checks = [
        ("G1 RECOMPOSE", check_g1_recompose),
        ("G2 NULL-DIST", check_g2_null_dist),
        ("G3 FIRE", check_g3_fire),
        ("G4 NO-FIRE", check_g4_no_fire),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as e:
            print(f"    ERROR: {e!r}")
            results[name] = False
    print("\n==== GATE B' MACHINERY TESTS ====")
    for name, _ in checks:
        print(f"  {name:<14} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
