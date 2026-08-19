#!/usr/bin/env python3
"""Control tests for the amended Gate B kept-set overlap statistic (M3B, CPU-only).

Pure set algebra on synthetic saliencies with a KNOWN kept set, so every expected
value is derived by hand. No checkpoint is loaded and no saliency is computed on the
real model.

    O1 IDENTITY     audio == text -> overlap 1.0 everywhere, adjusted 1.0, Gate B FAILS
                    (a criterion that cannot disagree must not pass a disagreement gate).
    O2 CHANCE       independent random saliencies at the real geometry (N=960, k=192,
                    12 layers) -> weighted overlap ~ chance 0.20, adjusted ~ 0, PASS.
    O3 PRUNE-FLOOR  the reported prune-set overlap matches brute force exactly, AND
                    reproduces audit finding G1: at N=960/k=192 it can never go below
                    0.75, so the master plan's 0.70 prune-set threshold is unreachable.
    O4 GATE LOGIC   both pre-registered conditions are necessary — a weighted value
                    within budget still FAILS with only one layer under the per-layer
                    threshold, and vice versa. Weighting is by k_l.
    O5 BOUNDARY     the thresholds are inclusive and exact at the boundary
                    (weighted == 0.80 and per-layer == 0.70 both PASS).
    O6 GUARDS       malformed input raises instead of silently producing a number.

Run: .venv/bin/python tests/research/test_overlap_gate_b.py
"""
from __future__ import annotations

import sys

import torch

from research_pruning.paired_modality import (
    GATE_B_LAYER_MAX,
    GATE_B_MIN_LAYERS,
    GATE_B_WEIGHTED_MAX,
    evaluate_gate_b,
    kept_set_overlap,
    weighted_overlap,
)

# The real M3B geometry: 12 ranking-driven layers, 960 channels, 192 kept.
N_REAL, K_REAL, L_REAL = 960, 192, 12


def make_saliency(n: int, keep) -> torch.Tensor:
    """A saliency vector whose top-|keep| channels are exactly `keep`.

    Kept channels get distinct values in (1, 2]; the rest distinct values in (0, 1].
    No ties, so `keep_topk` is deterministic and the test controls the kept set
    exactly rather than hoping argsort breaks ties a particular way.
    """
    keep = list(keep)
    keep_set = set(keep)
    if len(keep_set) != len(keep):
        raise ValueError("repeated index in keep")
    rest = [i for i in range(n) if i not in keep_set]
    step = 0.5 / max(n, 1)
    v = torch.zeros(n, dtype=torch.float64)
    for j, i in enumerate(keep):
        v[i] = 2.0 - j * step
    for j, i in enumerate(rest):
        v[i] = 1.0 - j * step
    return v


def pair_with_intersection(n: int, k: int, inter: int):
    """Two kept sets of size k over n channels sharing exactly `inter` channels."""
    if inter > k or 2 * k - inter > n:
        raise ValueError("infeasible intersection")
    keep_a = list(range(k))
    keep_t = list(range(inter)) + list(range(k, k + (k - inter)))
    assert len(set(keep_a) & set(keep_t)) == inter
    return make_saliency(n, keep_a), make_saliency(n, keep_t)


def build(spec, n=50, k=10):
    """spec: {layer_name: intersection}. Returns (sal_a, sal_t, k_per_layer)."""
    sal_a, sal_t, kk = {}, {}, {}
    for name, inter in spec.items():
        a, t = pair_with_intersection(n, k, inter)
        sal_a[name], sal_t[name], kk[name] = a, t, k
    return sal_a, sal_t, kk


# --------------------------------------------------------------------------- O1
def check_o1_identity() -> bool:
    sal = {f"layer.{i}": make_saliency(N_REAL, range(K_REAL)) for i in range(L_REAL)}
    res = evaluate_gate_b(sal, {k: v.clone() for k, v in sal.items()},
                          {k: K_REAL for k in sal})
    ok = True
    all_one = all(r.overlap == 1.0 and r.adjusted == 1.0 for r in res.per_layer)
    print(f"  layers={len(res.per_layer)} all overlap==1 and adjusted==1: {all_one}")
    print(f"  weighted={res.weighted_overlap:.6f} adjusted={res.weighted_adjusted:.6f} "
          f"passed={res.passed}")
    ok &= all_one
    ok &= res.weighted_overlap == 1.0 and res.weighted_adjusted == 1.0
    ok &= res.passed is False and res.layers_at_or_below_layer_max == []
    # identical prune sets too
    ok &= all(r.prune_overlap == 1.0 for r in res.per_layer)
    return ok


# --------------------------------------------------------------------------- O2
def check_o2_chance() -> bool:
    g = torch.Generator().manual_seed(20260819)
    sal_a, sal_t, kk = {}, {}, {}
    for i in range(L_REAL):
        name = f"layer.{i}"
        sal_a[name] = torch.rand(N_REAL, generator=g, dtype=torch.float64)
        sal_t[name] = torch.rand(N_REAL, generator=g, dtype=torch.float64)
        kk[name] = K_REAL
    res = evaluate_gate_b(sal_a, sal_t, kk)
    chance = K_REAL / N_REAL
    print(f"  weighted={res.weighted_overlap:.4f} (chance={chance:.4f}) "
          f"adjusted={res.weighted_adjusted:+.4f}")
    print(f"  layers <= {GATE_B_LAYER_MAX}: {len(res.layers_at_or_below_layer_max)}/{L_REAL} "
          f"passed={res.passed}")
    ok = abs(res.weighted_overlap - chance) < 0.05          # independent -> chance
    ok &= abs(res.weighted_adjusted) < 0.10                 # ~0 on the adjusted scale
    ok &= res.passed is True                                # independence must PASS
    ok &= len(res.layers_at_or_below_layer_max) == L_REAL   # every layer far below 0.70
    return ok


# --------------------------------------------------------------------------- O3
def check_o3_prune_floor() -> bool:
    ok = True
    # (a) reported prune_overlap == brute-force prune-set Jaccard-style overlap
    for inter in (0, 40, 96, 192):
        a, t = pair_with_intersection(N_REAL, K_REAL, inter)
        res = evaluate_gate_b({"L": a}, {"L": t}, {"L": K_REAL})
        r = res.per_layer[0]
        keep_a = set(torch.argsort(a, descending=True)[:K_REAL].tolist())
        keep_t = set(torch.argsort(t, descending=True)[:K_REAL].tolist())
        prune_a = set(range(N_REAL)) - keep_a
        prune_t = set(range(N_REAL)) - keep_t
        brute = len(prune_a & prune_t) / (N_REAL - K_REAL)
        match = abs(r.prune_overlap - brute) < 1e-12
        print(f"  inter={inter:3d}  kept_overlap={r.overlap:.4f}  "
              f"prune_overlap={r.prune_overlap:.6f}  brute={brute:.6f}  match={match}")
        ok &= match and r.intersection == inter
    # (b) finding G1, executable: the prune-set floor at this budget is 0.75, so the
    #     master plan's 0.70 prune-set threshold is unreachable for ANY intersection.
    floor = min(
        (N_REAL - 2 * K_REAL + i) / (N_REAL - K_REAL) for i in range(0, K_REAL + 1)
    )
    pigeonhole = (2 * (N_REAL - K_REAL) - N_REAL) / (N_REAL - K_REAL)
    print(f"  prune-set floor over all intersections = {floor:.4f}; "
          f"pigeonhole (2p-N)/p = {pigeonhole:.4f}; plan threshold 0.70 reachable: "
          f"{floor <= 0.70}")
    ok &= abs(floor - 0.75) < 1e-12 and abs(pigeonhole - 0.75) < 1e-12
    ok &= floor > 0.70          # the amendment's justification holds
    return ok


# --------------------------------------------------------------------------- O4
def check_o4_gate_logic() -> bool:
    ok = True
    cases = [
        # (name, per-layer intersections /10, expected weighted, expected pass)
        ("PASS both conditions",      {"a": 5, "b": 6, "c": 8, "d": 9}, 0.700, True),
        ("FAIL weighted too high",    {"a": 7, "b": 7, "c": 10, "d": 10}, 0.850, False),
        ("FAIL only 1 layer <= 0.70", {"a": 7, "b": 8, "c": 8, "d": 8}, 0.775, False),
    ]
    for label, spec, expected_w, expected_pass in cases:
        sal_a, sal_t, kk = build(spec)
        res = evaluate_gate_b(sal_a, sal_t, kk)
        below = len(res.layers_at_or_below_layer_max)
        good = (abs(res.weighted_overlap - expected_w) < 1e-12
                and res.passed is expected_pass)
        print(f"  {label:<28} weighted={res.weighted_overlap:.4f} "
              f"layers<=0.70={below} passed={res.passed} (want {expected_pass}) -> "
              f"{'OK' if good else 'BAD'}")
        ok &= good
    # weighting is by k_l, not a plain mean over layers
    sal_a, sal_t = {}, {}
    a1, t1 = pair_with_intersection(50, 10, 10)     # overlap 1.0, k=10
    a2, t2 = pair_with_intersection(150, 30, 6)     # overlap 0.2, k=30
    sal_a["small"], sal_t["small"] = a1, t1
    sal_a["big"], sal_t["big"] = a2, t2
    res = evaluate_gate_b(sal_a, sal_t, {"small": 10, "big": 30})
    expected = (10 + 6) / 40                        # 0.40, vs a plain mean of 0.60
    plain_mean = (1.0 + 0.2) / 2
    print(f"  k-weighting: weighted={res.weighted_overlap:.4f} expected={expected:.4f} "
          f"(unweighted mean would be {plain_mean:.4f})")
    ok &= abs(res.weighted_overlap - expected) < 1e-12
    return ok


# --------------------------------------------------------------------------- O5
def check_o5_boundary() -> bool:
    # overlaps 0.7, 0.7, 0.9, 0.9 -> weighted exactly 32/40 = 0.80, two layers at 0.70
    sal_a, sal_t, kk = build({"a": 7, "b": 7, "c": 9, "d": 9})
    res = evaluate_gate_b(sal_a, sal_t, kk)
    at_threshold = [r for r in res.per_layer if r.overlap == GATE_B_LAYER_MAX]
    print(f"  weighted={res.weighted_overlap!r} == {GATE_B_WEIGHTED_MAX!r}: "
          f"{res.weighted_overlap == GATE_B_WEIGHTED_MAX}")
    print(f"  layers exactly at {GATE_B_LAYER_MAX}: {len(at_threshold)} "
          f"counted as satisfying: {len(res.layers_at_or_below_layer_max)} "
          f"(need >= {GATE_B_MIN_LAYERS})  passed={res.passed}")
    ok = res.weighted_overlap == GATE_B_WEIGHTED_MAX     # exact, no FP drift
    ok &= len(at_threshold) == 2
    ok &= len(res.layers_at_or_below_layer_max) == 2      # inclusive per-layer bound
    ok &= res.passed is True                              # inclusive weighted bound
    # one notch worse must flip it
    sal_a, sal_t, kk = build({"a": 7, "b": 7, "c": 9, "d": 10})
    worse = evaluate_gate_b(sal_a, sal_t, kk)
    print(f"  one notch worse: weighted={worse.weighted_overlap:.4f} "
          f"passed={worse.passed}")
    ok &= worse.passed is False
    return ok


# --------------------------------------------------------------------------- O6
def check_o6_guards() -> bool:
    ok = True

    def expect_raises(label, fn):
        nonlocal ok
        try:
            fn()
        except (ValueError, KeyError) as exc:
            print(f"  {label:<34} raised {type(exc).__name__}: {str(exc)[:60]}")
            return
        print(f"  {label:<34} DID NOT RAISE")
        ok = False

    sal_a, sal_t, kk = build({"a": 5, "b": 5})

    expect_raises("k > N", lambda: kept_set_overlap(
        {"a": torch.arange(60)}, {"a": torch.arange(60)}, {"a": 50}))
    expect_raises("different layer sets", lambda: evaluate_gate_b(
        sal_a, {"a": sal_t["a"]}, kk))
    expect_raises("empty layer restriction", lambda: evaluate_gate_b(
        sal_a, sal_t, kk, layers=[]))
    expect_raises("unknown restricted layer", lambda: evaluate_gate_b(
        sal_a, sal_t, kk, layers=["a", "nope"]))
    expect_raises("unmatched per-layer k", lambda: kept_set_overlap(
        {"a": torch.arange(10)}, {"a": torch.arange(12)}, {"a": 50}))
    expect_raises("repeated channel index", lambda: kept_set_overlap(
        {"a": torch.tensor([1, 1, 2])}, {"a": torch.tensor([1, 2, 3])}, {"a": 50}))
    expect_raises("index out of range", lambda: kept_set_overlap(
        {"a": torch.tensor([1, 2, 99])}, {"a": torch.tensor([1, 2, 3])}, {"a": 50}))
    expect_raises("declared N != saliency length", lambda: evaluate_gate_b(
        sal_a, sal_t, kk, n_per_layer={"a": 40, "b": 50}))
    expect_raises("no layers to aggregate", lambda: weighted_overlap([]))

    # The layer restriction must actually restrict, and leaving the positional seams
    # in dilutes the gate: they agree by construction (overlap 1.0 — L1 itself is
    # positional there), so they can drag a genuinely disagreeing set over the
    # threshold. Here 3 ranked layers at 0.70 PASS on their own, but adding 3 seams
    # at 1.00 lifts the weighted value to 51/60 = 0.85 and flips the verdict to FAIL.
    ranked = ["ranked1", "ranked2", "ranked3"]
    sal_a3, sal_t3, kk3 = build(
        {"ranked1": 7, "ranked2": 7, "ranked3": 7, "pos1": 10, "pos2": 10, "pos3": 10}
    )
    full = evaluate_gate_b(sal_a3, sal_t3, kk3)
    restricted = evaluate_gate_b(sal_a3, sal_t3, kk3, layers=ranked)
    print(f"  restriction: all-6 weighted={full.weighted_overlap:.4f} passed={full.passed} "
          f"-> ranked-only weighted={restricted.weighted_overlap:.4f} "
          f"passed={restricted.passed}")
    ok &= len(restricted.per_layer) == 3
    ok &= abs(full.weighted_overlap - 51 / 60) < 1e-12
    ok &= abs(restricted.weighted_overlap - 0.70) < 1e-12
    ok &= full.passed is False and restricted.passed is True
    return ok


def test_o1_identity():
    assert check_o1_identity()


def test_o2_chance():
    assert check_o2_chance()


def test_o3_prune_floor():
    assert check_o3_prune_floor()


def test_o4_gate_logic():
    assert check_o4_gate_logic()


def test_o5_boundary():
    assert check_o5_boundary()


def test_o6_guards():
    assert check_o6_guards()


def main() -> int:
    checks = [
        ("O1 IDENTITY", check_o1_identity),
        ("O2 CHANCE", check_o2_chance),
        ("O3 PRUNE-FLOOR", check_o3_prune_floor),
        ("O4 GATE LOGIC", check_o4_gate_logic),
        ("O5 BOUNDARY", check_o5_boundary),
        ("O6 GUARDS", check_o6_guards),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        results[name] = bool(fn())
    print("\n==== M3B GATE-B OVERLAP TESTS ====")
    for name, _ in checks:
        print(f"  {name:<16} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
