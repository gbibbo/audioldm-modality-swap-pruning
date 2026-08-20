#!/usr/bin/env python3
"""Synthetic tests for the Gate E power simulation (CPU queue Q7). CPU-only, no model.

    E1 CALIBRATION  at delta=0 the exact rank test's rejection rate ≈ alpha (Type-I error
                    is controlled — the whole point of the exact test).
    E2 MONOTONE     power is (weakly) increasing in the effect delta; power at a large
                    effect clearly exceeds power at a small one.
    E3 MDE          find_mde returns a delta whose power >= target, and the grid point
                    just below it is < target (it is the threshold crossing), or None with
                    every power < target.
    E4 RESIZE       the M4-1b lever works: at a fixed effect, more prompts per event gives
                    >= power (a bigger, better-resolved panel is at least as sensitive).
    E5 DETERMINISM  same seed -> identical power.

Run: .venv/bin/python tests/research/test_gate_e_power.py
"""
from __future__ import annotations

import importlib.util
import sys


def _load():
    spec = importlib.util.spec_from_file_location("gate_e_power_sim",
                                                  "scripts/research/gate_e_power_sim.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = _load()
BASE = dict(n_events=20, n_prompts=15, k_rand=20, p_base=0.6, mu_loss=0.15,
            n_sim=2000, alpha=0.05, seed=20260818)


def check_e1_calibration() -> bool:
    fp = G.simulate_power(0.0, **BASE)
    print(f"    FP rate at delta=0 = {fp:.3f} (alpha=0.05)")
    return 0.02 <= fp <= 0.09


def check_e2_monotone() -> bool:
    lo = G.simulate_power(0.05, **BASE)
    mid = G.simulate_power(0.20, **BASE)
    hi = G.simulate_power(0.40, **BASE)
    print(f"    power(0.05)={lo:.3f}  power(0.20)={mid:.3f}  power(0.40)={hi:.3f}")
    return lo < mid < hi and hi > lo + 0.2


def check_e3_mde() -> bool:
    deltas = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5]
    mde, curve = G.find_mde(deltas=deltas, power_target=0.80, **BASE)
    powers = dict(curve)
    print(f"    MDE={mde}  curve={[f'{d}:{p:.2f}' for d, p in curve]}")
    if mde is None:
        return all(p < 0.80 for _, p in curve)
    # power at MDE >= target and at the previous grid delta < target
    idx = deltas.index(mde)
    ok = powers[mde] >= 0.80 and (idx == 0 or powers[deltas[idx - 1]] < 0.80)
    return ok


def check_e4_resize() -> bool:
    few = G.simulate_power(0.20, **{**BASE, "n_prompts": 15})
    many = G.simulate_power(0.20, **{**BASE, "n_prompts": 30})
    print(f"    power@delta0.20: 15 prompts={few:.3f}  30 prompts={many:.3f}")
    return many >= few + 0.1   # more prompts -> materially more power


def check_e5_determinism() -> bool:
    a = G.simulate_power(0.20, **BASE)
    b = G.simulate_power(0.20, **BASE)
    print(f"    power runs: {a:.6f} vs {b:.6f}")
    return a == b


def main() -> int:
    checks = [
        ("E1 CALIBRATION", check_e1_calibration),
        ("E2 MONOTONE", check_e2_monotone),
        ("E3 MDE", check_e3_mde),
        ("E4 RESIZE", check_e4_resize),
        ("E5 DETERMINISM", check_e5_determinism),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as e:
            print(f"    ERROR: {e!r}")
            results[name] = False
    print("\n==== GATE E POWER-SIM TESTS ====")
    for name, _ in checks:
        print(f"  {name:<16} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
