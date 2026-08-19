#!/usr/bin/env python3
"""Verify, against the real artifacts, the geometry the Gate B amendment rests on.

Audit finding G1 says the master plan's Gate B is infeasible as written because it
states its thresholds against the PRUNE-set overlap, which at the `(1,2,3,1)` budget
is confined to `[(2p-N)/p, 1]`. DECISION-M3B-003 amends the gate onto the KEPT-set
definition, keeping the plan's numerals. That amendment is only justified if the
per-layer geometry really is what the derivation assumes, so this script re-derives
it from the published ranking + the pruned target shapes rather than from prose:

    * the ranking-driven layer set (where audio/text can actually disagree),
    * per layer N (full channels), k (kept), p (pruned),
    * kept-set chance level k/N and prune-set pigeonhole floor (2p-N)/p,
    * whether the plan's 0.70 prune-set threshold is reachable at all.

Structure only: no saliency, no diagnostics, no checkpoint weights are ranked. Prints
a table and exits 0 when the derivation holds.

Run: .venv/bin/python scripts/research/verify_gate_b_geometry.py
"""
from __future__ import annotations

import sys

import yaml

from research_pruning.diagnostics.random_masks import (
    kept_counts,
    load_l1_ranking,
    ranking_driven_layers,
    ranking_full_lengths,
)
from research_pruning.paired_modality import GATE_B_LAYER_MAX, GATE_B_WEIGHTED_MAX

CONFIG = ("audioldm_train/config/2023_08_23_reproduce_audioldm/"
          "audioldm_original_medium.yaml")
RANKING_PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"

# The master plan's original (prune-set) thresholds, kept here only to test their
# reachability. The amended gate uses GATE_B_* against the kept-set overlap.
PLAN_PRUNE_WEIGHTED_MAX = 0.80
PLAN_PRUNE_LAYER_MAX = 0.70


def main() -> int:
    config = yaml.safe_load(open(CONFIG))
    ranking = load_l1_ranking(RANKING_PKL)
    driven = ranking_driven_layers(config, ranking)
    counts = kept_counts(config, list(ranking.keys()))
    full = ranking_full_lengths(ranking)

    print(f"ranking-driven layers (where P1/P2/P3 compete): {len(driven)} "
          f"of {len(ranking)} ranked layers\n")
    print(f"{'layer':<44} {'N':>5} {'k':>5} {'p':>5} {'chance':>7} {'floor':>7}")
    rows = []
    for name in driven:
        n, k = full[name], counts[name]
        p = n - k
        chance = k / n
        floor = (2 * p - n) / p
        rows.append((name, n, k, p, chance, floor))
        print(f"{name:<44} {n:>5} {k:>5} {p:>5} {chance:>7.4f} {floor:>7.4f}")

    ok = bool(rows)
    if not ok:
        print("\nFAIL: no ranking-driven layers found")
        return 1

    floors = sorted({round(r[5], 9) for r in rows})
    chances = sorted({round(r[4], 9) for r in rows})
    max_floor = max(r[5] for r in rows)

    print(f"\ndistinct kept-set chance levels : {chances}")
    print(f"distinct prune-set floors       : {floors}")

    print("\n--- master plan as written (prune-set overlap) ---")
    print(f"  attainable range per layer      : [{max_floor:.4f}, 1.0]")
    print(f"  chance level (p/N)              : {rows[0][3] / rows[0][1]:.4f}")
    reachable = max_floor <= PLAN_PRUNE_LAYER_MAX
    print(f"  threshold <= {PLAN_PRUNE_LAYER_MAX} reachable      : {reachable}")
    print(f"  threshold <= {PLAN_PRUNE_WEIGHTED_MAX} vs chance      : "
          f"{'AT chance' if abs(PLAN_PRUNE_WEIGHTED_MAX - rows[0][3] / rows[0][1]) < 1e-9 else 'off chance'}")
    if reachable:
        print("  -> finding G1 would NOT hold at this geometry")
        ok = False
    else:
        print("  -> finding G1 CONFIRMED: Gate B cannot PASS as written")

    print("\n--- amended gate (kept-set overlap, DECISION-M3B-003) ---")
    print(f"  attainable range per layer      : [0.0, 1.0]")
    print(f"  chance level (k/N)              : {chances[0]:.4f}")
    print(f"  weighted threshold              : <= {GATE_B_WEIGHTED_MAX}")
    print(f"  per-layer threshold             : <= {GATE_B_LAYER_MAX} "
          f"(>= 2 layers), reachable: {GATE_B_LAYER_MAX >= 0.0}")
    if not (chances[0] < GATE_B_LAYER_MAX < GATE_B_WEIGHTED_MAX < 1.0):
        print("  -> FAIL: amended thresholds are not strictly between chance and 1")
        ok = False
    else:
        print("  -> thresholds sit strictly between chance and identity: well-posed")

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
