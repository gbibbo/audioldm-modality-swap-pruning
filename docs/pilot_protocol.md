# M3 Pilot Protocol

**Status: UNFROZEN. M3 is blocked until this file is completed, reviewed against `docs/master_plan_v3.md`, and committed before saliency results are inspected.**

## Calibration budget

* Base slots `B`:
* P1: `2B` text gradient evaluations.
* P2/P3: `B` text + `B` audio gradient evaluations.
* Slot construction:
* Noise draw policy:
* Timestep draw/strata:

## Primary timestep aggregation

Equal-weight mean across preregistered timestep strata, followed by averaging across examples, unless the master plan is explicitly amended before results are inspected.

## M3A matched random null

* `Krand = 20` structured random masks at the `(1,2,3,1)` budget.
* Primary statistic: `Delta_swap = R_mod^L1 - E[R_mod^random | D_gen^L1]`.
* Bootstrap unit and seed policy:
* Gate A PASS requires 95% bootstrap CI above zero and standardized residual at least 0.5 random-control SD.

## M3B saliency disagreement

* Target prune tail definition:
* Targeted layers:
* Weighted overlap calculation:
* Gate B PASS requires weighted prune-set overlap `<= 0.80` and at least two key prunable layers with overlap `<= 0.70`.

## Frozen identifiers

* Calibration manifest SHA256:
* Timestep list/hash:
* Random mask seed list/hash:
* Code commit:
* Resolved config:
* Freeze commit:
* Freeze timestamp:
