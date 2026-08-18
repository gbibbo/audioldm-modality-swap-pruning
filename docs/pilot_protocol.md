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
* Targeted layers: **constrained by M0.** The public L1 baseline
  (`sorted_indexes_dict.pkl`, Zenodo 10.5281/zenodo.21376822) ranks exactly 28
  conv layers -- `input_blocks.7..11` (9), `middle_block` (4),
  `output_blocks.0..6` (15) -- with widths 384/576/960. P1, P2 and P3 must be
  computed over this same layer set with the same per-layer channel counts, or
  the comparison against L1 is not structure-matched. Full list and evidence:
  `docs/m0_baseline_reproduction/public_artifact_inventory.md`.
* Weighted overlap calculation:
* Gate B PASS requires weighted prune-set overlap `<= 0.80` and at least two key prunable layers with overlap `<= 0.70`.

## Frozen identifiers

* Reference architecture: `channel_mult = [1,2,3,1]`, `model_channels = 192`,
  verified from `l1_audioldm-m-full_p1.ckpt` (md5 `2666e6fc108a9c4fc0d19bbf26832905`).
* L1 index manifest md5: `a4cd11ff83438ee0f9aa5fe0917f39e3`
* Calibration manifest SHA256:
* Timestep list/hash:
* Random mask seed list/hash:
* Code commit:
* Resolved config:
* Freeze commit:
* Freeze timestamp:
