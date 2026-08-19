# Claims Matrix

Allowed status values: `unresolved`, `supported`, `rejected`, `exploratory`, `unavailable`.

| ID | Candidate claim | Required evidence / branch | Status | Evidence paths | Allowed paper wording |
|---|---|---|---|---|---|
| RQ1 | Structured pruning causes modality-dependent damage beyond the matched random-pruning null. | M3 Gate A | unresolved | | |
| RQ2a | Paired modality adds value beyond faithful text-only Taylor at matched gradient-evaluation budget. | M3 Gate B + M4 | unresolved | `docs/m0_baseline_reproduction/l1_pruning_direction_finding.md` | **P0/L1 baseline convention decided (2026-08-19, DECISION-M3B-002):** RQ2's L1 baseline is Arshdeep's official published artifact, which keeps the LOWEST-L1 filters, so the project's P0 adopts that inverted convention (`p0_importance(convention='published')`, verified to reproduce the published kept-set 12/12). Not 'standard' L1. |
| RQ2b | Max aggregation adds value beyond paired mean. | M4 P3 vs P2 | unresolved | | |
| RQ3 | Fixed parameter-efficient recovery restores residual pruning damage under the tested budget. | M5 | unresolved | `docs/m0_baseline_reproduction/public_artifact_inventory.md` | **Constrained by M0:** the recovered full-FT `(1,2,3,1)` checkpoint is proven not publicly available, so until Arshdeep supplies it RQ3 may only be phrased as a published-reference comparison. No exact percentage-of-full-FT recovery may be claimed from cross-pipeline numbers. |
| SEM | Specific semantic/event families remain differentially vulnerable. | M6 with support and uncertainty | unresolved | | |
| EFF | Reported pruning/recovery efficiency is supported by measured runtime, VRAM, GPU-hours, and storage. | M1/M4/M5/M6 | unresolved | `artifacts/m0_baseline_reproduction/architecture_check.log` | Parameter counts only are measured so far (U-Net 415.955 M -> 145.674 M, -65.0%). Runtime, VRAM and GPU-hours are unmeasured, so no efficiency claim may be written yet. |

No rejected or exploratory claim may be promoted to a headline claim.
