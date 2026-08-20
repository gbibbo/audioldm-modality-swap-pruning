# NEXT SESSION — start here (written 2026-08-20 14:10, HEAD d4ca0ae+)

Self-contained. A fresh session needs only this file; everything it points to is committed.

## 1. Read, in this order (15 min)

1. `PROGRESS.md` state block (injected by the hook).
2. `docs/HANDOFF.md` — only the head block dated 2026-08-20 14:00; the rest is history.
3. `docs/decisions/DECISION-V4_pending.md` — the decision sheet.
4. `docs/master_plan_v4.md` (rc4) — the **adopted contract** (DECISION-V4-00, 2026-08-20);
   `docs/master_plan_v3.md` is superseded, kept for history.
5. `docs/review/2026-08-20_reframing_round1.md` … `round5.md` — why the plan changed.
6. `docs/experiment_ledger.md` entries `M3B-SALIENCY-RUN`, `M3A-DIAG-RUN`,
   `M4-SCREEN-FOUND`, `REVIEW-001..005`.

Verify before anything else: `git status --short --branch` clean, `git log -1` at or after
`d4ca0ae`, `.venv/bin/python -c "import torch; print(torch.__version__)"` → `1.13.1+cu117`.

## 2. First action: the decision sheet — DONE (2026-08-20 14:13)

Gabriel adopted all eight recommended defaults verbatim (Tier 0 only, ~2.7 cr; balance
~3–5 cr; 2.0-cr reserve stands). Recorded as ledger `DECISION-V4-00..07` and propagated
(v4 = contract, v3 superseded, `AGENTS.md` step 2, `docs/claims_matrix.md`, PROGRESS,
HANDOFF). **No GPU job in this session unless Gabriel says so explicitly.**

## 3. CPU queue (credit-free) — ✅ COMPLETE (Q1–Q7 done 2026-08-20, one commit each; ledger Q1..Q7 + E-BLAS)

| # | Item | Done when |
|---|---|---|
| Q1 | Fix FAD/FD NaN (F-eval-3, VGGish `sqrtm`; Cnn14-FD also NaN) in the `audioldm_eval` path; re-score the existing `artifacts/m4_screening/` audio | finite FAD and FD for all 6 screening systems; numbers in the ledger as screening-only |
| Q2 | Freeze manifests, sha256 in the ledger: event set `E*` (`N_min`, `n_min`), strict + expanded synonym maps derived from the comma aliases in `data/dataset/metadata/audiocaps/class_labels_indices.csv`, covariate manifest (audio exposure, calibration-caption exposure, AudioSet exposure if the CSV is fetched), calibration/mechanism/holdout partition at source-wav level, sentinel panel, prompt manifests, seed table (seed pairing) | files under `configs/research/`, hashes recorded, no pruned generation inspected before the freeze |
| Q3 | FineLAP smoke: load `AndreasXi/FineLAP` under `torch 1.13.1`, `get_frame_level_score` on 5 known single-label clips; frozen score→duration rule | PASS/FAIL recorded → resolves V4-06 |
| Q4 | Parameterize `research_pruning/diagnostics/random_masks.py::build_pruned_unet` / `materialize` by `channel_mult`; regression test bit-exact at `(1,2,3,1)` (R5 equality), param count 317.308 M at `(1,2,3,4)` | tests PASS, ledger entry |
| Q5 | Per-slot saliency storage in `scripts/research/m3b_saliency.py` (28 layers × ≤960 ch × slots) + CPU recomposition + null-split overlap distribution (Gate B′ machinery, synthetic test) | test PASS on control model |
| Q6 | Template (`"a sound of [alias]"`) and counterfactual (`c_without_e`, `c_only_e`) text conditioning in `scripts/research/m3a_diagnostics.py`; signed `norm_E_a/norm_E_t` and per-stratum outputs (D1) | CPU dry-run PASS |
| Q7 | Power-simulation script for Gate E (parametric bootstrap from screening rates; sentinel design → power at MDE) | script + synthetic test |

Then stop and report. Tier-0 GPU jobs (D1+D2 forward job, Gate B′ saliency job, screening
generation) need a clean pushed commit, a CPU dry-run, a GPU smoke, and Gabriel's go.

## 4. Standing rules (unchanged)

CPU Studio for all development; GPU only through Lightning Jobs (recipe in `HANDOFF.md`);
`/auditar` before any scientific claim; every number traceable (commit, config, hashes,
seeds, job name, cost); negative results are valid; P0-published is worded "vs the published
L1 artefact"; never commit datasets, audio, checkpoints, secrets.
