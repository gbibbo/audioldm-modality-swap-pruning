# Hostile review — round 3 (2026-08-20, Reviewer B reply)

Output: `docs/master_plan_v4_draft.md` **rc2** (changelog at the top of that file).

## Accepted without reservation (both are real errors in rc1)

* **Gate E was not executable in Tier 0** — Tier 0 generates audio only for {base, P0-std,
  P1-nat}; RAND×5 are ε-forwards, no `L_RAND(e)` exists. rc2: Tier 0 runs a *heterogeneity
  screen* (dispersion + P0-std/P1-nat concordance, prompt-bootstrap CI) whose only role is
  to inform the Tier-1 funding decision; Gate E lives in Tier 1.
* **RAND×2 cannot give a between-mask 95th percentile.** rc2: `K_rand ≥ 10` (20 if
  funded), each mask generated on a 300-prompt subset, with P0-std's gate statistic
  computed on the same subset; prompt-bootstrap of two masks explicitly disallowed.
  Tier 1a re-costed to ≈23 credits.

## Accepted with verification

* **Occupancy estimator.** FineLAP (arXiv 2604.01155): code at github.com/xiquan-li/FineLAP,
  EAT + RoBERTa, evaluated on SED and text-to-audio grounding — **public checkpoint not
  confirmed from the paper page; the `get_frame_level_score` API A quotes is unverified by
  me.** rc2 makes FineLAP the primary estimator *conditional on a CPU validity smoke*, PANNs
  `Cnn14_DecisionLevelMax` framewise the sensitivity analysis, and drops occupancy if the
  smoke fails (DECISION-V4-06). Circularity argument accepted.
* **Synonym protocol.** Verified that `class_labels_indices.csv` ships comma aliases
  (`Gunshot, gunfire`, `Police car (siren)`, `Male speech, man speaking`). rc2: strict map
  from official aliases + minimal morphology; expanded map only as sensitivity; no
  LLM-generated synonyms.
* **Gate M as nested block LRTs** (tail / guidance / acoustic), α = 0.01 per block, AIC
  descriptive. Accepted verbatim.
* **Seeds.** The FAD 3-seed × 300-clip audio is scored for event recall as the
  seed-robustness check. Accepted.
* **FAD/FD rule** restated: fix before any scientifically-used generation; does not block
  D1/D2/Gate B′. The round-2 chat message was stricter than the draft; the draft wins.
* **Music drift** → §10b exploratory, promotable only with an independent evaluator and a
  real-audio PANNs control. Accepted.

## Measured this round (CPU, this session)

Mild-budget geometry from the frozen config: `(1,2,3,4)` = 317.308 M = **−23.7 %**;
`(1,2,3,3)` = 239.047 M = −42.5 %; `(1,2,3,2)` = 182.168 M = −56.2 %; `(1,2,3,1)` =
145.674 M = −65.0 %. A's "20–30 % structured reduction" is `(1,2,3,4)`, not the `(1,2,3,3)`
rc1 named. Engineering prerequisite: `build_pruned_unet`/`materialize` are hardcoded to
`(1,2,3,1)`; parameterizing them needs a bit-exact regression test at the old budget.

## Remaining disagreement

None of substance. Both reviewers now vote **adopt v4 (rc2)**, Tier 0 only with the
current balance, P0-standard primary, ELSA deferred, RQ-swap kept as one paragraph with D1
deciding the wording. Seven decisions (V4-00..06) are listed in the draft for Gabriel.

## Pre-execution checklist for Tier 0 (after DECISION-V4-00)

1. Fix FAD/FD NaN (F-eval-3) on CPU; re-score the existing screening audio.
2. Freeze: event set `E*` (`N_min=200`, `n_min=10`), strict/expanded synonym maps,
   covariate manifest, stratified prompt manifest — sha256 in the ledger.
3. FineLAP CPU validity smoke → DECISION-V4-06.
4. Parameterize the materializer for `(1,2,3,4)` + bit-exact test (needed for Tier 1 only,
   but cheap to do now).
5. Jobs, each after CPU dry-run + GPU smoke: D1+D2 forward job; Gate B′ saliency job with
   per-slot storage; screening generation (200 prompts × 3 systems).
