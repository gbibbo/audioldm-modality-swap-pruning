# `configs/research/`

Resolved research configs. Empty on purpose.

Every config added here must be referenceable from a `docs/experiment_ledger.md`
entry, and every number reported from a run must be traceable to the exact
resolved config committed here.

Pending, per `docs/master_plan_v3.md`:

* the `(1,2,3,1)` pruned architecture config, derived from
  `audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml`
  by replacing `channel_mult: [1,2,3,5]` with `channel_mult: [1,2,3,1]`;
* the fixed parameter-efficient recovery config (LoRA rank/alpha/targets,
  trainable biases, trainable GroupNorm affine);
* the M3 pilot calibration config (slots `B`, timestep strata, seeds).

`(1,2,1,1)` is outside the ICASSP core scope.

## Plan-v4 pre-registration manifests (frozen 2026-08-20, CPU queue Q2)

Built deterministically by `scripts/research/build_v4_manifests.py` (`--check` verifies
determinism). All sha256 recorded in `docs/experiment_ledger.md` (Q2) and in
`v4_manifests_index.json`. Master seed `20260818`. **No pruned generation was inspected
before this freeze.**

* `audioset_ontology.json` — official AudioSet ontology (632 nodes, 7 roots), frozen input for `family`.
* `event_synonyms_strict.json` — comma aliases from `class_labels_indices.csv` + plurals of single-word aliases (plan §4 minimal morphology; verb forms deferred to expanded).
* `event_synonyms_expanded.json` — strict + a small reviewed manual block; **sensitivity only, never the primary tail block** (no LLM used).
* `event_family_map.json` — mid → AudioSet top-level family.
* `event_set.json` — E* = `n_labelled ≥ 200` AND `n_requested ≥ n_min` (Tier 0 `n_min=10` → 101 events; Tier 1 `n_min=20` → 98).
* `event_covariates.json` — per-event exposures (log audio, log calibration-caption); acoustic/guidance covariates added later per plan §3.
* `data_partition.json` — calibration (256 natural from the M3B manifest + 256 tail-enriched) / mechanism (50 events × 20) / holdout (500), **pairwise-disjoint at source-wav id**, disjoint from the val split.
* `sentinel_panel.json` — 20 events × 15 prompts, stratified by exposure × family (Gate E, Tier 1).
* `prompts_heterogeneity_screen.json` — 200 stratified Tier-0 screen prompts.
* `seed_table.json` — master seed, per-prompt paired noise seeds, `K_rand=20` RAND mask seeds, 3-seed FAD panel.
* `v4_manifests_index.json` — sha256 of every file above.
