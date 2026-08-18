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
