# M1 checklist

Status legend: [x] done & tested on CPU · [~] implemented, needs real-U-Net/GPU
evidence · [ ] not started. See `docs/experiment_ledger.md` M1-005+ and
`docs/m1_scaffold_audit.md` (defects F1–F8).

## CPU (dummy-model verified — M1-005)
- [x] Linear LoRA merge/unmerge equivalence (L1).
- [x] Conv2d LoRA merge/unmerge equivalence + factorised-forward equivalence (L2, L3; F5).
- [x] Injection selection report (J1).
- [x] Base frozen after injection; LoRA preserved under mis-ordered freeze (J1, J4; F4).
- [x] Biases / GroupNorm / LayerNorm trainables reported separately (J1, J2; F2).
- [x] Stable auxiliary-trainable counts across repeated calls (J3; F3).
- [x] Adapter-only state save/load (S1).
- [x] Optimizer groups contain only intended parameters (S1).
- [x] Trainable-only EMA (S2).
- [x] Full resume state: adapter + optimizer + scheduler + EMA + step (S3; F7).
- [x] pytest-free runner (F1): `scripts/research/run_research_tests.py`.

## Real pruned U-Net (F6 — CPU, M1-006 DONE)
- [x] Inject on the real `(1,2,3,1)` U-Net; 284 modules wrapped (185L+99C);
      decomposition lora 3,718,784 / bias 108,680 / GroupNorm 48,768 /
      LayerNorm 0 (default), trainable 3,876,232 of 149,392,648 (R6a).
- [x] Merge/unmerge numerical equivalence on a real forward, max|Δ| 1.0e-7 (R6b).
- [x] train_layernorm_affine=True adds exactly 35,712 params (R6c).

## Integration (F8 — hooks + CPU tests DONE, M1-007; upstream patch deferred to GPU session)
- [x] Inject only after checkpoint load (`setup_peft`); post-load-order proven (I4).
- [x] PEFT-only optimizer groups (`build_peft_optimizer`), AdamW, per-group LRs (I2).
- [x] EMA constructed post-setup, trainable-only (`build_trainable_only_ema`; S2/S3).
- [x] Validation EMA store/copy/restore trainable-only (`TrainableOnlyEMA.scope`; S3).
- [x] Full resume state (`training_state_dict`/`load_training_state_dict`; S3).
- [x] Config parse from the research yaml (`peft_config_from_yaml`; I3).
- [ ] Apply the minimal upstream patch in `audioldm_train/` (deferred; keeps diff empty until deliberately made and reviewed on a GPU-enabled session).

## GPU benchmark (blocked — no GPU attached)
- [ ] 500 real optimization steps.
- [ ] Peak VRAM.
- [ ] Seconds/step.
- [ ] Resume test.
- [ ] Generation sec/clip or throughput.
- [ ] Saliency forward/backward benchmark.
