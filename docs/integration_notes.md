# AudioLDM PEFT integration notes

Precise upstream patch points for wiring parameter-efficient recovery into the
frozen AudioLDM training code. **No upstream file is modified yet** —
`git diff upstream-frozen -- audioldm_train/` must stay empty. These are the
minimal, reviewable hooks a future GPU-enabled session will apply; the CPU-side
logic they call already exists in `audioldm_peft/` and `audioldm_peft.integrate`
and is unit-tested.

Frozen upstream observations (item 3 independently confirmed against
`audioldm_train/modules/latent_diffusion/ddpm.py`):

1. `audioldm_train/train/latent_diffusion.py` instantiates `LatentDiffusion`, then
   loads an external checkpoint before `trainer.fit`.
2. PEFT injection must occur **after** external checkpoint loading, otherwise the
   wrapped parameter names no longer match the original checkpoint keys. Use
   `audioldm_peft.integrate.setup_peft(model, cfg)` at that point — it runs
   `freeze_for_peft -> inject_lora -> configure_auxiliary_trainables` in the
   correct order and asserts readiness.
3. `LatentDiffusion.configure_optimizers()` currently constructs AdamW from the
   complete diffusion model plus conditional-stage parameters. Replace its body
   with `audioldm_peft.integrate.build_peft_optimizer(self, cfg, ...)`, which uses
   only the three/four PEFT parameter groups.
4. `DDPM.__init__` constructs `LitEma(self.model)` during initialization. For PEFT
   this is too early (the base is still trainable). Disable the upstream EMA and
   construct `TrainableOnlyEMA` **after** `setup_peft`, or rebuild it there.
5. `ema_scope()` passes `self.model.parameters()` to `LitEma.store/restore`;
   upstream `store()` clones every parameter in the supplied iterable. Use
   `TrainableOnlyEMA.scope(model)` for validation so the frozen U-Net is never
   copied.

## Full-resume state

Lightning resume must round-trip the PEFT training state. `audioldm_peft.state`
provides `training_state_dict(model, optimizer, scheduler, ema, global_step)` and
`load_training_state_dict(...)`; hook these into the checkpoint save/load path (or
`on_save_checkpoint` / `on_load_checkpoint`). Post-load injection must happen
before the optimizer/EMA state is restored, so the trainable parameter set exists.

## Scope of the deliberate upstream patch (kept minimal)

- PEFT setup hook after checkpoint load (`setup_peft`);
- optimizer-group hook (`build_peft_optimizer`);
- EMA-safe initialization/scope (`TrainableOnlyEMA`);
- checkpoint save/load of the full resume state.

Everything else stays byte-identical to `upstream-frozen`. The GPU acceptance
(several hundred real steps, VRAM, sec/step, resume) runs only once a GPU is
attached and is recorded in `docs/compute_budget.md` from measured values.
