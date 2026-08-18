# M1 scaffold — recovery and audit

**Date:** 2026-08-18. **Auditor role, not implementation.** This document records
the recovery of the local-only M1 PEFT scaffold referenced by
`docs/master_plan_v3.md` (M1, "CPU scaffold exists and its initial tests have
passed locally") and an evidence-first audit of it against the **real** AudioLDM
U-Net. It supersedes `docs/HANDOFF.md` §5 and §12.B, which stated that M1 was
unrecoverable and blocked.

## 1. Provenance

Gabriel uploaded two archives to the repository root. Their payloads are
**byte-identical** (`diff -rq` clean, 61 404 B / 27 files each); one nests the
overlay under `scaffold/`, the other is flat.

| Archive | md5 |
|---|---|
| `audioldm_pruning_lora_repo_scaffold.zip` | `2fdc79fff6674489962791a02fde9e35` |
| `scaffold.zip` | `dc35dfe4acf4ced2e69419003706ade9` |

Internal file timestamps: **2026-08-13**. Both archives were deleted from the
repository root after extraction. The pristine overlay is preserved at
`_external/m1_scaffold_recovered/` (gitignored) and the audit scripts and logs at
`artifacts/m1_scaffold_recovery/` (gitignored).

**This is an overlay pack, not the evolved local repository.** It carries no Git
history, and its `docs/` copies are an *older* generation of our provenance
documents — its `master_plan_v3.md` is 33 415 B against our 36 854 B.

> **Adoption constraint.** Only `audioldm_peft/`, `tests/research/`,
> `configs/research/peft_r8_full_unet.yaml`, `scripts/research/cpu_smoke_peft.py`,
> `docs/integration_notes.md` and `docs/M1_CHECKLIST.md` may be adopted.
> Copying the overlay's `docs/master_plan_v3.md`, `compute_budget.md`,
> `claims_matrix.md`, `pilot_protocol.md` or `experiment_ledger.md` would
> silently regress current project state.

## 2. What the scaffold contains

`audioldm_peft/`: `config.py` (`PeftConfig`), `layers.py` (`LoRALinear`,
`LoRAConv2d`, merge/unmerge), `inject.py` (injector, freeze, auxiliary
trainables), `report.py`, `state.py` (adapter-only state dict), `optimizer.py`
(parameter groups), `ema.py` (`TrainableOnlyEMA`). Three CPU test modules and a
CPU smoke script. `configs/research/peft_r8_full_unet.yaml` matches the master
plan defaults exactly: rank 8, alpha 16, dropout 0.0, full U-Net scope,
`train_bias: true`, `train_groupnorm_affine: true`, trainable-only EMA.

## 3. Verification performed

### 3.1 Recovered tests execute and pass

5/5 pass on the project interpreter. Log:
`artifacts/m1_scaffold_recovery/recovered_tests.log`.

**They prove very little.** All three modules exercise 3–4 layer `nn.Sequential`
dummies, never AudioLDM. `pytest` is **not installed** in `.venv` (the frozen
`poetry.lock` does not carry it), so the tests were run with a stdlib runner,
`artifacts/m1_scaffold_recovery/run_tests.py`.

### 3.2 Injection on the real pruned U-Net

`UNetModel` rebuilt from the frozen config at `channel_mult=[1,2,3,1]`,
`model_channels=192`. Log: `artifacts/m1_scaffold_recovery/real_unet_audit.log`.

```text
module inventory   Linear 185 | Conv2d 99 | GroupNorm32 45 | GroupNorm 16 | LayerNorm 48
Conv2d groups != 1 0        -> LoRAConv2d's groups==1 restriction costs nothing
Conv1d             0        -> use_spatial_transformer=true removes AttentionBlock's
                               conv_nd(1,...) qkv/proj_out entirely; nothing is missed
modules wrapped    284      (185 Linear + 99 Conv2d, i.e. every eligible module)

total params       149,392,648      (= 145,673,864 base + 3,718,784 LoRA; the base
                                     figure matches the M0-verified 145.674 M)
trainable total      3,894,088      2.607 %
  LoRA               3,718,784      2.489 %
  bias                 126,536      0.085 %
  GroupNorm affine      48,768      0.033 %
  other trainable            0      <- injector leaves nothing unaccounted for
optimizer groups   lora / bias / groupnorm_affine, sizes as above
```

### 3.3 Merge equivalence on the real U-Net

Log: `artifacts/m1_scaffold_recovery/real_merge_audit.log`. `lora_B` was seeded
with `N(0, 0.02)` first, otherwise the delta is identically zero and the test is
vacuous.

```text
forward CPU, batch 1, latent [1,8,256,16]
  unmerged                 2.32 s
  merged                   1.56 s      (-33 %; merging matters for evaluation cost)
max |unmerged - merged|    9.779e-08   output std 4.018e-02  -> rel. err ~2.4e-06
max |unmerged - unmerge()| 6.706e-08   -> unmerge restores exactly in float32
max |LoRA - no LoRA|       2.144e-01   -> the adapter genuinely changes the output
```

The U-Net must be called as `unet(x, t, y=y, context_list=[],
context_attn_mask_list=[])`. `context_dim` is commented out in the frozen config,
so the model is **FiLM-only**: `SpatialTransformer` blocks exist but receive no
cross-attention context.

**Verdict: adopt and fix, do not rebuild.** The scaffold is coherent, matches the
master plan's engineering decisions, and is numerically correct on the real
pruned U-Net.

## 4. Defects found

| # | Severity | Finding |
|---|---|---|
| F1 | blocking | `pytest` absent from `.venv`; the M1 acceptance criterion "CPU tests pass" has no runner. Adding it is an environment deviation that must be recorded and must not relax any pinned scientific version. |
| F2 | scientific | **LayerNorm is left half-trained.** `train_bias` makes all 48 LayerNorm biases trainable (17 856 params, **14.1 %** of the `bias` bucket) while their gains stay frozen, because only `nn.GroupNorm` weights are unfrozen. Neither the plan nor `report.py` declares this. It needs an explicit config flag and a separate line in the parameter report, or the paper's "biases and GroupNorm affine parameters" description is inaccurate. |
| F3 | correctness | `configure_auxiliary_trainables` counts only parameters that were frozen at call time (`and not bias.requires_grad`). Calling it twice returns zeros; the reported counts depend on call order. Count unconditionally. |
| F4 | correctness | No ordering guard. `freeze_for_peft` after `inject_lora` silently freezes `lora_A`/`lora_B` and training becomes a no-op with no error. Needs an assertion and a regression test. |
| F5 | efficiency | `LoRAConv2d.forward` materialises the full `[out, in, kh, kw]` delta every step and runs a second full convolution. Factorising into `conv2d(x, A) -> conv2d(·, B)` (1x1) is mathematically identical and costs roughly `r/out` of the base convolution. At 100 k recovery steps × 3 variants this is real money. |
| F6 | evidence | Tests never touch AudioLDM. §3.2/§3.3 above must become repository tests. |
| F7 | gap | `adaptation_state_dict` saves adapter parameters only. M1 acceptance also requires **full resume state** (optimizer, scheduler, step, EMA). Not implemented. |
| F8 | gap | Upstream integration is notes only: `configure_optimizers()` hook, EMA initialisation order, checkpoint/resume path, post-checkpoint-load injection. This is the bulk of the remaining M1 work. |

`docs/integration_notes.md` in the overlay correctly identifies the upstream
patch points; item 3 (upstream `configure_optimizers()` collecting the whole
diffusion model plus conditional-stage parameters) was independently confirmed
against `audioldm_train/modules/latent_diffusion/ddpm.py`.

## 5. Consequence for the plan

M1 moves from **blocked** to **recovered, audited, adoption pending**. The
authorisation Gabriel gave to rebuild from scratch was deliberately not used: the
recovered artifact is real, and rebuilding would have discarded genuine
provenance and reproduced the same eight defects from memory.
