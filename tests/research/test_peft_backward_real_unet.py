#!/usr/bin/env python3
"""PEFT BACKWARD through the real checkpointed U-Net (CPU-only) — closes the F9/F10 gap.

`test_peft_real_unet.py` (R6a-c) wraps modules, counts parameters and checks merge
algebra, but it never performs a `backward`. That gap let two real defects reach the
first GPU session (ledger M1-008):

    F9  LoRA adapters were created on the default device, so injecting PEFT after the
        model had been moved to CUDA raised a cuda/cpu mismatch on the first forward.
    F10 `CheckpointFunction.backward` differentiated w.r.t. `input_tensors +
        input_params`; PEFT freezes the base weights, so autograd raised
        "One of the differentiated Tensors does not require grad". Fixed by the
        project's single upstream patch (DECISION-F10).

This module runs a REAL optimization step and asserts the gradients are not merely
non-crashing but numerically correct.

    R7a STEP        one full fwd+bwd+optimizer step on the real pruned U-Net with REAL
                    pretrained weights: every one of the 284 `lora_B` adapters receives
                    a non-zero gradient (so gradient really reaches through the
                    checkpointed transformer blocks), every `lora_A` gradient is exactly
                    zero (B is zero-initialised, so the adapter starts as identity),
                    frozen base weights get no gradient, and after `opt.step()` the LoRA
                    parameters moved while every base weight is bit-identical.
    R7b CKPT-EQUIV  gradients computed through the checkpointed path equal those from
                    the non-checkpointed path. This is what proves the F10 patch
                    computes the RIGHT gradients rather than just avoiding the raise.
    R7c ZEROINIT    the trap that would make this whole module vacuous: on a FRESHLY
                    INITIALISED U-Net the final output conv is `zero_module`-ed, so no
                    gradient propagates backward past it and only ONE adapter is
                    exercised. Asserted explicitly so nobody "simplifies" R7a/R7b back
                    to a random-init model and silently tests almost nothing.
    R7d FACTORY     F9 regression: adapters are built on the base layer's device AND
                    dtype, checked non-vacuously on CPU via a float64 base layer (and
                    additionally on CUDA when a GPU happens to be present).

Requires `data/checkpoints/l1_audioldm-m-full_p1.ckpt` (gitignored, fetched by
`scripts/research/fetch_public_artifacts.sh`). Skips with a clear message if absent.

Run: .venv/bin/python tests/research/test_peft_backward_real_unet.py
"""
from __future__ import annotations

import copy
import os
import sys

import torch
import yaml
from torch import nn

from audioldm_train.modules.diffusionmodules.attention import BasicTransformerBlock
from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
from audioldm_peft import PeftConfig, build_peft_optimizer, setup_peft
from audioldm_peft.layers import LoRAConv2d, LoRALinear

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
CKPT = "data/checkpoints/l1_audioldm-m-full_p1.ckpt"
PREFIX = "model.diffusion_model."

N_ADAPTERS = 284          # measured in R6a
N_TRANSFORMER_BLOCKS = 16  # every one has checkpoint=True by default (attention.py:379)


class Holder(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = unet


def _pruned_unet():
    with open(CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    params = copy.deepcopy(cfg["model"]["params"]["unet_config"]["params"])
    params["channel_mult"] = [1, 2, 3, 1]
    return UNetModel(**params)


def _load_real_weights(unet) -> None:
    """Strict-load the published pruned weights. Real weights matter here: see R7c."""
    obj = torch.load(CKPT, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    weights = {k[len(PREFIX):]: v for k, v in state.items() if k.startswith(PREFIX)}
    unet.load_state_dict(weights, strict=True)


def _peft_model(real_weights: bool, checkpoint_flag=None):
    torch.manual_seed(1234)
    unet = _pruned_unet()
    if real_weights:
        _load_real_weights(unet)
    holder = Holder(unet)
    setup_peft(holder, PeftConfig(root_path="model.diffusion_model", rank=8, alpha=16))
    unet = holder.model.diffusion_model
    unet.eval()                      # no dropout: the comparison in R7b must be exact
    if checkpoint_flag is not None:
        for mod in unet.modules():
            if isinstance(mod, BasicTransformerBlock):
                mod.checkpoint = checkpoint_flag
    return holder, unet


def _batch(batch=1):
    torch.manual_seed(7)
    return (torch.randn(batch, 8, 256, 16), torch.randint(0, 1000, (batch,)),
            torch.randn(batch, 512), torch.randn(batch, 8, 256, 16))


def _fwd_bwd(unet):
    z, t, y, eps = _batch()
    out = unet(z, t, y=y, context_list=[], context_attn_mask_list=[])
    torch.nn.functional.mse_loss(out, eps).backward()
    return out


def _named(unet, suffix):
    return [(n, p) for n, p in unet.named_parameters() if n.endswith(suffix)]


def _missing_ckpt() -> bool:
    if not os.path.exists(CKPT):
        print(f"  SKIP: {CKPT} not present (fetch_public_artifacts.sh)")
        return True
    return False


# --------------------------------------------------------------------------- R7a
def check_r7a_step() -> bool:
    if _missing_ckpt():
        return True
    holder, unet = _peft_model(real_weights=True)
    opt, _ = build_peft_optimizer(holder, lora_lr=1e-3, auxiliary_lr=1e-3)

    n_btb = sum(1 for m in unet.modules() if isinstance(m, BasicTransformerBlock))
    ckpt_on = {bool(m.checkpoint) for m in unet.modules()
               if isinstance(m, BasicTransformerBlock)}
    print(f"  BasicTransformerBlocks: {n_btb} (expect {N_TRANSFORMER_BLOCKS}), "
          f"checkpoint flags {ckpt_on}")

    base_before = {n: p.detach().clone() for n, p in unet.named_parameters()
                   if "lora_" not in n and not p.requires_grad}
    lora_before = {n: p.detach().clone() for n, p in unet.named_parameters()
                   if "lora_" in n}

    opt.zero_grad(set_to_none=True)
    _fwd_bwd(unet)   # this is what raised before the F10 patch

    grads_a = _named(unet, "lora_A")
    grads_b = _named(unet, "lora_B")
    nz_b = sum(1 for _, p in grads_b if p.grad is not None and p.grad.abs().sum().item() > 0)
    nz_a = sum(1 for _, p in grads_a if p.grad is not None and p.grad.abs().sum().item() > 0)
    none_b = sum(1 for _, p in grads_b if p.grad is None)
    frozen_with_grad = [n for n, p in unet.named_parameters()
                        if "lora_" not in n and not p.requires_grad and p.grad is not None]

    print(f"  lora_B: {len(grads_b)} total, {nz_b} with NON-ZERO grad, {none_b} with None")
    print(f"  lora_A: {len(grads_a)} total, {nz_a} with non-zero grad "
          f"(expected 0 — B is zero-initialised so dL/dA = 0 at step 0)")
    print(f"  frozen base params that received a gradient: {len(frozen_with_grad)}")

    ok = len(grads_b) == N_ADAPTERS and len(grads_a) == N_ADAPTERS
    ok &= n_btb == N_TRANSFORMER_BLOCKS and ckpt_on == {True}
    # THE point of this test: gradient reaches EVERY adapter, not just the last one.
    ok &= nz_b == N_ADAPTERS
    ok &= nz_a == 0
    ok &= none_b == 0
    ok &= not frozen_with_grad

    opt.step()
    moved = sum(1 for n, p in unet.named_parameters()
                if "lora_" in n and not torch.equal(p.detach(), lora_before[n]))
    base_changed = [n for n, p in unet.named_parameters()
                    if n in base_before and not torch.equal(p.detach(), base_before[n])]
    print(f"  after opt.step(): {moved}/{len(lora_before)} LoRA tensors changed, "
          f"{len(base_changed)} frozen base tensors changed (must be 0)")
    ok &= moved > 0 and not base_changed
    return ok


# --------------------------------------------------------------------------- R7b
def check_r7b_ckpt_equiv() -> bool:
    if _missing_ckpt():
        return True
    grads = {}
    for flag in (True, False):
        _, unet = _peft_model(real_weights=True, checkpoint_flag=flag)
        _fwd_bwd(unet)
        grads[flag] = {n: (p.grad.detach().clone() if p.grad is not None else None)
                       for n, p in unet.named_parameters() if "lora_" in n}

    shared = [n for n in grads[True]
              if grads[True][n] is not None and grads[False][n] is not None]
    max_delta = max((grads[True][n] - grads[False][n]).abs().max().item() for n in shared)
    max_mag = max(grads[False][n].abs().max().item() for n in shared)
    print(f"  compared {len(shared)} LoRA gradient tensors "
          f"(checkpointed vs non-checkpointed)")
    print(f"  max|Delta| = {max_delta:.3e}   max|grad| = {max_mag:.3e}")
    ok = len(shared) == 2 * N_ADAPTERS
    ok &= max_delta < 1e-6      # the patch must reproduce plain autograd
    ok &= max_mag > 0.0         # and the comparison must not be trivially all-zero
    return ok


# --------------------------------------------------------------------------- R7c
def check_r7c_zeroinit_trap() -> bool:
    """A freshly-initialised U-Net cannot test this: `zero_module` blocks the gradient."""
    _, unet = _peft_model(real_weights=False)
    final_conv = unet.out[2]
    # PEFT has wrapped it, so the real Conv2d is one level down.
    base_conv = final_conv.base if hasattr(final_conv, "base") else final_conv
    w_sum = base_conv.weight.abs().sum().item()
    print(f"  fresh model final out conv sum|W| = {w_sum} (zero_module -> {w_sum == 0.0})")
    _fwd_bwd(unet)
    nz = [n for n, p in _named(unet, "lora_B")
          if p.grad is not None and p.grad.abs().sum().item() > 0]
    print(f"  adapters with non-zero grad on a FRESH model: {len(nz)} of {N_ADAPTERS} "
          f"-> {nz}")
    ok = w_sum == 0.0 and len(nz) == 1 and nz[0].endswith("out.2.lora_B")

    if not _missing_ckpt():
        real = _pruned_unet()
        _load_real_weights(real)
        real_sum = real.out[2].weight.abs().sum().item()
        print(f"  real pretrained final conv sum|W| = {real_sum:.4f} "
              f"-> gradients propagate: {real_sum > 0}")
        ok &= real_sum > 0
    return ok


# --------------------------------------------------------------------------- R7d
def check_r7d_factory() -> bool:
    """F9: adapters must inherit the base layer's device AND dtype."""
    ok = True
    lin = nn.Linear(16, 32).to(torch.float64)
    wrapped = LoRALinear(lin, rank=4)
    conv = nn.Conv2d(4, 6, 3, padding=1).to(torch.float64)
    wrapped_c = LoRAConv2d(conv, rank=4)
    for name, mod, base in (("LoRALinear", wrapped, lin), ("LoRAConv2d", wrapped_c, conv)):
        same = (mod.lora_A.dtype == base.weight.dtype
                and mod.lora_B.dtype == base.weight.dtype
                and mod.lora_A.device == base.weight.device
                and mod.lora_B.device == base.weight.device)
        print(f"  {name}: base {base.weight.dtype}/{base.weight.device} -> "
              f"adapters {mod.lora_A.dtype}/{mod.lora_A.device}  match={same}")
        ok &= same

    if torch.cuda.is_available():
        cl = nn.Linear(16, 32).cuda()
        wc = LoRALinear(cl, rank=4)
        on_cuda = wc.lora_A.device == cl.weight.device and wc.lora_B.device == cl.weight.device
        print(f"  CUDA present: adapters on {wc.lora_A.device} match={on_cuda}")
        ok &= on_cuda
    else:
        print("  (no CUDA here; the dtype check above is what makes this non-vacuous)")
    return ok


def test_r7a_step():
    assert check_r7a_step()


def test_r7b_ckpt_equiv():
    assert check_r7b_ckpt_equiv()


def test_r7c_zeroinit_trap():
    assert check_r7c_zeroinit_trap()


def test_r7d_factory():
    assert check_r7d_factory()


def main() -> int:
    checks = [
        ("R7a STEP", check_r7a_step),
        ("R7b CKPT-EQUIV", check_r7b_ckpt_equiv),
        ("R7c ZEROINIT", check_r7c_zeroinit_trap),
        ("R7d FACTORY", check_r7d_factory),
    ]
    results = {}
    for name, fn in checks:
        print(f"\n[{name}]")
        results[name] = bool(fn())
    print("\n==== M1 PEFT BACKWARD TESTS (real U-Net) ====")
    for name, _ in checks:
        print(f"  {name:<16} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
