#!/usr/bin/env python3
"""M1 PEFT tests on the REAL pruned AudioLDM U-Net (CPU-only) — audit finding F6.

The dummy-model tests exercise 3–4 layer nn.Sequential stand-ins. This module
builds the actual `(1,2,3,1)` diffusion U-Net from the frozen upstream config
(no checkpoint needed; parameter counts and merge algebra are weight-independent)
and asserts:

    R6a DECOMPOSITION   inject over the whole U-Net wraps exactly 284 modules
                        (185 Linear + 99 Conv2d); the parameter report matches the
                        measured decomposition exactly, with no `other_trainable`
                        and internal sums consistent. The bias bucket is 108,680
                        (LayerNorm biases correctly excluded — F2), not the
                        126,536 the pre-fix scaffold reported.
    R6b MERGE-EQUIV     with a nonzero adapter, a real forward is invariant under
                        merge and exactly restored by unmerge (max|Δ| < 1e-5).
    R6c LAYERNORM-ON    train_layernorm_affine=True adds exactly 35,712 params
                        (48 LayerNorms, weight+bias) under `layernorm_affine`.

Run directly:

    .venv/bin/python tests/research/test_peft_real_unet.py
"""
from __future__ import annotations

import copy
import sys

import torch
import yaml
from torch import nn

from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
from audioldm_peft import (
    PeftConfig, freeze_for_peft, inject_lora, configure_auxiliary_trainables,
    assert_peft_ready, parameter_report, merge_all_lora, unmerge_all_lora,
    LoRALinear, LoRAConv2d,
)

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"

# Measured on the real pruned U-Net (M1-005 / F6); see docs/experiment_ledger.md.
EXPECT = {
    "base_total": 145_673_864,
    "wrapped": 284, "n_linear": 185, "n_conv2d": 99,
    "total": 149_392_648, "trainable_total": 3_876_232,
    "lora": 3_718_784, "bias": 108_680, "groupnorm_affine": 48_768,
    "layernorm_affine_on": 35_712,
}


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


def _setup(train_layernorm_affine=False, seed=0):
    torch.manual_seed(seed)
    unet = _pruned_unet()
    base_total = sum(p.numel() for p in unet.parameters())  # before injection
    m = Holder(unet)
    cfg = PeftConfig(root_path="model.diffusion_model", rank=8, alpha=16,
                     train_layernorm_affine=train_layernorm_affine)
    freeze_for_peft(m)
    inject_lora(m, cfg)
    aux = configure_auxiliary_trainables(m, cfg)
    assert_peft_ready(m, cfg)
    return m, unet, aux, base_total


def check_r6a_decomposition() -> bool:
    m, unet, _, base_total = _setup()
    n_lin = sum(1 for mod in m.modules() if isinstance(mod, LoRALinear))
    n_conv = sum(1 for mod in m.modules() if isinstance(mod, LoRAConv2d))
    rep = parameter_report(m)
    ln = rep.get("layernorm_affine", 0)
    other = rep.get("other_trainable", 0)
    consistent = (rep["lora"] + rep["bias"] + rep["groupnorm_affine"] + ln) == rep["trainable_total"]
    ok = (
        base_total == EXPECT["base_total"]
        and (n_lin + n_conv) == EXPECT["wrapped"]
        and n_lin == EXPECT["n_linear"] and n_conv == EXPECT["n_conv2d"]
        and rep["total"] == EXPECT["total"]
        and rep["trainable_total"] == EXPECT["trainable_total"]
        and rep["lora"] == EXPECT["lora"]
        and rep["bias"] == EXPECT["bias"]
        and rep["groupnorm_affine"] == EXPECT["groupnorm_affine"]
        and ln == 0 and other == 0
        and consistent
        and rep["total"] == base_total + rep["lora"]
    )
    print(f"  base={base_total} wrapped={n_lin+n_conv}({n_lin}L+{n_conv}C) "
          f"lora={rep['lora']} bias={rep['bias']} gn={rep['groupnorm_affine']} "
          f"ln={ln} other={other} trainable={rep['trainable_total']}/{rep['total']}")
    return bool(ok)


def check_r6b_merge_equiv() -> bool:
    m, unet, _, _ = _setup(seed=0)
    for mod in m.modules():
        if isinstance(mod, (LoRALinear, LoRAConv2d)):
            with torch.no_grad():
                mod.lora_B.normal_(0, 0.02)
    x = torch.randn(1, 8, 256, 16)
    t = torch.randint(0, 1000, (1,))
    y = torch.randn(1, 512)
    with torch.no_grad():
        y0 = unet(x, t, y=y, context_list=[], context_attn_mask_list=[])
        merge_all_lora(m)
        y1 = unet(x, t, y=y, context_list=[], context_attn_mask_list=[])
        unmerge_all_lora(m)
        y2 = unet(x, t, y=y, context_list=[], context_attn_mask_list=[])
    d_merge = (y0 - y1).abs().max().item()
    d_unmerge = (y0 - y2).abs().max().item()
    print(f"  out={tuple(y0.shape)} std={y0.std():.3e} "
          f"max|unmerged-merged|={d_merge:.2e} max|unmerged-unmerge()|={d_unmerge:.2e}")
    return bool(d_merge < 1e-5 and d_unmerge < 1e-5)


def check_r6c_layernorm_on() -> bool:
    m, _, aux, _ = _setup(train_layernorm_affine=True)
    rep = parameter_report(m)
    ln_count = rep.get("layernorm_affine", 0)
    trainable = all(
        (mod.weight.requires_grad and mod.bias.requires_grad)
        for mod in m.modules()
        if isinstance(mod, nn.LayerNorm) and mod.elementwise_affine
    )
    ok = (ln_count == EXPECT["layernorm_affine_on"]
          and aux["layernorm_affine"] == EXPECT["layernorm_affine_on"]
          and trainable and rep.get("other_trainable", 0) == 0)
    print(f"  layernorm_affine={ln_count} (expect {EXPECT['layernorm_affine_on']}) all_ln_trainable={trainable}")
    return bool(ok)


TESTS = [
    ("R6a DECOMPOSITION", check_r6a_decomposition),
    ("R6b MERGE-EQUIV", check_r6b_merge_equiv),
    ("R6c LAYERNORM-ON", check_r6c_layernorm_on),
]


def main() -> int:
    results = {}
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results[name] = False
    print("\n==== M1 REAL-U-NET PEFT TESTS (F6) ====")
    for name, _ in TESTS:
        print(f"  {name:<20} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
