#!/usr/bin/env python3
"""M1 injector / auxiliary-trainable tests (CPU-only).

    J1 INJECT-FREEZE    inject wraps Linear+Conv2d; base frozen; LoRA trainable;
                        GroupNorm affine reported; trainable_total < total.
    J2 F2-LAYERNORM     with train_bias=True but train_layernorm_affine=False, no
                        LayerNorm parameter is trainable (no half-trained norm);
                        the generic bias bucket excludes LayerNorm bias. With the
                        flag on, LayerNorm weight+bias appear under layernorm_affine.
    J3 F3-STABLE-COUNT  configure_auxiliary_trainables returns identical nonzero
                        counts on a second call (unconditional counting).
    J4 F4-ORDER-GUARD   calling freeze_for_peft AFTER inject keeps LoRA trainable;
                        assert_peft_ready passes. (Old scaffold silently no-op'd.)

Run directly:

    .venv/bin/python tests/research/test_injector.py
"""
from __future__ import annotations

import sys

import torch
from torch import nn

from audioldm_peft import (
    PeftConfig, freeze_for_peft, inject_lora, configure_auxiliary_trainables,
    parameter_report, assert_peft_ready,
)
from audioldm_peft.layers import LoRALinear, LoRAConv2d


class Dummy(nn.Module):
    """Has a Conv2d, a GroupNorm, a LayerNorm and a Linear under diffusion_model."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.diffusion_model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.GroupNorm(2, 4),
            nn.Flatten(),
            nn.LayerNorm(4 * 8 * 8),
            nn.Linear(4 * 8 * 8, 5),
        )


def _setup(cfg):
    m = Dummy()
    freeze_for_peft(m)
    names = inject_lora(m, cfg)
    aux = configure_auxiliary_trainables(m, cfg)
    return m, names, aux


def check_j1_inject_freeze() -> bool:
    cfg = PeftConfig(root_path="model.diffusion_model", rank=2, alpha=4)
    m, names, aux = _setup(cfg)
    report = parameter_report(m)
    ok = (
        len(names) == 2
        and isinstance(m.model.diffusion_model[0], LoRAConv2d)
        and isinstance(m.model.diffusion_model[4], LoRALinear)
        and report["lora"] > 0
        and report["trainable_total"] < report["total"]
        and aux["groupnorm_affine"] > 0
        and report.get("other_trainable", 0) == 0
    )
    print(f"  names={len(names)} report={{lora:{report['lora']}, gn:{aux['groupnorm_affine']}, "
          f"trainable:{report['trainable_total']}/{report['total']}}}")
    return bool(ok)


def check_j2_layernorm() -> bool:
    ln_dim = 4 * 8 * 8
    # flag OFF: no LayerNorm parameter trainable, and layernorm_affine count == 0
    cfg_off = PeftConfig(root_path="model.diffusion_model", rank=2, alpha=4,
                         train_bias=True, train_layernorm_affine=False)
    m_off, _, aux_off = _setup(cfg_off)
    ln = m_off.model.diffusion_model[3]
    off_ok = (
        aux_off["layernorm_affine"] == 0
        and not ln.weight.requires_grad
        and not ln.bias.requires_grad
    )
    rep_off = parameter_report(m_off)
    # LayerNorm has 2*ln_dim params; none should be counted as trainable bias.
    off_ok = off_ok and rep_off.get("other_trainable", 0) == 0

    # flag ON: LayerNorm weight+bias trainable under layernorm_affine
    cfg_on = PeftConfig(root_path="model.diffusion_model", rank=2, alpha=4,
                        train_bias=True, train_layernorm_affine=True)
    m_on, _, aux_on = _setup(cfg_on)
    ln_on = m_on.model.diffusion_model[3]
    on_ok = (
        aux_on["layernorm_affine"] == 2 * ln_dim
        and ln_on.weight.requires_grad and ln_on.bias.requires_grad
    )
    rep_on = parameter_report(m_on)
    on_ok = on_ok and rep_on["layernorm_affine"] == 2 * ln_dim and rep_on.get("other_trainable", 0) == 0
    print(f"  OFF layernorm_affine={aux_off['layernorm_affine']} (ln frozen={not ln.weight.requires_grad})  "
          f"ON layernorm_affine={aux_on['layernorm_affine']} (expect {2*ln_dim})")
    return bool(off_ok and on_ok)


def check_j3_stable_count() -> bool:
    cfg = PeftConfig(root_path="model.diffusion_model", rank=2, alpha=4)
    m, _, aux1 = _setup(cfg)
    aux2 = configure_auxiliary_trainables(m, cfg)  # second call
    ok = aux1 == aux2 and aux1["bias"] > 0 and aux1["groupnorm_affine"] > 0
    print(f"  call1={aux1} call2={aux2}")
    return bool(ok)


def check_j4_order_guard() -> bool:
    # Adversarial order: freeze AFTER inject. Must NOT disable LoRA training.
    cfg = PeftConfig(root_path="model.diffusion_model", rank=2, alpha=4)
    m = Dummy()
    inject_lora(m, cfg)
    freeze_for_peft(m)              # dangerous order
    configure_auxiliary_trainables(m, cfg)
    lora = [mod for mod in m.modules() if isinstance(mod, (LoRALinear, LoRAConv2d))]
    trainable = all(l.lora_A.requires_grad and l.lora_B.requires_grad for l in lora)
    guard_ok = True
    try:
        assert_peft_ready(m, cfg)
    except AssertionError:
        guard_ok = False
    print(f"  lora_trainable_after_late_freeze={trainable}  assert_peft_ready={guard_ok}")
    return bool(trainable and guard_ok)


TESTS = [
    ("J1 INJECT-FREEZE", check_j1_inject_freeze),
    ("J2 F2-LAYERNORM", check_j2_layernorm),
    ("J3 F3-STABLE-COUNT", check_j3_stable_count),
    ("J4 F4-ORDER-GUARD", check_j4_order_guard),
]


def main() -> int:
    results = {}
    for name, fn in TESTS:
        print(f"\n[{name}]")
        try:
            results[name] = bool(fn())
        except Exception as exc:  # noqa: BLE001
            print(f"  EXCEPTION: {exc}")
            results[name] = False
    print("\n==== M1 INJECTOR TESTS ====")
    for name, _ in TESTS:
        print(f"  {name:<20} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
