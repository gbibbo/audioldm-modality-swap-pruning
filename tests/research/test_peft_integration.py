#!/usr/bin/env python3
"""M1 PEFT upstream-integration hook tests (CPU-only) — audit finding F8.

Exercises the functions a GPU session will wire into LatentDiffusion, without
touching audioldm_train/:

    I1 SETUP           setup_peft runs freeze->inject->aux in the correct order,
                       leaves only PEFT params trainable, and asserts readiness.
    I2 OPTIMIZER       build_peft_optimizer returns AdamW over exactly the PEFT
                       groups (lora/bias/groupnorm_affine), per-group LRs applied,
                       and every optimized tensor is a trainable model parameter.
    I3 CONFIG-PARSE    peft_config_from_yaml parses the real research_peft yaml
                       into matching PeftConfig / optimizer / ema kwargs.
    I4 POST-LOAD-ORDER injection must follow checkpoint load: after wrapping, the
                       original checkpoint keys no longer strict-load (renamed to
                       ...base.weight), yet the base weights are preserved exactly
                       through wrapping (integration_notes.md item 2).

Run directly:

    .venv/bin/python tests/research/test_peft_integration.py
"""
from __future__ import annotations

import sys

import torch
import yaml
from torch import nn

from audioldm_peft import (
    PeftConfig, setup_peft, build_peft_optimizer, peft_config_from_yaml,
    build_trainable_only_ema, parameter_report,
)

CONFIG_YAML = "configs/research/peft_r8_full_unet.yaml"


class Dummy(nn.Module):
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


def _cfg():
    return PeftConfig(root_path="model.diffusion_model", rank=2, alpha=4)


def check_i1_setup() -> bool:
    m = Dummy()
    info = setup_peft(m, _cfg())
    rep = info["report"]
    ok = (
        len(info["injected"]) == 2
        and rep["lora"] > 0 and rep["bias"] > 0 and rep["groupnorm_affine"] > 0
        and rep.get("other_trainable", 0) == 0
        and rep["trainable_total"] < rep["total"]
    )
    print(f"  injected={len(info['injected'])} aux={info['aux']} "
          f"trainable={rep['trainable_total']}/{rep['total']} other={rep.get('other_trainable',0)}")
    return bool(ok)


def check_i2_optimizer() -> bool:
    m = Dummy()
    setup_peft(m, _cfg())
    opt, groups = build_peft_optimizer(m, lora_lr=1e-4, auxiliary_lr=5e-5)
    names = {g["group_name"] for g in groups}
    lrs = {g["group_name"]: g["lr"] for g in groups}
    trainable_ids = {id(p) for p in m.parameters() if p.requires_grad}
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    n_opt = sum(len(g["params"]) for g in opt.param_groups)
    n_trainable = sum(1 for p in m.parameters() if p.requires_grad)
    ok = (
        isinstance(opt, torch.optim.AdamW)
        and names == {"lora", "bias", "groupnorm_affine"}
        and lrs["lora"] == 1e-4 and lrs["bias"] == 5e-5 and lrs["groupnorm_affine"] == 5e-5
        and opt_ids == trainable_ids and n_opt == n_trainable
    )
    print(f"  groups={sorted(names)} lrs={lrs} opt_params={n_opt}==trainable={n_trainable} "
          f"ids_match={opt_ids == trainable_ids}")
    return bool(ok)


def check_i3_config_parse() -> bool:
    with open(CONFIG_YAML) as handle:
        doc = yaml.safe_load(handle)
    cfg, opt_kwargs, ema_kwargs = peft_config_from_yaml(doc)
    ok = (
        cfg.rank == 8 and cfg.alpha == 16.0 and cfg.dropout == 0.0
        and cfg.target_linear and cfg.target_conv2d
        and cfg.train_bias and cfg.train_groupnorm_affine
        and cfg.train_layernorm_affine is False
        and cfg.root_path == "model.diffusion_model"
        and opt_kwargs["lora_lr"] == 1e-4 and opt_kwargs["auxiliary_lr"] == 1e-4
        and opt_kwargs["lora_weight_decay"] == 0.01 and opt_kwargs["auxiliary_weight_decay"] == 0.0
        and ema_kwargs["policy"] == "trainable_only" and ema_kwargs["decay"] == 0.9999
    )
    print(f"  cfg(rank={cfg.rank},alpha={cfg.alpha},ln_affine={cfg.train_layernorm_affine}) "
          f"opt={opt_kwargs} ema={ema_kwargs}")
    return bool(ok)


def check_i4_post_load_order() -> bool:
    m = Dummy()
    # A pretend "external checkpoint": the base state BEFORE any PEFT wrapping.
    ckpt = {k: v.detach().clone() for k, v in m.state_dict().items()}
    conv_w_key = "model.diffusion_model.0.weight"
    conv_w_before = ckpt[conv_w_key].clone()

    setup_peft(m, _cfg())  # wrap AFTER "loading" the checkpoint

    # Loading the ORIGINAL checkpoint now must fail strict matching (keys renamed).
    order_matters = False
    try:
        m.load_state_dict(ckpt, strict=True)
    except RuntimeError:
        order_matters = True

    # ...but the base weights were preserved by wrapping (not reinitialised).
    wrapped_base_w = m.model.diffusion_model[0].base.weight.detach()
    preserved = torch.equal(wrapped_base_w, conv_w_before)
    # and the renamed key exists in the wrapped state_dict.
    renamed_present = "model.diffusion_model.0.base.weight" in m.state_dict()
    print(f"  strict_load_of_old_ckpt_fails={order_matters} base_preserved={preserved} "
          f"renamed_key_present={renamed_present}")
    return bool(order_matters and preserved and renamed_present)


TESTS = [
    ("I1 SETUP", check_i1_setup),
    ("I2 OPTIMIZER", check_i2_optimizer),
    ("I3 CONFIG-PARSE", check_i3_config_parse),
    ("I4 POST-LOAD-ORDER", check_i4_post_load_order),
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
    print("\n==== M1 PEFT INTEGRATION TESTS (F8) ====")
    for name, _ in TESTS:
        print(f"  {name:<20} {'PASS' if results[name] else 'FAIL'}")
    all_ok = all(results.values())
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
