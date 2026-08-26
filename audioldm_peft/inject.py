"""Injection, freezing and auxiliary-trainable selection.

Adopted from the recovered M1 scaffold; audited in ``docs/m1_scaffold_audit.md``.

Fix F3: ``configure_auxiliary_trainables`` counts the parameters it is responsible
for unconditionally, rather than only those that happened to be frozen at call
time. The old ``and not p.requires_grad`` guard made the reported counts depend on
call order and return zeros on a second call.

Fix F4: ``freeze_for_peft`` never freezes LoRA adapter parameters, so calling it
after ``inject_lora`` no longer silently turns training into a no-op. ``assert_peft_ready``
is a cheap post-setup guard (used by a regression test).

Fix F2: LayerNorm affine is handled explicitly (all-or-nothing under
``cfg.train_layernorm_affine``) and never half-trained through the generic bias
sweep. GroupNorm and LayerNorm are both excluded from the generic bias loop.
"""
from typing import Dict, List, Tuple

from torch import nn

from .config import PeftConfig
from .layers import LoRALinear, LoRAConv2d, lora_param_ids, iter_lora_modules


def resolve_module(root: nn.Module, dotted_path: str) -> nn.Module:
    module = root
    if not dotted_path:
        return module
    for part in dotted_path.split("."):
        module = getattr(module, part)
    return module


def _name_allowed(name: str, cfg: PeftConfig) -> bool:
    if cfg.include_name_substrings and not any(x in name for x in cfg.include_name_substrings):
        return False
    if cfg.exclude_name_substrings and any(x in name for x in cfg.exclude_name_substrings):
        return False
    return True


def freeze_for_peft(root: nn.Module) -> None:
    """Freeze every base parameter, preserving LoRA adapters as trainable (F4).

    Order-independent: safe to call before or after ``inject_lora``. Adapter
    parameters (``lora_A`` / ``lora_B``) are left with their existing
    ``requires_grad`` so a mis-ordered call cannot silently disable training.
    """
    keep_trainable = lora_param_ids(root)
    for p in root.parameters():
        if id(p) not in keep_trainable:
            p.requires_grad = False


def inject_lora(model: nn.Module, cfg: PeftConfig) -> List[str]:
    cfg.validate()
    target_root = resolve_module(model, cfg.root_path)
    replaced: List[str] = []
    candidates: List[Tuple[str, nn.Module]] = list(target_root.named_modules())
    for local_name, module in candidates:
        if not local_name or not _name_allowed(local_name, cfg):
            continue
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            continue
        wrapper = None
        if cfg.target_linear and isinstance(module, nn.Linear):
            wrapper = LoRALinear(module, cfg.rank, cfg.alpha, cfg.dropout, init=cfg.init_lora_weights)
        elif cfg.target_conv2d and isinstance(module, nn.Conv2d) and module.groups == 1:
            wrapper = LoRAConv2d(module, cfg.rank, cfg.alpha, cfg.dropout, init=cfg.init_lora_weights)
        if wrapper is None:
            continue
        parts = local_name.split(".")
        parent = target_root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)
        replaced.append(f"{cfg.root_path}.{local_name}")
    return replaced


def configure_auxiliary_trainables(model: nn.Module, cfg: PeftConfig) -> Dict[str, int]:
    """Unfreeze and count auxiliary trainables (F2 + F3).

    Counts are computed unconditionally over the parameters each option owns, so
    the report is stable across repeated calls and independent of prior
    ``requires_grad`` state. GroupNorm and LayerNorm are excluded from the generic
    bias loop; LayerNorm is trained (weight and bias together) only when
    ``cfg.train_layernorm_affine`` is set, and is reported as its own category.
    """
    target_root = resolve_module(model, cfg.root_path)
    bias_count = 0
    groupnorm_count = 0
    layernorm_count = 0

    if cfg.train_bias:
        for module in target_root.modules():
            if isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
                continue  # never half-train a norm layer through the generic bias sweep
            bias = getattr(module, "bias", None)
            if isinstance(bias, nn.Parameter):
                bias.requires_grad = True
                bias_count += bias.numel()

    if cfg.train_groupnorm_affine:
        for module in target_root.modules():
            if isinstance(module, nn.GroupNorm) and module.affine:
                if module.weight is not None:
                    module.weight.requires_grad = True
                    groupnorm_count += module.weight.numel()
                if module.bias is not None:
                    module.bias.requires_grad = True
                    groupnorm_count += module.bias.numel()

    if cfg.train_layernorm_affine:
        for module in target_root.modules():
            if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                if module.weight is not None:
                    module.weight.requires_grad = True
                    layernorm_count += module.weight.numel()
                if module.bias is not None:
                    module.bias.requires_grad = True
                    layernorm_count += module.bias.numel()

    return {
        "bias": bias_count,
        "groupnorm_affine": groupnorm_count,
        "layernorm_affine": layernorm_count,
    }


def assert_peft_ready(model: nn.Module, cfg: PeftConfig) -> None:
    """Guard (F4): after setup, every LoRA adapter parameter must be trainable and
    every wrapped base weight must be frozen. Raises AssertionError otherwise."""
    n_adapters = 0
    for m in iter_lora_modules(model):
        n_adapters += 1
        assert m.lora_A.requires_grad and m.lora_B.requires_grad, (
            "LoRA adapter parameters are frozen; freeze_for_peft was likely called "
            "in a way that disabled training. Adapters must stay trainable."
        )
        assert not m.base.weight.requires_grad, "wrapped base weight is not frozen"
    assert n_adapters > 0, "no LoRA modules present; inject_lora did not run"
