"""Per-category trainable-parameter accounting.

Adopted from the recovered M1 scaffold; audited in ``docs/m1_scaffold_audit.md``.
Fix F2: LayerNorm affine parameters are categorised as ``layernorm_affine`` rather
than silently folded into ``bias``, so the report separates LoRA, bias, GroupNorm
affine, and LayerNorm affine as the master plan requires.
"""
from collections import defaultdict
from typing import Dict

from torch import nn

from .layers import LoRALinear, LoRAConv2d


def parameter_categories(model: nn.Module):
    category_by_id = {}
    for module in model.modules():
        if isinstance(module, (LoRALinear, LoRAConv2d)):
            category_by_id[id(module.lora_A)] = "lora"
            category_by_id[id(module.lora_B)] = "lora"
            if module.base.bias is not None:
                category_by_id[id(module.base.bias)] = "bias"
        elif isinstance(module, nn.GroupNorm) and module.affine:
            if module.weight is not None:
                category_by_id[id(module.weight)] = "groupnorm_affine"
            if module.bias is not None:
                category_by_id[id(module.bias)] = "groupnorm_affine"
        elif isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            if module.weight is not None:
                category_by_id[id(module.weight)] = "layernorm_affine"
            if module.bias is not None:
                category_by_id[id(module.bias)] = "layernorm_affine"
        else:
            bias = getattr(module, "bias", None)
            if isinstance(bias, nn.Parameter):
                category_by_id.setdefault(id(bias), "bias")
    return category_by_id


def parameter_report(model: nn.Module) -> Dict[str, int]:
    out = defaultdict(int)
    categories = parameter_categories(model)
    for _, p in model.named_parameters():
        n = p.numel()
        out["total"] += n
        if p.requires_grad:
            out["trainable_total"] += n
            out[categories.get(id(p), "other_trainable")] += n
    return dict(out)
