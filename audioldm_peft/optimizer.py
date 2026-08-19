"""Optimizer parameter groups for PEFT recovery.

Adopted from the recovered M1 scaffold; audited in ``docs/m1_scaffold_audit.md``.
Fix F2: a ``layernorm_affine`` group is emitted when LayerNorm affine parameters
are trainable, so no intended trainable is ever silently misrouted or rejected as
``other_trainable``.
"""
from torch import nn

from .report import parameter_categories


def build_parameter_groups(model: nn.Module, lora_lr: float, auxiliary_lr: float,
                           lora_weight_decay: float = 0.01, auxiliary_weight_decay: float = 0.0):
    buckets = {"lora": [], "bias": [], "groupnorm_affine": [], "layernorm_affine": [], "other_trainable": []}
    categories = parameter_categories(model)
    for _, p in model.named_parameters():
        if p.requires_grad:
            buckets[categories.get(id(p), "other_trainable")].append(p)

    if buckets["other_trainable"]:
        raise RuntimeError("Unexpected trainable parameters remain outside LoRA/bias/GroupNorm/LayerNorm")

    groups = []
    if buckets["lora"]:
        groups.append({"params": buckets["lora"], "lr": lora_lr, "weight_decay": lora_weight_decay, "group_name": "lora"})
    if buckets["bias"]:
        groups.append({"params": buckets["bias"], "lr": auxiliary_lr, "weight_decay": auxiliary_weight_decay, "group_name": "bias"})
    if buckets["groupnorm_affine"]:
        groups.append({"params": buckets["groupnorm_affine"], "lr": auxiliary_lr, "weight_decay": auxiliary_weight_decay, "group_name": "groupnorm_affine"})
    if buckets["layernorm_affine"]:
        groups.append({"params": buckets["layernorm_affine"], "lr": auxiliary_lr, "weight_decay": auxiliary_weight_decay, "group_name": "layernorm_affine"})
    if not groups:
        raise RuntimeError("No trainable PEFT parameters found")
    return groups
