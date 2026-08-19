"""Parameter-efficient recovery for pruned AudioLDM (LoRA + trainable biases + trainable GroupNorm affine).

Adopted from the recovered local M1 scaffold (2026-08-13 archive) and audited
against the real pruned U-Net in ``docs/m1_scaffold_audit.md``. Defects F2, F3,
F4, F5 and F7 from that audit are fixed here; F6 (real-U-Net tests) and F8
(upstream integration) live in ``tests/research/`` and ``audioldm_peft.integrate``.

LoRA is *not* the claimed novelty: when biases and GroupNorm affine parameters are
trainable, this is parameter-efficient recovery, and LoRA / bias / GroupNorm /
LayerNorm / total trainable parameters are reported separately.
"""
from .config import PeftConfig
from .inject import (
    inject_lora, freeze_for_peft, configure_auxiliary_trainables,
    assert_peft_ready, resolve_module,
)
from .layers import (
    LoRALinear, LoRAConv2d, merge_all_lora, unmerge_all_lora,
    iter_lora_modules, lora_param_ids,
)
from .report import parameter_report, parameter_categories
from .state import (
    adaptation_state_dict, load_adaptation_state_dict,
    training_state_dict, load_training_state_dict,
)
from .optimizer import build_parameter_groups
from .ema import TrainableOnlyEMA
from .integrate import (
    setup_peft, build_peft_optimizer, build_trainable_only_ema, peft_config_from_yaml,
)

__all__ = [
    "PeftConfig", "inject_lora", "freeze_for_peft", "configure_auxiliary_trainables",
    "assert_peft_ready", "resolve_module",
    "LoRALinear", "LoRAConv2d", "merge_all_lora", "unmerge_all_lora",
    "iter_lora_modules", "lora_param_ids",
    "parameter_report", "parameter_categories",
    "adaptation_state_dict", "load_adaptation_state_dict",
    "training_state_dict", "load_training_state_dict",
    "build_parameter_groups", "TrainableOnlyEMA",
    "setup_peft", "build_peft_optimizer", "build_trainable_only_ema", "peft_config_from_yaml",
]
