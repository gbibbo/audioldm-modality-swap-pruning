"""Upstream-integration hooks for PEFT recovery (audit finding F8).

These functions are the CPU-testable core of the minimal upstream patch described
in ``docs/integration_notes.md``. They do **not** modify ``audioldm_train/`` —
``git diff upstream-frozen -- audioldm_train/`` must stay empty. A GPU-enabled
session wires them into ``LatentDiffusion`` at the documented patch points:

* ``setup_peft`` runs after the external checkpoint load (freeze -> inject ->
  auxiliaries, in the one correct order, with a readiness assertion);
* ``build_peft_optimizer`` replaces ``configure_optimizers`` so only PEFT groups
  are optimized;
* ``build_trainable_only_ema`` constructs the EMA *after* setup, avoiding the
  upstream "EMA shadows the frozen U-Net" trap;
* ``peft_config_from_yaml`` parses ``configs/research/peft_r8_full_unet.yaml``.

The full-resume state hooks live in ``audioldm_peft.state``.
"""
from typing import Any, Dict, Tuple

import torch
from torch import nn

from .config import PeftConfig
from .inject import (
    freeze_for_peft, inject_lora, configure_auxiliary_trainables, assert_peft_ready,
)
from .report import parameter_report
from .optimizer import build_parameter_groups
from .ema import TrainableOnlyEMA


def setup_peft(model: nn.Module, cfg: PeftConfig) -> Dict[str, Any]:
    """Freeze base -> inject LoRA -> unfreeze auxiliaries, in the single correct order.

    MUST be called AFTER the external base/pruned checkpoint has been loaded into
    ``model``: injection renames wrapped weights to ``...base.weight``, so loading
    the original checkpoint afterwards would fail strict key matching.
    """
    cfg.validate()
    freeze_for_peft(model)
    injected = inject_lora(model, cfg)
    aux = configure_auxiliary_trainables(model, cfg)
    assert_peft_ready(model, cfg)
    return {"injected": injected, "aux": aux, "report": parameter_report(model)}


def build_peft_optimizer(model: nn.Module,
                         lora_lr: float = 1e-4,
                         auxiliary_lr: float = 1e-4,
                         lora_weight_decay: float = 0.01,
                         auxiliary_weight_decay: float = 0.0,
                         betas: Tuple[float, float] = (0.9, 0.999),
                         eps: float = 1e-8):
    """AdamW over only the PEFT parameter groups (optimizer contract, master plan M1).

    Returns ``(optimizer, groups)``. ``build_parameter_groups`` raises if any
    trainable parameter falls outside the LoRA/bias/GroupNorm/LayerNorm buckets,
    so an accidental full-model optimizer can never be built silently.
    """
    groups = build_parameter_groups(
        model, lora_lr, auxiliary_lr, lora_weight_decay, auxiliary_weight_decay,
    )
    optimizer = torch.optim.AdamW(groups, betas=betas, eps=eps)
    return optimizer, groups


def build_trainable_only_ema(model: nn.Module, decay: float = 0.9999) -> TrainableOnlyEMA:
    """Construct the trainable-only EMA. Call AFTER ``setup_peft`` so it shadows
    only the PEFT-trainable parameters, never the frozen U-Net."""
    return TrainableOnlyEMA(model, decay=decay)


def peft_config_from_yaml(doc: Dict[str, Any]) -> Tuple[PeftConfig, Dict[str, float], Dict[str, Any]]:
    """Parse a ``research_peft`` config dict into (PeftConfig, optimizer_kwargs, ema_kwargs)."""
    rp = doc["research_peft"] if "research_peft" in doc else doc
    cfg = PeftConfig(
        rank=int(rp.get("rank", 8)),
        alpha=float(rp.get("alpha", 16.0)),
        dropout=float(rp.get("dropout", 0.0)),
        target_linear=bool(rp.get("target_linear", True)),
        target_conv2d=bool(rp.get("target_conv2d", True)),
        train_bias=bool(rp.get("train_bias", True)),
        train_groupnorm_affine=bool(rp.get("train_groupnorm_affine", True)),
        train_layernorm_affine=bool(rp.get("train_layernorm_affine", False)),
        root_path=str(rp.get("root_path", "model.diffusion_model")),
    )
    opt = rp.get("optimizer", {}) or {}
    optimizer_kwargs = dict(
        lora_lr=float(opt.get("lora_lr", 1e-4)),
        auxiliary_lr=float(opt.get("auxiliary_lr", 1e-4)),
        lora_weight_decay=float(opt.get("lora_weight_decay", 0.01)),
        auxiliary_weight_decay=float(opt.get("auxiliary_weight_decay", 0.0)),
    )
    ema = rp.get("ema", {}) or {}
    ema_kwargs = dict(
        policy=str(ema.get("policy", "trainable_only")),
        decay=float(ema.get("decay", 0.9999)),
    )
    return cfg, optimizer_kwargs, ema_kwargs
