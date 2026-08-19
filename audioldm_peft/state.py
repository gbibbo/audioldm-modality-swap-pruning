"""Adapter and full-resume state persistence.

Adopted from the recovered M1 scaffold; audited in ``docs/m1_scaffold_audit.md``.
Fix F7: ``adaptation_state_dict`` saves adapter parameters only, which is correct
for deployment but insufficient to *resume training*. ``training_state_dict`` /
``load_training_state_dict`` bundle the full resumable state — adapter parameters,
optimizer, scheduler, EMA, and global step — which the M1 acceptance criterion
("full resume state") requires.
"""
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


def adaptation_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
    state = model.state_dict()
    # Parameters only. Buffers/EMA belong in the resumable training checkpoint.
    return {k: v.detach().cpu().clone() for k, v in state.items() if k in trainable_names}


def load_adaptation_state_dict(model: nn.Module, state: Mapping[str, torch.Tensor], strict: bool = True):
    current = model.state_dict()
    missing = []
    for key, value in state.items():
        if key not in current:
            if strict:
                raise KeyError(f"Adapter key not found in model: {key}")
            continue
        if current[key].shape != value.shape:
            raise ValueError(f"Shape mismatch for {key}: model={current[key].shape}, adapter={value.shape}")
        current[key].copy_(value.to(device=current[key].device, dtype=current[key].dtype))

    if strict:
        expected = {name for name, p in model.named_parameters() if p.requires_grad}
        missing = sorted(expected.difference(state.keys()))
        if missing:
            raise KeyError(f"Missing trainable keys in adapter state: {missing[:10]}")
    return missing


def training_state_dict(model: nn.Module,
                        optimizer: Optional[torch.optim.Optimizer] = None,
                        scheduler: Optional[Any] = None,
                        ema: Optional[nn.Module] = None,
                        global_step: int = 0,
                        extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Full resumable training state (F7).

    ``adapter`` is the trainable-parameter state (CPU tensors). ``optimizer`` /
    ``scheduler`` / ``ema`` are ``state_dict()`` payloads when supplied. ``extra``
    carries anything the caller needs (e.g. RNG state, config hash).
    """
    return {
        "adapter": adaptation_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "global_step": int(global_step),
        "extra": dict(extra) if extra else {},
    }


def load_training_state_dict(model: nn.Module,
                             state: Mapping[str, Any],
                             optimizer: Optional[torch.optim.Optimizer] = None,
                             scheduler: Optional[Any] = None,
                             ema: Optional[nn.Module] = None,
                             strict: bool = True) -> int:
    """Restore full training state (F7). Returns the restored ``global_step``.

    The model must already have PEFT injected/frozen identically to the run that
    produced ``state`` (same trainable-parameter set), and ``ema`` must have been
    reconstructed on that same model so its shadow-buffer keys line up.
    """
    load_adaptation_state_dict(model, state["adapter"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    return int(state.get("global_step", 0))
