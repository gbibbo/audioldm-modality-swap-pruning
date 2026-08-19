"""Trainable-only EMA for PEFT recovery.

Adopted verbatim from the recovered M1 scaffold; audited in
``docs/m1_scaffold_audit.md``. Tracks only the parameters that are trainable when
the object is constructed, so it never creates a full frozen-U-Net shadow. It is
an ``nn.Module`` whose shadow tensors are registered buffers, so ``state_dict()`` /
``load_state_dict()`` round-trip it for full-resume checkpoints (F7).
"""
from contextlib import contextmanager

import torch
from torch import nn


class TrainableOnlyEMA(nn.Module):
    """EMA that tracks only parameters trainable when the object is created."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        super().__init__()
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        self.decay = float(decay)
        self.name_to_buffer = {}
        self._stored = {}
        for i, (name, p) in enumerate(model.named_parameters()):
            if p.requires_grad:
                bname = f"shadow_{i}"
                self.name_to_buffer[name] = bname
                self.register_buffer(bname, p.detach().clone())
        if not self.name_to_buffer:
            raise RuntimeError("EMA initialized with no trainable parameters")

    def update(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, bname in self.name_to_buffer.items():
                p = params[name]
                shadow = getattr(self, bname)
                shadow.lerp_(p.detach().to(shadow.dtype), 1.0 - self.decay)

    def store(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        self._stored = {name: params[name].detach().clone() for name in self.name_to_buffer}

    def copy_to(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, bname in self.name_to_buffer.items():
                params[name].copy_(getattr(self, bname).to(params[name].dtype))

    def restore(self, model: nn.Module) -> None:
        params = dict(model.named_parameters())
        with torch.no_grad():
            for name, saved in self._stored.items():
                params[name].copy_(saved.to(params[name].dtype))
        self._stored = {}

    @contextmanager
    def scope(self, model: nn.Module):
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)
