"""Channel gates for first-order Taylor pruning saliency (master plan §4).

A prunable output channel `c` of a Conv2d gets a multiplicative gate `g_c` (init
1.0): ``out = conv(x) * g``. The Taylor saliency of the channel under a loss `L`
is ``S_c = |g_c · dL/dg_c|`` evaluated at ``g_c = 1`` — i.e. the magnitude of the
gate gradient. Gating the *output* channel is equivalent to scoring the removal of
that channel (Diff-Pruning style), and keeps the score independent of how the
weight tensor is laid out.

This module only provides the gate mechanism and gradient plumbing. It computes
NO saliency on the real pruned checkpoint; that is an M3B/M4 scientific run, which
stays blocked until the pilot protocol is frozen. Tested on control models only.
"""
from typing import Dict, List

import torch
from torch import nn


class ChannelGate(nn.Module):
    """Wrap a Conv2d with a per-output-channel multiplicative gate (init 1.0)."""

    def __init__(self, base: nn.Conv2d):
        super().__init__()
        if not isinstance(base, nn.Conv2d):
            raise TypeError("ChannelGate wraps nn.Conv2d only")
        self.base = base
        self.gate = nn.Parameter(torch.ones(base.out_channels))

    def forward(self, x):
        return self.base(x) * self.gate.view(1, -1, 1, 1)


def _get_parent(root: nn.Module, dotted: str):
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def attach_gates(model: nn.Module, layer_names: List[str]) -> Dict[str, ChannelGate]:
    """Replace each named Conv2d in `model` with a ChannelGate wrapper, in place.

    `layer_names` are dotted module paths (relative to `model`) that must resolve
    to nn.Conv2d. Returns {name: ChannelGate}. Raises if a name is missing or not
    a Conv2d, so a mis-specified prunable-layer set fails loudly rather than
    silently scoring nothing.
    """
    gates: Dict[str, ChannelGate] = {}
    for name in layer_names:
        parent, attr = _get_parent(model, name)
        module = getattr(parent, attr)
        if isinstance(module, ChannelGate):
            gates[name] = module
            continue
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"{name} is {type(module).__name__}, expected nn.Conv2d")
        wrapper = ChannelGate(module)
        setattr(parent, attr, wrapper)
        gates[name] = wrapper
    return gates


def remove_gates(model: nn.Module, gates: Dict[str, ChannelGate]) -> None:
    """Restore the bare Conv2d for each gated layer (inverse of attach_gates)."""
    for name, gate in gates.items():
        parent, attr = _get_parent(model, name)
        if isinstance(getattr(parent, attr), ChannelGate):
            setattr(parent, attr, gate.base)


def zero_gate_grads(gates: Dict[str, ChannelGate]) -> None:
    for g in gates.values():
        g.gate.grad = None


def conv_modules(gates: Dict[str, ChannelGate]) -> Dict[str, nn.Conv2d]:
    """The underlying Conv2d of each gated layer (for data-free P0 L1 magnitude)."""
    return {name: g.base for name, g in gates.items()}
