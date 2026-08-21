"""Block removal by identity substitution (protocol section 1.3, 10).

`ContinuousTransformer.forward` iterates `self.layers` (nn.ModuleList of residual
TransformerBlocks) and calls `layer(x, **layer_kwargs)`. Removing block g == replacing it by a
module that returns its input unchanged. `BlockMask` swaps the entries of `layers` in place
and restores them on exit; with an empty mask the model is bit-identical to the original.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Set

import torch.nn as nn


class IdentityBlock(nn.Module):
    """Returns x unchanged; accepts and ignores every keyword the real block receives."""

    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x, *args, **kwargs):
        return x


def find_layers(module: nn.Module) -> nn.ModuleList:
    """Locate the `transformer.layers` ModuleList inside a DiffusionTransformer / wrapper."""
    for name, sub in module.named_modules():
        if name.endswith("transformer.layers") or name == "layers":
            if isinstance(sub, nn.ModuleList):
                return sub
    raise AttributeError("no `transformer.layers` ModuleList found")


@contextmanager
def block_mask(module: nn.Module, skip: Iterable[int]):
    """Temporarily replace `transformer.layers[i]` by IdentityBlock for i in `skip`."""
    layers = find_layers(module)
    skip_set: Set[int] = set(int(i) for i in skip)
    bad = [i for i in skip_set if i < 0 or i >= len(layers)]
    if bad:
        raise IndexError(f"block indices out of range: {bad} (depth={len(layers)})")
    saved = {i: layers[i] for i in skip_set}
    try:
        for i in skip_set:
            layers[i] = IdentityBlock(i)
        yield module
    finally:
        for i, blk in saved.items():
            layers[i] = blk


def depth(module: nn.Module) -> int:
    return len(find_layers(module))
