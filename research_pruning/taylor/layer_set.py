"""The structure-matched prunable layer set for P0-P3 (finding 9.4).

For P1/P2/P3 to be structure-matched to the published L1 pruning, they must rank
channels over exactly the layer set the L1 method ranks — the 28 keys in the public
`sorted_indexes_dict.pkl`. This module derives the corresponding Conv2d module paths
and verifies them against the real U-Net.

It reads only the architecture (module names, channel counts) and the public L1
ranking. It computes NO saliency and NO P0-P3 ranking on the real model — those are
the M3B/M4 scientific run, blocked until the pilot protocol is frozen.
"""
from typing import Dict, List

from torch import nn

from research_pruning.diagnostics.random_masks import load_l1_ranking, ranking_full_lengths


def l1_prunable_layer_names(ranking: dict) -> List[str]:
    """Module paths (relative to the UNetModel) of the 28 L1-ranked prunable convs.

    The ranking keys are weight-tensor names (``...weight``); the gated module is the
    owning Conv2d, i.e. the key without the trailing ``.weight``.
    """
    names = []
    for key in ranking:
        names.append(key[:-len(".weight")] if key.endswith(".weight") else key)
    return names


def verify_prunable_layers(unet: nn.Module, ranking: dict) -> Dict[str, int]:
    """Check every L1-ranked layer resolves to a Conv2d whose out_channels equals the
    ranking's full length. Returns {module_path: out_channels}. Raises on any mismatch
    so a wrong layer set fails loudly rather than silently scoring the wrong channels.
    """
    modules = dict(unet.named_modules())
    lengths = ranking_full_lengths(ranking)
    out: Dict[str, int] = {}
    for key, name in zip(ranking, l1_prunable_layer_names(ranking)):
        mod = modules.get(name)
        if mod is None:
            raise KeyError(f"L1 layer {name!r} not found in the U-Net")
        if not isinstance(mod, nn.Conv2d):
            raise TypeError(f"L1 layer {name!r} is {type(mod).__name__}, expected Conv2d")
        if mod.out_channels != lengths[key]:
            raise ValueError(
                f"{name!r} out_channels {mod.out_channels} != L1 full length {lengths[key]}"
            )
        out[name] = mod.out_channels
    return out


def load_and_verify(unet: nn.Module, ranking_pkl: str) -> Dict[str, int]:
    """Convenience: load the public L1 ranking and verify it against `unet`."""
    return verify_prunable_layers(unet, load_l1_ranking(ranking_pkl))
