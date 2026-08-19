#!/usr/bin/env python3
"""Verify the ADOPTED P0 convention reproduces Arshdeep's published L1 kept-set.

Finding M3B-002 established that the published PruningAudioLDM L1 checkpoint keeps
the LOWEST-L1 filters per pruned layer. Gabriel's decision (2026-08-19): because
RQ2's L1 baseline IS that official artifact, the project's P0 adopts the "published"
convention (keep lowest-L1). `research_pruning.taylor.p0_importance` implements it as
importance = -L1, so `keep_topk` keeps the low-L1 channels.

This script proves the adopted convention is faithful: on the real base (1,2,3,5)
U-Net, for every ranking-driven pruned layer, the set `keep_topk(p0_importance
('published'), k)` EQUALS the published kept-set `ranking[:k]` (exact set equality),
and the 'standard' convention is its complement-direction (disjoint from the published
kept-set wherever k < full/2). Machinery/baseline validation only: data-free P0 on the
base weights vs the public ranking artifact. NO diagnostics, NO calibration slots, NO
saliency on the L1 checkpoint.

    .venv/bin/python scripts/research/verify_p0_convention.py
"""
from __future__ import annotations

import copy
import sys

import torch
import yaml

from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
from research_pruning.diagnostics.random_masks import (
    load_l1_ranking, kept_counts, ranking_driven_layers, kept_sets,
)
from research_pruning.taylor import (
    l1_prunable_layer_names, p0_importance, keep_topk, attach_gates, conv_modules,
)

CONFIG = "audioldm_train/config/2023_08_23_reproduce_audioldm/audioldm_original_medium.yaml"
BASE_CKPT = "data/checkpoints/audioldm-m-full.ckpt"
RANKING_PKL = "artifacts/m0_baseline_reproduction/sorted_indexes_dict.pkl"


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> int:
    with open(CONFIG) as handle:
        cfg = yaml.safe_load(handle)
    params = copy.deepcopy(cfg["model"]["params"]["unet_config"]["params"])
    params["channel_mult"] = [1, 2, 3, 5]
    unet = UNetModel(**params)
    sd = _torch_load(BASE_CKPT)
    sd = sd.get("state_dict", sd)
    pref = "model.diffusion_model."
    unet.load_state_dict({k[len(pref):]: v for k, v in sd.items() if k.startswith(pref)}, strict=True)

    ranking = load_l1_ranking(RANKING_PKL)
    names = l1_prunable_layer_names(ranking)
    name_of = dict(zip(ranking, names))
    gates = attach_gates(unet, names)
    convs = conv_modules(gates)

    counts = kept_counts(cfg, list(ranking.keys()))
    driven = ranking_driven_layers(cfg, ranking)          # layers where the ranking selects rows
    published = kept_sets(ranking, counts, layers=driven)  # frozenset(ranking[k][:k]) = lowest-L1

    imp_pub = p0_importance(convs, "published")
    imp_std = p0_importance(convs, "standard")
    k_by_name = {name_of[k]: counts[k] for k in driven}
    keep_pub = keep_topk({name_of[k]: imp_pub[name_of[k]] for k in driven}, k_by_name)
    keep_std = keep_topk({name_of[k]: imp_std[name_of[k]] for k in driven}, k_by_name)

    reproduced = 0
    std_disjoint = 0
    checkable_disjoint = 0
    for k in driven:
        name = name_of[k]
        pub_set = published[k]
        got_pub = frozenset(keep_pub[name].tolist())
        got_std = frozenset(keep_std[name].tolist())
        if got_pub == pub_set:
            reproduced += 1
        full = len(ranking[k])
        # standard keeps highest-L1; where fewer than half are kept, it cannot overlap
        # the lowest-L1 published kept-set.
        if counts[k] <= full - counts[k]:
            checkable_disjoint += 1
            if got_std.isdisjoint(pub_set):
                std_disjoint += 1

    n = len(driven)
    print(f"ranking-driven pruned layers: {n}")
    print(f"P0 'published' reproduces the published kept-set EXACTLY: {reproduced}/{n}")
    print(f"P0 'standard' kept-set disjoint from published (where k<=full/2): "
          f"{std_disjoint}/{checkable_disjoint}")
    ok = reproduced == n and n > 0 and std_disjoint == checkable_disjoint
    print(f"\nVERDICT: adopted P0 convention reproduces Arshdeep's published L1 baseline: "
          f"{'CONFIRMED' if ok else 'NOT confirmed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
