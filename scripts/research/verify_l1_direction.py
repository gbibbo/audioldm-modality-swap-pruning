#!/usr/bin/env python3
"""Verify the pruning DIRECTION of the published PruningAudioLDM L1 baseline.

FINDING (four independent, mutually consistent lines of evidence): the published
pruned checkpoint keeps the k channels of LOWEST conv-filter L1 magnitude per pruned
layer and removes the highest — inverted from standard L1 magnitude pruning, which
keeps the highest-magnitude filters.

Evidence produced by this script:
  1. Rank correlation: the per-layer published ranking vs my P0 (per-output-channel
     conv-weight L1, descending) has Spearman == -1 on every layer (exact reversal).
  2. Kept-vs-pruned: on every ACTUALLY-pruned layer (k < full), the kept set
     (ranking[:k]) has LOWER mean L1 than the pruned set.
Reference code (ground truth, `_external/PruningAudioLDM/scripts/
layerwise_sorted_index_generation.py`): `l1_imp_index` = per-filter sum(|w|) (== this
P0); `sorted_idx = np.argsort(scores)` sorts ASCENDING; the frozen materializer
(random_masks.prune_with_indices) keeps `out_idx_full[:out_k]` and is bit-exact to the
published checkpoint (ledger M3-002, 690/690). Ascending argsort + keep[:k] => keep lowest.

This computes only P0 (data-free) on the base weights and compares to the public
ranking artifact. It is a machinery/baseline validation, NOT an M3 experiment: no
diagnostics, no calibration slots, no saliency on the L1 checkpoint.

    .venv/bin/python scripts/research/verify_l1_direction.py
"""
from __future__ import annotations

import copy
import sys

import numpy as np
import torch
import yaml

from audioldm_train.modules.diffusionmodules.openaimodel import UNetModel
from research_pruning.diagnostics.random_masks import load_l1_ranking, kept_counts
from research_pruning.taylor import (
    l1_prunable_layer_names, p0_l1_magnitude, attach_gates, conv_modules,
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
    gates = attach_gates(unet, names)
    p0 = p0_l1_magnitude(conv_modules(gates))
    kc = kept_counts(cfg, list(ranking.keys()))

    spearmans, pruned_layers, kept_lower = [], 0, 0
    for key, name in zip(ranking, names):
        pub = list(ranking[key])
        mine = torch.argsort(p0[name], descending=True).tolist()
        a = np.array([pub.index(c) for c in range(len(pub))])
        b = np.array([mine.index(c) for c in range(len(pub))])
        spearmans.append(float(np.corrcoef(a, b)[0, 1]))
        k = kc[key]
        if k < len(pub):
            pruned_layers += 1
            kept = float(p0[name][pub[:k]].mean())
            pruned = float(p0[name][pub[k:]].mean())
            if kept < pruned:
                kept_lower += 1

    mean_sp = float(np.mean(spearmans))
    print(f"layers: {len(ranking)}")
    print(f"mean Spearman(published ranking vs my P0 descending): {mean_sp:.6f}  (min {min(spearmans):.6f})")
    print(f"actually-pruned layers (k<full): {pruned_layers}")
    print(f"  of those, KEPT set has LOWER mean L1 than PRUNED set: {kept_lower}/{pruned_layers}")
    inverted = mean_sp < -0.999 and kept_lower == pruned_layers and pruned_layers > 0
    print(f"\nVERDICT: published L1 baseline keeps the LOWEST-magnitude filters "
          f"(inverted from standard L1): {'CONFIRMED' if inverted else 'NOT confirmed'}")
    print("Whether this is intentional or a direction bug in the reference is a question "
          "for Gabriel/Arshdeep; see docs/m0_baseline_reproduction/l1_pruning_direction_finding.md")
    return 0 if inverted else 1


if __name__ == "__main__":
    sys.exit(main())
